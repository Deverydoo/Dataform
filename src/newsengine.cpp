#include "newsengine.h"
#include "memorystore.h"
#include "thoughtengine.h"
#include "llmprovider.h"
#include "settingsmanager.h"
#include <QDebug>
#include <QSqlQuery>
#include <QSqlError>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QTimer>
#include <QNetworkRequest>

NewsEngine::NewsEngine(QObject *parent)
    : QObject(parent)
    , m_networkManager(new QNetworkAccessManager(this))
{
}

NewsEngine::~NewsEngine()
{
}

void NewsEngine::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void NewsEngine::setLLMProvider(LLMProviderManager *provider) {
    m_llmProvider = provider;
    if (provider) {
        connect(provider, &LLMProviderManager::backgroundResponseReceived,
                this, [this](const QString &owner, const QString &response) {
            if (owner == "NewsEngine") onLLMResponse(response);
        });
        connect(provider, &LLMProviderManager::backgroundErrorOccurred,
                this, [this](const QString &owner, const QString &error) {
            if (owner == "NewsEngine") onLLMError(error);
        });
    }
}
void NewsEngine::setThoughtEngine(ThoughtEngine *engine) { m_thoughtEngine = engine; }
void NewsEngine::setSettingsManager(SettingsManager *settings) { m_settingsManager = settings; }

// --- Idle window handling ---

void NewsEngine::onIdleWindowOpened()
{
    m_idleWindowOpen = true;
    m_paused = false;

    if (canStartCycle()) {
        startCycle();
    }
}

void NewsEngine::onIdleWindowClosed()
{
    m_idleWindowOpen = false;
    m_consecutiveErrors = 0;  // Fresh idle window gets a fresh chance

    if (m_isFetchingNews) {
        qDebug() << "NewsEngine: idle window closed, pausing after current operation";
        m_paused = true;
    }
}

// --- Cycle control ---

bool NewsEngine::canStartCycle() const
{
    if (!m_settingsManager || !m_settingsManager->newsEnabled()) return false;
    if (!m_memoryStore || !m_llmProvider || !m_thoughtEngine) return false;
    if (m_isFetchingNews) return false;
    if (m_paused) return false;

    // Error recovery: pause after too many consecutive errors
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) return false;

    // Privacy check
    if (m_settingsManager->privacyLevel() == "minimal") return false;

    // Daily limit
    if (m_cyclesCompletedToday >= m_settingsManager->maxNewsPerDay()) return false;

    // Cooldown
    if (m_lastCycleEndTime.isValid()) {
        qint64 elapsed = m_lastCycleEndTime.secsTo(QDateTime::currentDateTime());
        if (elapsed < CYCLE_COOLDOWN_SEC) return false;
    }

    return true;
}

void NewsEngine::startCycle()
{
    resetDailyCounterIfNeeded();

    if (!canStartCycle()) return;

    qDebug() << "NewsEngine: starting news cycle";
    m_isFetchingNews = true;
    emit isFetchingNewsChanged();

    // Reset cycle state
    m_allHeadlines.clear();
    m_newHeadlines.clear();
    m_selectedHeadlines.clear();
    m_currentHeadlineIndex = 0;

    // Prune old entries on first cycle of the day
    if (m_cyclesCompletedToday == 0) {
        pruneOldHeadlines();
    }

    setPhase(FetchHeadlines);
    phaseFetchHeadlines();
}

void NewsEngine::advancePhase()
{
    if (m_paused || !m_idleWindowOpen) {
        qDebug() << "NewsEngine: stopping (paused or idle window closed)";
        setPhase(Idle);
        return;
    }

    switch (m_phase) {
    case FetchHeadlines:
        setPhase(FilterNew);
        phaseFilterNew();
        break;
    case FilterNew:
        setPhase(SelectInteresting);
        phaseSelectInteresting();
        break;
    case SelectInteresting:
        setPhase(GenerateThought);
        phaseGenerateThought();
        break;
    case GenerateThought:
        setPhase(Store);
        phaseStore();
        break;
    case Store:
        // Cycle complete
        m_cyclesCompletedToday++;
        m_lastCycleEndTime = QDateTime::currentDateTime();
        emit cyclesCompletedTodayChanged();
        qDebug() << "NewsEngine: cycle complete. Cycles today:" << m_cyclesCompletedToday;
        setPhase(Idle);

        // Try another cycle if still idle
        if (m_idleWindowOpen && canStartCycle()) {
            QTimer::singleShot(10000, this, [this]() {
                if (m_idleWindowOpen && canStartCycle()) startCycle();
            });
        }
        break;
    default:
        break;
    }
}

void NewsEngine::setPhase(Phase phase)
{
    m_phase = phase;

    static const QStringList phaseNames = {
        "Idle", "Fetching headlines", "Filtering new",
        "Selecting interesting", "Generating thought", "Storing"
    };

    m_currentPhase = phaseNames.value(static_cast<int>(phase), "Idle");
    emit currentPhaseChanged();

    if (phase == Idle && m_isFetchingNews) {
        m_isFetchingNews = false;
        emit isFetchingNewsChanged();
    }
}

// --- Phase 1: Fetch Headlines ---

void NewsEngine::phaseFetchHeadlines()
{
    // Get feed URL from settings, cycling round-robin
    QStringList feeds;
    if (m_settingsManager) {
        feeds = m_settingsManager->newsFeeds();
    }
    if (feeds.isEmpty()) {
        feeds.append("https://www.allsides.com/rss/news");
    }

    if (m_currentFeedIndex >= feeds.size()) {
        m_currentFeedIndex = 0;
    }
    QString feedUrl = feeds[m_currentFeedIndex];
    m_currentFeedIndex = (m_currentFeedIndex + 1) % feeds.size();

    qDebug() << "NewsEngine: fetching RSS from" << feedUrl;

    QUrl url(feedUrl);
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::UserAgentHeader, "DATAFORM/1.0");
    request.setTransferTimeout(15000);

    QNetworkReply *reply = m_networkManager->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, feedUrl]() {
        reply->deleteLater();

        if (reply->error() != QNetworkReply::NoError) {
            qWarning() << "NewsEngine: RSS fetch error from" << feedUrl << ":" << reply->errorString();
            emit newsError(reply->errorString());
            setPhase(Idle);
            return;
        }

        QByteArray xml = reply->read(256 * 1024); // 256KB max for RSS
        m_allHeadlines = parseRssFeed(xml);

        qDebug() << "NewsEngine: parsed" << m_allHeadlines.size() << "headlines from" << feedUrl;

        if (m_allHeadlines.isEmpty()) {
            qDebug() << "NewsEngine: no headlines in RSS feed";
            setPhase(Idle);
            return;
        }

        advancePhase(); // -> FilterNew
    });
}

QList<NewsHeadline> NewsEngine::parseRssFeed(const QByteArray &xml) const
{
    QList<NewsHeadline> headlines;
    QXmlStreamReader reader(xml);
    QSet<QString> seenUrls;
    bool inItem = false;
    QString currentTitle, currentLink, currentDescription, currentPubDate;

    while (!reader.atEnd() && headlines.size() < MAX_HEADLINES_TO_PARSE) {
        reader.readNext();

        if (reader.isStartElement()) {
            if (reader.name() == u"item") {
                inItem = true;
                currentTitle.clear();
                currentLink.clear();
                currentDescription.clear();
                currentPubDate.clear();
            } else if (inItem) {
                if (reader.name() == u"title") {
                    currentTitle = reader.readElementText().trimmed();
                } else if (reader.name() == u"link") {
                    currentLink = reader.readElementText().trimmed();
                } else if (reader.name() == u"description") {
                    currentDescription = reader.readElementText().trimmed();
                    // Strip any HTML tags from description
                    currentDescription.remove(QRegularExpression("<[^>]*>"));
                } else if (reader.name() == u"pubDate") {
                    currentPubDate = reader.readElementText().trimmed();
                }
            }
        } else if (reader.isEndElement() && reader.name() == u"item") {
            inItem = false;

            if (currentTitle.length() < 10 || currentLink.isEmpty()) continue;
            if (seenUrls.contains(currentLink)) continue;
            seenUrls.insert(currentLink);

            NewsHeadline headline;
            headline.title = currentTitle;
            headline.url = currentLink;
            headline.description = currentDescription.left(300);
            headline.pubDate = QDateTime::fromString(currentPubDate, Qt::RFC2822Date);
            headlines.append(headline);
        }
    }

    if (reader.hasError()) {
        qWarning() << "NewsEngine: RSS parse error:" << reader.errorString();
    }

    return headlines;
}

// --- Phase 2: Filter New ---

void NewsEngine::phaseFilterNew()
{
    m_newHeadlines.clear();

    for (const NewsHeadline &h : m_allHeadlines) {
        if (!isHeadlineSeen(h.url)) {
            m_newHeadlines.append(h);
        }
    }

    qDebug() << "NewsEngine:" << m_newHeadlines.size()
             << "new headlines out of" << m_allHeadlines.size();

    if (m_newHeadlines.isEmpty()) {
        qDebug() << "NewsEngine: no new headlines, ending cycle";
        setPhase(Idle);
        return;
    }

    advancePhase(); // -> SelectInteresting
}

// --- Phase 3: Select Interesting (LLM) ---

void NewsEngine::phaseSelectInteresting()
{
    if (!m_llmProvider) {
        setPhase(Idle);
        return;
    }

    // Build trait context
    QString traitContext;
    if (m_memoryStore) {
        QList<TraitRecord> traits = m_memoryStore->getAllTraits();
        int count = 0;
        for (const TraitRecord &t : traits) {
            if (count >= 5) break;
            traitContext += "- " + t.statement + "\n";
            count++;
        }
    }

    // Build headline list for LLM (include descriptions for better selection)
    QString headlineList;
    int maxToShow = qMin(15, m_newHeadlines.size());
    for (int i = 0; i < maxToShow; ++i) {
        headlineList += QString("%1. %2\n").arg(i + 1).arg(m_newHeadlines[i].title);
        if (!m_newHeadlines[i].description.isEmpty()) {
            headlineList += QString("   %1\n").arg(m_newHeadlines[i].description.left(150));
        }
    }

    QString prompt = QString(
        "Headlines:\n%1\n"
        "User interests:\n%2\n"
        "Pick 1-2 headlines most interesting to this user.\n"
        "Output a JSON object. Example: {\"selected\":[1,5]}\n"
        "IMPORTANT: Output ONLY the JSON object starting with { and ending with }. No other text."
    ).arg(headlineList,
          traitContext.isEmpty() ? "(no traits known yet)" : traitContext);

    QJsonArray msgs;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    msgs.append(msg);
    m_llmProvider->sendBackgroundRequest("NewsEngine",
        "You are a JSON-only API. You output valid JSON and nothing else. Only JSON.",
        msgs, LLMProviderManager::PriorityLow);
}

// --- Phase 4: Generate Thought (LLM) ---

void NewsEngine::phaseGenerateThought()
{
    if (m_selectedHeadlines.isEmpty() ||
        m_currentHeadlineIndex >= m_selectedHeadlines.size()) {
        advancePhase(); // -> Store
        return;
    }

    const NewsHeadline &headline = m_selectedHeadlines[m_currentHeadlineIndex];

    QString traitContext;
    if (m_memoryStore) {
        QList<TraitRecord> traits = m_memoryStore->getAllTraits();
        int count = 0;
        for (const TraitRecord &t : traits) {
            if (count >= 3) break;
            traitContext += "- " + t.statement + "\n";
            count++;
        }
    }

    QString prompt = QString(
        "News headline: \"%1\"\n"
        "Source: RSS news feed\n"
        "Known user interests:\n%2\n\n"
        "Write a brief, conversational opener (2-3 sentences) to start a "
        "discussion about this headline with the user. Be curious and "
        "inviting, not preachy. Ask the user what they think.\n"
        "Respond ONLY with the conversation opener text, nothing else."
    ).arg(headline.title,
          traitContext.isEmpty() ? "(none yet)" : traitContext);

    m_llmProvider->sendBackgroundPrompt("NewsEngine", prompt, LLMProviderManager::PriorityLow);
}

// --- Phase 5: Store ---

void NewsEngine::phaseStore()
{
    // All thought insertion happens in onLLMResponse for GenerateThought.
    // This just advances to cycle completion.
    advancePhase();
}

// --- LLM response handling ---

void NewsEngine::onLLMResponse(const QString &response)
{
    m_consecutiveErrors = 0;  // Reset on any successful LLM response

    // Strip <think>...</think> reasoning blocks (qwen3 and other reasoning models)
    static const QRegularExpression thinkRegex(
        "<think>.*?</think>",
        QRegularExpression::DotMatchesEverythingOption);
    QString cleanedResponse = response;
    cleanedResponse.remove(thinkRegex);
    cleanedResponse = cleanedResponse.trimmed();

    if (m_phase == SelectInteresting) {
        // Parse JSON to get selected headline indices
        m_selectedHeadlines.clear();

        QString jsonStr = cleanedResponse;
        // Extract JSON if wrapped in markdown code block
        static QRegularExpression jsonExtract("\\{[^}]*\"selected\"[^}]*\\}");
        QRegularExpressionMatch match = jsonExtract.match(jsonStr);
        if (match.hasMatch()) {
            jsonStr = match.captured(0);
        }

        QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8());
        if (doc.isObject()) {
            QJsonArray selected = doc.object().value("selected").toArray();
            for (const QJsonValue &v : selected) {
                int idx = v.toInt() - 1; // 1-indexed -> 0-indexed
                if (idx >= 0 && idx < m_newHeadlines.size()) {
                    m_selectedHeadlines.append(m_newHeadlines[idx]);
                }
            }
        }

        // Fallback: if parsing failed, just take the first headline
        if (m_selectedHeadlines.isEmpty() && !m_newHeadlines.isEmpty()) {
            m_selectedHeadlines.append(m_newHeadlines.first());
        }

        // Cap at 2 headlines per cycle
        while (m_selectedHeadlines.size() > 2) {
            m_selectedHeadlines.removeLast();
        }

        qDebug() << "NewsEngine: selected" << m_selectedHeadlines.size() << "headlines";
        m_currentHeadlineIndex = 0;
        advancePhase(); // -> GenerateThought

    } else if (m_phase == GenerateThought) {
        // Store the thought and mark headline as seen
        if (m_currentHeadlineIndex < m_selectedHeadlines.size() && m_thoughtEngine) {
            const NewsHeadline &headline = m_selectedHeadlines[m_currentHeadlineIndex];

            ThoughtRecord record;
            record.type = "news_insight";
            record.sourceType = "news";
            record.priority = 0.7;
            record.title = headline.title;
            record.content = cleanedResponse;
            record.generatedBy = "news_cycle";

            m_thoughtEngine->insertThought(record);
            markHeadlineSeen(headline.url);

            emit newsCycleComplete(headline.title, headline.url);
            qDebug() << "NewsEngine: created thought for headline:" << headline.title;
        }

        m_currentHeadlineIndex++;

        // Process next selected headline, or advance to Store
        if (m_currentHeadlineIndex < m_selectedHeadlines.size()) {
            phaseGenerateThought();
        } else {
            advancePhase(); // -> Store
        }
    }
}

void NewsEngine::onLLMError(const QString &error)
{
    m_consecutiveErrors++;
    qWarning() << "NewsEngine: LLM error:" << error;
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        qWarning() << "NewsEngine: pausing after" << m_consecutiveErrors << "consecutive errors";
    }
    emit newsError(error);

    if (m_phase == SelectInteresting) {
        // Fallback: take first new headline
        m_selectedHeadlines.clear();
        if (!m_newHeadlines.isEmpty()) {
            m_selectedHeadlines.append(m_newHeadlines.first());
        }
        m_currentHeadlineIndex = 0;
        advancePhase(); // -> GenerateThought

    } else if (m_phase == GenerateThought) {
        // Skip LLM formatting, use headline title directly
        if (m_currentHeadlineIndex < m_selectedHeadlines.size() && m_thoughtEngine) {
            const NewsHeadline &headline = m_selectedHeadlines[m_currentHeadlineIndex];

            ThoughtRecord record;
            record.type = "news_insight";
            record.sourceType = "news";
            record.priority = 0.6;
            record.title = headline.title;
            record.content = QString("I noticed this headline and thought you might find it "
                                     "interesting: \"%1\". What do you think?").arg(headline.title);
            record.generatedBy = "news_cycle";

            m_thoughtEngine->insertThought(record);
            markHeadlineSeen(headline.url);
        }

        m_currentHeadlineIndex++;
        if (m_currentHeadlineIndex < m_selectedHeadlines.size()) {
            phaseGenerateThought();
        } else {
            advancePhase(); // -> Store
        }

    } else {
        setPhase(Idle);
    }
}

// --- DB helpers ---

bool NewsEngine::isHeadlineSeen(const QString &url) const
{
    if (!m_memoryStore) return false;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("SELECT COUNT(*) FROM news_seen WHERE url = :url");
    q.bindValue(":url", url);

    if (q.exec() && q.next()) {
        return q.value(0).toInt() > 0;
    }
    return false;
}

void NewsEngine::markHeadlineSeen(const QString &url)
{
    if (!m_memoryStore) return;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("INSERT OR IGNORE INTO news_seen (url) VALUES (:url)");
    q.bindValue(":url", url);
    q.exec();
}

void NewsEngine::pruneOldHeadlines()
{
    if (!m_memoryStore) return;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.exec("DELETE FROM news_seen WHERE seen_ts < datetime('now', '-30 days')");
    int pruned = q.numRowsAffected();
    if (pruned > 0) {
        qDebug() << "NewsEngine: pruned" << pruned << "old headline entries";
    }
}

// --- Daily tracking ---

void NewsEngine::resetDailyCounterIfNeeded()
{
    QDate today = QDate::currentDate();
    if (m_lastCycleDate != today) {
        m_cyclesCompletedToday = 0;
        m_lastCycleDate = today;
        emit cyclesCompletedTodayChanged();
    }
}
