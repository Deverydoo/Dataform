#include "researchengine.h"
#include "fuzzylogic.h"
#include "llmresponseparser.h"
#include "memorystore.h"
#include "llmprovider.h"
#include "whyengine.h"
#include "settingsmanager.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QRandomGenerator>

ResearchEngine::ResearchEngine(QObject *parent)
    : QObject(parent)
{
}

ResearchEngine::~ResearchEngine()
{
}

void ResearchEngine::setWebSearchEngine(WebSearchEngine *engine)
{
    m_webSearch = engine;
    if (m_webSearch) {
        connect(m_webSearch, &WebSearchEngine::searchResultsReady,
                this, &ResearchEngine::onSearchResultsReady);
        connect(m_webSearch, &WebSearchEngine::pageContentReady,
                this, &ResearchEngine::onPageContentReady);
        connect(m_webSearch, &WebSearchEngine::searchError,
                this, [this](const QString &err) {
            qWarning() << "ResearchEngine: search error:" << err;
            emit researchError(err);
            setPhase(Idle);
        });
    }
}

void ResearchEngine::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void ResearchEngine::setResearchStore(ResearchStore *store) { m_researchStore = store; }
void ResearchEngine::setLLMProvider(LLMProviderManager *provider) {
    m_llmProvider = provider;
    if (provider) {
        connect(provider, &LLMProviderManager::backgroundResponseReceived,
                this, [this](const QString &owner, const QString &response) {
            if (owner == "ResearchEngine") onLLMResponse(response);
        });
        connect(provider, &LLMProviderManager::backgroundErrorOccurred,
                this, [this](const QString &owner, const QString &error) {
            if (owner == "ResearchEngine") onLLMError(error);
        });
    }
}
void ResearchEngine::setWhyEngine(WhyEngine *engine) { m_whyEngine = engine; }
void ResearchEngine::setSettingsManager(SettingsManager *settings) { m_settingsManager = settings; }

// --- Idle window handling ---

void ResearchEngine::onIdleWindowOpened()
{
    m_idleWindowOpen = true;
    m_paused = false;
    // Auto-start removed — coordinator calls requestStart()
}

void ResearchEngine::requestStart()
{
    if (canStartCycle()) startCycle();
    else emit cycleFinished();
}

void ResearchEngine::onIdleWindowClosed()
{
    m_idleWindowOpen = false;
    m_consecutiveErrors = 0;  // Fresh idle window gets a fresh chance

    if (m_isResearching) {
        qDebug() << "ResearchEngine: idle window closed, pausing after current operation";
        m_paused = true;
        // Don't abort mid-request; the current network/LLM op will complete
        // but advancePhase() will check m_paused and stop
    }
}

void ResearchEngine::pauseResearch()
{
    m_paused = true;
    if (m_isResearching) {
        qDebug() << "ResearchEngine: pausing research";
        setPhase(Idle);
    }
}

// --- Cycle control ---

bool ResearchEngine::canStartCycle() const
{
    if (!m_settingsManager || !m_settingsManager->researchEnabled()) return false;
    if (!m_webSearch || !m_memoryStore || !m_researchStore || !m_llmProvider) return false;
    if (m_isResearching) return false;
    if (m_paused) return false;

    // Error recovery: pause after too many consecutive errors
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) return false;

    // Privacy check
    if (m_settingsManager->privacyLevel() == "minimal") return false;

    // Daily limit
    if (m_cyclesCompletedToday >= m_settingsManager->maxResearchPerDay()) return false;

    // Soft cooldown centered at 300s (5 min)
    if (m_lastCycleEndTime.isValid()) {
        qint64 elapsed = m_lastCycleEndTime.secsTo(QDateTime::currentDateTime());
        double cooldownReady = Fuzzy::sigmoid(static_cast<double>(elapsed), 300.0, 60.0);
        if (QRandomGenerator::global()->generateDouble() > cooldownReady) return false;
    }

    return true;
}

void ResearchEngine::startCycle()
{
    resetDailyCounterIfNeeded();

    if (!canStartCycle()) return;

    qDebug() << "ResearchEngine: starting research cycle";
    m_isResearching = true;
    emit isResearchingChanged();

    // Reset cycle state
    m_currentResults.clear();
    m_fetchedPages.clear();
    m_currentQueries.clear();
    m_currentQueryIndex = 0;
    m_currentPageIndex = 0;
    m_currentFindingsCount = 0;

    setPhase(SelectTopic);
    phaseSelectTopic();
}

void ResearchEngine::advancePhase()
{
    if (m_paused || !m_idleWindowOpen) {
        qDebug() << "ResearchEngine: stopping (paused or idle window closed)";
        setPhase(Idle);
        return;
    }

    switch (m_phase) {
    case SelectTopic:
        setPhase(GenerateQuery);
        phaseGenerateQuery();
        break;
    case GenerateQuery:
        setPhase(ExecuteSearch);
        phaseExecuteSearch();
        break;
    case ExecuteSearch:
        setPhase(FetchAndSummarize);
        phaseFetchAndSummarize();
        break;
    case FetchAndSummarize:
        setPhase(Store);
        phaseStore();
        break;
    case Store:
        // Cycle complete
        m_cyclesCompletedToday++;
        m_lastCycleEndTime = QDateTime::currentDateTime();
        emit cyclesCompletedTodayChanged();
        emit researchCycleComplete(m_currentTopic, m_currentFindingsCount);
        qDebug() << "ResearchEngine: cycle complete. Topic:" << m_currentTopic
                 << "Findings:" << m_currentFindingsCount
                 << "Cycles today:" << m_cyclesCompletedToday;
        setPhase(Idle);
        // Auto-retry removed — coordinator handles scheduling
        break;
    default:
        break;
    }
}

void ResearchEngine::setPhase(Phase phase)
{
    Phase old = m_phase;
    m_phase = phase;

    static const QStringList phaseNames = {
        "Idle", "Selecting topic", "Generating query",
        "Searching", "Analyzing pages", "Storing findings"
    };

    m_currentPhase = phaseNames.value(static_cast<int>(phase), "Idle");
    emit currentPhaseChanged();

    if (phase == Idle && m_isResearching) {
        m_isResearching = false;
        emit isResearchingChanged();
    }

    if (phase == Idle && old != Idle) {
        emit cycleFinished();
    }
}

// --- Phase 1: Select Topic ---

void ResearchEngine::phaseSelectTopic()
{
    buildTopicQueue();

    if (m_topicQueue.isEmpty()) {
        qDebug() << "ResearchEngine: no topics to research";
        setPhase(Idle);
        return;
    }

    // Pick the highest-priority topic not researched recently
    for (int i = 0; i < m_topicQueue.size(); ++i) {
        const QString &topic = m_topicQueue[i].topic;
        if (!m_researchStore->hasRecentResearch(topic, 72)) {
            m_currentTopic = topic;
            m_topicQueue.removeAt(i);
            emit currentTopicChanged();
            emit topicQueueChanged();
            qDebug() << "ResearchEngine: selected topic:" << m_currentTopic;
            advancePhase();
            return;
        }
    }

    // All topics researched recently — use the highest priority anyway
    m_currentTopic = m_topicQueue.first().topic;
    m_topicQueue.removeFirst();
    emit currentTopicChanged();
    emit topicQueueChanged();
    qDebug() << "ResearchEngine: selected topic (all recent):" << m_currentTopic;
    advancePhase();
}

void ResearchEngine::buildTopicQueue()
{
    m_topicQueue.clear();

    // Source 1: Novel topics from WhyEngine
    if (m_whyEngine) {
        QStringList novelTopics = m_whyEngine->getNovelTopicsForResearch();
        for (const QString &t : novelTopics) {
            m_topicQueue.append({t, 0.8});
        }
    }

    // Source 2: Traits with low evidence count
    if (m_memoryStore) {
        QList<TraitRecord> traits = m_memoryStore->getAllTraits();
        for (const TraitRecord &t : traits) {
            if (t.evidenceEpisodeIds.size() <= 2 && t.confidence >= 0.3) {
                // Extract key phrase from trait statement
                QString keywords = t.statement;
                if (keywords.length() > 60) keywords = keywords.left(60);
                m_topicQueue.append({keywords, 0.6});
            }
        }
    }

    // Source 3: Recent conversation keywords from WhyEngine
    if (m_whyEngine) {
        QStringList recentKw = m_whyEngine->recentTopicKeywords();
        if (!recentKw.isEmpty()) {
            QString topicPhrase = recentKw.mid(0, qMin(5, recentKw.size())).join(' ');
            m_topicQueue.append({topicPhrase, 0.5});
        }
    }

    // Source 4: Recent episode keywords (fallback when other sources are empty)
    if (m_topicQueue.isEmpty() && m_memoryStore) {
        QList<EpisodicRecord> recentEps = m_memoryStore->getRecentEpisodes(5);
        for (const EpisodicRecord &ep : recentEps) {
            if (ep.userText.isEmpty()) continue;
            QStringList words = ep.userText.toLower().split(
                QRegularExpression("\\W+"), Qt::SkipEmptyParts);
            QStringList keywords;
            for (const QString &w : words) {
                if (w.length() > 5) keywords.append(w);
            }
            if (keywords.size() >= 2) {
                QString phrase = keywords.mid(0, qMin(4, keywords.size())).join(' ');
                m_topicQueue.append({phrase, 0.4});
                if (m_topicQueue.size() >= 3) break;
            }
        }
    }

    // Sort by priority (highest first)
    std::sort(m_topicQueue.begin(), m_topicQueue.end(),
              [](const TopicEntry &a, const TopicEntry &b) {
        return a.priority > b.priority;
    });

    // Cap
    while (m_topicQueue.size() > 20) {
        m_topicQueue.removeLast();
    }

    emit topicQueueChanged();
    qDebug() << "ResearchEngine: topic queue built with" << m_topicQueue.size() << "entries";
}

// --- Phase 2: Generate Search Query ---

void ResearchEngine::phaseGenerateQuery()
{
    if (!m_llmProvider) {
        setPhase(Idle);
        return;
    }

    // Build a prompt to generate search queries
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

    // Graduated lean balancing: light nudge at |0.3|, stronger at |0.6|
    QString leanHint;
    if (m_memoryStore) {
        QVariantMap lean = m_memoryStore->getLatestLeanForQml();
        if (!lean.isEmpty()) {
            double score = lean["leanScore"].toDouble();
            double absScore = qAbs(score);
            QString direction = score < 0 ? "left" : "right";
            QString opposite = score < 0 ? "center or right-of-center" : "center or left-of-center";
            if (absScore > 0.6) {
                leanHint = QString(
                    "The user's views lean strongly %1. Ensure at least one query "
                    "explores %2 perspectives on this topic to broaden understanding.\n"
                ).arg(direction, opposite);
            } else if (absScore > 0.3) {
                leanHint = QString(
                    "The user's views lean slightly %1. When relevant, consider "
                    "%2 angles on this topic for balance.\n"
                ).arg(direction, opposite);
            }
        }
    }

    QString prompt = QString(
        "Given user interest: \"%1\"\n"
        "Known traits:\n%2\n%3"
        "Generate 1-2 DuckDuckGo search queries to learn more about this topic "
        "from the user's perspective. Return ONLY the queries, one per line. "
        "No numbering, no explanation."
    ).arg(m_currentTopic,
          traitContext.isEmpty() ? "(none yet)" : traitContext,
          leanHint);

    m_llmProvider->sendBackgroundPrompt("ResearchEngine", prompt, LLMProviderManager::PriorityNormal);
}

// --- Phase 3: Execute Search ---

void ResearchEngine::phaseExecuteSearch()
{
    if (m_currentQueries.isEmpty()) {
        qDebug() << "ResearchEngine: no queries generated, skipping";
        setPhase(Idle);
        return;
    }

    m_currentQueryIndex = 0;
    m_currentResults.clear();

    if (m_webSearch->canSearch()) {
        qDebug() << "ResearchEngine: searching:" << m_currentQueries.first();
        m_webSearch->search(m_currentQueries.first(), 5);
    } else {
        // Wait for rate limiter — capture query now in case state changes before timer fires
        QString queryToUse = m_currentQueries.first();
        QTimer::singleShot(11000, this, [this, queryToUse]() {
            if (m_paused || !m_idleWindowOpen) {
                setPhase(Idle);
                return;
            }
            if (m_webSearch && m_webSearch->canSearch()) {
                m_webSearch->search(queryToUse, 5);
            } else {
                setPhase(Idle);
            }
        });
    }
}

// --- Phase 4: Fetch + Summarize ---

void ResearchEngine::phaseFetchAndSummarize()
{
    if (m_currentResults.isEmpty()) {
        qDebug() << "ResearchEngine: no search results, ending cycle";
        // Skip to store (which will just finish with 0 findings)
        advancePhase();
        return;
    }

    // Fetch up to 3 pages
    m_fetchedPages.clear();
    m_currentPageIndex = 0;

    int maxPages = qMin(3, m_currentResults.size());
    if (maxPages > 0) {
        m_webSearch->fetchPage(m_currentResults[0].url);
    } else {
        advancePhase();
    }
}

// --- Phase 5: Store ---

void ResearchEngine::phaseStore()
{
    // All findings have been stored during onLLMResponse in the summarize phase
    // Just advance to complete the cycle
    advancePhase();
}

// --- Signal handlers ---

void ResearchEngine::onSearchResultsReady(const QList<SearchResult> &results)
{
    if (m_phase != ExecuteSearch) return;

    m_currentResults = results;
    qDebug() << "ResearchEngine: got" << results.size() << "results for" << m_currentTopic;

    // If we have more queries and few results, search the next query
    m_currentQueryIndex++;
    if (m_currentResults.size() < 3 && m_currentQueryIndex < m_currentQueries.size()) {
        if (m_webSearch->canSearch()) {
            m_webSearch->search(m_currentQueries[m_currentQueryIndex], 5);
        } else {
            // Proceed with what we have
            advancePhase();
        }
    } else {
        // Deduplicate by URL
        QStringList seenUrls;
        QList<SearchResult> unique;
        for (const SearchResult &r : m_currentResults) {
            if (!seenUrls.contains(r.url)) {
                seenUrls.append(r.url);
                unique.append(r);
            }
        }
        m_currentResults = unique;
        advancePhase();
    }
}

void ResearchEngine::onPageContentReady(const PageContent &content)
{
    if (m_phase != FetchAndSummarize) return;

    m_fetchedPages.append(content);
    qDebug() << "ResearchEngine: fetched page" << (m_fetchedPages.size())
             << "of" << qMin(3, m_currentResults.size());

    int maxPages = qMin(3, m_currentResults.size());

    if (m_fetchedPages.size() < maxPages) {
        // Fetch next page
        int nextIdx = m_fetchedPages.size();
        m_webSearch->fetchPage(m_currentResults[nextIdx].url);
    } else {
        // All pages fetched — now summarize each with LLM
        m_currentPageIndex = 0;

        if (m_fetchedPages.isEmpty()) {
            advancePhase();
            return;
        }

        // Summarize first page
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

        // Truncate page text to avoid overwhelming the LLM
        QString pageText = m_fetchedPages[0].plainText;
        if (pageText.length() > 3000) pageText = pageText.left(3000);

        QString prompt = QString(
            "User interested in: \"%1\"\nTraits:\n%2\n"
            "Content from %3:\n\n%4\n\n"
            "Output a JSON object. Example: {\"summary\":\"Brief summary here\",\"relevance\":0.8,\"reason\":\"Why relevant\"}\n"
            "IMPORTANT: Output ONLY the JSON object starting with { and ending with }. No other text."
        ).arg(m_currentTopic,
              traitContext.isEmpty() ? "(none)" : traitContext,
              m_fetchedPages[0].url,
              pageText);

        QJsonArray msgs;
        QJsonObject msg;
        msg["role"] = "user";
        msg["content"] = prompt;
        msgs.append(msg);
        m_llmProvider->sendBackgroundRequest("ResearchEngine",
            "You are a JSON-only API. You output valid JSON and nothing else. Only JSON.",
            msgs, LLMProviderManager::PriorityNormal);
    }
}

void ResearchEngine::onLLMResponse(const QString &response)
{
    m_consecutiveErrors = 0;  // Reset on any successful LLM response

    QString cleanedResponse = LLMResponseParser::stripThinkTags(response);

    if (m_phase == GenerateQuery) {
        // Parse queries from response
        m_currentQueries.clear();
        QStringList lines = cleanedResponse.split('\n', Qt::SkipEmptyParts);
        for (QString line : lines) {
            line = line.trimmed();
            // Strip numbering like "1. " or "- "
            line.remove(QRegularExpression("^[\\d]+[\\.\\)\\-]\\s*"));
            line.remove(QRegularExpression("^[-*]\\s*"));
            line = line.trimmed();
            if (!line.isEmpty() && line.length() > 5) {
                m_currentQueries.append(line);
            }
        }

        // Cap at 2 queries
        while (m_currentQueries.size() > 2) m_currentQueries.removeLast();

        qDebug() << "ResearchEngine: generated queries:" << m_currentQueries;
        advancePhase();

    } else if (m_phase == FetchAndSummarize) {
        // Parse JSON summary
        QJsonDocument doc = LLMResponseParser::extractJsonObject(cleanedResponse, "ResearchEngine");
        if (!doc.isNull()) {
            QJsonObject obj = doc.object();
            double relevance = obj.value("relevance").toDouble(0.0);

            // Soft storage threshold centered at 0.4
            double storeProbability = Fuzzy::sigmoid(relevance, 0.4, 0.08);
            bool shouldStore = (QRandomGenerator::global()->generateDouble() < storeProbability);

            if (shouldStore && m_currentPageIndex < m_fetchedPages.size()) {
                ResearchFinding finding;
                finding.topic = m_currentTopic;
                finding.searchQuery = m_currentQueries.value(0);
                finding.sourceUrl = m_fetchedPages[m_currentPageIndex].url;
                finding.sourceTitle = m_fetchedPages[m_currentPageIndex].title;
                finding.rawSnippet = m_fetchedPages[m_currentPageIndex].plainText.left(500);
                finding.llmSummary = obj.value("summary").toString();
                finding.relevanceReason = obj.value("reason").toString();
                finding.relevanceScore = relevance;
                finding.modelId = m_settingsManager ? m_settingsManager->model() : "";

                // Soft auto-approve centered at 0.7
                double approveProbability = Fuzzy::sigmoid(relevance, 0.7, 0.08);
                if (m_settingsManager && m_settingsManager->privacyLevel() == "full"
                    && QRandomGenerator::global()->generateDouble() < approveProbability) {
                    finding.status = 1;  // approved
                } else {
                    finding.status = 0;  // pending
                }

                qint64 id = m_researchStore->insertFinding(finding);
                if (id >= 0) m_currentFindingsCount++;

                qDebug() << "ResearchEngine: stored finding (relevance:"
                         << relevance << "storePr:" << storeProbability
                         << ") from" << finding.sourceUrl;
            }
        } else {
            qDebug() << "ResearchEngine: failed to parse LLM summary JSON";
        }

        // Move to next page
        m_currentPageIndex++;
        if (m_currentPageIndex < m_fetchedPages.size()) {
            // Summarize next page
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

            QString pageText = m_fetchedPages[m_currentPageIndex].plainText;
            if (pageText.length() > 3000) pageText = pageText.left(3000);

            QString prompt = QString(
                "User interested in: \"%1\"\nTraits:\n%2\n"
                "Content from %3:\n\n%4\n\n"
                "Output a JSON object. Example: {\"summary\":\"Brief summary here\",\"relevance\":0.8,\"reason\":\"Why relevant\"}\n"
                "IMPORTANT: Output ONLY the JSON object starting with { and ending with }. No other text."
            ).arg(m_currentTopic,
                  traitContext.isEmpty() ? "(none)" : traitContext,
                  m_fetchedPages[m_currentPageIndex].url,
                  pageText);

            QJsonArray msgs;
            QJsonObject msg;
            msg["role"] = "user";
            msg["content"] = prompt;
            msgs.append(msg);
            m_llmProvider->sendBackgroundRequest("ResearchEngine",
                "You are a JSON-only API. You output valid JSON and nothing else. Only JSON.",
                msgs, LLMProviderManager::PriorityNormal);
        } else {
            // All pages summarized
            advancePhase();
        }
    }
}

void ResearchEngine::onLLMError(const QString &error)
{
    m_consecutiveErrors++;
    qWarning() << "ResearchEngine: LLM error:" << error;
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        qWarning() << "ResearchEngine: pausing after" << m_consecutiveErrors << "consecutive errors";
    }
    emit researchError(error);

    // Try to continue — skip current phase
    if (m_phase == GenerateQuery) {
        // Use topic directly as a search query
        m_currentQueries = {m_currentTopic};
        advancePhase();
    } else if (m_phase == FetchAndSummarize) {
        // Skip to next page or finish
        m_currentPageIndex++;
        if (m_currentPageIndex < m_fetchedPages.size()) {
            // Retry with next page but skip LLM — just store raw snippet
            advancePhase();
        } else {
            advancePhase();
        }
    } else {
        setPhase(Idle);
    }
}

// --- Research context for conversation ---

QString ResearchEngine::buildResearchContext(const QString &topic, int maxFindings) const
{
    if (!m_researchStore || topic.isEmpty()) return QString();

    QList<ResearchFinding> findings = m_researchStore->getApprovedFindingsForTopic(topic, maxFindings);
    if (findings.isEmpty()) return QString();

    QString context = "Your research on topics relevant to this conversation:\n";
    for (const ResearchFinding &f : findings) {
        context += QString("- %1 (source: %2)\n").arg(f.llmSummary, f.sourceUrl);
    }
    context += "You looked into this on your own time. If relevant, mention it casually, "
                "e.g. 'I was thinking about this while you were away' or 'I looked into that a bit more.' "
                "Do not say 'I researched' -- keep it conversational.";
    return context;
}

// --- Queue management ---

void ResearchEngine::queueTopic(const QString &topic, double priority)
{
    if (topic.isEmpty()) return;

    // Check for duplicates
    for (const TopicEntry &e : m_topicQueue) {
        if (e.topic.toLower() == topic.toLower()) return;
    }

    m_topicQueue.append({topic, priority});
    std::sort(m_topicQueue.begin(), m_topicQueue.end(),
              [](const TopicEntry &a, const TopicEntry &b) {
        return a.priority > b.priority;
    });

    emit topicQueueChanged();
    qDebug() << "ResearchEngine: queued topic:" << topic << "priority:" << priority;
}

// --- Daily counter ---

void ResearchEngine::resetDailyCounterIfNeeded()
{
    QDate today = QDate::currentDate();
    if (m_lastCycleDate != today) {
        m_cyclesCompletedToday = 0;
        m_lastCycleDate = today;
        emit cyclesCompletedTodayChanged();
    }
}
