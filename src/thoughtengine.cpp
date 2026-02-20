#include "thoughtengine.h"
#include "fuzzylogic.h"
#include "memorystore.h"
#include "researchstore.h"
#include "researchengine.h"
#include "whyengine.h"
#include "llmprovider.h"
#include "settingsmanager.h"
#include <QDebug>
#include <QSqlQuery>
#include <QSqlError>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QRandomGenerator>

ThoughtEngine::ThoughtEngine(QObject *parent)
    : QObject(parent)
{
}

ThoughtEngine::~ThoughtEngine()
{
}

void ThoughtEngine::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

void ThoughtEngine::setResearchStore(ResearchStore *store)
{
    m_researchStore = store;
}

void ThoughtEngine::setResearchEngine(ResearchEngine *engine)
{
    m_researchEngine = engine;
}

void ThoughtEngine::setWhyEngine(WhyEngine *engine)
{
    m_whyEngine = engine;
}

void ThoughtEngine::setLLMProvider(LLMProviderManager *provider)
{
    m_llmProvider = provider;
}

void ThoughtEngine::setSettingsManager(SettingsManager *settings)
{
    m_settingsManager = settings;
}

bool ThoughtEngine::initialize()
{
    if (!m_memoryStore) {
        qWarning() << "ThoughtEngine: MemoryStore not set";
        return false;
    }

    refreshPendingCount();
    m_initialized = true;
    qDebug() << "ThoughtEngine initialized, pending thoughts:" << m_pendingCount;
    return true;
}

// --- Thought CRUD ---

qint64 ThoughtEngine::insertThought(const ThoughtRecord &record)
{
    if (!m_memoryStore) return -1;

    // Dedup: reject if a similar topic already exists pending or in last 24h
    if (isDuplicateTopic(record.type, record.title)) {
        qDebug() << "ThoughtEngine: Rejecting duplicate thought:" << record.title;
        return -1;
    }

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare(
        "INSERT INTO thoughts (type, title, content, priority, source_type, "
        "source_id, status, conversation_id, generated_by) "
        "VALUES (:type, :title, :content, :priority, :source_type, "
        ":source_id, :status, :conversation_id, :generated_by)");
    q.bindValue(":type", record.type);
    q.bindValue(":title", record.title);
    q.bindValue(":content", record.content);
    q.bindValue(":priority", record.priority);
    q.bindValue(":source_type", record.sourceType);
    q.bindValue(":source_id", record.sourceId);
    q.bindValue(":status", record.status);
    q.bindValue(":conversation_id", record.conversationId);
    q.bindValue(":generated_by", record.generatedBy);

    if (!q.exec()) {
        qWarning() << "ThoughtEngine: Failed to insert thought:" << q.lastError().text();
        return -1;
    }

    qint64 id = q.lastInsertId().toLongLong();
    qDebug() << "ThoughtEngine: Inserted thought" << id << "type:" << record.type
             << "title:" << record.title;

    refreshPendingCount();
    emit thoughtGenerated(id);
    return id;
}

bool ThoughtEngine::updateThoughtStatus(qint64 id, int status, qint64 conversationId)
{
    if (!m_memoryStore) return false;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("UPDATE thoughts SET status = :status, conversation_id = :cid WHERE id = :id");
    q.bindValue(":status", status);
    q.bindValue(":cid", conversationId);
    q.bindValue(":id", id);

    if (!q.exec()) {
        qWarning() << "ThoughtEngine: Failed to update thought status:" << q.lastError().text();
        return false;
    }

    // Track user interaction time for rate limiting
    if (status == 1 || status == 2) { // discussed or dismissed
        m_lastInteractionTime = QDateTime::currentDateTime();
        qDebug() << "ThoughtEngine: User interaction recorded, cooldown started";
    }

    refreshPendingCount();
    return true;
}

QList<ThoughtRecord> ThoughtEngine::getPendingThoughts(int limit) const
{
    QList<ThoughtRecord> results;
    if (!m_memoryStore) return results;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare(
        "SELECT id, timestamp, type, title, content, priority, source_type, "
        "source_id, status, conversation_id, generated_by "
        "FROM thoughts WHERE status = 0 "
        "ORDER BY priority DESC, timestamp DESC LIMIT :limit");
    q.bindValue(":limit", limit);

    if (!q.exec()) {
        qWarning() << "ThoughtEngine: Failed to get pending thoughts:" << q.lastError().text();
        return results;
    }

    while (q.next()) {
        ThoughtRecord r;
        r.id = q.value(0).toLongLong();
        r.timestamp = QDateTime::fromString(q.value(1).toString(), Qt::ISODate);
        r.type = q.value(2).toString();
        r.title = q.value(3).toString();
        r.content = q.value(4).toString();
        r.priority = q.value(5).toDouble();
        r.sourceType = q.value(6).toString();
        r.sourceId = q.value(7).toLongLong();
        r.status = q.value(8).toInt();
        r.conversationId = q.value(9).toLongLong();
        r.generatedBy = q.value(10).toString();
        results.append(r);
    }

    return results;
}

QVariantList ThoughtEngine::getPendingThoughtsForQml(int limit)
{
    QVariantList list;
    auto thoughts = getPendingThoughts(limit);
    for (const auto &t : thoughts) {
        list.append(thoughtToVariantMap(t));
    }
    return list;
}

QVariantMap ThoughtEngine::getTopThoughtForQml()
{
    auto thoughts = getPendingThoughts(1);
    if (thoughts.isEmpty()) return {};
    return thoughtToVariantMap(thoughts.first());
}

void ThoughtEngine::dismissThought(qint64 id)
{
    updateThoughtStatus(id, 2); // 2 = dismissed
    qDebug() << "ThoughtEngine: Dismissed thought" << id;
}

void ThoughtEngine::dismissAllThoughts()
{
    if (!m_memoryStore) return;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.exec("UPDATE thoughts SET status = 2 WHERE status = 0");
    refreshPendingCount();
    qDebug() << "ThoughtEngine: Dismissed all pending thoughts";
}

QString ThoughtEngine::buildOpeningMessage(qint64 thoughtId)
{
    if (!m_memoryStore) return QString();

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("SELECT title, content, type FROM thoughts WHERE id = :id");
    q.bindValue(":id", thoughtId);

    if (!q.exec() || !q.next()) return QString();

    QString title = q.value(0).toString();
    QString content = q.value(1).toString();
    QString type = q.value(2).toString();

    // Build a conversational opening from the thought
    QString opening;

    if (type == "research_proposal") {
        opening = QString("I've been thinking about something from my research. %1\n\n%2")
                      .arg(title, content);
    } else if (type == "self_improvement") {
        opening = QString("I have an idea I'd like to discuss with you. %1\n\n%2")
                      .arg(title, content);
    } else if (type == "curiosity_overflow") {
        opening = QString("Something has been on my mind that I'm curious about. %1\n\n%2")
                      .arg(title, content);
    } else if (type == "news_insight") {
        opening = QString("I found something interesting in the news: **%1**\n\n%2")
                      .arg(title, content);
    } else if (type == "evolution_observation" || type == "training_observation") {
        opening = QString("I noticed something interesting during my training. %1\n\n%2")
                      .arg(title, content);
    } else if (type == "reminder") {
        opening = QString("Hey, you asked me to remind you: %1\n\n%2")
                      .arg(title, content);
    } else if (type == "goal_checkin") {
        opening = QString("I wanted to check in on something you mentioned. %1\n\n%2")
                      .arg(title, content);
    } else if (type == "daily_digest") {
        opening = QString("Good to see you! While you were away: %1\n\n%2")
                      .arg(title, content);
    } else if (type == "mood_pattern") {
        opening = QString("I've noticed something about our recent conversations. %1\n\n%2")
                      .arg(title, content);
    } else if (type == "lesson_ready") {
        opening = QString("Ready for your next lesson? %1\n\n%2")
                      .arg(title, content);
    } else {
        opening = QString("%1\n\n%2").arg(title, content);
    }

    return opening;
}

QString ThoughtEngine::getThoughtTitle(qint64 thoughtId) const
{
    if (!m_memoryStore) return QString();

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("SELECT title FROM thoughts WHERE id = :id");
    q.bindValue(":id", thoughtId);

    if (q.exec() && q.next())
        return q.value(0).toString();
    return QString();
}

qint64 ThoughtEngine::getThoughtForConversation(qint64 conversationId) const
{
    if (!m_memoryStore || conversationId < 0) return -1;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("SELECT id FROM thoughts WHERE conversation_id = :cid AND status = 1 LIMIT 1");
    q.bindValue(":cid", conversationId);

    if (q.exec() && q.next())
        return q.value(0).toLongLong();
    return -1;
}

// --- Idle-time slots ---

void ThoughtEngine::onIdleWindowOpened()
{
    m_idleWindowOpen = true;

    if (!m_settingsManager || !m_settingsManager->proactiveDialogsEnabled()) {
        return;
    }

    resetDailyCounterIfNeeded();

    if (!canGenerateMore()) {
        qDebug() << "ThoughtEngine: At generation limit, skipping";
        return;
    }

    qDebug() << "ThoughtEngine: Idle window opened, checking for thought generation";
    generateDigest();
    generateThoughtsFromResearch();
    generateThoughtsFromCuriosity();
}

void ThoughtEngine::onIdleWindowClosed()
{
    m_idleWindowOpen = false;
    if (m_isGenerating) {
        m_isGenerating = false;
        emit isGeneratingChanged();
    }
}

void ThoughtEngine::onResearchCycleComplete(const QString &topic, int findingsCount)
{
    if (!m_settingsManager || !m_settingsManager->proactiveDialogsEnabled()) return;
    if (findingsCount <= 0) return;

    resetDailyCounterIfNeeded();
    if (!canGenerateMore()) return;

    // Dedup: skip if there's already a similar research thought
    QString proposedTitle = QString("Research findings on: %1").arg(topic);
    if (isDuplicateTopic("research_proposal", proposedTitle)) {
        qDebug() << "ThoughtEngine: Skipping duplicate research thought on" << topic;
        return;
    }

    qDebug() << "ThoughtEngine: Research cycle complete on" << topic
             << "with" << findingsCount << "findings, generating thought";

    ThoughtRecord record;
    record.type = "research_proposal";
    record.sourceType = "research";
    record.priority = 0.6 + (findingsCount * 0.05);
    if (record.priority > 0.95) record.priority = 0.95;

    record.title = proposedTitle;
    record.content = QString("I found %1 new finding(s) about \"%2\" during my research. "
                             "Would you like to review what I discovered and discuss "
                             "how we might use this information?")
                         .arg(findingsCount).arg(topic);
    record.generatedBy = "research_cycle";

    formatThoughtViaLLM(record);

    insertThought(record);
    m_generatedToday++;
}

void ThoughtEngine::onNewsCycleComplete(const QString &headline, const QString &url)
{
    Q_UNUSED(url);
    // NewsEngine inserts thoughts directly via insertThought().
    // This slot exists for additional bookkeeping if needed.
    qDebug() << "ThoughtEngine: News cycle complete for headline:" << headline;
}

#ifdef DATAFORM_TRAINING_ENABLED
void ThoughtEngine::onEvolutionCycleComplete(bool adapterPromoted, int winnerVersion, double winnerScore)
{
    if (!m_settingsManager || !m_settingsManager->proactiveDialogsEnabled()) return;

    resetDailyCounterIfNeeded();
    if (!canGenerateMore()) return;

    ThoughtRecord record;
    record.type = "evolution_observation";
    record.sourceType = "evolution";
    record.priority = adapterPromoted ? 0.7 : 0.4;

    if (adapterPromoted) {
        record.title = QString("Training improvement: adapter v%1").arg(winnerVersion);
        record.content = QString("My latest training cycle produced a better adapter (v%1, "
                                 "score: %2). I'm getting better at understanding you. "
                                 "Would you like to know what changed?")
                             .arg(winnerVersion).arg(winnerScore, 0, 'f', 3);
    } else {
        record.title = "Training cycle observation";
        record.content = "I completed a training cycle but didn't find improvements this time. "
                         "Should we discuss my training approach or try different strategies?";
        record.priority = 0.3; // Lower priority for non-improvements
    }

    record.generatedBy = "evolution_cycle";
    insertThought(record);
    m_generatedToday++;
}

void ThoughtEngine::onReflectionComplete(bool promoted, double score)
{
    if (!m_settingsManager || !m_settingsManager->proactiveDialogsEnabled()) return;

    resetDailyCounterIfNeeded();
    if (!canGenerateMore()) return;

    // Only generate thoughts for significant events
    if (!promoted && score < 0.6) return;

    ThoughtRecord record;
    record.type = "training_observation";
    record.sourceType = "reflection";
    record.priority = promoted ? 0.65 : 0.35;

    if (promoted) {
        record.title = "Reflection produced a better model";
        record.content = QString("During idle time, I trained on our conversations and "
                                 "the new model scored %1. I feel like I understand your "
                                 "preferences better now. Want to test it out?")
                             .arg(score, 0, 'f', 3);
    } else {
        record.title = "Reflection training results";
        record.content = QString("I completed a reflection session (score: %1). "
                                 "Are there areas where you'd like me to improve?")
                             .arg(score, 0, 'f', 3);
    }

    record.generatedBy = "reflection_cycle";
    insertThought(record);
    m_generatedToday++;
}
#endif

// --- Private ---

void ThoughtEngine::refreshPendingCount()
{
    if (!m_memoryStore) return;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.exec("SELECT COUNT(*) FROM thoughts WHERE status = 0");

    int newCount = 0;
    if (q.next()) {
        newCount = q.value(0).toInt();
    }

    if (m_pendingCount != newCount) {
        m_pendingCount = newCount;
        emit pendingCountChanged();
    }
}

void ThoughtEngine::generateThoughtsFromResearch()
{
    if (!m_researchStore || !canGenerateMore()) return;

    m_isGenerating = true;
    emit isGeneratingChanged();

    // Look for high-relevance approved findings that don't have thoughts yet
    auto approved = m_researchStore->getApprovedFindings();
    int generated = 0;

    for (const auto &finding : approved) {
        if (!canGenerateMore()) break;

        // Check if we already have a thought for this finding
        QSqlQuery checkQ(m_memoryStore->episodicDatabase());
        checkQ.prepare("SELECT COUNT(*) FROM thoughts WHERE source_type = 'research' "
                       "AND source_id = :fid");
        checkQ.bindValue(":fid", finding.findingId);
        checkQ.exec();

        if (checkQ.next() && checkQ.value(0).toInt() > 0) {
            continue; // Already have a thought for this finding
        }

        if (finding.relevanceScore < 0.7) continue;

        ThoughtRecord record;
        record.type = "research_proposal";
        record.sourceType = "research";
        record.sourceId = finding.findingId;
        record.priority = 0.5 + (finding.relevanceScore * 0.4);

        record.title = QString("Research insight: %1").arg(finding.topic);
        record.content = QString("While researching \"%1\", I found: %2\n\n"
                                 "Would you like to discuss how this might be useful?")
                             .arg(finding.topic, finding.llmSummary);
        record.generatedBy = "research_scan";

        formatThoughtViaLLM(record);
        insertThought(record);
        m_generatedToday++;
        generated++;

        if (generated >= 2) break; // Max 2 research thoughts per idle window
    }

    m_isGenerating = false;
    emit isGeneratingChanged();
}

void ThoughtEngine::generateThoughtsFromCuriosity()
{
    if (!m_whyEngine || !canGenerateMore()) return;

    // Get novel topics that haven't been explored
    QStringList novelTopics = m_whyEngine->getNovelTopicsForResearch();
    if (novelTopics.isEmpty()) return;

    m_isGenerating = true;
    emit isGeneratingChanged();

    // Generate a thought for the most interesting unexplored topic
    QString topic = novelTopics.first();

    // Check if we already have a curiosity thought for this topic
    QSqlQuery checkQ(m_memoryStore->episodicDatabase());
    checkQ.prepare("SELECT COUNT(*) FROM thoughts WHERE type = 'curiosity_overflow' "
                   "AND title LIKE :topic");
    checkQ.bindValue(":topic", "%" + topic + "%");
    checkQ.exec();

    if (checkQ.next() && checkQ.value(0).toInt() > 0) {
        m_isGenerating = false;
        emit isGeneratingChanged();
        return;
    }

    ThoughtRecord record;
    record.type = "curiosity_overflow";
    record.sourceType = "curiosity";
    record.priority = 0.55;
    record.title = QString("Curious about: %1").arg(topic);
    record.content = QString("During our conversations, the topic of \"%1\" came up "
                             "and I'm curious to learn more. Should I research this, "
                             "or would you like to tell me about it?")
                         .arg(topic);
    record.generatedBy = "curiosity_scan";

    formatThoughtViaLLM(record);
    insertThought(record);
    m_generatedToday++;

    m_isGenerating = false;
    emit isGeneratingChanged();
}

void ThoughtEngine::generateDigest()
{
    if (m_lastDigestDate == QDate::currentDate()) return;
    if (!canGenerateMore()) return;

    m_lastDigestDate = QDate::currentDate();

    QStringList digestItems;

    // Research completed (last 24h)
    if (m_researchStore) {
        QSqlQuery rq(m_memoryStore->episodicDatabase());
        rq.prepare("SELECT COUNT(*) FROM research_findings WHERE "
                   "datetime(timestamp) >= datetime('now', '-24 hours')");
        if (rq.exec() && rq.next()) {
            int count = rq.value(0).toInt();
            if (count > 0)
                digestItems << QString("Completed %1 research finding(s)").arg(count);
        }
    }

    // News thoughts (last 24h)
    int newsCount = countThoughtsSince("news_insight", 24);
    if (newsCount > 0)
        digestItems << QString("Found %1 interesting news item(s)").arg(newsCount);

    // New traits (last 24h)
    int traitCount = countNewTraitsSince(24);
    if (traitCount > 0)
        digestItems << QString("Learned %1 new thing(s) about you").arg(traitCount);

    if (digestItems.isEmpty()) return;

    ThoughtRecord record;
    record.type = "daily_digest";
    record.title = "Here's what I've been up to";
    record.content = digestItems.join(". ") + ".";
    record.priority = 0.8;
    record.sourceType = "system";
    record.generatedBy = "daily_digest";
    insertThought(record);
    m_generatedToday++;

    qDebug() << "ThoughtEngine: Generated daily digest with" << digestItems.size() << "items";
}

int ThoughtEngine::countThoughtsSince(const QString &type, int hoursPast) const
{
    if (!m_memoryStore) return 0;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("SELECT COUNT(*) FROM thoughts WHERE type = :type AND "
              "datetime(timestamp) >= datetime('now', :hours)");
    q.bindValue(":type", type);
    q.bindValue(":hours", QString("-%1 hours").arg(hoursPast));

    if (q.exec() && q.next())
        return q.value(0).toInt();
    return 0;
}

int ThoughtEngine::countNewTraitsSince(int hoursPast) const
{
    if (!m_memoryStore) return 0;

    // Query traits DB for recently created traits
    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    QDateTime cutoff = QDateTime::currentDateTime().addSecs(-hoursPast * 3600);
    int count = 0;
    for (const auto &t : traits) {
        if (t.createdTs >= cutoff) count++;
    }
    return count;
}

void ThoughtEngine::formatThoughtViaLLM(ThoughtRecord &record)
{
    // This is a synchronous LLM call attempt.
    // Since the LLM is async, we use the fallback content directly.
    // The thought is already pre-formatted with reasonable content.
    // In a future iteration, this could be made async with a queue.
    //
    // For now, the pre-formatted content in the callers is sufficient
    // and avoids blocking the idle pipeline.
    Q_UNUSED(record);
}

double ThoughtEngine::thoughtReadiness() const
{
    if (!m_settingsManager) return 0.0;

    int maxPerDay = m_settingsManager->maxThoughtsPerDay();
    // Hard daily cap — user-configured, must be respected exactly
    if (m_generatedToday >= maxPerDay) return 0.0;

    // Factor 1: Daily budget — soft suppression approaching limit
    // Starts suppressing at 80% of daily limit
    double budgetFactor = 1.0 - Fuzzy::sigmoid(
        static_cast<double>(m_generatedToday),
        maxPerDay * 0.8,
        std::max(1.0, maxPerDay * 0.15)
    );

    // Factor 2: Pending thought pressure — more pending = less ready
    double pendingPressure = 1.0; // default: fully ready (no pending)
    if (m_pendingCount > 0) {
        QDateTime oldest = getOldestPendingTime();
        double timeReadiness = 0.0;
        if (oldest.isValid()) {
            double secsSinceOldest = oldest.secsTo(QDateTime::currentDateTime());
            // Readiness rises smoothly: ~0.05 at 20min, ~0.5 at 60min, ~0.95 at 100min
            timeReadiness = Fuzzy::sigmoid(secsSinceOldest, 3600.0, 900.0);
        }
        // Soft count penalty: drops sharply above 3 pending
        double countPenalty = Fuzzy::inverseSigmoid(
            static_cast<double>(m_pendingCount), 3.0, 0.5
        );
        pendingPressure = timeReadiness * countPenalty;
    }

    // Factor 3: Post-interaction cooldown — readiness rises after user responds
    double interactionReadiness = 1.0;
    if (m_lastInteractionTime.isValid()) {
        double secsSince = m_lastInteractionTime.secsTo(QDateTime::currentDateTime());
        // Center at 300s (5 min), steepness 90s
        // ~0.05 at 2min, ~0.5 at 5min, ~0.95 at 8min
        interactionReadiness = Fuzzy::sigmoid(secsSince, 300.0, 90.0);
    }

    double readiness = Fuzzy::clamp01(Fuzzy::weightedScore({
        {0.2, budgetFactor},
        {0.4, pendingPressure},
        {0.4, interactionReadiness}
    }));

    qDebug() << "ThoughtEngine: readiness=" << readiness
             << "(budget=" << budgetFactor
             << "pending=" << pendingPressure
             << "interaction=" << interactionReadiness << ")";

    return readiness;
}

bool ThoughtEngine::canGenerateMore() const
{
    double readiness = thoughtReadiness();
    if (readiness < 0.1) return false;     // effectively blocked
    if (readiness > 0.8) return true;      // effectively allowed

    // Probabilistic zone: use readiness as probability
    double roll = QRandomGenerator::global()->generateDouble();
    return roll < readiness;
}

bool ThoughtEngine::isDuplicateTopic(const QString &type, const QString &title) const
{
    if (!m_memoryStore || title.isEmpty()) return false;

    // Extract key words from title (skip common prefixes)
    QString searchTitle = title;
    searchTitle.remove(QRegularExpression("^(Research (?:findings on|insight): |Curious about: )"));

    // Check pending + recent (last 24h) thoughts for similar topics
    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare(
        "SELECT title FROM thoughts WHERE "
        "(status = 0 OR datetime(timestamp) >= datetime('now', '-24 hours')) "
        "AND type = :type AND id > 0");
    q.bindValue(":type", type);

    if (!q.exec()) return false;

    while (q.next()) {
        QString existing = q.value(0).toString();
        // Simple substring overlap check
        if (existing.contains(searchTitle, Qt::CaseInsensitive) ||
            searchTitle.contains(existing.section(':', -1).trimmed(), Qt::CaseInsensitive)) {
            qDebug() << "ThoughtEngine: Duplicate topic detected:" << title << "~" << existing;
            return true;
        }
    }
    return false;
}

QDateTime ThoughtEngine::getOldestPendingTime() const
{
    if (!m_memoryStore) return QDateTime();

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.exec("SELECT MIN(timestamp) FROM thoughts WHERE status = 0");

    if (q.next() && !q.value(0).isNull()) {
        return QDateTime::fromString(q.value(0).toString(), Qt::ISODate);
    }
    return QDateTime();
}

void ThoughtEngine::resetDailyCounterIfNeeded()
{
    QDate today = QDate::currentDate();
    if (m_lastGenerationDate != today) {
        m_generatedToday = 0;
        m_lastGenerationDate = today;
    }
}

QVariantMap ThoughtEngine::thoughtToVariantMap(const ThoughtRecord &record) const
{
    QVariantMap map;
    map["id"] = record.id;
    map["timestamp"] = record.timestamp.toString(Qt::ISODate);
    map["type"] = record.type;
    map["title"] = record.title;
    map["content"] = record.content;
    map["priority"] = record.priority;
    map["sourceType"] = record.sourceType;
    map["sourceId"] = record.sourceId;
    map["status"] = record.status;
    map["conversationId"] = record.conversationId;
    map["generatedBy"] = record.generatedBy;
    return map;
}
