#include "distillationmanager.h"
#include "memorystore.h"
#include "llmprovider.h"
#include "settingsmanager.h"
#include "thoughtengine.h"
#include "embeddingmanager.h"
#ifdef DATAFORM_TRAINING_ENABLED
#include "ortinferencemanager.h"
#endif
#include <QRandomGenerator>
#include <QSqlQuery>
#include <QDebug>

const QString DistillationManager::OWNER_TAG = QStringLiteral("Distiller");

DistillationManager::DistillationManager(QObject *parent)
    : QObject(parent)
{
    m_cycleTimer.setSingleShot(true);
    connect(&m_cycleTimer, &QTimer::timeout, this, &DistillationManager::startCycle);
}

DistillationManager::~DistillationManager()
{
}

void DistillationManager::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
    if (store) {
        // Load existing source episode IDs to avoid re-distilling
        QList<qint64> ids = store->getDistillationSourceEpisodeIds();
        for (qint64 id : ids)
            m_usedSourceEpisodeIds.insert(id);
        refreshCounts();
    }
}

void DistillationManager::setLLMProvider(LLMProviderManager *provider)
{
    m_llmProvider = provider;
    if (provider) {
        connect(provider, &LLMProviderManager::teacherResponseReceived,
                this, &DistillationManager::onTeacherResponse);
        connect(provider, &LLMProviderManager::teacherErrorOccurred,
                this, &DistillationManager::onTeacherError);
    }
}

void DistillationManager::setSettingsManager(SettingsManager *settings)
{
    m_settingsManager = settings;
}

void DistillationManager::setThoughtEngine(ThoughtEngine *engine)
{
    m_thoughtEngine = engine;
}

void DistillationManager::setEmbeddingManager(EmbeddingManager *embeddings)
{
    m_embeddingManager = embeddings;
}

#ifdef DATAFORM_TRAINING_ENABLED
void DistillationManager::setOrtInferenceManager(OrtInferenceManager *inference)
{
    m_ortInference = inference;
}
#endif

// --- Idle Window Integration ---

void DistillationManager::onIdleWindowOpened()
{
    m_idleWindowOpen = true;
    m_pairsThisSession = 0;
    // Auto-start removed — coordinator calls requestStart()
}

void DistillationManager::requestStart()
{
    if (!canStartCycle()) {
        emit cycleFinished();
        return;
    }
    startCycle();
}

void DistillationManager::onIdleWindowClosed()
{
    m_idleWindowOpen = false;
    m_cycleTimer.stop();

    if (m_isDistilling) {
        setPhase(Idle);
        setStatus("Paused — user active");
    }
}

void DistillationManager::distillNow()
{
    if (m_isDistilling) {
        qDebug() << "DistillationManager: already distilling";
        return;
    }
    if (!m_llmProvider || !m_memoryStore) {
        setStatus("Missing dependencies");
        return;
    }
    startCycle();
}

// --- Cycle Management ---

bool DistillationManager::canStartCycle() const
{
    if (!m_settingsManager || !m_memoryStore || !m_llmProvider)
        return false;

    if (!m_settingsManager->distillationEnabled())
        return false;

    // Check daily limit
    if (m_lastCycleDate == QDate::currentDate() &&
        m_cyclesCompletedToday >= m_settingsManager->distillationDailyCycles() * MAX_PAIRS_PER_SESSION)
        return false;

    // Check session limit
    if (m_pairsThisSession >= MAX_PAIRS_PER_SESSION)
        return false;

    // Check cooldown
    if (m_lastCycleEndTime.isValid() &&
        m_lastCycleEndTime.secsTo(QDateTime::currentDateTime()) < CYCLE_COOLDOWN_SEC)
        return false;

    // Need at least some episodes to distill from
    if (m_memoryStore->episodeCount() < 3)
        return false;

    return true;
}

void DistillationManager::resetDailyCounterIfNeeded()
{
    QDate today = QDate::currentDate();
    if (m_lastCycleDate != today) {
        m_cyclesCompletedToday = 0;
        m_lastCycleDate = today;
        m_graduationThoughtGenerated = false;
    }
}

void DistillationManager::startCycle()
{
    resetDailyCounterIfNeeded();

    if (!canStartCycle()) {
        setStatus("Daily limit reached or conditions not met");
        emit cycleFinished();
        return;
    }

    m_isDistilling = true;
    emit isDistillingChanged();
    m_consecutiveErrors = 0;

    phaseSelectSource();
}

// --- Phase 1: Select Source ---

void DistillationManager::phaseSelectSource()
{
    setPhase(SelectSource);
    setStatus("Selecting conversation for distillation...");

    qint64 episodeId = -1;
    QString userPrompt;
    QString sourceType;

    // Weighted random selection of source type
    int roll = QRandomGenerator::global()->bounded(100);

    if (roll < 60) {
        // 60%: high-signal episode
        userPrompt = selectFromHighSignalEpisode(episodeId);
        sourceType = "episode";
    } else if (roll < 85) {
        // 25%: synthesize from traits
        userPrompt = synthesizeFromTraits();
        sourceType = "trait_qa";
    } else {
        // 15%: synthesize from research
        userPrompt = synthesizeFromResearch();
        sourceType = "synthetic";
    }

    if (userPrompt.isEmpty()) {
        // Fallback: try other sources
        if (sourceType != "episode") {
            userPrompt = selectFromHighSignalEpisode(episodeId);
            sourceType = "episode";
        }
        if (userPrompt.isEmpty()) {
            userPrompt = synthesizeFromTraits();
            sourceType = "trait_qa";
        }
    }

    if (userPrompt.isEmpty()) {
        setStatus("No suitable source material found");
        m_isDistilling = false;
        emit isDistillingChanged();
        emit cycleFinished();
        return;
    }

    phaseEnhancePrompt(episodeId, userPrompt, sourceType);
}

QString DistillationManager::selectFromHighSignalEpisode(qint64 &episodeId)
{
    if (!m_memoryStore) return QString();

    QList<EpisodicRecord> episodes = m_memoryStore->getHighSignalEpisodes(0, 50);
    if (episodes.isEmpty()) {
        // Fallback to recent episodes
        episodes = m_memoryStore->getRecentEpisodes(20);
    }

    // Find an episode not yet used for distillation
    for (const EpisodicRecord &ep : episodes) {
        if (!m_usedSourceEpisodeIds.contains(ep.id) && !ep.userText.isEmpty()) {
            episodeId = ep.id;
            return ep.userText;
        }
    }

    // All episodes used — pick a random one (allow re-distillation for variety)
    if (!episodes.isEmpty()) {
        int idx = QRandomGenerator::global()->bounded(episodes.size());
        episodeId = episodes[idx].id;
        return episodes[idx].userText;
    }

    return QString();
}

QString DistillationManager::synthesizeFromTraits()
{
    if (!m_memoryStore) return QString();

    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    if (traits.isEmpty()) return QString();

    // Filter to high-confidence traits
    QList<TraitRecord> goodTraits;
    for (const TraitRecord &t : traits) {
        if (t.confidence >= 0.3)
            goodTraits.append(t);
    }
    if (goodTraits.isEmpty()) goodTraits = traits;

    const TraitRecord &trait = goodTraits[QRandomGenerator::global()->bounded(goodTraits.size())];

    // Generate a question about this trait topic
    static const QStringList templates = {
        "What are your thoughts on %1?",
        "Can you explain your perspective on %1?",
        "I'd like to understand more about %1. What do you think?",
        "How do you feel about %1?",
        "Tell me about %1 — what's important to know?",
        "What would you say to someone curious about %1?",
        "Can you share your views on %1?"
    };

    int idx = QRandomGenerator::global()->bounded(templates.size());
    return templates[idx].arg(trait.statement);
}

QString DistillationManager::synthesizeFromResearch()
{
    if (!m_memoryStore) return QString();

    // Query approved research findings for topics
    QSqlQuery q(m_memoryStore->episodicDatabase());
    if (q.exec("SELECT topic, llm_summary FROM research_findings "
               "WHERE status = 1 ORDER BY RANDOM() LIMIT 1")) {
        if (q.next()) {
            QString topic = q.value(0).toString();
            static const QStringList templates = {
                "What do you know about %1?",
                "Tell me something interesting about %1.",
                "I've been reading about %1. What are your thoughts?",
                "Can you give me a summary of %1?",
                "What should I know about %1?"
            };
            int idx = QRandomGenerator::global()->bounded(templates.size());
            return templates[idx].arg(topic);
        }
    }

    return QString();
}

// --- Phase 2: Enhance Prompt ---

void DistillationManager::phaseEnhancePrompt(qint64 episodeId, const QString &userPrompt,
                                              const QString &sourceType)
{
    setPhase(EnhancePrompt);
    setStatus("Building enhanced context...");

    QString identityContext = buildIdentityContext();

    QString systemPrompt = QString(
        "You are DATAFORM, a learning AI companion who adapts to each user over time. "
        "Respond naturally and helpfully, showing genuine understanding of the user.\n\n"
        "%1\n\n"
        "Provide a thorough, thoughtful, personalized response. "
        "Draw on your knowledge of this user's interests and preferences. "
        "Be specific and insightful rather than generic."
    ).arg(identityContext);

    m_pendingUserPrompt = userPrompt;
    m_pendingSystemContext = systemPrompt;
    m_pendingSourceEpisodeId = episodeId;
    m_pendingSourceType = sourceType;

    phaseGenerateTeacher(systemPrompt, userPrompt, episodeId, sourceType);
}

QString DistillationManager::buildIdentityContext() const
{
    if (!m_memoryStore) return QString();

    QString context;

    // Add top traits
    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    if (!traits.isEmpty()) {
        context += "Known traits about this user:\n";
        int count = 0;
        for (const TraitRecord &t : traits) {
            if (t.confidence >= 0.3 && count < 10) {
                context += QString("- [%1] %2 (confidence: %3)\n")
                               .arg(t.type, t.statement)
                               .arg(t.confidence, 0, 'f', 2);
                ++count;
            }
        }
    }

    // Add recent research topics
    QSqlQuery q(m_memoryStore->episodicDatabase());
    if (q.exec("SELECT DISTINCT topic FROM research_findings "
               "WHERE status = 1 ORDER BY timestamp DESC LIMIT 5")) {
        QStringList topics;
        while (q.next())
            topics.append(q.value(0).toString());
        if (!topics.isEmpty()) {
            context += "\nRecent research interests: " + topics.join(", ") + "\n";
        }
    }

    // Add active goals
    QList<GoalRecord> goals = m_memoryStore->getActiveGoals();
    if (!goals.isEmpty()) {
        context += "\nActive goals:\n";
        for (const GoalRecord &g : goals) {
            context += "- " + g.title + "\n";
        }
    }

    return context;
}

// --- Phase 3: Generate Teacher Response ---

void DistillationManager::phaseGenerateTeacher(const QString &systemPrompt,
                                                const QString &userPrompt,
                                                qint64 sourceEpisodeId,
                                                const QString &sourceType)
{
    setPhase(GenerateTeacher);
    setStatus("Generating teacher response...");

    if (!m_llmProvider) {
        setStatus("No LLM provider available");
        m_isDistilling = false;
        emit isDistillingChanged();
        emit cycleFinished();
        return;
    }

    QJsonArray messages;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = userPrompt;
    messages.append(msg);

    m_llmProvider->sendTeacherRequest(OWNER_TAG, systemPrompt, messages);
}

// --- Teacher Response Handler ---

void DistillationManager::onTeacherResponse(const QString &owner, const QString &response)
{
    if (owner != OWNER_TAG) return;

    m_consecutiveErrors = 0;
    phaseQualityCheck(m_pendingUserPrompt, response,
                      m_pendingSystemContext, m_pendingSourceEpisodeId,
                      m_pendingSourceType);
}

void DistillationManager::onTeacherError(const QString &owner, const QString &error)
{
    if (owner != OWNER_TAG) return;

    ++m_consecutiveErrors;
    qWarning() << "DistillationManager: teacher error:" << error;
    setStatus("Teacher error: " + error);

    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        setStatus("Too many errors — pausing distillation");
        m_isDistilling = false;
        emit isDistillingChanged();
        emit cycleFinished();
        return;
    }

    // Retry with cooldown
    m_lastCycleEndTime = QDateTime::currentDateTime();
    if (m_idleWindowOpen && canStartCycle()) {
        m_cycleTimer.start(CYCLE_COOLDOWN_SEC * 1000);
    } else {
        m_isDistilling = false;
        emit isDistillingChanged();
        emit cycleFinished();
    }
}

// --- Phase 4: Quality Check ---

void DistillationManager::phaseQualityCheck(const QString &userPrompt,
                                             const QString &teacherResponse,
                                             const QString &systemContext,
                                             qint64 sourceEpisodeId,
                                             const QString &sourceType)
{
    setPhase(QualityCheck);
    setStatus("Checking response quality...");

    // Reject empty or too-short responses
    if (teacherResponse.trimmed().length() < 50) {
        qDebug() << "DistillationManager: response too short, skipping";
        setStatus("Response too short — skipping");
        finishCycleAndScheduleNext();
        return;
    }

    // Reject error-like responses
    QString lower = teacherResponse.toLower();
    if (lower.startsWith("i cannot") || lower.startsWith("i don't have") ||
        lower.startsWith("i'm not able") || lower.contains("ollama_error") ||
        lower.contains("raw_parse_fail")) {
        qDebug() << "DistillationManager: error response detected, skipping";
        setStatus("Error response — skipping");
        finishCycleAndScheduleNext();
        return;
    }

    // Quality score: length-based with floor
    double lengthScore = qMin(1.0, teacherResponse.length() / 500.0);
    double qualityScore = lengthScore * 0.7 + 0.3;

    phaseStore(userPrompt, teacherResponse, systemContext,
               sourceEpisodeId, sourceType, qualityScore);
}

// --- Phase 5: Store ---

void DistillationManager::phaseStore(const QString &userPrompt,
                                      const QString &teacherResponse,
                                      const QString &systemContext,
                                      qint64 sourceEpisodeId,
                                      const QString &sourceType,
                                      double qualityScore)
{
    setPhase(StorePair);
    setStatus("Storing distillation pair...");

    qint64 pairId = m_memoryStore->insertDistillationPair(
        userPrompt, teacherResponse, systemContext,
        sourceEpisodeId, sourceType, qualityScore);

    if (pairId >= 0) {
        if (sourceEpisodeId >= 0)
            m_usedSourceEpisodeIds.insert(sourceEpisodeId);

        ++m_pairsThisSession;
        ++m_cyclesCompletedToday;
        refreshCounts();

        emit distillationPairStored(pairId);
        qDebug() << "DistillationManager: pair stored, id:" << pairId
                 << "quality:" << qualityScore
                 << "session total:" << m_pairsThisSession;

        // Check if we should run graduation eval
        if (m_pairsCollected > 0 && m_pairsCollected % GRADUATION_EVAL_INTERVAL == 0) {
            runGraduationEval();
        }
    }

    finishCycleAndScheduleNext();
}

void DistillationManager::finishCycleAndScheduleNext()
{
    m_lastCycleEndTime = QDateTime::currentDateTime();

    setPhase(Idle);
    if (m_pairsThisSession >= MAX_PAIRS_PER_SESSION)
        setStatus(QString("Session complete — %1 pairs collected").arg(m_pairsThisSession));
    else
        setStatus("Distillation idle");
    m_isDistilling = false;
    emit isDistillingChanged();
    // Auto-retry removed — coordinator handles scheduling
    emit cycleFinished();
}

// --- Graduation Evaluation ---

void DistillationManager::runGraduationEval()
{
    // For now, compute readiness from stored eval scores
    // Full teacher-vs-student comparison requires OrtInferenceManager + EmbeddingManager
    // which may not always be available. Use average quality score as proxy.
    computeReadiness();
}

void DistillationManager::computeReadiness()
{
    if (!m_memoryStore) return;

    double avgScore = m_memoryStore->getAverageDistillationScore(20);

    // If we have eval data, use it
    if (avgScore > 0.0) {
        m_readinessScore = avgScore;
    } else {
        // Proxy: use average quality score of collected pairs as rough estimate
        // More pairs = closer to readiness (asymptotic)
        double pairFactor = 1.0 - (1.0 / (1.0 + m_pairsCollected / 50.0));
        m_readinessScore = pairFactor * 0.5;  // max 0.5 from pair count alone
    }

    emit readinessScoreChanged();
    qDebug() << "DistillationManager: readiness score:" << m_readinessScore
             << "pairs:" << m_pairsCollected;

    // Check graduation threshold
    if (m_readinessScore >= GRADUATION_THRESHOLD && !m_graduationThoughtGenerated) {
        if (m_thoughtEngine) {
            ThoughtRecord thought;
            thought.type = "distillation_ready";
            thought.title = "Personal Model Progress";
            thought.content = "Your personal model is showing strong convergence with your chat model. "
                             "You may be ready to try offline mode — switch your provider to 'Local' in Settings "
                             "to use your personalized model without an internet connection.";
            thought.priority = 0.9;
            thought.sourceType = "distillation";
            thought.generatedBy = "DistillationManager";
            m_thoughtEngine->insertThought(thought);
            m_graduationThoughtGenerated = true;
            qDebug() << "DistillationManager: graduation thought generated!";
        }
    }
}

// --- Helpers ---

void DistillationManager::setPhase(Phase phase)
{
    m_phase = phase;
    switch (phase) {
    case Idle:            m_currentPhase = "Idle"; break;
    case SelectSource:    m_currentPhase = "Selecting Source"; break;
    case EnhancePrompt:   m_currentPhase = "Enhancing Prompt"; break;
    case GenerateTeacher: m_currentPhase = "Generating Teacher"; break;
    case QualityCheck:    m_currentPhase = "Quality Check"; break;
    case StorePair:       m_currentPhase = "Storing Pair"; break;
    }
    emit currentPhaseChanged();
}

void DistillationManager::setStatus(const QString &status)
{
    if (m_statusMessage != status) {
        m_statusMessage = status;
        emit statusMessageChanged();
    }
}

void DistillationManager::refreshCounts()
{
    if (!m_memoryStore) return;
    int total = m_memoryStore->distillationPairCount();
    int used = m_memoryStore->usedDistillationPairCount();
    if (total != m_pairsCollected) {
        m_pairsCollected = total;
        emit pairsCollectedChanged();
    }
    if (used != m_pairsUsedInTraining) {
        m_pairsUsedInTraining = used;
        emit pairsUsedInTrainingChanged();
    }
}
