#include "goaltracker.h"
#include "memorystore.h"
#include "llmprovider.h"
#include "thoughtengine.h"
#include "settingsmanager.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QRegularExpression>
#include <QTimer>

GoalTracker::GoalTracker(QObject *parent)
    : QObject(parent)
{
}

GoalTracker::~GoalTracker()
{
}

void GoalTracker::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void GoalTracker::setLLMProvider(LLMProviderManager *provider) {
    m_llmProvider = provider;
    if (provider) {
        connect(provider, &LLMProviderManager::backgroundResponseReceived,
                this, [this](const QString &owner, const QString &response) {
            if (owner == "GoalTracker") onLLMResponse(response);
        });
        connect(provider, &LLMProviderManager::backgroundErrorOccurred,
                this, [this](const QString &owner, const QString &error) {
            if (owner == "GoalTracker") onLLMError(error);
        });
    }
}
void GoalTracker::setThoughtEngine(ThoughtEngine *engine) { m_thoughtEngine = engine; }
void GoalTracker::setSettingsManager(SettingsManager *settings) { m_settingsManager = settings; }

void GoalTracker::onIdleWindowOpened()
{
    m_idleWindowOpen = true;
    m_paused = false;
    if (canStartCycle()) startCycle();
}

void GoalTracker::onIdleWindowClosed()
{
    m_idleWindowOpen = false;
    m_consecutiveErrors = 0;  // Fresh idle window gets a fresh chance
    if (m_isProcessing) m_paused = true;
}

bool GoalTracker::canStartCycle() const
{
    if (!m_settingsManager || !m_settingsManager->goalsEnabled()) return false;
    if (m_settingsManager && m_settingsManager->privacyLevel() == "minimal") return false;
    if (!m_memoryStore || !m_llmProvider || !m_thoughtEngine) return false;
    if (m_isProcessing || m_paused) return false;
    // Error recovery: pause after too many consecutive errors
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) return false;
    if (m_lastCycleEndTime.isValid()) {
        qint64 elapsed = m_lastCycleEndTime.secsTo(QDateTime::currentDateTime());
        if (elapsed < CYCLE_COOLDOWN_SEC) return false;
    }
    return true;
}

void GoalTracker::startCycle()
{
    m_isProcessing = true;
    setPhase(DetectGoals);
    phaseDetectGoals();
}

void GoalTracker::setPhase(Phase phase)
{
    m_phase = phase;
}

void GoalTracker::advancePhase()
{
    if (m_paused || !m_idleWindowOpen) {
        m_isProcessing = false;
        setPhase(Idle);
        return;
    }

    switch (m_phase) {
    case DetectGoals:
        setPhase(GenerateCheckin);
        phaseGenerateCheckin();
        break;
    case GenerateCheckin:
        m_isProcessing = false;
        m_lastCycleEndTime = QDateTime::currentDateTime();
        setPhase(Idle);
        refreshGoalCount();
        // Auto-retry
        if (m_idleWindowOpen && canStartCycle()) {
            QTimer::singleShot(10000, this, [this]() {
                if (m_idleWindowOpen && canStartCycle()) startCycle();
            });
        }
        break;
    default:
        m_isProcessing = false;
        setPhase(Idle);
        break;
    }
}

void GoalTracker::phaseDetectGoals()
{
    if (!m_memoryStore || !m_llmProvider) {
        advancePhase();
        return;
    }

    // Get recent episodes that haven't been scanned
    QList<EpisodicRecord> episodes = m_memoryStore->getRecentEpisodes(10);
    QStringList userMessages;
    for (const auto &ep : episodes) {
        if (m_scannedEpisodeIds.contains(ep.id)) continue;
        if (!ep.userText.isEmpty()) {
            userMessages.append(ep.userText.left(200));
            m_scannedEpisodeIds.insert(ep.id);
        }
    }

    if (userMessages.isEmpty()) {
        advancePhase();
        return;
    }

    // Keep scanned set bounded
    while (m_scannedEpisodeIds.size() > 200) {
        m_scannedEpisodeIds.erase(m_scannedEpisodeIds.begin());
    }

    QString prompt = QString(
        "Identify goals or intentions in these user messages.\n\n"
        "%1\n\n"
        "Output a JSON array. Example: [{\"title\":\"Learn Spanish\",\"description\":\"User wants to learn Spanish for travel\"}]\n"
        "If no goals found, return []\n"
        "IMPORTANT: Output ONLY the JSON array starting with [ and ending with ]. No other text."
    ).arg(userMessages.join("\n---\n"));

    QJsonArray msgs;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    msgs.append(msg);
    m_llmProvider->sendBackgroundRequest("GoalTracker",
        "You are a JSON-only API. You output valid JSON arrays and nothing else. "
        "Never include explanation, analysis, or commentary. Only JSON.",
        msgs, LLMProviderManager::PriorityLow);
}

void GoalTracker::phaseGenerateCheckin()
{
    if (!m_memoryStore || !m_thoughtEngine) {
        advancePhase();
        return;
    }

    QList<GoalRecord> staleGoals = m_memoryStore->getGoalsNeedingCheckin(CHECKIN_INTERVAL_DAYS);
    if (staleGoals.isEmpty()) {
        advancePhase();
        return;
    }

    // Generate check-in thought for the first stale goal
    const GoalRecord &goal = staleGoals.first();
    m_currentCheckinGoalId = goal.id;
    m_currentCheckinGoalTitle = goal.title;

    ThoughtRecord thought;
    thought.type = "goal_checkin";
    thought.title = QString("Checking in: %1").arg(goal.title);
    thought.content = QString("You mentioned \"%1\" a while ago. How's that going? "
                              "I'd love to hear about your progress.").arg(goal.title);
    thought.priority = 0.75;
    thought.sourceType = "goal";
    thought.sourceId = goal.id;
    thought.generatedBy = "goal_tracker";
    m_thoughtEngine->insertThought(thought);

    m_memoryStore->updateGoalCheckin(goal.id);
    emit checkinGenerated(goal.id);
    qDebug() << "GoalTracker: generated check-in for goal:" << goal.title;

    advancePhase();
}

void GoalTracker::onLLMResponse(const QString &response)
{
    m_consecutiveErrors = 0;  // Reset on any successful LLM response

    if (m_phase != DetectGoals) return;

    // Parse JSON array of goals
    QString cleaned = response.trimmed();
    // Strip <think>...</think> reasoning blocks (qwen3 and other reasoning models)
    static const QRegularExpression thinkRegex(
        "<think>.*?</think>",
        QRegularExpression::DotMatchesEverythingOption);
    cleaned.remove(thinkRegex);
    cleaned = cleaned.trimmed();
    // Strip markdown fences
    if (cleaned.startsWith("```")) {
        int start = cleaned.indexOf('\n') + 1;
        int end = cleaned.lastIndexOf("```");
        if (end > start) cleaned = cleaned.mid(start, end - start).trimmed();
    }

    QJsonDocument doc = QJsonDocument::fromJson(cleaned.toUtf8());
    if (!doc.isArray()) {
        qDebug() << "GoalTracker: LLM response not a JSON array, skipping";
        advancePhase();
        return;
    }

    QJsonArray goals = doc.array();
    int detected = 0;
    for (const QJsonValue &val : goals) {
        QJsonObject obj = val.toObject();
        QString title = obj.value("title").toString().trimmed();
        QString description = obj.value("description").toString().trimmed();
        if (title.isEmpty()) continue;

        // Check for duplicate goals
        QList<GoalRecord> existing = m_memoryStore->getActiveGoals();
        bool isDuplicate = false;
        for (const auto &g : existing) {
            if (g.title.toLower().contains(title.toLower().left(20)) ||
                title.toLower().contains(g.title.toLower().left(20))) {
                isDuplicate = true;
                break;
            }
        }

        if (!isDuplicate) {
            m_memoryStore->insertGoal(title, description);
            emit goalDetected(title);
            detected++;
            qDebug() << "GoalTracker: detected new goal:" << title;
        }
    }

    if (detected > 0) refreshGoalCount();
    advancePhase();
}

void GoalTracker::onLLMError(const QString &error)
{
    m_consecutiveErrors++;
    qDebug() << "GoalTracker: LLM error:" << error;
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        qWarning() << "GoalTracker: pausing after" << m_consecutiveErrors << "consecutive errors";
    }
    advancePhase();
}

void GoalTracker::refreshGoalCount()
{
    if (!m_memoryStore) return;
    int count = m_memoryStore->getActiveGoals().size();
    if (m_activeGoalCount != count) {
        m_activeGoalCount = count;
        emit activeGoalCountChanged();
    }
}
