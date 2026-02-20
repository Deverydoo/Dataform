#include "goaltracker.h"
#include "llmresponseparser.h"
#include "memorystore.h"
#include "llmprovider.h"
#include "thoughtengine.h"
#include "settingsmanager.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
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
    // Auto-start removed — coordinator calls requestStart()
}

void GoalTracker::requestStart()
{
    if (canStartCycle()) startCycle();
    else emit cycleFinished();
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
    Phase old = m_phase;
    m_phase = phase;
    if (phase == Idle && old != Idle) {
        emit cycleFinished();
    }
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
        // Auto-retry removed — coordinator handles scheduling
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
            m_scannedEpisodeIds.append(ep.id);
        }
    }

    if (userMessages.isEmpty()) {
        advancePhase();
        return;
    }

    // Keep scanned list bounded (FIFO eviction preserves insertion order)
    while (m_scannedEpisodeIds.size() > 200) {
        m_scannedEpisodeIds.removeFirst();
    }

    QString prompt = QString(
        "Identify goals or intentions in these user messages.\n\n"
        "%1\n\n"
        "For each goal, estimate how often we should check in based on its nature:\n"
        "- Daily habits (exercise, journaling, meditation): checkin_days = 1-2\n"
        "- Weekly tasks (projects, learning): checkin_days = 7\n"
        "- Growing/planting/seasonal (gardens, crops): checkin_days = 14-30\n"
        "- Long-term aspirations (career, big projects): checkin_days = 30-90\n\n"
        "Output a JSON array. Example:\n"
        "[{\"title\":\"Exercise daily\",\"description\":\"Daily morning routine\",\"checkin_days\":1},\n"
        " {\"title\":\"Grow sunchokes\",\"description\":\"Planting for harvest\",\"checkin_days\":21}]\n"
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

    QList<GoalRecord> staleGoals = m_memoryStore->getGoalsNeedingCheckin();
    if (staleGoals.isEmpty()) {
        advancePhase();
        return;
    }

    // Generate check-in thought for the first stale goal
    const GoalRecord &goal = staleGoals.first();
    m_currentCheckinGoalId = goal.id;
    m_currentCheckinGoalTitle = goal.title;

    // Calculate time since goal was created
    qint64 daysAgo = goal.createdTs.daysTo(QDateTime::currentDateTime());
    QString timeDesc;
    if (daysAgo < 1)
        timeDesc = "earlier today";
    else if (daysAgo == 1)
        timeDesc = "yesterday";
    else if (daysAgo < 7)
        timeDesc = QString("%1 days ago").arg(daysAgo);
    else if (daysAgo < 30)
        timeDesc = QString("about %1 week%2 ago").arg(daysAgo / 7).arg(daysAgo / 7 > 1 ? "s" : "");
    else
        timeDesc = QString("about %1 month%2 ago").arg(daysAgo / 30).arg(daysAgo / 30 > 1 ? "s" : "");

    // Timing-aware thought message based on check-in interval
    QString content;
    int interval = goal.checkinIntervalDays;
    if (interval <= 2) {
        // Daily habits — encouraging, routine tone
        content = QString("Daily check-in: How did \"%1\" go today? "
                          "Keeping the streak going!").arg(goal.title);
    } else if (interval <= 7) {
        // Weekly tasks — progress-focused
        content = QString("You've been working on \"%1\" (started %2). "
                          "How's this week's progress?").arg(goal.title, timeDesc);
    } else if (interval <= 30) {
        // Growing/planting/medium-term — patience-aware
        content = QString("It's been a little while since we talked about \"%1\" "
                          "(started %2). These things take time — any updates?")
                      .arg(goal.title, timeDesc);
    } else {
        // Long-term aspirations — big picture
        content = QString("Thinking about your long-term goal \"%1\" "
                          "(started %2). Any milestones or shifts in direction?")
                      .arg(goal.title, timeDesc);
    }

    ThoughtRecord thought;
    thought.type = "goal_checkin";
    thought.title = QString("Checking in: %1").arg(goal.title);
    thought.content = content;
    thought.priority = (interval <= 2) ? 0.8 : 0.75;  // Slight boost for daily habits
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

    if (m_phase != DetectGoals || !m_memoryStore) return;

    // Parse JSON array of goals
    QJsonDocument doc = LLMResponseParser::extractJsonArray(response, "GoalTracker");
    if (doc.isNull()) {
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
            int checkinDays = obj.value("checkin_days").toInt(7);
            checkinDays = qBound(1, checkinDays, 365);
            m_memoryStore->insertGoal(title, description, -1, checkinDays);
            emit goalDetected(title);
            detected++;
            qDebug() << "GoalTracker: detected new goal:" << title
                     << "checkin every" << checkinDays << "days";
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
