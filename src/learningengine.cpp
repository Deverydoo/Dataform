#include "learningengine.h"
#include "llmresponseparser.h"
#include "memorystore.h"
#include "llmprovider.h"
#include "thoughtengine.h"
#include "researchengine.h"
#include "settingsmanager.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QTimer>

LearningEngine::LearningEngine(QObject *parent)
    : QObject(parent)
{
}

LearningEngine::~LearningEngine()
{
}

void LearningEngine::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void LearningEngine::setLLMProvider(LLMProviderManager *provider) {
    m_llmProvider = provider;
    if (provider) {
        connect(provider, &LLMProviderManager::backgroundResponseReceived,
                this, [this](const QString &owner, const QString &response) {
            if (owner == "LearningEngine") onLLMResponse(response);
        });
        connect(provider, &LLMProviderManager::backgroundErrorOccurred,
                this, [this](const QString &owner, const QString &error) {
            if (owner == "LearningEngine") onLLMError(error);
        });
    }
}
void LearningEngine::setThoughtEngine(ThoughtEngine *engine) { m_thoughtEngine = engine; }
void LearningEngine::setResearchEngine(ResearchEngine *engine) { m_researchEngine = engine; }
void LearningEngine::setSettingsManager(SettingsManager *settings) { m_settingsManager = settings; }

void LearningEngine::onIdleWindowOpened()
{
    m_idleWindowOpen = true;
    m_paused = false;

    // During idle, check if there's an active plan needing a lesson thought
    if (canStartCycle()) {
        refreshState();
        if (m_hasActivePlan && m_currentPlanId > 0) {
            m_isProcessing = true;
            setPhase(GenerateThought);
            phaseGenerateThought();
        }
    }
}

void LearningEngine::onIdleWindowClosed()
{
    m_idleWindowOpen = false;
    m_consecutiveErrors = 0;  // Fresh idle window gets a fresh chance
    if (m_isProcessing) m_paused = true;
}

bool LearningEngine::canStartCycle() const
{
    if (!m_settingsManager || !m_settingsManager->teachMeEnabled()) return false;
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

void LearningEngine::startLearningPlan(const QString &topic)
{
    if (!m_llmProvider || !m_memoryStore) return;
    if (topic.trimmed().isEmpty()) return;

    m_pendingTopic = topic.trimmed();
    m_isProcessing = true;
    setPhase(GeneratePlan);
    phaseGeneratePlan();
}

QVariantList LearningEngine::getActivePlansForQml()
{
    if (!m_memoryStore) return {};
    return m_memoryStore->getActivePlansForQml();
}

void LearningEngine::setPhase(Phase phase)
{
    m_phase = phase;
}

void LearningEngine::advancePhase()
{
    if (m_paused || !m_idleWindowOpen) {
        m_isProcessing = false;
        setPhase(Idle);
        return;
    }

    switch (m_phase) {
    case GeneratePlan:
        // After plan created, research lesson topics
        setPhase(ResearchLesson);
        phaseResearchLesson();
        break;
    case ResearchLesson:
        setPhase(GenerateThought);
        phaseGenerateThought();
        break;
    case GenerateThought:
        m_isProcessing = false;
        m_lastCycleEndTime = QDateTime::currentDateTime();
        setPhase(Idle);
        break;
    default:
        m_isProcessing = false;
        setPhase(Idle);
        break;
    }
}

void LearningEngine::phaseGeneratePlan()
{
    if (m_pendingTopic.isEmpty()) {
        m_isProcessing = false;
        setPhase(Idle);
        return;
    }

    QString prompt = QString(
        "Create a 5-lesson learning plan for: \"%1\"\n\n"
        "Output a JSON object. Example: {\"lessons\":[{\"title\":\"Intro\",\"description\":\"Basics\",\"keyTopics\":[\"topic1\"]}]}\n"
        "Make exactly 5 lessons, each building on the previous.\n"
        "IMPORTANT: Output ONLY the JSON object starting with { and ending with }. No other text."
    ).arg(m_pendingTopic);

    QJsonArray msgs;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    msgs.append(msg);
    m_llmProvider->sendBackgroundRequest("LearningEngine",
        "You are a JSON-only API. You output valid JSON and nothing else. Only JSON.",
        msgs, LLMProviderManager::PriorityLow);
}

void LearningEngine::phaseResearchLesson()
{
    if (!m_researchEngine || !m_memoryStore) {
        advancePhase();
        return;
    }

    refreshState();
    if (!m_hasActivePlan || m_currentPlanId < 0) {
        advancePhase();
        return;
    }

    // Get current lesson's key topics and queue them for research
    QList<LearningPlanRecord> plans = m_memoryStore->getActiveLearningPlans();
    for (const auto &plan : plans) {
        QJsonDocument doc = QJsonDocument::fromJson(plan.planJson.toUtf8());
        if (!doc.isObject()) continue;

        QJsonArray lessons = doc.object().value("lessons").toArray();
        if (plan.currentLesson < lessons.size()) {
            QJsonObject lesson = lessons[plan.currentLesson].toObject();
            QJsonArray keyTopics = lesson.value("keyTopics").toArray();
            for (const auto &topic : keyTopics) {
                QString t = topic.toString();
                if (!t.isEmpty()) {
                    m_researchEngine->queueTopic(t, 0.9);
                    qDebug() << "LearningEngine: queued research topic:" << t;
                }
            }
        }
        break; // Only process first active plan
    }

    advancePhase();
}

void LearningEngine::phaseGenerateThought()
{
    if (!m_thoughtEngine || !m_memoryStore) {
        advancePhase();
        return;
    }

    refreshState();
    if (!m_hasActivePlan || m_currentPlanId < 0) {
        m_isProcessing = false;
        setPhase(Idle);
        return;
    }

    // Get current lesson info
    QList<LearningPlanRecord> plans = m_memoryStore->getActiveLearningPlans();
    for (const auto &plan : plans) {
        QJsonDocument doc = QJsonDocument::fromJson(plan.planJson.toUtf8());
        if (!doc.isObject()) continue;

        QJsonArray lessons = doc.object().value("lessons").toArray();
        if (plan.currentLesson < lessons.size()) {
            QJsonObject lesson = lessons[plan.currentLesson].toObject();
            QString lessonTitle = lesson.value("title").toString();
            QString lessonDesc = lesson.value("description").toString();

            ThoughtRecord thought;
            thought.type = "lesson_ready";
            thought.title = QString("Lesson %1: %2").arg(plan.currentLesson + 1).arg(lessonTitle);
            thought.content = QString("Ready to continue learning about %1? "
                                      "Today's lesson covers: %2").arg(plan.topic, lessonDesc);
            thought.priority = 0.85;
            thought.sourceType = "learning";
            thought.sourceId = plan.id;
            thought.generatedBy = "learning_engine";
            m_thoughtEngine->insertThought(thought);

            emit lessonReady(plan.topic, plan.currentLesson + 1);
            qDebug() << "LearningEngine: lesson thought ready for" << plan.topic
                     << "lesson" << plan.currentLesson + 1;
        }
        break;
    }

    m_isProcessing = false;
    m_lastCycleEndTime = QDateTime::currentDateTime();
    setPhase(Idle);
}

void LearningEngine::onLLMResponse(const QString &response)
{
    m_consecutiveErrors = 0;  // Reset on any successful LLM response

    if (m_phase != GeneratePlan) return;

    QJsonDocument doc = LLMResponseParser::extractJsonObject(response, "LearningEngine");
    if (doc.isNull()) {
        m_isProcessing = false;
        setPhase(Idle);
        return;
    }

    QJsonArray lessons = doc.object().value("lessons").toArray();
    int totalLessons = lessons.size();
    if (totalLessons == 0) {
        qDebug() << "LearningEngine: no lessons in plan";
        m_isProcessing = false;
        setPhase(Idle);
        return;
    }

    qint64 planId = m_memoryStore->insertLearningPlan(
        m_pendingTopic, QString(QJsonDocument(doc.object()).toJson(QJsonDocument::Compact)), totalLessons);

    if (planId >= 0) {
        emit planCreated(m_pendingTopic, totalLessons);
        qDebug() << "LearningEngine: created plan for" << m_pendingTopic << "with" << totalLessons << "lessons";
        refreshState();
    }

    m_pendingTopic.clear();

    // If idle, continue to research phase
    if (m_idleWindowOpen) {
        advancePhase();
    } else {
        m_isProcessing = false;
        setPhase(Idle);
    }
}

void LearningEngine::onLLMError(const QString &error)
{
    m_consecutiveErrors++;
    qDebug() << "LearningEngine: LLM error:" << error;
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        qWarning() << "LearningEngine: pausing after" << m_consecutiveErrors << "consecutive errors";
    }
    m_isProcessing = false;
    m_pendingTopic.clear();
    setPhase(Idle);
}

void LearningEngine::refreshState()
{
    if (!m_memoryStore) return;

    QList<LearningPlanRecord> plans = m_memoryStore->getActiveLearningPlans();
    bool hadPlan = m_hasActivePlan;

    if (!plans.isEmpty()) {
        const auto &plan = plans.first();
        m_hasActivePlan = true;
        m_currentPlanId = plan.id;
        if (m_currentTopic != plan.topic) { m_currentTopic = plan.topic; emit currentTopicChanged(); }
        if (m_currentLesson != plan.currentLesson) { m_currentLesson = plan.currentLesson; emit currentLessonChanged(); }
        if (m_totalLessons != plan.totalLessons) { m_totalLessons = plan.totalLessons; emit totalLessonsChanged(); }
    } else {
        m_hasActivePlan = false;
        m_currentPlanId = -1;
        m_currentTopic.clear();
        m_currentLesson = 0;
        m_totalLessons = 0;
        emit currentTopicChanged();
        emit currentLessonChanged();
        emit totalLessonsChanged();
    }

    if (hadPlan != m_hasActivePlan) emit hasActivePlanChanged();
}
