#ifndef LEARNINGENGINE_H
#define LEARNINGENGINE_H

#include <QObject>
#include <QString>
#include <QDateTime>

class MemoryStore;
class LLMProviderManager;
class ThoughtEngine;
class ResearchEngine;
class SettingsManager;

class LearningEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool hasActivePlan READ hasActivePlan NOTIFY hasActivePlanChanged)
    Q_PROPERTY(QString currentTopic READ currentTopic NOTIFY currentTopicChanged)
    Q_PROPERTY(int currentLesson READ currentLesson NOTIFY currentLessonChanged)
    Q_PROPERTY(int totalLessons READ totalLessons NOTIFY totalLessonsChanged)

public:
    explicit LearningEngine(QObject *parent = nullptr);
    ~LearningEngine();

    void setMemoryStore(MemoryStore *store);
    void setLLMProvider(LLMProviderManager *provider);
    void setThoughtEngine(ThoughtEngine *engine);
    void setResearchEngine(ResearchEngine *engine);
    void setSettingsManager(SettingsManager *settings);

    bool hasActivePlan() const { return m_hasActivePlan; }
    QString currentTopic() const { return m_currentTopic; }
    int currentLesson() const { return m_currentLesson; }
    int totalLessons() const { return m_totalLessons; }

    Q_INVOKABLE void startLearningPlan(const QString &topic);
    Q_INVOKABLE QVariantList getActivePlansForQml();
    bool canStartCycle() const;
    void requestStart();

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();

signals:
    void hasActivePlanChanged();
    void currentTopicChanged();
    void currentLessonChanged();
    void totalLessonsChanged();
    void planCreated(const QString &topic, int lessons);
    void lessonReady(const QString &topic, int lessonNum);
    void cycleFinished();

private slots:
    void onLLMResponse(const QString &response);
    void onLLMError(const QString &error);

private:
    enum Phase { Idle, GeneratePlan, ResearchLesson, GenerateThought };

    void advancePhase();
    void setPhase(Phase phase);

    void phaseGeneratePlan();
    void phaseResearchLesson();
    void phaseGenerateThought();

    void refreshState();

    MemoryStore *m_memoryStore = nullptr;
    LLMProviderManager *m_llmProvider = nullptr;
    ThoughtEngine *m_thoughtEngine = nullptr;
    ResearchEngine *m_researchEngine = nullptr;
    SettingsManager *m_settingsManager = nullptr;

    bool m_idleWindowOpen = false;
    bool m_paused = false;
    bool m_isProcessing = false;
    Phase m_phase = Idle;

    bool m_hasActivePlan = false;
    QString m_currentTopic;
    int m_currentLesson = 0;
    int m_totalLessons = 0;
    qint64 m_currentPlanId = -1;
    QString m_pendingTopic;

    QDateTime m_lastCycleEndTime;
    static constexpr int CYCLE_COOLDOWN_SEC = 3600;

    // Error recovery
    int m_consecutiveErrors = 0;
    static constexpr int MAX_CONSECUTIVE_ERRORS = 3;
};

#endif // LEARNINGENGINE_H
