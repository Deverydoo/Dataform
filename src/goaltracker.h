#ifndef GOALTRACKER_H
#define GOALTRACKER_H

#include <QObject>
#include <QString>
#include <QList>
#include <QDate>
#include <QDateTime>

class MemoryStore;
class LLMProviderManager;
class ThoughtEngine;
class SettingsManager;

class GoalTracker : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int activeGoalCount READ activeGoalCount NOTIFY activeGoalCountChanged)

public:
    explicit GoalTracker(QObject *parent = nullptr);
    ~GoalTracker();

    void setMemoryStore(MemoryStore *store);
    void setLLMProvider(LLMProviderManager *provider);
    void setThoughtEngine(ThoughtEngine *engine);
    void setSettingsManager(SettingsManager *settings);

    int activeGoalCount() const { return m_activeGoalCount; }
    bool canStartCycle() const;
    void requestStart();

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();

signals:
    void activeGoalCountChanged();
    void goalDetected(const QString &title);
    void checkinGenerated(qint64 goalId);
    void cycleFinished();

private slots:
    void onLLMResponse(const QString &response);
    void onLLMError(const QString &error);

private:
    enum Phase { Idle, DetectGoals, GenerateCheckin };

    void startCycle();
    void advancePhase();
    void setPhase(Phase phase);

    void phaseDetectGoals();
    void phaseGenerateCheckin();

    void refreshGoalCount();

    MemoryStore *m_memoryStore = nullptr;
    LLMProviderManager *m_llmProvider = nullptr;
    ThoughtEngine *m_thoughtEngine = nullptr;
    SettingsManager *m_settingsManager = nullptr;

    bool m_idleWindowOpen = false;
    bool m_paused = false;
    bool m_isProcessing = false;
    Phase m_phase = Idle;
    int m_activeGoalCount = 0;

    QList<qint64> m_scannedEpisodeIds;
    qint64 m_currentCheckinGoalId = -1;
    QString m_currentCheckinGoalTitle;

    QDateTime m_lastCycleEndTime;
    static constexpr int CYCLE_COOLDOWN_SEC = 1800;

    // Error recovery
    int m_consecutiveErrors = 0;
    static constexpr int MAX_CONSECUTIVE_ERRORS = 3;
};

#endif // GOALTRACKER_H
