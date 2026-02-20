#ifndef DISTILLATIONMANAGER_H
#define DISTILLATIONMANAGER_H

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QTimer>
#include <QSet>

class MemoryStore;
class LLMProviderManager;
class SettingsManager;
class ThoughtEngine;
class EmbeddingManager;

#ifdef DATAFORM_TRAINING_ENABLED
class OrtInferenceManager;
#endif

class DistillationManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isDistilling READ isDistilling NOTIFY isDistillingChanged)
    Q_PROPERTY(QString currentPhase READ currentPhase NOTIFY currentPhaseChanged)
    Q_PROPERTY(int pairsCollected READ pairsCollected NOTIFY pairsCollectedChanged)
    Q_PROPERTY(int pairsUsedInTraining READ pairsUsedInTraining NOTIFY pairsUsedInTrainingChanged)
    Q_PROPERTY(double readinessScore READ readinessScore NOTIFY readinessScoreChanged)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)

public:
    explicit DistillationManager(QObject *parent = nullptr);
    ~DistillationManager();

    void setMemoryStore(MemoryStore *store);
    void setLLMProvider(LLMProviderManager *provider);
    void setSettingsManager(SettingsManager *settings);
    void setThoughtEngine(ThoughtEngine *engine);
    void setEmbeddingManager(EmbeddingManager *embeddings);

#ifdef DATAFORM_TRAINING_ENABLED
    void setOrtInferenceManager(OrtInferenceManager *inference);
#endif

    bool isDistilling() const { return m_isDistilling; }
    QString currentPhase() const { return m_currentPhase; }
    int pairsCollected() const { return m_pairsCollected; }
    int pairsUsedInTraining() const { return m_pairsUsedInTraining; }
    double readinessScore() const { return m_readinessScore; }
    QString statusMessage() const { return m_statusMessage; }

    Q_INVOKABLE void distillNow();
    bool canStartCycle() const;
    void requestStart();

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();

signals:
    void isDistillingChanged();
    void currentPhaseChanged();
    void pairsCollectedChanged();
    void pairsUsedInTrainingChanged();
    void readinessScoreChanged();
    void statusMessageChanged();
    void distillationPairStored(qint64 pairId);
    void graduationEvalComplete(double score);
    void cycleFinished();

private slots:
    void onTeacherResponse(const QString &owner, const QString &response);
    void onTeacherError(const QString &owner, const QString &error);

private:
    enum Phase {
        Idle,
        SelectSource,
        EnhancePrompt,
        GenerateTeacher,
        QualityCheck,
        StorePair
    };

    // Pipeline phases
    void startCycle();
    void phaseSelectSource();
    void phaseEnhancePrompt(qint64 episodeId, const QString &userPrompt,
                            const QString &sourceType);
    void phaseGenerateTeacher(const QString &systemPrompt, const QString &userPrompt,
                              qint64 sourceEpisodeId, const QString &sourceType);
    void phaseQualityCheck(const QString &userPrompt, const QString &teacherResponse,
                           const QString &systemContext, qint64 sourceEpisodeId,
                           const QString &sourceType);
    void phaseStore(const QString &userPrompt, const QString &teacherResponse,
                    const QString &systemContext, qint64 sourceEpisodeId,
                    const QString &sourceType, double qualityScore);

    // Graduation evaluation
    void runGraduationEval();
    void computeReadiness();

    // Source selection
    QString selectFromHighSignalEpisode(qint64 &episodeId);
    QString synthesizeFromTraits();
    QString synthesizeFromResearch();

    // Helpers
    QString buildIdentityContext() const;
    void resetDailyCounterIfNeeded();
    void setPhase(Phase phase);
    void setStatus(const QString &status);
    void refreshCounts();
    void finishCycleAndScheduleNext();

    // Dependencies
    MemoryStore *m_memoryStore = nullptr;
    LLMProviderManager *m_llmProvider = nullptr;
    SettingsManager *m_settingsManager = nullptr;
    ThoughtEngine *m_thoughtEngine = nullptr;
    EmbeddingManager *m_embeddingManager = nullptr;

#ifdef DATAFORM_TRAINING_ENABLED
    OrtInferenceManager *m_ortInference = nullptr;
#endif

    // State
    bool m_isDistilling = false;
    bool m_idleWindowOpen = false;
    Phase m_phase = Idle;
    QString m_currentPhase;
    QString m_statusMessage;

    // Current cycle state
    QString m_pendingUserPrompt;
    QString m_pendingSystemContext;
    qint64 m_pendingSourceEpisodeId = -1;
    QString m_pendingSourceType;

    // Tracking
    int m_pairsCollected = 0;
    int m_pairsUsedInTraining = 0;
    int m_pairsThisSession = 0;
    int m_cyclesCompletedToday = 0;
    QDate m_lastCycleDate;
    double m_readinessScore = 0.0;
    bool m_graduationThoughtGenerated = false;

    // Episodes already used as distillation sources
    QSet<qint64> m_usedSourceEpisodeIds;

    // Cooldown
    QDateTime m_lastCycleEndTime;
    QTimer m_cycleTimer;

    // Constants
    static constexpr int MAX_PAIRS_PER_SESSION = 5;
    static constexpr int GRADUATION_EVAL_INTERVAL = 10;
    static constexpr double GRADUATION_THRESHOLD = 0.75;
    static constexpr int CYCLE_COOLDOWN_SEC = 60;

    // Error recovery
    int m_consecutiveErrors = 0;
    static constexpr int MAX_CONSECUTIVE_ERRORS = 3;

    static const QString OWNER_TAG;
};

#endif // DISTILLATIONMANAGER_H
