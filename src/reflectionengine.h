#ifndef REFLECTIONENGINE_H
#define REFLECTIONENGINE_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>

class MemoryStore;
class Tokenizer;
class TrainingDataGenerator;
class OrtTrainingManager;
class AdapterManager;
class EvalSuite;
class IdleScheduler;
class ProfileManager;
class SettingsManager;
class LLMProviderManager;
class OrtInferenceManager;

class ReflectionEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString phase READ phase NOTIFY phaseChanged)
    Q_PROPERTY(bool isReflecting READ isReflecting NOTIFY isReflectingChanged)
    Q_PROPERTY(float trainingLoss READ trainingLoss NOTIFY trainingLossChanged)
    Q_PROPERTY(int trainingStep READ trainingStep NOTIFY trainingStepChanged)
    Q_PROPERTY(int totalTrainingSteps READ totalTrainingSteps NOTIFY totalTrainingStepsChanged)
    Q_PROPERTY(QString reflectionStatus READ reflectionStatus NOTIFY reflectionStatusChanged)
    Q_PROPERTY(int sessionsCompleted READ sessionsCompleted NOTIFY sessionsCompletedChanged)

public:
    explicit ReflectionEngine(QObject *parent = nullptr);
    ~ReflectionEngine();

    // Dependency injection
    void setMemoryStore(MemoryStore *store);
    void setTokenizer(Tokenizer *tokenizer);
    void setTrainingDataGenerator(TrainingDataGenerator *generator);
    void setOrtTrainingManager(OrtTrainingManager *trainingManager);
    void setAdapterManager(AdapterManager *adapterManager);
    void setEvalSuite(EvalSuite *evalSuite);
    void setIdleScheduler(IdleScheduler *scheduler);
    void setProfileManager(ProfileManager *profileManager);
    void setSettingsManager(SettingsManager *settingsManager);
    void setLLMProvider(LLMProviderManager *llmProvider);
    void setOrtInferenceManager(OrtInferenceManager *mgr);

    // Property getters
    QString phase() const { return m_currentPhase; }
    bool isReflecting() const { return m_isReflecting; }
    float trainingLoss() const;
    int trainingStep() const;
    int totalTrainingSteps() const;
    QString reflectionStatus() const { return m_reflectionStatus; }
    int sessionsCompleted() const { return m_sessionsCompleted; }

    // Manual triggers
    Q_INVOKABLE void triggerReflection();
    Q_INVOKABLE void pauseReflection();

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();

signals:
    void phaseChanged();
    void isReflectingChanged();
    void trainingLossChanged();
    void trainingStepChanged();
    void totalTrainingStepsChanged();
    void reflectionStatusChanged();
    void sessionsCompletedChanged();
    void reflectionComplete(bool adapterPromoted, double evalScore);
    void reflectionError(const QString &error);
    void adapterPromoted(int version, double score);

private slots:
    void onTrainingComplete(int stepsCompleted, float finalLoss);
    void onTrainingPaused(int stepsCompleted);
    void onTrainingError(const QString &error);

private:
    // Pipeline phases
    void runPhase1_SelectEpisodes();
    void runPhase2_GenerateTrainingData();
    void runPhase3_TrainAdapter();
    void runPhase4_EvaluateCandidate();
    void runPhase5_PromoteOrReject(int candidateVersion, double evalScore);

    bool hasEnoughNewData() const;
    int computeMaxSteps() const;
    void setPhase(const QString &phase);
    void setReflectionStatus(const QString &status);

    // Dependencies
    MemoryStore *m_memoryStore = nullptr;
    Tokenizer *m_tokenizer = nullptr;
    TrainingDataGenerator *m_dataGenerator = nullptr;
    OrtTrainingManager *m_trainingManager = nullptr;
    AdapterManager *m_adapterManager = nullptr;
    EvalSuite *m_evalSuite = nullptr;
    IdleScheduler *m_idleScheduler = nullptr;
    ProfileManager *m_profileManager = nullptr;
    SettingsManager *m_settingsManager = nullptr;
    LLMProviderManager *m_llmProvider = nullptr;
    OrtInferenceManager *m_ortInference = nullptr;

    // State
    bool m_isReflecting = false;
    QString m_currentPhase = "idle";
    QString m_reflectionStatus = "Waiting for idle window";
    int m_sessionsCompleted = 0;

    // Training result tracking
    int m_candidateVersion = -1;
    float m_lastTrainingLoss = 0.0f;

    static constexpr int MIN_NEW_EPISODES = 5;
    static constexpr int MIN_HIGH_SIGNAL = 3;
    static constexpr double PROMOTION_MARGIN = 0.02;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // REFLECTIONENGINE_H
