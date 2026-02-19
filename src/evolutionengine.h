#ifndef EVOLUTIONENGINE_H
#define EVOLUTIONENGINE_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>
#include <QList>
#include "populationstate.h"

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
class WeightMerger;
class LineageTracker;
class OrtInferenceManager;
struct TrainingExample;

class EvolutionEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString evolutionStage READ evolutionStage NOTIFY evolutionStageChanged)
    Q_PROPERTY(bool isEvolving READ isEvolving NOTIFY isEvolvingChanged)
    Q_PROPERTY(int populationSize READ populationSize NOTIFY populationSizeChanged)
    Q_PROPERTY(int currentVariantIndex READ currentVariantIndex NOTIFY currentVariantIndexChanged)
    Q_PROPERTY(int totalVariants READ totalVariants NOTIFY totalVariantsChanged)
    Q_PROPERTY(int cyclesCompleted READ cyclesCompleted NOTIFY cyclesCompletedChanged)
    Q_PROPERTY(int consolidationsCompleted READ consolidationsCompleted NOTIFY consolidationsCompletedChanged)
    Q_PROPERTY(float trainingLoss READ trainingLoss NOTIFY trainingLossChanged)
    Q_PROPERTY(int trainingStep READ trainingStep NOTIFY trainingStepChanged)
    Q_PROPERTY(int totalTrainingSteps READ totalTrainingSteps NOTIFY totalTrainingStepsChanged)
    Q_PROPERTY(QString evolutionStatus READ evolutionStatus NOTIFY evolutionStatusChanged)

public:
    explicit EvolutionEngine(QObject *parent = nullptr);
    ~EvolutionEngine();

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
    void setWeightMerger(WeightMerger *weightMerger);
    void setLineageTracker(LineageTracker *tracker);
    void setOrtInferenceManager(OrtInferenceManager *mgr);

    // Property getters
    QString evolutionStage() const { return m_population.cycleStage; }
    bool isEvolving() const { return m_isEvolving; }
    int populationSize() const;
    int currentVariantIndex() const { return m_population.currentVariantIndex; }
    int totalVariants() const { return m_population.variants.size(); }
    int cyclesCompleted() const { return m_cyclesCompleted; }
    int consolidationsCompleted() const { return m_consolidationsCompleted; }
    float trainingLoss() const;
    int trainingStep() const;
    int totalTrainingSteps() const;
    QString evolutionStatus() const { return m_evolutionStatus; }

    // Manual controls
    Q_INVOKABLE void triggerEvolutionCycle();
    Q_INVOKABLE void pauseEvolution();
    Q_INVOKABLE void triggerConsolidation();
    Q_INVOKABLE QVariantList getPopulationForQml() const;

    // Persistence
    void loadPopulationState();
    void savePopulationState();

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();

signals:
    void evolutionStageChanged();
    void isEvolvingChanged();
    void populationSizeChanged();
    void currentVariantIndexChanged();
    void totalVariantsChanged();
    void cyclesCompletedChanged();
    void consolidationsCompletedChanged();
    void trainingLossChanged();
    void trainingStepChanged();
    void totalTrainingStepsChanged();
    void evolutionStatusChanged();
    void evolutionCycleComplete(bool adapterPromoted, int winnerVersion, double winnerScore);
    void evolutionError(const QString &error);
    void consolidationComplete(int eraSnapshotVersion);
    void variantTrained(int variantIndex, float loss, int steps);
    void variantEvaluated(int variantIndex, double score);

private slots:
    void onTrainingComplete(int stepsCompleted, float finalLoss);
    void onTrainingPaused(int stepsCompleted);
    void onTrainingError(const QString &error);

private:
    // Evolution cycle phases
    void resumeOrStartCycle();
    void seedPopulation();
    void trainNextVariant();
    void onVariantTrainingDone(float finalLoss, int steps);
    void evaluateAllVariants();
    void selectWinner();
    void checkConsolidation();

    // Variant creation
    QList<VariantSpec> generateVariantSpecs(int count);
    QList<TrainingExample> subsampleExamples(const QList<TrainingExample> &all,
                                              float ratio, quint32 seed);

    // Consolidation
    void runConsolidation();
    void createEraSnapshot(const QString &label);

    // Data gating
    bool hasEnoughNewData() const;
    bool shouldConsolidate() const;
    int computeMaxStepsPerVariant() const;
    int computePopulationSize() const;

    // Status helpers
    void setEvolutionStage(const QString &stage);
    void setEvolutionStatus(const QString &status);
    void saveEvolutionLog();
    void cleanupCycleData();

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
    WeightMerger *m_weightMerger = nullptr;
    LineageTracker *m_lineageTracker = nullptr;
    OrtInferenceManager *m_ortInference = nullptr;

    // Evolution state
    PopulationState m_population;
    bool m_isEvolving = false;
    bool m_idleWindowOpen = false;
    QString m_evolutionStatus = "Waiting for idle window";
    int m_cyclesCompleted = 0;
    int m_consolidationsCompleted = 0;

    // Cached training data (generated once per cycle)
    QList<TrainingExample> m_cycleExamples;

    // Constants
    static constexpr int DEFAULT_POPULATION_SIZE = 4;
    static constexpr int MAX_POPULATION_SIZE = 8;
    static constexpr int MIN_NEW_EPISODES = 5;
    static constexpr int MIN_HIGH_SIGNAL = 3;
    static constexpr double PROMOTION_MARGIN = 0.02;
    static constexpr float LR_PERTURBATION_MIN = 0.5f;
    static constexpr float LR_PERTURBATION_MAX = 3.0f;
    static constexpr float MIN_SUBSAMPLE_RATIO = 0.75f;
    static constexpr int CONSOLIDATION_INTERVAL_CYCLES = 4;
    static constexpr int ERA_SNAPSHOT_INTERVAL = 4;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // EVOLUTIONENGINE_H
