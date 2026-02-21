#ifdef DATAFORM_TRAINING_ENABLED

#include "reflectionengine.h"
#include "memorystore.h"
#include "tokenizer.h"
#include "trainingdatagenerator.h"
#include "orttrainingmanager.h"
#include "adaptermanager.h"
#include "evalsuite.h"
#include "idlescheduler.h"
#include "profilemanager.h"
#include "settingsmanager.h"
#include "llmprovider.h"
#include "ortinferencemanager.h"
#include <QDebug>

ReflectionEngine::ReflectionEngine(QObject *parent)
    : QObject(parent)
{
}

ReflectionEngine::~ReflectionEngine() = default;

void ReflectionEngine::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void ReflectionEngine::setTokenizer(Tokenizer *tokenizer) { m_tokenizer = tokenizer; }
void ReflectionEngine::setTrainingDataGenerator(TrainingDataGenerator *g) { m_dataGenerator = g; }
void ReflectionEngine::setOrtTrainingManager(OrtTrainingManager *m) { m_trainingManager = m; }
void ReflectionEngine::setAdapterManager(AdapterManager *m) { m_adapterManager = m; }
void ReflectionEngine::setEvalSuite(EvalSuite *e) { m_evalSuite = e; }
void ReflectionEngine::setIdleScheduler(IdleScheduler *s) { m_idleScheduler = s; }
void ReflectionEngine::setProfileManager(ProfileManager *p) { m_profileManager = p; }
void ReflectionEngine::setSettingsManager(SettingsManager *s) { m_settingsManager = s; }
void ReflectionEngine::setLLMProvider(LLMProviderManager *l) { m_llmProvider = l; }
void ReflectionEngine::setOrtInferenceManager(OrtInferenceManager *m) { m_ortInference = m; }

float ReflectionEngine::trainingLoss() const
{
    return m_trainingManager ? m_trainingManager->currentLoss() : 0.0f;
}

int ReflectionEngine::trainingStep() const
{
    return m_trainingManager ? m_trainingManager->currentStep() : 0;
}

int ReflectionEngine::totalTrainingSteps() const
{
    return m_trainingManager ? m_trainingManager->totalSteps() : 0;
}

bool ReflectionEngine::hasEnoughNewData() const
{
    if (!m_memoryStore || !m_dataGenerator) return false;

    int totalEpisodes = m_memoryStore->episodeCount();
    if (totalEpisodes < MIN_NEW_EPISODES) return false;

    // Check high-signal episodes since last training
    auto highSignal = m_memoryStore->getHighSignalEpisodes(
        m_dataGenerator->lastProcessedEpisodeId(), MIN_HIGH_SIGNAL + 1);

    return highSignal.size() >= MIN_HIGH_SIGNAL;
}

int ReflectionEngine::computeMaxSteps() const
{
    int budget = 10;

    if (m_settingsManager) {
        budget = m_settingsManager->computeBudgetPercent();
    }

    int baseSteps = 25;
    int maxSteps = static_cast<int>(baseSteps * (budget / 10.0));

    return qBound(5, maxSteps, 200);
}

void ReflectionEngine::setPhase(const QString &phase)
{
    if (m_currentPhase != phase) {
        m_currentPhase = phase;
        emit phaseChanged();
    }
}

void ReflectionEngine::setReflectionStatus(const QString &status)
{
    if (m_reflectionStatus != status) {
        m_reflectionStatus = status;
        emit reflectionStatusChanged();
    }
}

// --- Idle window handlers ---

void ReflectionEngine::onIdleWindowOpened()
{
    qDebug() << "ReflectionEngine: idle window opened";

    if (m_isReflecting) {
        qDebug() << "Already reflecting, skipping";
        return;
    }

    if (!hasEnoughNewData()) {
        setReflectionStatus("Idle - insufficient new data for training");
        return;
    }

    // Check that training artifacts exist
    if (m_profileManager) {
        QString artifactsDir = m_profileManager->profilePath() + "/training_artifacts";
        if (!QFile::exists(artifactsDir + "/training_model.onnx")) {
            setReflectionStatus("No training artifacts found - run generate_artifacts.py first");
            return;
        }
    }

    m_isReflecting = true;
    emit isReflectingChanged();

    runPhase1_SelectEpisodes();
}

void ReflectionEngine::onIdleWindowClosed()
{
    qDebug() << "ReflectionEngine: idle window closed";

    if (!m_isReflecting) return;

    if (m_trainingManager && m_trainingManager->isTraining()) {
        m_trainingManager->requestPause();
        setReflectionStatus("Pausing training - user active");
    } else {
        m_isReflecting = false;
        emit isReflectingChanged();
        setReflectionStatus("Paused - user active");
    }
}

void ReflectionEngine::triggerReflection()
{
    qDebug() << "ReflectionEngine: manual trigger";

    if (m_isReflecting) {
        qDebug() << "Already reflecting";
        return;
    }

    m_isReflecting = true;
    emit isReflectingChanged();
    runPhase1_SelectEpisodes();
}

void ReflectionEngine::pauseReflection()
{
    if (m_trainingManager && m_trainingManager->isTraining()) {
        m_trainingManager->requestPause();
    }
    m_isReflecting = false;
    emit isReflectingChanged();
    setReflectionStatus("Paused by user");
}

// --- Pipeline phases ---

void ReflectionEngine::runPhase1_SelectEpisodes()
{
    setPhase("Episode Selection");
    setReflectionStatus("Selecting high-signal episodes...");

    if (!m_dataGenerator) {
        emit reflectionError("TrainingDataGenerator not available");
        setPhase("idle");
        m_isReflecting = false;
        emit isReflectingChanged();
        return;
    }

    auto episodeIds = m_dataGenerator->selectHighSignalEpisodes(50);

    if (episodeIds.size() < MIN_HIGH_SIGNAL) {
        setReflectionStatus(QString("Not enough high-signal episodes (%1 found, need %2)")
            .arg(episodeIds.size()).arg(MIN_HIGH_SIGNAL));
        setPhase("idle");
        m_isReflecting = false;
        emit isReflectingChanged();
        return;
    }

    qDebug() << "ReflectionEngine: selected" << episodeIds.size() << "episodes";
    runPhase2_GenerateTrainingData();
}

void ReflectionEngine::runPhase2_GenerateTrainingData()
{
    setPhase("Data Generation");
    setReflectionStatus("Generating training examples...");

    int maxExamples = computeMaxSteps() * 4;  // batchSize * steps
    m_dataGenerator->generateTrainingBatch(maxExamples);

    int count = m_dataGenerator->exampleCount();
    if (count < 4) {  // Less than one batch
        setReflectionStatus("Insufficient training examples generated");
        setPhase("idle");
        m_isReflecting = false;
        emit isReflectingChanged();
        return;
    }

    qDebug() << "ReflectionEngine: generated" << count << "training examples";
    runPhase3_TrainAdapter();
}

void ReflectionEngine::runPhase3_TrainAdapter()
{
    setPhase("Training");
    setReflectionStatus("Initializing training...");

    if (!m_trainingManager || !m_profileManager) {
        emit reflectionError("Training manager or profile manager not available");
        setPhase("idle");
        m_isReflecting = false;
        emit isReflectingChanged();
        return;
    }

    TrainingConfig config;
    config.artifactsDir = m_profileManager->profilePath() + "/training_artifacts";
    config.checkpointDir = m_profileManager->checkpointsPath() + "/current";
    config.outputDir = m_profileManager->checkpointsPath() + "/export";
    config.maxStepsPerSession = computeMaxSteps();
    config.learningRate = 1e-4f;
    config.batchSize = 4;
    config.intraOpThreads = 2;

    // Initialize if needed
    if (!m_trainingManager->isInitialized()) {
        if (!m_trainingManager->initialize(config)) {
            setReflectionStatus("Failed to initialize ORT Training - check artifacts");
            emit reflectionError("ORT Training initialization failed");
            setPhase("idle");
            m_isReflecting = false;
            emit isReflectingChanged();
            return;
        }
    }

    // Connect training signals (one-shot)
    connect(m_trainingManager, &OrtTrainingManager::trainingComplete,
            this, &ReflectionEngine::onTrainingComplete, Qt::UniqueConnection);
    connect(m_trainingManager, &OrtTrainingManager::trainingPaused,
            this, &ReflectionEngine::onTrainingPaused, Qt::UniqueConnection);
    connect(m_trainingManager, &OrtTrainingManager::trainingError,
            this, &ReflectionEngine::onTrainingError, Qt::UniqueConnection);

    // Forward progress signals for QML binding
    connect(m_trainingManager, &OrtTrainingManager::currentLossChanged,
            this, &ReflectionEngine::trainingLossChanged, Qt::UniqueConnection);
    connect(m_trainingManager, &OrtTrainingManager::currentStepChanged,
            this, &ReflectionEngine::trainingStepChanged, Qt::UniqueConnection);
    connect(m_trainingManager, &OrtTrainingManager::totalStepsChanged,
            this, &ReflectionEngine::totalTrainingStepsChanged, Qt::UniqueConnection);

    setReflectionStatus("Training adapter...");
    m_trainingManager->startTraining(m_dataGenerator->examples(), config);
}

void ReflectionEngine::onTrainingComplete(int stepsCompleted, float finalLoss)
{
    qDebug() << "ReflectionEngine: training complete, steps:" << stepsCompleted
             << "loss:" << finalLoss;

    m_lastTrainingLoss = finalLoss;
    runPhase4_EvaluateCandidate();
}

void ReflectionEngine::onTrainingPaused(int stepsCompleted)
{
    qDebug() << "ReflectionEngine: training paused at step" << stepsCompleted;
    setReflectionStatus(QString("Training paused at step %1 - will resume next idle window")
        .arg(stepsCompleted));

    setPhase("idle");
    m_isReflecting = false;
    emit isReflectingChanged();
}

void ReflectionEngine::onTrainingError(const QString &error)
{
    qWarning() << "ReflectionEngine: training error:" << error;
    setReflectionStatus("Training error: " + error);
    emit reflectionError(error);

    setPhase("idle");
    m_isReflecting = false;
    emit isReflectingChanged();
}

void ReflectionEngine::runPhase4_EvaluateCandidate()
{
    setPhase("Evaluation");
    setReflectionStatus("Evaluating candidate adapter...");

    if (!m_trainingManager || !m_adapterManager || !m_evalSuite || !m_profileManager) {
        emit reflectionError("Missing dependencies for evaluation");
        setPhase("idle");
        m_isReflecting = false;
        emit isReflectingChanged();
        return;
    }

    // Export trained model for inference
    QString exportPath = m_profileManager->checkpointsPath() + "/export/candidate.onnx";
    m_trainingManager->exportForInference(exportPath);

    // Register as candidate
    AdapterMetadata meta;
    meta.adapterName = "communication_style";
    meta.trainingDate = QDateTime::currentDateTime();
    meta.trainingSteps = m_trainingManager->currentStep();
    meta.finalLoss = m_lastTrainingLoss;
    meta.exportedModelPath = exportPath;
    meta.checkpointPath = m_profileManager->checkpointsPath() + "/current";
    meta.parentVersion = m_adapterManager->activeVersion();
    meta.status = "candidate";

    m_candidateVersion = m_adapterManager->registerCandidate(meta);

    // Run eval suite
    EvalReport report = m_evalSuite->runFullSuite();
    double candidateScore = report.overallScore;

    qDebug() << "ReflectionEngine: candidate v" << m_candidateVersion
             << "eval score:" << candidateScore;

    // Decide promotion
    runPhase5_PromoteOrReject(m_candidateVersion, candidateScore);
}

void ReflectionEngine::runPhase5_PromoteOrReject(int candidateVersion, double evalScore)
{
    // Get current champion score
    AdapterMetadata currentActive = m_adapterManager->activeAdapterMetadata();
    double championScore = currentActive.evalScore;

    bool shouldPromote = false;

    if (currentActive.version < 0) {
        // No current adapter - promote if score is reasonable
        shouldPromote = (evalScore >= 0.4);
    } else {
        // Must beat current by margin
        shouldPromote = (evalScore >= championScore + PROMOTION_MARGIN);
    }

    // Safety check: all safety tests must pass
    // (already part of eval score, but check explicitly)
    shouldPromote = shouldPromote && (evalScore >= 0.3);

    if (shouldPromote) {
        setPhase("Promotion");
        setReflectionStatus(QString("Promoting v%1 (score: %2)")
            .arg(candidateVersion).arg(evalScore, 0, 'f', 2));

        // Update candidate eval score
        // (metadata was already saved by registerCandidate, update it)
        AdapterMetadata updatedMeta = m_adapterManager->getVersionMetadata(candidateVersion);
        updatedMeta.evalScore = evalScore;

        // Promote
        m_adapterManager->promoteVersion(candidateVersion);

        // Set the exported model path for ORT inference
        if (m_llmProvider && !updatedMeta.exportedModelPath.isEmpty()) {
            m_llmProvider->setOrtModelPath(updatedMeta.exportedModelPath);
        }

        // Hot-swap local inference model
        if (m_ortInference && !updatedMeta.exportedModelPath.isEmpty()) {
            m_ortInference->loadModel(updatedMeta.exportedModelPath);
            qDebug() << "[ReflectionEngine] Hot-swapped inference model to"
                     << updatedMeta.exportedModelPath;
        }

        setReflectionStatus(QString("Adapter v%1 promoted (score: %2)")
            .arg(candidateVersion).arg(evalScore, 0, 'f', 2));
        emit adapterPromoted(candidateVersion, evalScore);

    } else {
        setReflectionStatus(QString("Candidate v%1 rejected (score: %2, needed: %3)")
            .arg(candidateVersion)
            .arg(evalScore, 0, 'f', 2)
            .arg(championScore + PROMOTION_MARGIN, 0, 'f', 2));
    }

    // Prune old versions
    m_adapterManager->pruneVersions(3);

    // Clean up
    m_dataGenerator->clear();

    m_sessionsCompleted++;
    emit sessionsCompletedChanged();
    m_isReflecting = false;
    emit isReflectingChanged();
    emit reflectionComplete(shouldPromote, evalScore);
}

#endif // DATAFORM_TRAINING_ENABLED
