#ifdef DATAFORM_TRAINING_ENABLED

#include "evolutionengine.h"
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
#include "weightmerger.h"
#include "lineagetracker.h"
#include "ortinferencemanager.h"
#include <QDebug>
#include <QFile>
#include <QDir>
#include <QJsonDocument>
#include <QRandomGenerator>
#include <QtMath>
#include <algorithm>

EvolutionEngine::EvolutionEngine(QObject *parent)
    : QObject(parent)
{
}

EvolutionEngine::~EvolutionEngine() = default;

// --- Dependency injection ---

void EvolutionEngine::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void EvolutionEngine::setTokenizer(Tokenizer *tokenizer) { m_tokenizer = tokenizer; }
void EvolutionEngine::setTrainingDataGenerator(TrainingDataGenerator *g) { m_dataGenerator = g; }
void EvolutionEngine::setOrtTrainingManager(OrtTrainingManager *m) { m_trainingManager = m; }
void EvolutionEngine::setAdapterManager(AdapterManager *m) { m_adapterManager = m; }
void EvolutionEngine::setEvalSuite(EvalSuite *e) { m_evalSuite = e; }
void EvolutionEngine::setIdleScheduler(IdleScheduler *s) { m_idleScheduler = s; }
void EvolutionEngine::setProfileManager(ProfileManager *p) { m_profileManager = p; }
void EvolutionEngine::setSettingsManager(SettingsManager *s) { m_settingsManager = s; }
void EvolutionEngine::setLLMProvider(LLMProviderManager *l) { m_llmProvider = l; }
void EvolutionEngine::setWeightMerger(WeightMerger *w) { m_weightMerger = w; }
void EvolutionEngine::setLineageTracker(LineageTracker *t) { m_lineageTracker = t; }
void EvolutionEngine::setOrtInferenceManager(OrtInferenceManager *m) { m_ortInference = m; }

// --- Property getters ---

int EvolutionEngine::populationSize() const
{
    return computePopulationSize();
}

float EvolutionEngine::trainingLoss() const
{
    return m_trainingManager ? m_trainingManager->currentLoss() : 0.0f;
}

int EvolutionEngine::trainingStep() const
{
    return m_trainingManager ? m_trainingManager->currentStep() : 0;
}

int EvolutionEngine::totalTrainingSteps() const
{
    return m_trainingManager ? m_trainingManager->totalSteps() : 0;
}

int EvolutionEngine::computePopulationSize() const
{
    int size = m_settingsManager ? m_settingsManager->populationSize() : DEFAULT_POPULATION_SIZE;
    return qBound(1, size, MAX_POPULATION_SIZE);
}

int EvolutionEngine::computeMaxStepsPerVariant() const
{
    int budget = 10;

    if (m_settingsManager) {
        budget = m_settingsManager->computeBudgetPercent();
    }

    int baseSteps = 25;
    int maxSteps = static_cast<int>(baseSteps * (budget / 10.0));

    // Divide budget across population
    int popSize = computePopulationSize();
    maxSteps = maxSteps / popSize;

    return qBound(5, maxSteps, 100);
}

bool EvolutionEngine::hasEnoughNewData() const
{
    if (!m_memoryStore || !m_dataGenerator) return false;

    int totalEpisodes = m_memoryStore->episodeCount();
    if (totalEpisodes < MIN_NEW_EPISODES) return false;

    auto highSignal = m_memoryStore->getHighSignalEpisodes(
        m_dataGenerator->lastProcessedEpisodeId(), MIN_HIGH_SIGNAL + 1);

    return highSignal.size() >= MIN_HIGH_SIGNAL;
}

bool EvolutionEngine::shouldConsolidate() const
{
    return m_cyclesCompleted > 0 &&
           m_cyclesCompleted % CONSOLIDATION_INTERVAL_CYCLES == 0;
}

// --- Status helpers ---

void EvolutionEngine::setEvolutionStage(const QString &stage)
{
    if (m_population.cycleStage != stage) {
        m_population.cycleStage = stage;
        emit evolutionStageChanged();
    }
}

void EvolutionEngine::setEvolutionStatus(const QString &status)
{
    if (m_evolutionStatus != status) {
        m_evolutionStatus = status;
        emit evolutionStatusChanged();
    }
}

// --- Idle window handlers ---

void EvolutionEngine::onIdleWindowOpened()
{
    qDebug() << "EvolutionEngine: idle window opened";
    m_idleWindowOpen = true;

    if (m_isEvolving) {
        qDebug() << "Already evolving, resuming...";
        resumeOrStartCycle();
        return;
    }

    // Check if evolution is enabled
    if (m_settingsManager && !m_settingsManager->evolutionEnabled()) {
        setEvolutionStatus("Evolution disabled in settings");
        return;
    }

    // Check for in-progress cycle
    if (m_population.cycleStage != "idle") {
        m_isEvolving = true;
        emit isEvolvingChanged();
        resumeOrStartCycle();
        return;
    }

    // Check data gate
    if (!hasEnoughNewData()) {
        setEvolutionStatus("Idle - insufficient new data for evolution");
        return;
    }

    // Check training artifacts
    if (m_profileManager) {
        QString artifactsDir = m_profileManager->profilePath() + "/training_artifacts";
        if (!QFile::exists(artifactsDir + "/training_model.onnx")) {
            setEvolutionStatus("No training artifacts found - run generate_artifacts.py first");
            return;
        }
    }

    m_isEvolving = true;
    emit isEvolvingChanged();
    seedPopulation();
}

void EvolutionEngine::onIdleWindowClosed()
{
    qDebug() << "EvolutionEngine: idle window closed";
    m_idleWindowOpen = false;

    if (!m_isEvolving) return;

    if (m_trainingManager && m_trainingManager->isTraining()) {
        m_trainingManager->requestPause();
        m_population.variantTrainingPaused = true;
        savePopulationState();
        setEvolutionStatus("Pausing training - user active");
    } else {
        m_isEvolving = false;
        emit isEvolvingChanged();
        setEvolutionStatus("Paused - user active");
    }
}

void EvolutionEngine::triggerEvolutionCycle()
{
    qDebug() << "EvolutionEngine: manual trigger";

    if (m_isEvolving) {
        qDebug() << "Already evolving";
        return;
    }

    m_isEvolving = true;
    m_idleWindowOpen = true;  // Treat manual trigger as open window
    emit isEvolvingChanged();

    if (m_population.cycleStage != "idle") {
        resumeOrStartCycle();
    } else {
        seedPopulation();
    }
}

void EvolutionEngine::pauseEvolution()
{
    if (m_trainingManager && m_trainingManager->isTraining()) {
        m_trainingManager->requestPause();
        m_population.variantTrainingPaused = true;
    }
    m_isEvolving = false;
    m_idleWindowOpen = false;
    emit isEvolvingChanged();
    savePopulationState();
    setEvolutionStatus("Paused by user");
}

// --- Cycle state machine ---

void EvolutionEngine::resumeOrStartCycle()
{
    QString stage = m_population.cycleStage;
    qDebug() << "EvolutionEngine: resuming at stage" << stage;

    if (stage == "training") {
        trainNextVariant();
    } else if (stage == "evaluating") {
        evaluateAllVariants();
    } else if (stage == "selecting") {
        selectWinner();
    } else if (stage == "consolidating") {
        runConsolidation();
    } else if (stage == "seeding") {
        seedPopulation();
    } else {
        // Unknown state, start fresh
        seedPopulation();
    }
}

void EvolutionEngine::seedPopulation()
{
    setEvolutionStage("seeding");
    setEvolutionStatus("Seeding population...");

    if (!m_dataGenerator) {
        emit evolutionError("TrainingDataGenerator not available");
        m_isEvolving = false;
        emit isEvolvingChanged();
        return;
    }

    // Select episodes
    auto episodeIds = m_dataGenerator->selectHighSignalEpisodes(50);
    if (episodeIds.size() < MIN_HIGH_SIGNAL) {
        setEvolutionStatus(QString("Not enough high-signal episodes (%1 found, need %2)")
            .arg(episodeIds.size()).arg(MIN_HIGH_SIGNAL));
        m_isEvolving = false;
        emit isEvolvingChanged();
        setEvolutionStage("idle");
        return;
    }

    // Generate training data once for the whole cycle
    int popSize = computePopulationSize();
    int maxExamples = computeMaxStepsPerVariant() * 4 * popSize;  // Enough for all variants
    m_dataGenerator->generateTrainingBatch(maxExamples);
    m_cycleExamples = m_dataGenerator->examples();

    if (m_cycleExamples.size() < 4) {
        setEvolutionStatus("Insufficient training examples generated");
        m_isEvolving = false;
        emit isEvolvingChanged();
        setEvolutionStage("idle");
        return;
    }

    qDebug() << "EvolutionEngine: generated" << m_cycleExamples.size()
             << "training examples for population of" << popSize;

    // Generate variant specs
    m_population.variants = generateVariantSpecs(popSize);
    m_population.populationSize = popSize;
    m_population.cycleStartTime = QDateTime::currentDateTime();
    m_population.currentVariantIndex = 0;
    m_population.variantTrainingPaused = false;

    // Get current champion info
    if (m_adapterManager) {
        m_population.championVersion = m_adapterManager->activeVersion();
        auto activeMeta = m_adapterManager->activeAdapterMetadata();
        m_population.championScore = activeMeta.evalScore;
    }

    emit totalVariantsChanged();
    emit populationSizeChanged();

    setEvolutionStage("training");
    savePopulationState();
    trainNextVariant();
}

QList<VariantSpec> EvolutionEngine::generateVariantSpecs(int count)
{
    QList<VariantSpec> specs;
    float baseLR = 1e-4f;

    for (int i = 0; i < count; ++i) {
        VariantSpec spec;
        spec.variantIndex = i;
        spec.dataSeed = QRandomGenerator::global()->generate();

        if (i == 0) {
            // Control variant: base learning rate, full data
            spec.learningRate = baseLR;
            spec.dataSubsampleRatio = 1.0f;
        } else {
            // Perturbed variant: log-uniform LR, optional subsampling
            float logMin = qLn(baseLR * LR_PERTURBATION_MIN);
            float logMax = qLn(baseLR * LR_PERTURBATION_MAX);
            float logLR = logMin + QRandomGenerator::global()->generateDouble() * (logMax - logMin);
            spec.learningRate = qExp(logLR);

            // Subsample for populations >= 4
            if (count >= 4) {
                spec.dataSubsampleRatio = MIN_SUBSAMPLE_RATIO +
                    QRandomGenerator::global()->generateDouble() * (1.0f - MIN_SUBSAMPLE_RATIO);
            } else {
                spec.dataSubsampleRatio = 1.0f;
            }
        }

        spec.status = "pending";

        // Set checkpoint path
        if (m_profileManager) {
            spec.checkpointPath = m_profileManager->evolutionPath() +
                QString("/cycle_%1/variant_%2").arg(m_population.cycleId).arg(i);
        }

        qDebug() << "EvolutionEngine: variant" << i
                 << "LR=" << spec.learningRate
                 << "seed=" << spec.dataSeed
                 << "subsample=" << spec.dataSubsampleRatio;

        specs.append(spec);
    }

    return specs;
}

QList<TrainingExample> EvolutionEngine::subsampleExamples(
    const QList<TrainingExample> &all, float ratio, quint32 seed)
{
    if (ratio >= 1.0f || all.isEmpty()) return all;

    int targetCount = qMax(4, static_cast<int>(all.size() * ratio));
    QList<TrainingExample> copy = all;

    // Fisher-Yates shuffle with seeded generator
    QRandomGenerator rng(seed);
    for (int i = copy.size() - 1; i > 0; --i) {
        int j = rng.bounded(i + 1);
        copy.swapItemsAt(i, j);
    }

    return copy.mid(0, targetCount);
}

// --- Training ---

void EvolutionEngine::trainNextVariant()
{
    // Find next variant needing training
    int nextIdx = -1;
    for (int i = 0; i < m_population.variants.size(); ++i) {
        const auto &v = m_population.variants[i];
        if (v.status == "pending" || v.status == "training") {
            nextIdx = i;
            break;
        }
    }

    if (nextIdx < 0) {
        // All variants trained, move to evaluation
        qDebug() << "EvolutionEngine: all variants trained, evaluating...";
        setEvolutionStage("evaluating");
        savePopulationState();
        evaluateAllVariants();
        return;
    }

    auto &variant = m_population.variants[nextIdx];
    m_population.currentVariantIndex = nextIdx;
    emit currentVariantIndexChanged();

    setEvolutionStatus(QString("Training variant %1/%2 (LR: %3)")
        .arg(nextIdx + 1).arg(m_population.variants.size())
        .arg(variant.learningRate, 0, 'e', 1));

    if (!m_trainingManager || !m_profileManager) {
        emit evolutionError("Training manager or profile manager not available");
        m_isEvolving = false;
        emit isEvolvingChanged();
        return;
    }

    // Prepare variant's training config
    TrainingConfig config;
    config.artifactsDir = m_profileManager->profilePath() + "/training_artifacts";
    config.checkpointDir = variant.checkpointPath;
    config.outputDir = variant.checkpointPath + "/export";
    config.maxStepsPerSession = computeMaxStepsPerVariant();
    config.learningRate = variant.learningRate;
    config.batchSize = 4;
    config.intraOpThreads = 2;

    // Shutdown previous training session if needed
    if (m_trainingManager->isInitialized()) {
        m_trainingManager->shutdown();
    }

    // Copy base checkpoint to variant directory if first time
    if (variant.status == "pending") {
        QDir().mkpath(variant.checkpointPath);

        // Use artifacts checkpoint as base
        QString baseCheckpoint = config.artifactsDir + "/checkpoint";
        if (m_population.championVersion >= 0 && m_adapterManager) {
            auto championMeta = m_adapterManager->getVersionMetadata(m_population.championVersion);
            if (!championMeta.checkpointPath.isEmpty() && QDir(championMeta.checkpointPath).exists()) {
                baseCheckpoint = championMeta.checkpointPath;
            }
        }

        // Copy checkpoint files to variant directory
        QDir baseDir(baseCheckpoint);
        if (baseDir.exists()) {
            for (const auto &fileName : baseDir.entryList(QDir::Files)) {
                QFile::copy(baseCheckpoint + "/" + fileName,
                           variant.checkpointPath + "/" + fileName);
            }
        }
    }

    // Initialize ORT with variant's checkpoint
    if (!m_trainingManager->initialize(config)) {
        setEvolutionStatus(QString("Failed to init training for variant %1").arg(nextIdx));
        emit evolutionError("ORT Training initialization failed for variant");
        variant.status = "rejected";
        savePopulationState();
        // Try next variant
        trainNextVariant();
        return;
    }

    // Connect signals
    connect(m_trainingManager, &OrtTrainingManager::trainingComplete,
            this, &EvolutionEngine::onTrainingComplete, Qt::UniqueConnection);
    connect(m_trainingManager, &OrtTrainingManager::trainingPaused,
            this, &EvolutionEngine::onTrainingPaused, Qt::UniqueConnection);
    connect(m_trainingManager, &OrtTrainingManager::trainingError,
            this, &EvolutionEngine::onTrainingError, Qt::UniqueConnection);

    // Forward progress signals
    connect(m_trainingManager, &OrtTrainingManager::currentLossChanged,
            this, &EvolutionEngine::trainingLossChanged, Qt::UniqueConnection);
    connect(m_trainingManager, &OrtTrainingManager::currentStepChanged,
            this, &EvolutionEngine::trainingStepChanged, Qt::UniqueConnection);
    connect(m_trainingManager, &OrtTrainingManager::totalStepsChanged,
            this, &EvolutionEngine::totalTrainingStepsChanged, Qt::UniqueConnection);

    variant.status = "training";
    savePopulationState();

    // Subsample data for this variant
    auto variantExamples = subsampleExamples(m_cycleExamples,
                                              variant.dataSubsampleRatio,
                                              variant.dataSeed);

    m_trainingManager->startTraining(variantExamples, config);
}

void EvolutionEngine::onTrainingComplete(int stepsCompleted, float finalLoss)
{
    int idx = m_population.currentVariantIndex;
    if (idx < 0 || idx >= m_population.variants.size()) return;

    auto &variant = m_population.variants[idx];
    variant.finalLoss = finalLoss;
    variant.trainingSteps = stepsCompleted;
    variant.status = "trained";

    qDebug() << "EvolutionEngine: variant" << idx << "complete, steps:"
             << stepsCompleted << "loss:" << finalLoss;

    onVariantTrainingDone(finalLoss, stepsCompleted);
}

void EvolutionEngine::onTrainingPaused(int stepsCompleted)
{
    int idx = m_population.currentVariantIndex;
    if (idx >= 0 && idx < m_population.variants.size()) {
        m_population.variants[idx].trainingSteps = stepsCompleted;
        m_population.variantTrainingPaused = true;
    }

    qDebug() << "EvolutionEngine: variant" << idx << "paused at step" << stepsCompleted;
    setEvolutionStatus(QString("Variant %1 paused at step %2 - will resume next idle window")
        .arg(idx + 1).arg(stepsCompleted));

    savePopulationState();
    m_isEvolving = false;
    emit isEvolvingChanged();
}

void EvolutionEngine::onTrainingError(const QString &error)
{
    int idx = m_population.currentVariantIndex;
    qWarning() << "EvolutionEngine: variant" << idx << "training error:" << error;

    if (idx >= 0 && idx < m_population.variants.size()) {
        m_population.variants[idx].status = "rejected";
    }

    setEvolutionStatus("Training error: " + error);
    savePopulationState();

    // Try next variant
    if (m_idleWindowOpen) {
        trainNextVariant();
    } else {
        m_isEvolving = false;
        emit isEvolvingChanged();
    }
}

void EvolutionEngine::onVariantTrainingDone(float finalLoss, int steps)
{
    int idx = m_population.currentVariantIndex;
    auto &variant = m_population.variants[idx];

    // Export model for inference evaluation
    QString exportPath = variant.checkpointPath + "/export/candidate.onnx";
    if (m_trainingManager) {
        m_trainingManager->exportForInference(exportPath);
    }
    variant.exportedModelPath = exportPath;

    // Register as candidate in adapter manager
    if (m_adapterManager) {
        AdapterMetadata meta;
        meta.adapterName = "evolution_variant";
        meta.trainingDate = QDateTime::currentDateTime();
        meta.trainingSteps = steps;
        meta.finalLoss = finalLoss;
        meta.exportedModelPath = exportPath;
        meta.checkpointPath = variant.checkpointPath;
        meta.parentVersion = m_population.championVersion;
        meta.status = "candidate";

        variant.adapterVersion = m_adapterManager->registerCandidate(meta);

        // Record lineage node
        if (m_lineageTracker) {
            LineageNode node;
            node.version = variant.adapterVersion;
            node.parentVersion = meta.parentVersion;
            node.adapterName = meta.adapterName;
            node.trainingDate = meta.trainingDate;
            node.evalScore = meta.evalScore;
            node.finalLoss = finalLoss;
            node.trainingSteps = steps;
            node.status = meta.status;
            node.origin = "evolution";
            node.generationId = 0;
            node.cycleId = m_population.cycleId;
            node.variantIndex = idx;
            m_lineageTracker->recordNode(node);
        }
    }

    emit variantTrained(idx, finalLoss, steps);
    savePopulationState();

    // Shutdown training session before next variant
    if (m_trainingManager) {
        m_trainingManager->shutdown();
    }

    // Continue with next variant if idle window still open
    if (m_idleWindowOpen) {
        trainNextVariant();
    } else {
        m_isEvolving = false;
        emit isEvolvingChanged();
        setEvolutionStatus("Waiting for next idle window to continue evolution");
    }
}

// --- Evaluation ---

void EvolutionEngine::evaluateAllVariants()
{
    setEvolutionStatus("Evaluating all variants...");

    if (!m_evalSuite) {
        emit evolutionError("EvalSuite not available");
        m_isEvolving = false;
        emit isEvolvingChanged();
        return;
    }

    for (int i = 0; i < m_population.variants.size(); ++i) {
        auto &variant = m_population.variants[i];
        if (variant.status != "trained") continue;

        setEvolutionStatus(QString("Evaluating variant %1/%2...")
            .arg(i + 1).arg(m_population.variants.size()));

        EvalReport report = m_evalSuite->runFullSuite();
        variant.evalScore = report.overallScore;
        variant.status = "evaluated";

        // Update adapter metadata with eval score
        if (m_adapterManager && variant.adapterVersion >= 0) {
            AdapterMetadata meta = m_adapterManager->getVersionMetadata(variant.adapterVersion);
            meta.evalScore = variant.evalScore;
            m_adapterManager->updateVersionMetadata(variant.adapterVersion, meta);
        }

        emit variantEvaluated(i, variant.evalScore);
        qDebug() << "EvolutionEngine: variant" << i << "eval score:" << variant.evalScore;
    }

    setEvolutionStage("selecting");
    savePopulationState();
    selectWinner();
}

// --- Selection ---

void EvolutionEngine::selectWinner()
{
    setEvolutionStatus("Selecting winner...");

    // Collect evaluated variants
    QList<int> evaluatedIndices;
    for (int i = 0; i < m_population.variants.size(); ++i) {
        if (m_population.variants[i].status == "evaluated") {
            evaluatedIndices.append(i);
        }
    }

    if (evaluatedIndices.isEmpty()) {
        setEvolutionStatus("No variants evaluated - cycle failed");
        m_isEvolving = false;
        emit isEvolvingChanged();
        setEvolutionStage("idle");
        m_population.cycleId++;
        savePopulationState();
        return;
    }

    // Sort by eval score descending
    std::sort(evaluatedIndices.begin(), evaluatedIndices.end(),
              [this](int a, int b) {
                  return m_population.variants[a].evalScore > m_population.variants[b].evalScore;
              });

    int bestIdx = evaluatedIndices[0];
    auto &bestVariant = m_population.variants[bestIdx];
    double bestScore = bestVariant.evalScore;

    qDebug() << "EvolutionEngine: best variant" << bestIdx
             << "score:" << bestScore << "champion score:" << m_population.championScore;

    bool shouldPromote = false;
    if (m_population.championVersion < 0) {
        // No current adapter - promote if reasonable
        shouldPromote = (bestScore >= 0.4);
    } else {
        shouldPromote = (bestScore >= m_population.championScore + PROMOTION_MARGIN);
    }
    shouldPromote = shouldPromote && (bestScore >= 0.3);

    if (shouldPromote && m_adapterManager && bestVariant.adapterVersion >= 0) {
        bestVariant.status = "selected";
        m_adapterManager->promoteVersion(bestVariant.adapterVersion);

        if (m_llmProvider && !bestVariant.exportedModelPath.isEmpty()) {
            m_llmProvider->setOrtModelPath(bestVariant.exportedModelPath);
        }

        // Hot-swap local inference model
        if (m_ortInference && !bestVariant.exportedModelPath.isEmpty()) {
            m_ortInference->loadModel(bestVariant.exportedModelPath);
            qDebug() << "[EvolutionEngine] Hot-swapped inference model to"
                     << bestVariant.exportedModelPath;
        }

        setEvolutionStatus(QString("Variant %1 promoted as v%2 (score: %3)")
            .arg(bestIdx).arg(bestVariant.adapterVersion)
            .arg(bestScore, 0, 'f', 2));

        qDebug() << "EvolutionEngine: promoted variant" << bestIdx
                 << "as adapter v" << bestVariant.adapterVersion;
    } else {
        setEvolutionStatus(QString("No variant beat champion (best: %1, needed: %2)")
            .arg(bestScore, 0, 'f', 2)
            .arg(m_population.championScore + PROMOTION_MARGIN, 0, 'f', 2));
    }

    // Mark all non-selected variants as rejected
    for (auto &variant : m_population.variants) {
        if (variant.status == "evaluated") {
            variant.status = "rejected";
        }
    }

    // Prune old adapter versions
    if (m_adapterManager) {
        m_adapterManager->pruneVersions(5);
    }

    // Cleanup cycle data
    cleanupCycleData();

    m_cyclesCompleted++;
    m_population.cycleId++;
    emit cyclesCompletedChanged();

    // Check if consolidation is due
    if (shouldConsolidate()) {
        setEvolutionStage("consolidating");
        savePopulationState();
        checkConsolidation();
    } else {
        setEvolutionStage("idle");
        m_isEvolving = false;
        emit isEvolvingChanged();
        savePopulationState();
        saveEvolutionLog();
        emit evolutionCycleComplete(shouldPromote,
            shouldPromote ? bestVariant.adapterVersion : -1, bestScore);
    }
}

void EvolutionEngine::cleanupCycleData()
{
    // Clear cached training examples
    m_cycleExamples.clear();
    if (m_dataGenerator) {
        m_dataGenerator->clear();
    }

    // Remove rejected variant checkpoint directories
    for (const auto &variant : m_population.variants) {
        if (variant.status == "rejected" && !variant.checkpointPath.isEmpty()) {
            QDir(variant.checkpointPath).removeRecursively();
        }
    }
}

// --- Consolidation ---

void EvolutionEngine::checkConsolidation()
{
    if (!shouldConsolidate()) {
        setEvolutionStage("idle");
        m_isEvolving = false;
        emit isEvolvingChanged();
        savePopulationState();
        return;
    }

    runConsolidation();
}

void EvolutionEngine::runConsolidation()
{
    setEvolutionStatus("Consolidating top adapters...");

    if (!m_adapterManager || !m_weightMerger || !m_profileManager) {
        setEvolutionStatus("Missing dependencies for consolidation");
        setEvolutionStage("idle");
        m_isEvolving = false;
        emit isEvolvingChanged();
        return;
    }

    // Get top adapter versions
    auto topVersions = m_adapterManager->getTopVersions(3);

    // Filter to versions that have valid checkpoint paths
    QList<WeightMerger::MergeInput> mergeInputs;
    for (const auto &meta : topVersions) {
        if (!meta.checkpointPath.isEmpty() && QDir(meta.checkpointPath).exists()) {
            WeightMerger::MergeInput input;
            input.checkpointPath = meta.checkpointPath;
            input.weight = meta.evalScore;  // Weight by eval score
            mergeInputs.append(input);
        }
    }

    if (mergeInputs.size() < 2) {
        setEvolutionStatus("Not enough checkpoints for consolidation (need 2+)");
        setEvolutionStage("idle");
        m_isEvolving = false;
        emit isEvolvingChanged();
        savePopulationState();
        return;
    }

    QString mergedCheckpointPath = m_profileManager->evolutionPath() +
        QString("/consolidated_%1").arg(m_consolidationsCompleted);
    QString artifactsDir = m_profileManager->profilePath() + "/training_artifacts";

    if (!m_weightMerger->mergeCheckpoints(mergeInputs, mergedCheckpointPath, artifactsDir)) {
        setEvolutionStatus("Consolidation merge failed: " + m_weightMerger->lastError());
        setEvolutionStage("idle");
        m_isEvolving = false;
        emit isEvolvingChanged();
        savePopulationState();
        return;
    }

    // Export merged model
    QString mergedModelPath = mergedCheckpointPath + "/consolidated.onnx";
    if (!m_weightMerger->exportMerged(mergedCheckpointPath, artifactsDir, mergedModelPath)) {
        setEvolutionStatus("Consolidation export failed: " + m_weightMerger->lastError());
        setEvolutionStage("idle");
        m_isEvolving = false;
        emit isEvolvingChanged();
        savePopulationState();
        return;
    }

    // Register consolidated adapter
    AdapterMetadata meta;
    meta.adapterName = "consolidated";
    meta.trainingDate = QDateTime::currentDateTime();
    meta.exportedModelPath = mergedModelPath;
    meta.checkpointPath = mergedCheckpointPath;
    meta.parentVersion = m_adapterManager->activeVersion();
    meta.status = "candidate";

    int consolidatedVersion = m_adapterManager->registerCandidate(meta);

    // Record lineage node for consolidated adapter
    if (m_lineageTracker) {
        LineageNode node;
        node.version = consolidatedVersion;
        node.parentVersion = meta.parentVersion;
        node.adapterName = "consolidated";
        node.trainingDate = meta.trainingDate;
        node.status = "candidate";
        node.origin = "consolidation";
        node.generationId = 0;
        node.cycleId = m_population.cycleId;
        m_lineageTracker->recordNode(node);
    }

    // Evaluate consolidated adapter
    if (m_evalSuite) {
        EvalReport report = m_evalSuite->runFullSuite();
        meta.evalScore = report.overallScore;
        m_adapterManager->updateVersionMetadata(consolidatedVersion, meta);

        // Promote if it beats current champion
        double championScore = m_adapterManager->activeAdapterMetadata().evalScore;
        if (meta.evalScore >= championScore + PROMOTION_MARGIN) {
            m_adapterManager->promoteVersion(consolidatedVersion);
            if (m_llmProvider) {
                m_llmProvider->setOrtModelPath(mergedModelPath);
            }
            // Hot-swap local inference model
            if (m_ortInference) {
                m_ortInference->loadModel(mergedModelPath);
                qDebug() << "[EvolutionEngine] Hot-swapped inference model (consolidation) to"
                         << mergedModelPath;
            }
            setEvolutionStatus(QString("Consolidated adapter v%1 promoted (score: %2)")
                .arg(consolidatedVersion).arg(meta.evalScore, 0, 'f', 2));
        } else {
            setEvolutionStatus(QString("Consolidated adapter v%1 not promoted (score: %2)")
                .arg(consolidatedVersion).arg(meta.evalScore, 0, 'f', 2));
        }
    }

    m_consolidationsCompleted++;
    emit consolidationsCompletedChanged();

    // Era snapshot every N consolidations
    if (m_consolidationsCompleted % ERA_SNAPSHOT_INTERVAL == 0) {
        QDateTime now = QDateTime::currentDateTime();
        QString label = now.toString("yyyy_MM");
        createEraSnapshot(label);
        emit consolidationComplete(m_consolidationsCompleted);
    }

    setEvolutionStage("idle");
    m_isEvolving = false;
    emit isEvolvingChanged();
    savePopulationState();
    saveEvolutionLog();
}

void EvolutionEngine::triggerConsolidation()
{
    if (m_isEvolving) return;

    m_isEvolving = true;
    emit isEvolvingChanged();
    setEvolutionStage("consolidating");
    runConsolidation();
}

void EvolutionEngine::createEraSnapshot(const QString &label)
{
    if (!m_profileManager || !m_adapterManager) return;

    int activeVersion = m_adapterManager->activeVersion();
    if (activeVersion < 0) return;

    auto meta = m_adapterManager->getVersionMetadata(activeVersion);
    if (meta.checkpointPath.isEmpty()) return;

    QString snapshotDir = m_profileManager->eraSnapshotsPath() + "/era_" + label;
    QString snapshotCheckpoint = snapshotDir + "/checkpoint";
    QDir().mkpath(snapshotCheckpoint);

    // Copy checkpoint files
    QDir srcDir(meta.checkpointPath);
    for (const auto &fileName : srcDir.entryList(QDir::Files)) {
        QFile::copy(meta.checkpointPath + "/" + fileName,
                   snapshotCheckpoint + "/" + fileName);
    }

    // Save snapshot metadata
    QJsonObject snapshotMeta;
    snapshotMeta["label"] = label;
    snapshotMeta["sourceVersion"] = activeVersion;
    snapshotMeta["evalScore"] = meta.evalScore;
    snapshotMeta["createdAt"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    snapshotMeta["cyclesCompleted"] = m_cyclesCompleted;
    snapshotMeta["consolidationsCompleted"] = m_consolidationsCompleted;

    QFile metaFile(snapshotDir + "/metadata.json");
    if (metaFile.open(QIODevice::WriteOnly)) {
        metaFile.write(QJsonDocument(snapshotMeta).toJson(QJsonDocument::Indented));
        metaFile.close();
    }

    qDebug() << "EvolutionEngine: created era snapshot" << label
             << "from adapter v" << activeVersion;
}

// --- Persistence ---

void EvolutionEngine::loadPopulationState()
{
    if (!m_profileManager) return;

    QString statePath = m_profileManager->evolutionPath() + "/population_state.json";
    QFile file(statePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "EvolutionEngine: no population state found - starting fresh";
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();

    if (doc.isObject()) {
        m_population = PopulationState::fromJson(doc.object());

        // Also load cycle counter from the log
        QJsonObject root = doc.object();
        m_cyclesCompleted = root.value("cyclesCompleted").toInt(0);
        m_consolidationsCompleted = root.value("consolidationsCompleted").toInt(0);

        qDebug() << "EvolutionEngine: loaded state, stage:" << m_population.cycleStage
                 << "cycle:" << m_population.cycleId
                 << "cycles completed:" << m_cyclesCompleted;
    }
}

void EvolutionEngine::savePopulationState()
{
    if (!m_profileManager) return;

    QJsonObject root = m_population.toJson();
    root["cyclesCompleted"] = m_cyclesCompleted;
    root["consolidationsCompleted"] = m_consolidationsCompleted;

    QString statePath = m_profileManager->evolutionPath() + "/population_state.json";
    QDir().mkpath(m_profileManager->evolutionPath());

    QFile file(statePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        file.close();
    }
}

void EvolutionEngine::saveEvolutionLog()
{
    if (!m_profileManager) return;

    QString logPath = m_profileManager->evolutionPath() + "/evolution_log.jsonl";
    QFile logFile(logPath);
    if (logFile.open(QIODevice::Append)) {
        QJsonObject entry;
        entry["cycleId"] = m_population.cycleId - 1;  // Already incremented
        entry["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        entry["populationSize"] = m_population.populationSize;
        entry["championVersion"] = m_population.championVersion;

        // Summary of variant scores
        QJsonArray variantScores;
        for (const auto &v : m_population.variants) {
            QJsonObject vs;
            vs["variantIndex"] = v.variantIndex;
            vs["lr"] = static_cast<double>(v.learningRate);
            vs["score"] = v.evalScore;
            vs["loss"] = static_cast<double>(v.finalLoss);
            vs["status"] = v.status;
            variantScores.append(vs);
        }
        entry["variants"] = variantScores;

        logFile.write(QJsonDocument(entry).toJson(QJsonDocument::Compact));
        logFile.write("\n");
        logFile.close();
    }
}

QVariantList EvolutionEngine::getPopulationForQml() const
{
    QVariantList result;
    for (const auto &v : m_population.variants) {
        QVariantMap map;
        map["variantIndex"] = v.variantIndex;
        map["learningRate"] = static_cast<double>(v.learningRate);
        map["status"] = v.status;
        map["evalScore"] = v.evalScore;
        map["finalLoss"] = static_cast<double>(v.finalLoss);
        map["trainingSteps"] = v.trainingSteps;
        result.append(map);
    }
    return result;
}

#endif // DATAFORM_TRAINING_ENABLED
