#ifdef DATAFORM_TRAINING_ENABLED

#include "orttrainingmanager.h"
#include "trainingdatagenerator.h"
#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <QMetaObject>
#include <QThread>
#include <algorithm>
#include <numeric>

OrtTrainingManager::OrtTrainingManager(QObject *parent)
    : QObject(parent)
{
}

OrtTrainingManager::~OrtTrainingManager()
{
    shutdown();
}

bool OrtTrainingManager::initialize(const TrainingConfig &config)
{
    if (m_initialized) return true;

    m_activeConfig = config;

    try {
        // 1. Create ORT environment
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DataformTraining");

        // 2. Session options - limit CPU usage
        m_sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_sessionOptions->SetIntraOpNumThreads(config.intraOpThreads);
        m_sessionOptions->SetInterOpNumThreads(1);

        // 3. Resolve artifact paths
        QString trainingModelPath = config.artifactsDir + "/training_model.onnx";
        QString evalModelPath = config.artifactsDir + "/eval_model.onnx";
        QString optimizerModelPath = config.artifactsDir + "/optimizer_model.onnx";

        // Check if artifacts exist
        if (!QFile::exists(trainingModelPath)) {
            setTrainingStatus("Missing training_model.onnx in artifacts directory");
            qWarning() << "OrtTrainingManager: training model not found:" << trainingModelPath;
            return false;
        }

        // 4. Load checkpoint
        QString checkpointPath = config.checkpointDir;
        if (!QDir(checkpointPath).exists()) {
            checkpointPath = config.artifactsDir + "/checkpoint";
        }

        if (QDir(checkpointPath).exists()) {
            m_checkpointState = std::make_unique<Ort::CheckpointState>(
                Ort::CheckpointState::LoadCheckpoint(checkpointPath.toStdWString()));
        } else {
            setTrainingStatus("No checkpoint found - cannot initialize");
            qWarning() << "OrtTrainingManager: no checkpoint at:" << checkpointPath;
            return false;
        }

        // 5. Create training session
        m_trainingSession = std::make_unique<Ort::TrainingSession>(
            *m_env,
            *m_sessionOptions,
            *m_checkpointState,
            trainingModelPath.toStdWString(),
            evalModelPath.toStdWString(),
            optimizerModelPath.toStdWString());

        m_initialized = true;
        setTrainingStatus("Initialized - ready for training");
        emit isInitializedChanged();
        qDebug() << "OrtTrainingManager: initialized successfully";
        return true;

    } catch (const Ort::Exception &e) {
        QString error = QString("ORT initialization failed: %1").arg(e.what());
        setTrainingStatus(error);
        qWarning() << error;
        return false;
    } catch (const std::exception &e) {
        QString error = QString("Initialization failed: %1").arg(e.what());
        setTrainingStatus(error);
        qWarning() << error;
        return false;
    }
}

void OrtTrainingManager::shutdown()
{
    if (isTraining()) {
        requestPause();
        // Wait briefly for the training thread to finish
        QThread::msleep(3000);
    }

    m_trainingSession.reset();
    m_checkpointState.reset();
    m_sessionOptions.reset();
    m_env.reset();
    m_initialized = false;
}

bool OrtTrainingManager::loadCheckpoint(const QString &checkpointPath)
{
    if (!m_env) return false;

    try {
        m_checkpointState = std::make_unique<Ort::CheckpointState>(
            Ort::CheckpointState::LoadCheckpoint(checkpointPath.toStdWString()));
        qDebug() << "OrtTrainingManager: checkpoint loaded from" << checkpointPath;
        return true;
    } catch (const Ort::Exception &e) {
        qWarning() << "OrtTrainingManager: failed to load checkpoint:" << e.what();
        return false;
    }
}

bool OrtTrainingManager::saveCheckpoint(const QString &checkpointPath)
{
    if (!m_checkpointState) return false;

    try {
        QDir().mkpath(checkpointPath);
        Ort::CheckpointState::SaveCheckpoint(*m_checkpointState,
                                              checkpointPath.toStdWString());
        qDebug() << "OrtTrainingManager: checkpoint saved to" << checkpointPath;
        emit checkpointSaved(checkpointPath);
        return true;
    } catch (const Ort::Exception &e) {
        qWarning() << "OrtTrainingManager: failed to save checkpoint:" << e.what();
        return false;
    }
}

void OrtTrainingManager::prepareBatch(
    const QList<TrainingExample> &examples,
    int batchStart, int batchSize,
    std::vector<int64_t> &inputIds,
    std::vector<int64_t> &labels,
    std::vector<int64_t> &attentionMask,
    int &seqLength) const
{
    // Find sequence length from first example
    seqLength = 0;
    if (!examples.isEmpty()) {
        seqLength = static_cast<int>(examples[0].inputIds.size());
    }
    if (seqLength == 0) return;

    int actualBatchSize = qMin(batchSize, examples.size() - batchStart);
    inputIds.resize(actualBatchSize * seqLength);
    labels.resize(actualBatchSize * seqLength);
    attentionMask.resize(actualBatchSize * seqLength);

    for (int b = 0; b < actualBatchSize; ++b) {
        int idx = (batchStart + b) % examples.size();
        const auto &ex = examples[idx];

        int copyLen = qMin(seqLength, static_cast<int>(ex.inputIds.size()));
        for (int s = 0; s < copyLen; ++s) {
            inputIds[b * seqLength + s] = ex.inputIds[s];
            labels[b * seqLength + s] = ex.labels[s];
            attentionMask[b * seqLength + s] = ex.attentionMask[s];
        }
        // Pad remainder
        for (int s = copyLen; s < seqLength; ++s) {
            inputIds[b * seqLength + s] = 0;
            labels[b * seqLength + s] = -100;
            attentionMask[b * seqLength + s] = 0;
        }
    }
}

float OrtTrainingManager::executeTrainStep(
    const std::vector<int64_t> &inputIds,
    const std::vector<int64_t> &labels,
    const std::vector<int64_t> &attentionMask,
    int batchSize, int seqLength)
{
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                  OrtMemType::OrtMemTypeDefault);

    std::array<int64_t, 2> inputShape = {batchSize, seqLength};

    // Create input tensors
    Ort::Value inputIdsTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo,
        const_cast<int64_t *>(inputIds.data()),
        inputIds.size(),
        inputShape.data(), inputShape.size());

    Ort::Value attentionMaskTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo,
        const_cast<int64_t *>(attentionMask.data()),
        attentionMask.size(),
        inputShape.data(), inputShape.size());

    Ort::Value labelsTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo,
        const_cast<int64_t *>(labels.data()),
        labels.size(),
        inputShape.data(), inputShape.size());

    // Build feeds vector
    std::vector<Ort::Value> feeds;
    feeds.push_back(std::move(inputIdsTensor));
    feeds.push_back(std::move(attentionMaskTensor));
    feeds.push_back(std::move(labelsTensor));

    // TrainStep: forward + backward pass, returns loss
    std::vector<Ort::Value> fetches = m_trainingSession->TrainStep(feeds);

    float loss = 0.0f;
    if (!fetches.empty() && fetches[0].IsTensor()) {
        loss = *fetches[0].GetTensorMutableData<float>();
    }

    // OptimizerStep: apply gradient updates
    m_trainingSession->OptimizerStep();

    // LazyResetGrad: zero gradients for next step
    m_trainingSession->LazyResetGrad();

    return loss;
}

void OrtTrainingManager::startTraining(const QList<TrainingExample> &examples,
                                        const TrainingConfig &config)
{
    if (isTraining()) {
        qWarning() << "OrtTrainingManager: already training";
        return;
    }

    if (!m_initialized) {
        emit trainingError("Not initialized - call initialize() first");
        return;
    }

    if (examples.isEmpty()) {
        emit trainingError("No training examples provided");
        return;
    }

    m_pauseRequested.storeRelaxed(0);
    m_isTraining.storeRelaxed(1);
    emit isTrainingChanged();

    // Launch training on a background thread
    auto *thread = QThread::create([this, examples, config]() {
        trainingLoop(examples, config);
    });
    connect(thread, &QThread::finished, thread, &QThread::deleteLater);
    thread->start();
}

void OrtTrainingManager::trainingLoop(QList<TrainingExample> examples,
                                       TrainingConfig config)
{
    int numExamples = examples.size();
    int batchSize = qMin(config.batchSize, numExamples);
    int totalSteps = qMin(config.maxStepsPerSession,
                          (numExamples + batchSize - 1) / batchSize);

    m_progress.setTotalSteps(totalSteps);
    QMetaObject::invokeMethod(this, [this]() {
        emit totalStepsChanged();
    }, Qt::QueuedConnection);

    float lossSum = 0.0f;
    int completedSteps = 0;

    setTrainingStatus(QString("Training: 0/%1 steps").arg(totalSteps));

    for (int step = 0; step < totalSteps; ++step) {
        // Check for pause request
        if (m_pauseRequested.loadRelaxed()) {
            saveCheckpoint(config.checkpointDir);
            setTrainingStatus(QString("Paused at step %1/%2").arg(step).arg(totalSteps));

            m_isTraining.storeRelaxed(0);
            QMetaObject::invokeMethod(this, [this, step]() {
                emit isTrainingChanged();
                emit trainingPaused(step);
            }, Qt::QueuedConnection);
            return;
        }

        try {
            // Prepare batch
            std::vector<int64_t> batchInputIds, batchLabels, batchAttentionMask;
            int seqLength;
            int batchStart = (step * batchSize) % numExamples;

            prepareBatch(examples, batchStart, batchSize,
                        batchInputIds, batchLabels, batchAttentionMask, seqLength);

            if (seqLength == 0) continue;

            // Execute training step
            float loss = executeTrainStep(batchInputIds, batchLabels, batchAttentionMask,
                                          batchSize, seqLength);

            lossSum += loss;
            completedSteps = step + 1;

            // Update progress (thread-safe)
            m_progress.setCurrentStep(completedSteps);
            m_progress.setCurrentLoss(loss);

            QMetaObject::invokeMethod(this, [this, completedSteps, totalSteps, loss]() {
                emit currentStepChanged();
                emit currentLossChanged();
                setTrainingStatus(QString("Training: %1/%2 steps (loss: %3)")
                    .arg(completedSteps).arg(totalSteps)
                    .arg(loss, 0, 'f', 4));
            }, Qt::QueuedConnection);

            // Periodic checkpoint
            if (completedSteps % config.checkpointIntervalSteps == 0) {
                saveCheckpoint(config.checkpointDir);
            }

        } catch (const Ort::Exception &e) {
            QString error = QString("Training step %1 failed: %2").arg(step).arg(e.what());
            qWarning() << error;
            m_isTraining.storeRelaxed(0);
            QMetaObject::invokeMethod(this, [this, error]() {
                emit isTrainingChanged();
                emit trainingError(error);
            }, Qt::QueuedConnection);
            return;
        }
    }

    // Training complete
    float avgLoss = (completedSteps > 0) ? lossSum / completedSteps : 0.0f;
    saveCheckpoint(config.checkpointDir);

    m_isTraining.storeRelaxed(0);
    QMetaObject::invokeMethod(this, [this, completedSteps, avgLoss]() {
        emit isTrainingChanged();
        setTrainingStatus(QString("Complete: %1 steps, avg loss: %2")
            .arg(completedSteps).arg(avgLoss, 0, 'f', 4));
        emit trainingComplete(completedSteps, avgLoss);
    }, Qt::QueuedConnection);
}

void OrtTrainingManager::requestPause()
{
    m_pauseRequested.storeRelaxed(1);
    qDebug() << "OrtTrainingManager: pause requested";
}

bool OrtTrainingManager::exportForInference(const QString &outputPath)
{
    if (!m_trainingSession) {
        qWarning() << "OrtTrainingManager: cannot export - no training session";
        return false;
    }

    try {
        QDir().mkpath(QFileInfo(outputPath).absolutePath());

        // Export with output names matching the model graph
        std::vector<std::string> outputNames = {"logits"};
        m_trainingSession->ExportModelForInferencing(
            outputPath.toStdWString(), outputNames);

        m_exportedModelPath = outputPath;
        emit modelExported(outputPath);
        qDebug() << "OrtTrainingManager: model exported to" << outputPath;
        return true;

    } catch (const Ort::Exception &e) {
        qWarning() << "OrtTrainingManager: export failed:" << e.what();
        return false;
    }
}

void OrtTrainingManager::setTrainingStatus(const QString &status)
{
    m_progress.setStatus(status);
    QMetaObject::invokeMethod(this, [this]() {
        emit trainingStatusChanged();
    }, Qt::QueuedConnection);
}

#endif // DATAFORM_TRAINING_ENABLED
