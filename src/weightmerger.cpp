#ifdef DATAFORM_TRAINING_ENABLED

#include "weightmerger.h"
#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <onnxruntime_training_cxx_api.h>
#include <vector>
#include <numeric>

WeightMerger::WeightMerger(QObject *parent)
    : QObject(parent)
{
}

WeightMerger::~WeightMerger() = default;

bool WeightMerger::mergeCheckpoints(const QList<MergeInput> &inputs,
                                     const QString &outputCheckpointPath,
                                     const QString &artifactsDir)
{
    if (inputs.size() < 2) {
        m_lastError = "Need at least 2 checkpoints to merge";
        emit mergeError(m_lastError);
        return false;
    }

    // Normalize weights
    double weightSum = 0.0;
    for (const auto &input : inputs) {
        weightSum += input.weight;
    }
    if (weightSum <= 0.0) {
        m_lastError = "Merge weights must be positive";
        emit mergeError(m_lastError);
        return false;
    }

    QList<double> normalizedWeights;
    for (const auto &input : inputs) {
        normalizedWeights.append(input.weight / weightSum);
    }

    try {
        // Create ORT environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "WeightMerger");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);

        // Resolve artifact paths
        QString trainingModelPath = artifactsDir + "/training_model.onnx";
        QString evalModelPath = artifactsDir + "/eval_model.onnx";
        QString optimizerModelPath = artifactsDir + "/optimizer_model.onnx";

        if (!QFile::exists(trainingModelPath)) {
            m_lastError = "Training model not found: " + trainingModelPath;
            emit mergeError(m_lastError);
            return false;
        }

        // Step 1: Load first checkpoint, create session, get parameter size
        auto firstCheckpoint = std::make_unique<Ort::CheckpointState>(
            Ort::CheckpointState::LoadCheckpoint(inputs[0].checkpointPath.toStdWString()));

        auto firstSession = std::make_unique<Ort::TrainingSession>(
            env, sessionOptions, *firstCheckpoint,
            trainingModelPath.toStdWString(),
            evalModelPath.toStdWString(),
            optimizerModelPath.toStdWString());

        // Step 2: Read first checkpoint's parameters via ToBuffer
        Ort::Value firstBuffer = firstSession->ToBuffer(/*only_trainable=*/true);

        auto tensorInfo = firstBuffer.GetTensorTypeAndShapeInfo();
        size_t paramSize = tensorInfo.GetElementCount();

        if (paramSize == 0) {
            m_lastError = "No trainable parameters found";
            emit mergeError(m_lastError);
            return false;
        }

        qDebug() << "WeightMerger: parameter size =" << paramSize << "floats";

        const float *firstData = firstBuffer.GetTensorData<float>();
        std::vector<float> averaged(paramSize, 0.0f);

        // Scale first checkpoint's parameters and add to accumulator
        float w0 = static_cast<float>(normalizedWeights[0]);
        for (size_t i = 0; i < paramSize; ++i) {
            averaged[i] += firstData[i] * w0;
        }

        // Release first session
        firstSession.reset();
        firstCheckpoint.reset();

        // Step 3: Load remaining checkpoints and accumulate
        for (int c = 1; c < inputs.size(); ++c) {
            auto checkpoint = std::make_unique<Ort::CheckpointState>(
                Ort::CheckpointState::LoadCheckpoint(inputs[c].checkpointPath.toStdWString()));

            auto session = std::make_unique<Ort::TrainingSession>(
                env, sessionOptions, *checkpoint,
                trainingModelPath.toStdWString(),
                evalModelPath.toStdWString(),
                optimizerModelPath.toStdWString());

            Ort::Value paramBuffer = session->ToBuffer(/*only_trainable=*/true);
            const float *paramData = paramBuffer.GetTensorData<float>();

            float wc = static_cast<float>(normalizedWeights[c]);
            for (size_t i = 0; i < paramSize; ++i) {
                averaged[i] += paramData[i] * wc;
            }

            session.reset();
            checkpoint.reset();

            qDebug() << "WeightMerger: accumulated checkpoint" << c
                     << "with weight" << normalizedWeights[c];
        }

        // Step 4: Write averaged parameters back to a new checkpoint
        // Load any checkpoint as the base structure
        QDir().mkpath(outputCheckpointPath);

        auto outputCheckpoint = std::make_unique<Ort::CheckpointState>(
            Ort::CheckpointState::LoadCheckpoint(inputs[0].checkpointPath.toStdWString()));

        auto outputSession = std::make_unique<Ort::TrainingSession>(
            env, sessionOptions, *outputCheckpoint,
            trainingModelPath.toStdWString(),
            evalModelPath.toStdWString(),
            optimizerModelPath.toStdWString());

        // Create tensor from averaged data and write back
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                   OrtMemType::OrtMemTypeDefault);
        std::vector<int64_t> paramShape = {static_cast<int64_t>(paramSize)};
        Ort::Value avgTensor = Ort::Value::CreateTensor<float>(
            memInfo, averaged.data(), averaged.size(),
            paramShape.data(), paramShape.size());
        outputSession->FromBuffer(avgTensor);

        // Save the merged checkpoint
        Ort::CheckpointState::SaveCheckpoint(*outputCheckpoint,
                                              outputCheckpointPath.toStdWString());

        outputSession.reset();
        outputCheckpoint.reset();

        qDebug() << "WeightMerger: merged" << inputs.size()
                 << "checkpoints to" << outputCheckpointPath;
        emit mergeComplete(outputCheckpointPath);
        return true;

    } catch (const Ort::Exception &e) {
        m_lastError = QString("ORT merge failed: %1").arg(e.what());
        qWarning() << m_lastError;
        emit mergeError(m_lastError);
        return false;
    } catch (const std::exception &e) {
        m_lastError = QString("Merge failed: %1").arg(e.what());
        qWarning() << m_lastError;
        emit mergeError(m_lastError);
        return false;
    }
}

bool WeightMerger::exportMerged(const QString &checkpointPath,
                                 const QString &artifactsDir,
                                 const QString &outputModelPath)
{
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "WeightMergerExport");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);

        QString trainingModelPath = artifactsDir + "/training_model.onnx";
        QString evalModelPath = artifactsDir + "/eval_model.onnx";
        QString optimizerModelPath = artifactsDir + "/optimizer_model.onnx";

        auto checkpoint = std::make_unique<Ort::CheckpointState>(
            Ort::CheckpointState::LoadCheckpoint(checkpointPath.toStdWString()));

        auto session = std::make_unique<Ort::TrainingSession>(
            env, sessionOptions, *checkpoint,
            trainingModelPath.toStdWString(),
            evalModelPath.toStdWString(),
            optimizerModelPath.toStdWString());

        QDir().mkpath(QFileInfo(outputModelPath).absolutePath());

        std::vector<std::string> outputNames = {"logits"};
        session->ExportModelForInferencing(
            outputModelPath.toStdWString(), outputNames);

        qDebug() << "WeightMerger: exported merged model to" << outputModelPath;
        return true;

    } catch (const Ort::Exception &e) {
        m_lastError = QString("Export failed: %1").arg(e.what());
        qWarning() << m_lastError;
        emit mergeError(m_lastError);
        return false;
    }
}

#endif // DATAFORM_TRAINING_ENABLED
