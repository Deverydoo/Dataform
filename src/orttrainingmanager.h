#ifndef ORTTRAININGMANAGER_H
#define ORTTRAININGMANAGER_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>
#include <QThread>
#include <QMutex>
#include <QAtomicInt>
#include <QList>
#include <vector>
#include <memory>

#include <onnxruntime_training_cxx_api.h>

struct TrainingExample;

struct TrainingConfig {
    float learningRate = 1e-4f;
    int batchSize = 4;
    int maxStepsPerSession = 50;
    int checkpointIntervalSteps = 10;
    int intraOpThreads = 2;
    QString artifactsDir;
    QString checkpointDir;
    QString outputDir;
};

class OrtTrainingManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isInitialized READ isInitialized NOTIFY isInitializedChanged)
    Q_PROPERTY(bool isTraining READ isTraining NOTIFY isTrainingChanged)
    Q_PROPERTY(float currentLoss READ currentLoss NOTIFY currentLossChanged)
    Q_PROPERTY(int currentStep READ currentStep NOTIFY currentStepChanged)
    Q_PROPERTY(int totalSteps READ totalSteps NOTIFY totalStepsChanged)
    Q_PROPERTY(QString trainingStatus READ trainingStatus NOTIFY trainingStatusChanged)

public:
    explicit OrtTrainingManager(QObject *parent = nullptr);
    ~OrtTrainingManager();

    // Initialize from training artifacts directory
    bool initialize(const TrainingConfig &config);
    void shutdown();

    // Load/save checkpoint for pause/resume
    bool loadCheckpoint(const QString &checkpointPath);
    bool saveCheckpoint(const QString &checkpointPath);

    // Start training (runs on background thread)
    void startTraining(const QList<TrainingExample> &examples, const TrainingConfig &config);

    // Request graceful pause
    void requestPause();

    // Export trained model for inference
    bool exportForInference(const QString &outputPath);
    QString exportedModelPath() const { return m_exportedModelPath; }

    // Property getters
    bool isInitialized() const { return m_initialized; }
    bool isTraining() const { return m_isTraining.loadRelaxed(); }
    float currentLoss() const { return m_currentLoss; }
    int currentStep() const { return m_currentStep; }
    int totalSteps() const { return m_totalSteps; }
    QString trainingStatus() const { return m_trainingStatus; }

signals:
    void isInitializedChanged();
    void isTrainingChanged();
    void currentLossChanged();
    void currentStepChanged();
    void totalStepsChanged();
    void trainingStatusChanged();
    void trainingComplete(int stepsCompleted, float finalLoss);
    void trainingPaused(int stepsCompleted);
    void trainingError(const QString &error);
    void checkpointSaved(const QString &path);
    void modelExported(const QString &path);

private:
    // Background training loop
    void trainingLoop(QList<TrainingExample> examples, TrainingConfig config);

    // Single training step
    float executeTrainStep(const std::vector<int64_t> &inputIds,
                           const std::vector<int64_t> &labels,
                           const std::vector<int64_t> &attentionMask,
                           int batchSize, int seqLength);

    // Prepare a batch from examples
    void prepareBatch(const QList<TrainingExample> &examples,
                      int batchStart, int batchSize,
                      std::vector<int64_t> &inputIds,
                      std::vector<int64_t> &labels,
                      std::vector<int64_t> &attentionMask,
                      int &seqLength) const;

    void setTrainingStatus(const QString &status);

    // ORT objects
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::SessionOptions> m_sessionOptions;
    std::unique_ptr<Ort::CheckpointState> m_checkpointState;
    std::unique_ptr<Ort::TrainingSession> m_trainingSession;

    // Thread safety
    QAtomicInt m_pauseRequested{0};
    QAtomicInt m_isTraining{0};

    // State
    bool m_initialized = false;
    float m_currentLoss = 0.0f;
    int m_currentStep = 0;
    int m_totalSteps = 0;
    QString m_trainingStatus = "Not initialized";
    QString m_exportedModelPath;
    TrainingConfig m_activeConfig;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // ORTTRAININGMANAGER_H
