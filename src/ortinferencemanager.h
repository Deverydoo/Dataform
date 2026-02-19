#ifndef ORTINFERENCEMANAGER_H
#define ORTINFERENCEMANAGER_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>
#include <QJsonArray>
#include <QJsonObject>
#include <QAtomicInt>
#include <vector>
#include <memory>
#include <random>

// Must use the training header (not just onnxruntime_cxx_api.h) because
// the training header forward-declares OrtRelease(OrtTrainingSession*) before
// onnxruntime_cxx_api.h processes Base<T> — needed for MOC compilation order.
#include <onnxruntime_training_cxx_api.h>

class Tokenizer;

class OrtInferenceManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isModelLoaded READ isModelLoaded NOTIFY isModelLoadedChanged)
    Q_PROPERTY(bool isGenerating READ isGenerating NOTIFY isGeneratingChanged)
    Q_PROPERTY(QString modelPath READ modelPath NOTIFY modelPathChanged)

public:
    explicit OrtInferenceManager(QObject *parent = nullptr);
    ~OrtInferenceManager();

    void setTokenizer(Tokenizer *tokenizer);

    bool loadModel(const QString &onnxPath);
    void unloadModel();
    bool isModelLoaded() const { return m_session != nullptr; }
    bool isGenerating() const { return m_isGenerating.loadRelaxed(); }
    QString modelPath() const { return m_modelPath; }

    // Async generation on background thread
    void generate(const QString &systemPrompt, const QJsonArray &messages,
                  int maxNewTokens = 512, float temperature = 0.7f,
                  int topK = 40, float topP = 0.9f);

    void cancelGeneration();

signals:
    void isModelLoadedChanged();
    void isGeneratingChanged();
    void modelPathChanged();
    void tokenGenerated(const QString &token);
    void generationComplete(const QString &fullText);
    void generationError(const QString &error);

private:
    // Build ChatML-formatted prompt tokens from system prompt + messages
    std::vector<int64_t> buildPromptTokens(const QString &systemPrompt,
                                            const QJsonArray &messages);

    // Sample a token from logits with temperature, top-k, top-p
    int64_t sampleToken(const float *logits, int vocabSize,
                        float temperature, int topK, float topP);

    // Background generation loop
    void generationLoop(std::vector<int64_t> promptTokens,
                        int maxNewTokens, float temperature,
                        int topK, float topP);

    // ORT objects
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;

    Tokenizer *m_tokenizer = nullptr;
    QAtomicInt m_cancelRequested{0};
    QAtomicInt m_isGenerating{0};
    QString m_modelPath;

    std::mt19937 m_rng;

    static constexpr int MAX_CONTEXT_LENGTH = 512;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // ORTINFERENCEMANAGER_H
