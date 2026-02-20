#ifndef LLAMACPPMANAGER_H
#define LLAMACPPMANAGER_H

#ifdef DATAFORM_LLAMACPP_ENABLED

#include <QObject>
#include <QString>
#include <QJsonArray>
#include <QJsonObject>
#include <QAtomicInt>
#include <string>
#include <vector>

#include "sessionguard.h"

struct llama_model;
struct llama_context;
struct llama_sampler;

class LlamaCppManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isModelLoaded READ isModelLoaded NOTIFY isModelLoadedChanged)
    Q_PROPERTY(bool isGenerating READ isGenerating NOTIFY isGeneratingChanged)
    Q_PROPERTY(QString modelPath READ modelPath NOTIFY modelPathChanged)

public:
    explicit LlamaCppManager(QObject *parent = nullptr);
    ~LlamaCppManager();

    bool loadModel(const QString &ggufPath, int nCtx = 4096, int nThreads = 4);
    void unloadModel();
    bool isModelLoaded() const { return m_model != nullptr; }
    bool isGenerating() const { return m_isGenerating.loadRelaxed(); }
    QString modelPath() const { return m_modelPath; }

    Q_INVOKABLE QStringList availableModels() const;
    Q_INVOKABLE QString loadedModelName() const;

    // Async generation on background thread
    void generate(const QString &systemPrompt, const QJsonArray &messages,
                  int maxNewTokens = 1024, float temperature = 0.3f);

    void cancelGeneration();

signals:
    void isModelLoadedChanged();
    void isGeneratingChanged();
    void modelPathChanged();
    void tokenGenerated(const QString &token);
    void generationComplete(const QString &fullText);
    void generationError(const QString &error);

private:
    std::string buildChatPrompt(const QString &systemPrompt, const QJsonArray &messages);
    void generationLoop(std::string prompt, int maxNewTokens, float temperature);

    llama_model *m_model = nullptr;
    llama_context *m_ctx = nullptr;
    QAtomicInt m_isGenerating{0};
    QAtomicInt m_cancelRequested{0};
    SessionGuard m_sessionGuard;
    QString m_modelPath;
    int m_nCtx = 4096;
};

#endif // DATAFORM_LLAMACPP_ENABLED
#endif // LLAMACPPMANAGER_H
