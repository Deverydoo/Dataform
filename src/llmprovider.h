#ifndef LLMPROVIDER_H
#define LLMPROVIDER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QQueue>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#ifdef DATAFORM_TRAINING_ENABLED
class OrtInferenceManager;
#endif

class LLMProviderManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString currentProvider READ currentProvider WRITE setCurrentProvider NOTIFY currentProviderChanged)
    Q_PROPERTY(QString currentModel READ currentModel WRITE setCurrentModel NOTIFY currentModelChanged)
    Q_PROPERTY(QStringList availableModels READ availableModels NOTIFY availableModelsChanged)
    Q_PROPERTY(bool isConnected READ isConnected NOTIFY isConnectedChanged)
    Q_PROPERTY(QString connectionStatus READ connectionStatus NOTIFY connectionStatusChanged)

public:
    explicit LLMProviderManager(QObject *parent = nullptr);
    ~LLMProviderManager();

    enum Provider {
        LMStudio,
        Ollama,
        OpenAI,
        Anthropic,
        Local
    };
    Q_ENUM(Provider)

    enum BackgroundPriority {
        PriorityHigh = 0,    // TraitExtractor, SentimentTracker
        PriorityNormal = 1,  // ResearchEngine
        PriorityLow = 2      // GoalTracker, NewsEngine, LearningEngine
    };

    QString currentProvider() const { return m_currentProvider; }
    void setCurrentProvider(const QString &provider);

    QString currentModel() const { return m_currentModel; }
    void setCurrentModel(const QString &model);

    QStringList availableModels() const { return m_availableModels; }

    bool isConnected() const { return m_isConnected; }
    QString connectionStatus() const { return m_connectionStatus; }

    // Single-turn prompt (backward compatibility)
    Q_INVOKABLE void sendPrompt(const QString &prompt);

    // Multi-turn conversation with system prompt and message history
    Q_INVOKABLE void sendConversation(const QString &systemPrompt,
                                       const QJsonArray &messages,
                                       const QString &requestTag = QString());

    // Background request queue — serializes non-chat LLM requests
    // and routes responses via backgroundResponseReceived(owner, response)
    void sendBackgroundRequest(const QString &owner,
                               const QString &systemPrompt,
                               const QJsonArray &messages,
                               BackgroundPriority priority = PriorityNormal);
    void sendBackgroundPrompt(const QString &owner, const QString &prompt,
                              BackgroundPriority priority = PriorityNormal);
    bool isBackgroundBusy() const { return m_backgroundBusy; }

    Q_INVOKABLE void refreshModels();
    Q_INVOKABLE void testConnection();

    // Allow setting API keys and URLs from settings
    void setOpenAIKey(const QString &key) { m_openAIKey = key; }
    void setAnthropicKey(const QString &key) { m_anthropicKey = key; }
    void setLmStudioUrl(const QString &url) { m_lmStudioUrl = url; }
    void setOllamaUrl(const QString &url) { m_ollamaUrl = url; }

    // Phase 2: ORT adapted model path
    void setOrtModelPath(const QString &path) { m_ortModelPath = path; }

#ifdef DATAFORM_TRAINING_ENABLED
    // Phase 6: Local inference via ORT
    void setOrtInferenceManager(OrtInferenceManager *mgr);
#endif

signals:
    void currentProviderChanged();
    void currentModelChanged();
    void availableModelsChanged();
    void isConnectedChanged();
    void connectionStatusChanged();
    void modelsRefreshed(bool success, const QString &error);
    void connectionTested(bool success, const QString &message);
    void responseReceived(const QString &response);
    void tokenStreamed(const QString &token);
    void errorOccurred(const QString &error);
    void backgroundResponseReceived(const QString &owner, const QString &response);
    void backgroundErrorOccurred(const QString &owner, const QString &error);

private slots:
    void onNetworkReply(QNetworkReply *reply);
    void onTestConnectionReply(QNetworkReply *reply);
    void onRefreshModelsReply(QNetworkReply *reply);
    void onPromptReply(QNetworkReply *reply);

private:
    void scanLMStudioModels();
    void scanOllamaModels();
    void setConnectionStatus(bool connected, const QString &status);
    Provider getProviderEnum() const;
    QString getProviderBaseUrl() const;
    QJsonArray formatMessagesForProvider(const QString &provider, const QJsonArray &messages) const;
    void processNextBackgroundRequest();

    QNetworkAccessManager *m_networkManager;
    QString m_currentProvider;
    QString m_currentModel;
    QStringList m_availableModels;
    bool m_isConnected;
    QString m_connectionStatus;

    // API endpoints
    QString m_lmStudioUrl = "http://localhost:1234";
    QString m_ollamaUrl = "http://localhost:11434";
    QString m_openAIUrl = "https://api.openai.com/v1";
    QString m_anthropicUrl = "https://api.anthropic.com/v1";

    // API keys
    QString m_openAIKey;
    QString m_anthropicKey;

    // Phase 2: ORT adapted model
    QString m_ortModelPath;

    // Background request queue
    static constexpr int MAX_BACKGROUND_QUEUE = 20;
    struct BackgroundRequest {
        QString owner;
        QString systemPrompt;
        QJsonArray messages;
        BackgroundPriority priority = PriorityNormal;
    };
    QQueue<BackgroundRequest> m_backgroundQueue;
    QString m_activeBackgroundOwner;
    bool m_backgroundBusy = false;
    QString m_pendingLocalTag;  // For local (ORT) provider background routing

#ifdef DATAFORM_TRAINING_ENABLED
    OrtInferenceManager *m_ortInference = nullptr;
#endif
};

#endif // LLMPROVIDER_H
