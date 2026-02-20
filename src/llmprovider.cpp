#include "llmprovider.h"
#ifdef DATAFORM_TRAINING_ENABLED
#include "ortinferencemanager.h"
#endif
#ifdef DATAFORM_LLAMACPP_ENABLED
#include "llamacppmanager.h"
#endif
#include <QNetworkRequest>
#include <QDebug>

LLMProviderManager::LLMProviderManager(QObject *parent)
    : QObject(parent)
    , m_networkManager(new QNetworkAccessManager(this))
    , m_currentProvider("Ollama")
    , m_isConnected(false)
    , m_connectionStatus("Not connected")
{
    connect(m_networkManager, &QNetworkAccessManager::finished,
            this, &LLMProviderManager::onNetworkReply);

    // Default models for Ollama (DATAFORM's primary provider)
    m_availableModels = {"llama3.1:8b", "mistral:7b", "codellama:13b", "qwen2.5-coder:7b"};
}

LLMProviderManager::~LLMProviderManager()
{
}

void LLMProviderManager::setCurrentProvider(const QString &provider)
{
    if (m_currentProvider != provider) {
        m_currentProvider = provider;
        emit currentProviderChanged();

        if (provider == "LM Studio") {
            m_availableModels = {"llama-3.1-8b-instruct", "mistral-7b-instruct", "codellama-13b"};
        } else if (provider == "Ollama") {
            m_availableModels = {"llama3.1:8b", "mistral:7b", "codellama:13b", "qwen2.5-coder:7b"};
        } else if (provider == "OpenAI") {
            m_availableModels = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"};
        } else if (provider == "Anthropic") {
            m_availableModels = {"claude-sonnet-4-5-20250929", "claude-sonnet-3-5-20241022",
                                 "claude-opus-3-20240229", "claude-haiku-3-5-20241022"};
        } else if (provider == "Local") {
            m_availableModels = {"personal-model"};
        }

        emit availableModelsChanged();
        setConnectionStatus(false, "Not connected");
    }
}

void LLMProviderManager::setCurrentModel(const QString &model)
{
    if (m_currentModel != model) {
        m_currentModel = model;
        emit currentModelChanged();
    }
}

void LLMProviderManager::refreshModels()
{
    qDebug() << "Refreshing models for provider:" << m_currentProvider;

    if (m_currentProvider == "LM Studio") {
        scanLMStudioModels();
    } else if (m_currentProvider == "Ollama") {
        scanOllamaModels();
    } else if (m_currentProvider == "Local") {
        m_availableModels = {"personal-model"};
        emit availableModelsChanged();
        emit modelsRefreshed(true, "");
    } else {
        emit modelsRefreshed(true, "");
    }
}

void LLMProviderManager::scanLMStudioModels()
{
    QNetworkRequest request(QUrl(m_lmStudioUrl + "/v1/models"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply *reply = m_networkManager->get(request);
    reply->setProperty("requestType", "refreshModels");
    reply->setProperty("provider", "LMStudio");
}

void LLMProviderManager::scanOllamaModels()
{
    QNetworkRequest request(QUrl(m_ollamaUrl + "/api/tags"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply *reply = m_networkManager->get(request);
    reply->setProperty("requestType", "refreshModels");
    reply->setProperty("provider", "Ollama");
}

void LLMProviderManager::testConnection()
{
    qDebug() << "Testing connection to:" << m_currentProvider << "with model:" << m_currentModel;

    if (m_currentProvider == "Local") {
#ifdef DATAFORM_TRAINING_ENABLED
        bool loaded = m_ortInference && m_ortInference->isModelLoaded();
        setConnectionStatus(loaded, loaded ? "Local model loaded" : "No model loaded");
        emit connectionTested(loaded, loaded ? "Local model ready" : "No model loaded. Train a model first.");
#else
        setConnectionStatus(false, "Training support disabled");
        emit connectionTested(false, "Local inference requires DATAFORM_ENABLE_TRAINING=ON");
#endif
        return;
    }

    QString baseUrl = getProviderBaseUrl();
    QNetworkRequest request;

    if (m_currentProvider == "LM Studio") {
        request.setUrl(QUrl(baseUrl + "/v1/models"));
    } else if (m_currentProvider == "Ollama") {
        request.setUrl(QUrl(baseUrl + "/api/tags"));
    } else if (m_currentProvider == "OpenAI") {
        request.setUrl(QUrl(baseUrl + "/models"));
        request.setRawHeader("Authorization", QString("Bearer %1").arg(m_openAIKey).toUtf8());
    } else if (m_currentProvider == "Anthropic") {
        request.setUrl(QUrl(baseUrl + "/messages"));
        request.setRawHeader("x-api-key", m_anthropicKey.toUtf8());
        request.setRawHeader("anthropic-version", "2023-06-01");
    }

    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply *reply = m_networkManager->get(request);
    reply->setProperty("requestType", "testConnection");
}

void LLMProviderManager::sendPrompt(const QString &prompt)
{
    // Single-turn convenience: wraps into sendConversation
    QJsonArray messages;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    messages.append(msg);

    sendConversation(QString(), messages);
}

void LLMProviderManager::sendBackgroundRequest(const QString &owner,
                                                 const QString &systemPrompt,
                                                 const QJsonArray &messages,
                                                 BackgroundPriority priority)
{
#ifdef DATAFORM_LLAMACPP_ENABLED
    // Route to embedded llama.cpp if available (separate from chat provider)
    if (m_llamaCpp && m_llamaCpp->isModelLoaded()) {
        if (m_llamaCppBusy) {
            if (m_llamaCppQueue.size() >= MAX_BACKGROUND_QUEUE) {
                qWarning() << "LLM: llama.cpp background queue full, dropping request from" << owner;
                emit backgroundErrorOccurred(owner, "Background queue full");
                return;
            }
            BackgroundRequest req{owner, systemPrompt, messages, priority};
            bool inserted = false;
            for (int i = 0; i < m_llamaCppQueue.size(); ++i) {
                if (m_llamaCppQueue[i].priority > priority) {
                    m_llamaCppQueue.insert(i, req);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                m_llamaCppQueue.enqueue(req);
            }
            qDebug() << "LLM: queued llama.cpp background request from" << owner
                     << "priority:" << priority
                     << "(queue size:" << m_llamaCppQueue.size() << ")";
            return;
        }

        m_llamaCppBusy = true;
        m_pendingLlamaCppTag = owner;
        qDebug() << "LLM: sending background request to llama.cpp from" << owner;
        m_llamaCpp->generate(systemPrompt, messages);
        return;
    }
#endif

    if (m_backgroundBusy) {
        if (m_backgroundQueue.size() >= MAX_BACKGROUND_QUEUE) {
            qWarning() << "LLM: background queue full, dropping request from" << owner;
            emit backgroundErrorOccurred(owner, "Background queue full");
            return;
        }
        // Insert by priority (lower enum value = higher priority)
        BackgroundRequest req{owner, systemPrompt, messages, priority};
        bool inserted = false;
        for (int i = 0; i < m_backgroundQueue.size(); ++i) {
            if (m_backgroundQueue[i].priority > priority) {
                // Use QQueue's underlying QList to insert at position
                m_backgroundQueue.insert(i, req);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            m_backgroundQueue.enqueue(req);
        }
        qDebug() << "LLM: queued background request from" << owner
                 << "priority:" << priority
                 << "(queue size:" << m_backgroundQueue.size() << ")";
        return;
    }

    m_backgroundBusy = true;
    m_activeBackgroundOwner = owner;
    qDebug() << "LLM: sending background request from" << owner;
    sendConversation(systemPrompt, messages, owner);
}

void LLMProviderManager::sendBackgroundPrompt(const QString &owner, const QString &prompt,
                                                BackgroundPriority priority)
{
    QJsonArray messages;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    messages.append(msg);
    sendBackgroundRequest(owner, QString(), messages, priority);
}

void LLMProviderManager::processNextBackgroundRequest()
{
    if (m_backgroundQueue.isEmpty()) return;
    if (m_foregroundActive) return;  // Wait for foreground chat to complete

    BackgroundRequest req = m_backgroundQueue.dequeue();
    m_backgroundBusy = true;
    m_activeBackgroundOwner = req.owner;
    qDebug() << "LLM: processing queued background request from" << req.owner
             << "(remaining:" << m_backgroundQueue.size() << ")";
    sendConversation(req.systemPrompt, req.messages, req.owner);
}

void LLMProviderManager::sendTeacherRequest(const QString &owner,
                                              const QString &systemPrompt,
                                              const QJsonArray &messages)
{
    if (m_teacherBusy) {
        qWarning() << "LLM: teacher request dropped from" << owner << "- teacher busy";
        emit teacherErrorOccurred(owner, "Teacher model busy");
        return;
    }

    // The teacher MUST use the external provider, not Local/ORT
    if (m_currentProvider == "Local") {
        qWarning() << "LLM: teacher request from" << owner
                   << "rejected — current provider is Local (student). Need external provider.";
        emit teacherErrorOccurred(owner, "Cannot distill from Local provider. Switch chat provider to Ollama or another external model.");
        return;
    }

    m_teacherBusy = true;
    m_activeTeacherOwner = owner;
    qDebug() << "LLM: sending teacher request from" << owner;

    // Use TEACHER: prefix tag so onPromptReply routes to teacher signals
    sendConversation(systemPrompt, messages, QString("TEACHER:%1").arg(owner));
}

void LLMProviderManager::sendConversation(const QString &systemPrompt,
                                            const QJsonArray &messages,
                                            const QString &requestTag)
{
    qDebug() << "Sending conversation to" << m_currentProvider
             << "model:" << m_currentModel
             << "messages:" << messages.size();

    // Foreground priority: abort any in-flight background HTTP request
    // so the chat response isn't blocked behind a background task on the same server
    bool isForeground = requestTag.isEmpty();
    if (isForeground && m_currentProvider != "Local") {
        m_foregroundActive = true;
        if (m_pendingBackgroundReply) {
            qDebug() << "LLM: aborting background request from" << m_activeBackgroundOwner
                     << "to prioritize foreground chat";
            m_pendingBackgroundReply->abort();
            m_pendingBackgroundReply = nullptr;
        }
    }

    if (m_currentProvider == "Local") {
#ifdef DATAFORM_TRAINING_ENABLED
        if (!m_ortInference || !m_ortInference->isModelLoaded()) {
            if (!requestTag.isEmpty()) {
                emit backgroundErrorOccurred(requestTag, "No local model available.");
                m_backgroundBusy = false;
                m_activeBackgroundOwner.clear();
                processNextBackgroundRequest();
            } else {
                emit errorOccurred("No local model available. Train a model first or switch to another provider.");
            }
            return;
        }
        m_pendingLocalTag = requestTag;
        m_ortInference->generate(systemPrompt, messages);
        return;
#else
        if (!requestTag.isEmpty()) {
            emit backgroundErrorOccurred(requestTag, "Local inference requires training support.");
            m_backgroundBusy = false;
            m_activeBackgroundOwner.clear();
            processNextBackgroundRequest();
        } else {
            emit errorOccurred("Local inference requires training support. Rebuild with DATAFORM_ENABLE_TRAINING=ON.");
        }
        return;
#endif
    }

    if (m_currentModel.isEmpty()) {
        if (!requestTag.isEmpty()) {
            // Background request with no model — reset busy flag and notify owner
            qWarning() << "LLM: background request from" << requestTag << "failed: no model selected";
            m_backgroundBusy = false;
            m_activeBackgroundOwner.clear();
            emit backgroundErrorOccurred(requestTag, "No model selected");
            processNextBackgroundRequest();
        } else {
            emit errorOccurred("No model selected. Please select a model from the Settings panel.");
        }
        return;
    }

    QString baseUrl = getProviderBaseUrl();
    QNetworkRequest request;
    QJsonObject jsonBody;

    // Build full messages array with system prompt prepended
    QJsonArray fullMessages;
    if (!systemPrompt.isEmpty()) {
        QJsonObject sysMsg;
        sysMsg["role"] = "system";
        sysMsg["content"] = systemPrompt;
        fullMessages.append(sysMsg);
    }
    for (const QJsonValue &msg : messages) {
        fullMessages.append(msg);
    }

    // Format messages for the target provider (handles multimodal images)
    QJsonArray formattedFull = formatMessagesForProvider(m_currentProvider, fullMessages);
    QJsonArray formattedOriginal = formatMessagesForProvider(m_currentProvider, messages);

    if (m_currentProvider == "LM Studio") {
        request.setUrl(QUrl(baseUrl + "/v1/chat/completions"));
        jsonBody["model"] = m_currentModel;
        jsonBody["messages"] = formattedFull;
        jsonBody["temperature"] = 0.7;
        jsonBody["max_tokens"] = 2000;

    } else if (m_currentProvider == "Ollama") {
        // Use /api/chat for multi-turn conversation support
        request.setUrl(QUrl(baseUrl + "/api/chat"));
        jsonBody["model"] = m_currentModel;

        // Disable structured thinking for reasoning models (Qwen3+).
        // These models use Ollama's "thinking" JSON field (not inline <think> tags),
        // and will burn the entire num_predict budget on reasoning with zero visible
        // content. The "think": false API parameter disables this at the Ollama level.
        if (m_currentModel.contains("qwen3", Qt::CaseInsensitive)
            || m_currentModel.contains("deepseek", Qt::CaseInsensitive)) {
            jsonBody["think"] = false;
        }
        jsonBody["messages"] = formattedFull;
        jsonBody["stream"] = false;
        // Cap response length and penalize repetition
        QJsonObject options;
        options["num_predict"] = 2048;
        options["repeat_penalty"] = 1.15;
        jsonBody["options"] = options;

    } else if (m_currentProvider == "OpenAI") {
        request.setUrl(QUrl(baseUrl + "/chat/completions"));
        request.setRawHeader("Authorization", QString("Bearer %1").arg(m_openAIKey).toUtf8());
        jsonBody["model"] = m_currentModel;
        jsonBody["messages"] = formattedFull;
        jsonBody["temperature"] = 0.7;
        jsonBody["max_tokens"] = 2000;

    } else if (m_currentProvider == "Anthropic") {
        request.setUrl(QUrl(baseUrl + "/messages"));
        request.setRawHeader("x-api-key", m_anthropicKey.toUtf8());
        request.setRawHeader("anthropic-version", "2023-06-01");

        // Anthropic uses "system" as a top-level field, not in messages
        if (!systemPrompt.isEmpty()) {
            jsonBody["system"] = systemPrompt;
        }

        // Pass only non-system messages to Anthropic
        jsonBody["model"] = m_currentModel;
        jsonBody["messages"] = formattedOriginal;
        jsonBody["max_tokens"] = 2000;
    }

    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QJsonDocument doc(jsonBody);
    QByteArray payload = doc.toJson();

    // --- Detailed request logging ---
    if (isForeground) {
        qDebug() << "\n========== LLM REQUEST (CHAT) ==========";
        qDebug() << "Provider:" << m_currentProvider << "Model:" << m_currentModel;
        qDebug() << "URL:" << request.url().toString();

        // Log system prompt separately for readability
        if (!systemPrompt.isEmpty()) {
            qDebug() << "--- SYSTEM PROMPT (" << systemPrompt.length() << "chars) ---";
            qDebug().noquote() << systemPrompt;
            qDebug() << "--- END SYSTEM PROMPT ---";
        }

        // Log each message in the conversation
        qDebug() << "--- MESSAGES (" << messages.size() << "total) ---";
        for (int i = 0; i < messages.size(); ++i) {
            QJsonObject m = messages[i].toObject();
            QString role = m["role"].toString();
            QString content = m["content"].toString();
            qDebug().noquote() << QStringLiteral("[%1] %2: %3")
                .arg(i).arg(role, content.length() > 500 ? content.left(500) + "..." : content);
        }
        qDebug() << "--- END MESSAGES ---";

        // Log Ollama options
        if (jsonBody.contains("options")) {
            qDebug() << "Options:" << QJsonDocument(jsonBody["options"].toObject()).toJson(QJsonDocument::Compact);
        }
        qDebug() << "Total payload bytes:" << payload.size();
        qDebug() << "========================================\n";
    }

    QNetworkReply *reply = m_networkManager->post(request, payload);
    reply->setProperty("requestType", "prompt");
    if (!requestTag.isEmpty()) {
        reply->setProperty("backgroundTag", requestTag);
        // Track in-flight background reply for potential foreground preemption
        if (!requestTag.startsWith("TEACHER:")) {
            m_pendingBackgroundReply = reply;
        }
    }
}

void LLMProviderManager::onNetworkReply(QNetworkReply *reply)
{
    if (!reply) return;

    QString requestType = reply->property("requestType").toString();

    if (requestType == "testConnection") {
        onTestConnectionReply(reply);
    } else if (requestType == "refreshModels") {
        onRefreshModelsReply(reply);
    } else if (requestType == "prompt") {
        onPromptReply(reply);
    }

    reply->deleteLater();
}

void LLMProviderManager::onTestConnectionReply(QNetworkReply *reply)
{
    if (!reply) return;

    if (reply->error() == QNetworkReply::NoError) {
        setConnectionStatus(true, "Connected");
        emit connectionTested(true, "Connection successful!");
    } else {
        setConnectionStatus(false, "Connection failed");
        emit connectionTested(false, QString("Connection failed: %1").arg(reply->errorString()));
    }
}

void LLMProviderManager::onRefreshModelsReply(QNetworkReply *reply)
{
    if (!reply) return;

    QString provider = reply->property("provider").toString();

    if (reply->error() == QNetworkReply::NoError) {
        QByteArray responseData = reply->readAll();
        QJsonDocument doc = QJsonDocument::fromJson(responseData);
        QJsonObject obj = doc.object();

        QStringList models;

        if (provider == "LMStudio") {
            QJsonArray dataArray = obj["data"].toArray();
            for (const QJsonValue &value : dataArray) {
                models.append(value.toObject()["id"].toString());
            }
        } else if (provider == "Ollama") {
            QJsonArray modelsArray = obj["models"].toArray();
            for (const QJsonValue &value : modelsArray) {
                models.append(value.toObject()["name"].toString());
            }
        }

        if (!models.isEmpty()) {
            m_availableModels = models;
            emit availableModelsChanged();
            emit modelsRefreshed(true, "");
            qDebug() << "Models refreshed:" << models.size() << "found";
        } else {
            emit modelsRefreshed(false, "No models found");
        }
    } else {
        emit modelsRefreshed(false, QString("Failed to refresh: %1").arg(reply->errorString()));
    }
}

void LLMProviderManager::onPromptReply(QNetworkReply *reply)
{
    if (!reply) return;

    // Clear background reply tracking
    if (reply == m_pendingBackgroundReply) {
        m_pendingBackgroundReply = nullptr;
    }

    QString bgTag = reply->property("backgroundTag").toString();

    if (reply->error() == QNetworkReply::NoError) {
        QByteArray responseData = reply->readAll();
        QJsonDocument doc = QJsonDocument::fromJson(responseData);
        QJsonObject obj = doc.object();

        QString responseText;

        if (m_currentProvider == "LM Studio" || m_currentProvider == "OpenAI") {
            QJsonArray choices = obj["choices"].toArray();
            if (!choices.isEmpty()) {
                responseText = choices[0].toObject()["message"].toObject()["content"].toString();
            }
        } else if (m_currentProvider == "Ollama") {
            if (obj.contains("message")) {
                QJsonObject msgObj = obj["message"].toObject();
                responseText = msgObj["content"].toString();
                // Safety net: if content is empty but structured thinking field exists,
                // the model used Ollama's thinking mode and burned all tokens on reasoning.
                if (responseText.isEmpty() && msgObj.contains("thinking")) {
                    QString thinking = msgObj["thinking"].toString();
                    qWarning() << "LLM: Ollama response has empty content but"
                               << thinking.length() << "chars in 'thinking' field"
                               << "— model burned all tokens on reasoning. Consider think:false.";
                }
            } else {
                responseText = obj["response"].toString();
            }
        } else if (m_currentProvider == "Anthropic") {
            QJsonArray content = obj["content"].toArray();
            if (!content.isEmpty()) {
                responseText = content[0].toObject()["text"].toString();
            }
        }

        // --- Detailed response logging for chat requests ---
        if (bgTag.isEmpty()) {
            qDebug() << "\n========== LLM RESPONSE (CHAT) ==========";
            qDebug() << "Provider:" << m_currentProvider << "Model:" << m_currentModel;
            qDebug() << "Raw response bytes:" << responseData.size();
            qDebug() << "JSON keys:" << obj.keys();

            // Log Ollama-specific metadata
            if (m_currentProvider == "Ollama") {
                qDebug() << "done:" << obj["done"].toBool()
                         << "done_reason:" << obj["done_reason"].toString();
                qDebug() << "total_duration:" << obj["total_duration"].toDouble() / 1e9 << "s"
                         << "eval_count:" << obj["eval_count"].toInt()
                         << "eval_duration:" << obj["eval_duration"].toDouble() / 1e9 << "s"
                         << "prompt_eval_count:" << obj["prompt_eval_count"].toInt()
                         << "prompt_eval_duration:" << obj["prompt_eval_duration"].toDouble() / 1e9 << "s";
            }

            qDebug() << "Parsed response length:" << responseText.length();
            qDebug() << "--- FULL RESPONSE TEXT ---";
            qDebug().noquote() << responseText;
            qDebug() << "--- END RESPONSE TEXT ---";
            qDebug() << "==========================================\n";
        }

        // Diagnostic: if response is empty, log raw data for debugging
        if (responseText.isEmpty()) {
            qWarning() << "LLM: EMPTY response"
                       << (bgTag.isEmpty() ? "for chat" : QStringLiteral("for background %1").arg(bgTag))
                       << "provider:" << m_currentProvider
                       << "model:" << m_currentModel
                       << "raw bytes:" << responseData.size()
                       << "raw data:" << responseData.left(500);
            qWarning() << "LLM: JSON keys:" << obj.keys()
                       << "has 'message':" << obj.contains("message")
                       << "has 'error':" << obj.contains("error");
            if (obj.contains("error")) {
                responseText = QString("OLLAMA_ERROR: %1").arg(obj["error"].toString());
            } else if (responseData.size() > 0 && !bgTag.isEmpty()) {
                responseText = QString("RAW_PARSE_FAIL(%1 bytes): %2")
                                   .arg(responseData.size())
                                   .arg(QString::fromUtf8(responseData.left(500)));
            }
        }

        if (bgTag.startsWith("TEACHER:")) {
            // Teacher request (distillation) — route to teacher signals
            QString teacherOwner = bgTag.mid(8);  // strip "TEACHER:" prefix
            m_teacherBusy = false;
            m_activeTeacherOwner.clear();
            emit teacherResponseReceived(teacherOwner, responseText);
            qDebug() << "Teacher response for" << teacherOwner << ":" << responseText.left(80) << "...";
        } else if (!bgTag.isEmpty()) {
            // Background request — route to dedicated signal
            m_backgroundBusy = false;
            m_activeBackgroundOwner.clear();
            emit backgroundResponseReceived(bgTag, responseText);
            // Process next ONLY if nobody re-queued during the signal emission
            if (!m_backgroundBusy) {
                processNextBackgroundRequest();
            }
            qDebug() << "Background response for" << bgTag << ":" << responseText.left(80) << "...";
        } else {
            // Foreground response — clear priority flag so background can resume
            m_foregroundActive = false;
            emit responseReceived(responseText);
            qDebug() << "Response received:" << responseText.left(100) << "...";
        }
    } else {
        // Check for foreground-preemption abort of a background request
        if (reply->error() == QNetworkReply::OperationCanceledError
            && !bgTag.isEmpty() && !bgTag.startsWith("TEACHER:")) {
            qDebug() << "LLM: background request from" << bgTag
                     << "aborted for foreground priority";
            m_backgroundBusy = false;
            m_activeBackgroundOwner.clear();
            // Emit error so the engine can transition to Idle and emit cycleFinished.
            // The coordinator will reschedule it after the chat completes.
            emit backgroundErrorOccurred(bgTag, "Preempted by foreground chat");
            // Don't process next — foreground is active
            return;
        }

        QString errorMsg = QString("Request failed: %1").arg(reply->errorString());
        if (bgTag.startsWith("TEACHER:")) {
            QString teacherOwner = bgTag.mid(8);
            m_teacherBusy = false;
            m_activeTeacherOwner.clear();
            emit teacherErrorOccurred(teacherOwner, errorMsg);
        } else if (!bgTag.isEmpty()) {
            m_backgroundBusy = false;
            m_activeBackgroundOwner.clear();
            emit backgroundErrorOccurred(bgTag, errorMsg);
            // Process next ONLY if nobody re-queued during the signal emission
            if (!m_backgroundBusy) {
                processNextBackgroundRequest();
            }
        } else {
            m_foregroundActive = false;
            emit errorOccurred(errorMsg);
        }
    }
}

#ifdef DATAFORM_TRAINING_ENABLED
void LLMProviderManager::setOrtInferenceManager(OrtInferenceManager *mgr)
{
    m_ortInference = mgr;
    if (mgr) {
        connect(mgr, &OrtInferenceManager::generationComplete,
                this, [this](const QString &response) {
            if (!m_pendingLocalTag.isEmpty()) {
                QString tag = m_pendingLocalTag;
                m_pendingLocalTag.clear();
                m_backgroundBusy = false;
                m_activeBackgroundOwner.clear();
                emit backgroundResponseReceived(tag, response);
                // Process next ONLY if nobody re-queued during the signal emission
                if (!m_backgroundBusy) {
                    processNextBackgroundRequest();
                }
            } else {
                emit responseReceived(response);
            }
        });
        connect(mgr, &OrtInferenceManager::generationError,
                this, [this](const QString &error) {
            if (!m_pendingLocalTag.isEmpty()) {
                QString tag = m_pendingLocalTag;
                m_pendingLocalTag.clear();
                m_backgroundBusy = false;
                m_activeBackgroundOwner.clear();
                emit backgroundErrorOccurred(tag, error);
                // Process next ONLY if nobody re-queued during the signal emission
                if (!m_backgroundBusy) {
                    processNextBackgroundRequest();
                }
            } else {
                emit errorOccurred(error);
            }
        });
        connect(mgr, &OrtInferenceManager::tokenGenerated,
                this, &LLMProviderManager::tokenStreamed);
    }
}
#endif

#ifdef DATAFORM_LLAMACPP_ENABLED
void LLMProviderManager::setLlamaCppManager(LlamaCppManager *mgr)
{
    m_llamaCpp = mgr;
    if (mgr) {
        connect(mgr, &LlamaCppManager::generationComplete,
                this, [this](const QString &response) {
            if (!m_pendingLlamaCppTag.isEmpty()) {
                QString tag = m_pendingLlamaCppTag;
                m_pendingLlamaCppTag.clear();
                m_llamaCppBusy = false;
                emit backgroundResponseReceived(tag, response);
                if (!m_llamaCppBusy) {
                    processNextLlamaCppRequest();
                }
            }
        });
        connect(mgr, &LlamaCppManager::generationError,
                this, [this](const QString &error) {
            if (!m_pendingLlamaCppTag.isEmpty()) {
                QString tag = m_pendingLlamaCppTag;
                m_pendingLlamaCppTag.clear();
                m_llamaCppBusy = false;
                emit backgroundErrorOccurred(tag, error);
                if (!m_llamaCppBusy) {
                    processNextLlamaCppRequest();
                }
            }
        });
    }
}

void LLMProviderManager::processNextLlamaCppRequest()
{
    if (m_llamaCppQueue.isEmpty()) return;

    BackgroundRequest req = m_llamaCppQueue.dequeue();
    m_llamaCppBusy = true;
    m_pendingLlamaCppTag = req.owner;
    qDebug() << "LLM: processing queued llama.cpp request from" << req.owner
             << "(remaining:" << m_llamaCppQueue.size() << ")";
    m_llamaCpp->generate(req.systemPrompt, req.messages);
}
#endif

QJsonArray LLMProviderManager::formatMessagesForProvider(const QString &provider,
                                                          const QJsonArray &messages) const
{
    QJsonArray result;
    for (const QJsonValue &val : messages) {
        QJsonObject msg = val.toObject();
        QJsonArray images = msg["images"].toArray();

        // No images — pass through as-is (strip empty images key)
        if (images.isEmpty()) {
            msg.remove("images");
            result.append(msg);
            continue;
        }

        if (provider == "Ollama") {
            // Ollama natively supports "images" array on message objects
            // Pass through unchanged — already in the right format
            result.append(msg);

        } else if (provider == "OpenAI" || provider == "LM Studio") {
            // OpenAI/LM Studio: content becomes an array of content parts
            QJsonArray contentParts;

            QString text = msg["content"].toString();
            if (!text.isEmpty()) {
                QJsonObject textPart;
                textPart["type"] = "text";
                textPart["text"] = text;
                contentParts.append(textPart);
            }

            for (const QJsonValue &img : images) {
                QJsonObject imagePart;
                imagePart["type"] = "image_url";
                QJsonObject imageUrl;
                imageUrl["url"] = QString("data:image/png;base64,%1").arg(img.toString());
                imagePart["image_url"] = imageUrl;
                contentParts.append(imagePart);
            }

            msg.remove("images");
            msg["content"] = contentParts;
            result.append(msg);

        } else if (provider == "Anthropic") {
            // Anthropic: content becomes an array of content blocks
            QJsonArray contentParts;

            QString text = msg["content"].toString();
            if (!text.isEmpty()) {
                QJsonObject textPart;
                textPart["type"] = "text";
                textPart["text"] = text;
                contentParts.append(textPart);
            }

            for (const QJsonValue &img : images) {
                QJsonObject imagePart;
                imagePart["type"] = "image";
                QJsonObject source;
                source["type"] = "base64";
                source["media_type"] = "image/png";
                source["data"] = img.toString();
                imagePart["source"] = source;
                contentParts.append(imagePart);
            }

            msg.remove("images");
            msg["content"] = contentParts;
            result.append(msg);

        } else {
            // Local / unknown: strip images, keep text only
            msg.remove("images");
            result.append(msg);
        }
    }
    return result;
}

void LLMProviderManager::setConnectionStatus(bool connected, const QString &status)
{
    if (m_isConnected != connected) {
        m_isConnected = connected;
        emit isConnectedChanged();
    }
    if (m_connectionStatus != status) {
        m_connectionStatus = status;
        emit connectionStatusChanged();
    }
}

LLMProviderManager::Provider LLMProviderManager::getProviderEnum() const
{
    if (m_currentProvider == "LM Studio") return LMStudio;
    if (m_currentProvider == "Ollama") return Ollama;
    if (m_currentProvider == "OpenAI") return OpenAI;
    if (m_currentProvider == "Anthropic") return Anthropic;
    if (m_currentProvider == "Local") return Local;
    return Ollama; // Default to Ollama for DATAFORM
}

QString LLMProviderManager::getProviderBaseUrl() const
{
    if (m_currentProvider == "LM Studio") return m_lmStudioUrl;
    if (m_currentProvider == "Ollama") return m_ollamaUrl;
    if (m_currentProvider == "OpenAI") return m_openAIUrl;
    if (m_currentProvider == "Anthropic") return m_anthropicUrl;
    return m_ollamaUrl;
}
