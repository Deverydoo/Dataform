#ifdef DATAFORM_LLAMACPP_ENABLED

#include "llamacppmanager.h"
#include "llmresponseparser.h"
#include <QDebug>
#include <QThread>
#include <QMetaObject>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <llama.h>
#include <string>
#include <vector>

LlamaCppManager::LlamaCppManager(QObject *parent)
    : QObject(parent)
{
}

LlamaCppManager::~LlamaCppManager()
{
    cancelGeneration();
    // Wait briefly for generation thread to finish
    int tries = 0;
    while (m_isGenerating.loadRelaxed() && tries++ < 50) {
        QThread::msleep(100);
    }
    unloadModel();
}

bool LlamaCppManager::loadModel(const QString &ggufPath, int nCtx, int nThreads)
{
    if (m_isGenerating.loadRelaxed()) {
        qWarning() << "LlamaCpp: cannot load model while generating";
        return false;
    }

    unloadModel();

    std::string path = ggufPath.toStdString();
    qDebug() << "LlamaCpp: loading model from" << ggufPath;

    // Model params — use mmap for efficient loading
    llama_model_params mparams = llama_model_default_params();
    auto *newModel = llama_model_load_from_file(path.c_str(), mparams);
    if (!newModel) {
        qWarning() << "LlamaCpp: failed to load model from" << ggufPath;
        return false;
    }

    // Context params
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx;
    cparams.n_threads = nThreads;
    cparams.n_threads_batch = nThreads;

    auto *newCtx = llama_init_from_model(newModel, cparams);
    if (!newCtx) {
        qWarning() << "LlamaCpp: failed to create context";
        llama_model_free(newModel);
        return false;
    }

    // Write-lock protects against background thread reading during swap
    {
        QWriteLocker lock(m_sessionGuard.rwLock());
        m_model = newModel;
        m_ctx = newCtx;
    }

    m_nCtx = nCtx;
    m_modelPath = ggufPath;
    emit modelPathChanged();
    emit isModelLoadedChanged();

    qDebug() << "LlamaCpp: model loaded successfully, ctx=" << nCtx << "threads=" << nThreads;
    return true;
}

void LlamaCppManager::unloadModel()
{
    // Write-lock ensures background thread finishes before we destroy
    QWriteLocker lock(m_sessionGuard.rwLock());
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model) {
        llama_model_free(m_model);
        m_model = nullptr;
    }
    if (!m_modelPath.isEmpty()) {
        m_modelPath.clear();
        emit modelPathChanged();
        emit isModelLoadedChanged();
    }
}

QStringList LlamaCppManager::availableModels() const
{
    QString dir = QCoreApplication::applicationDirPath() + "/models/background_llm";
    QDir d(dir);
    if (!d.exists()) return {};
    QStringList ggufFiles = d.entryList({"*.gguf"}, QDir::Files, QDir::Name);
    return ggufFiles;
}

QString LlamaCppManager::loadedModelName() const
{
    if (m_modelPath.isEmpty()) return {};
    return QFileInfo(m_modelPath).fileName();
}

std::string LlamaCppManager::buildChatPrompt(const QString &systemPrompt, const QJsonArray &messages)
{
    // Build a vector of chat messages for llama_chat_apply_template
    struct ChatMsg {
        std::string role;
        std::string content;
    };
    std::vector<ChatMsg> chatMsgs;

    // System prompt
    if (!systemPrompt.isEmpty()) {
        chatMsgs.push_back({"system", systemPrompt.toStdString()});
    }

    // User/assistant messages
    for (const QJsonValue &val : messages) {
        QJsonObject msg = val.toObject();
        std::string role = msg["role"].toString().toStdString();
        std::string content = msg["content"].toString().toStdString();
        chatMsgs.push_back({role, content});
    }

    // Convert to llama_chat_message array
    std::vector<llama_chat_message> llamaMsgs;
    llamaMsgs.reserve(chatMsgs.size());
    for (const auto &cm : chatMsgs) {
        llama_chat_message m;
        m.role = cm.role.c_str();
        m.content = cm.content.c_str();
        llamaMsgs.push_back(m);
    }

    // Apply chat template — pass nullptr for tmpl to use model's built-in template
    const char * tmpl = llama_model_chat_template(m_model, nullptr);

    // First call to get required buffer size
    std::vector<char> buf(4096);
    int32_t res = llama_chat_apply_template(
        tmpl, llamaMsgs.data(), llamaMsgs.size(),
        true, // add_ass = true (add assistant prompt prefix)
        buf.data(), buf.size()
    );

    if (res > static_cast<int32_t>(buf.size())) {
        buf.resize(res + 1);
        res = llama_chat_apply_template(
            tmpl, llamaMsgs.data(), llamaMsgs.size(),
            true, buf.data(), buf.size()
        );
    }

    if (res < 0) {
        // Fallback: build ChatML manually
        qWarning() << "LlamaCpp: chat template failed, using ChatML fallback";
        std::string prompt;
        if (!systemPrompt.isEmpty()) {
            prompt += "<|im_start|>system\n" + systemPrompt.toStdString() + "<|im_end|>\n";
        }
        for (const auto &cm : chatMsgs) {
            if (cm.role == "system") continue; // already added
            prompt += "<|im_start|>" + cm.role + "\n" + cm.content + "<|im_end|>\n";
        }
        prompt += "<|im_start|>assistant\n";
        return prompt;
    }

    return std::string(buf.data(), res);
}

void LlamaCppManager::generate(const QString &systemPrompt, const QJsonArray &messages,
                                int maxNewTokens, float temperature)
{
    if (!m_model || !m_ctx) {
        emit generationError("No model loaded");
        return;
    }
    if (m_isGenerating.loadRelaxed()) {
        emit generationError("Generation already in progress");
        return;
    }

    m_isGenerating.storeRelaxed(1);
    m_cancelRequested.storeRelaxed(0);
    emit isGeneratingChanged();

    std::string prompt = buildChatPrompt(systemPrompt, messages);

    // Launch generation on background thread
    QThread *thread = QThread::create([this, prompt, maxNewTokens, temperature]() {
        generationLoop(prompt, maxNewTokens, temperature);
    });
    connect(thread, &QThread::finished, thread, &QThread::deleteLater);
    thread->start();
}

void LlamaCppManager::cancelGeneration()
{
    m_cancelRequested.storeRelaxed(1);
}

void LlamaCppManager::generationLoop(std::string prompt, int maxNewTokens, float temperature)
{
    // Read-lock prevents model unload during generation
    QReadLocker sessionLock(m_sessionGuard.rwLock());
    if (!m_model || !m_ctx) {
        QMetaObject::invokeMethod(this, [this]() {
            m_isGenerating.storeRelaxed(0);
            emit isGeneratingChanged();
            emit generationError("Model unloaded during generation");
        }, Qt::QueuedConnection);
        return;
    }

    try {
        const llama_vocab *vocab = llama_model_get_vocab(m_model);

        // Tokenize the prompt
        int n_prompt_max = prompt.size() + 256;
        std::vector<llama_token> promptTokens(n_prompt_max);
        int n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                       promptTokens.data(), n_prompt_max,
                                       true,  // add_special (BOS if model expects it)
                                       true); // parse_special (handle special tokens)
        if (n_prompt < 0) {
            // Buffer too small, resize and retry
            promptTokens.resize(-n_prompt);
            n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                       promptTokens.data(), promptTokens.size(),
                                       true, true);
        }
        if (n_prompt < 0) {
            QMetaObject::invokeMethod(this, [this]() {
                m_isGenerating.storeRelaxed(0);
                emit isGeneratingChanged();
                emit generationError("Failed to tokenize prompt");
            }, Qt::QueuedConnection);
            return;
        }
        promptTokens.resize(n_prompt);

        // Check if prompt fits in context
        if (n_prompt >= m_nCtx) {
            // Truncate from the beginning, keep the end (most recent context)
            int keep = m_nCtx - maxNewTokens - 64; // leave room for generation
            if (keep < 64) keep = 64;
            promptTokens.erase(promptTokens.begin(),
                               promptTokens.begin() + (n_prompt - keep));
            n_prompt = promptTokens.size();
            qDebug() << "LlamaCpp: prompt truncated to" << n_prompt << "tokens";
        }

        // Clear KV cache for fresh generation
        llama_memory_clear(llama_get_memory(m_ctx), true);

        // Set up sampler chain
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        llama_sampler *smpl = llama_sampler_chain_init(sparams);

        // For background JSON tasks, use low temperature + greedy-ish sampling
        if (temperature < 0.01f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(0)); // seed=0 for random
        }

        // Decode prompt in one batch
        llama_batch batch = llama_batch_get_one(promptTokens.data(), n_prompt);
        if (llama_decode(m_ctx, batch) != 0) {
            llama_sampler_free(smpl);
            QMetaObject::invokeMethod(this, [this]() {
                m_isGenerating.storeRelaxed(0);
                emit isGeneratingChanged();
                emit generationError("Failed to decode prompt");
            }, Qt::QueuedConnection);
            return;
        }

        // Autoregressive generation loop
        std::string fullText;
        char tokenBuf[256];

        for (int i = 0; i < maxNewTokens; ++i) {
            if (m_cancelRequested.loadRelaxed()) {
                qDebug() << "LlamaCpp: generation cancelled";
                break;
            }

            // Sample next token
            llama_token newToken = llama_sampler_sample(smpl, m_ctx, -1);

            // Check for end of generation
            if (llama_vocab_is_eog(vocab, newToken)) {
                break;
            }

            // Accept the token in the sampler
            llama_sampler_accept(smpl, newToken);

            // Decode token to text
            int n = llama_token_to_piece(vocab, newToken, tokenBuf, sizeof(tokenBuf), 0, true);
            if (n > 0) {
                std::string piece(tokenBuf, n);
                fullText += piece;

                // Emit token on main thread
                QString qPiece = QString::fromUtf8(piece.c_str(), piece.size());
                QMetaObject::invokeMethod(this, [this, qPiece]() {
                    emit tokenGenerated(qPiece);
                }, Qt::QueuedConnection);
            }

            // Prepare batch for next token
            llama_batch nextBatch = llama_batch_get_one(&newToken, 1);
            if (llama_decode(m_ctx, nextBatch) != 0) {
                qWarning() << "LlamaCpp: decode failed at token" << i;
                break;
            }
        }

        llama_sampler_free(smpl);

        // Strip <think>...</think> blocks from response (for qwen3 and similar models)
        QString result = LLMResponseParser::stripThinkTags(
            QString::fromUtf8(fullText.c_str(), fullText.size()));

        QMetaObject::invokeMethod(this, [this, result]() {
            m_isGenerating.storeRelaxed(0);
            emit isGeneratingChanged();
            emit generationComplete(result);
            qDebug() << "LlamaCpp: generation complete," << result.size() << "chars";
        }, Qt::QueuedConnection);

    } catch (const std::exception &e) {
        QString error = QString("LlamaCpp exception: %1").arg(e.what());
        QMetaObject::invokeMethod(this, [this, error]() {
            m_isGenerating.storeRelaxed(0);
            emit isGeneratingChanged();
            emit generationError(error);
        }, Qt::QueuedConnection);
    }
}

#endif // DATAFORM_LLAMACPP_ENABLED
