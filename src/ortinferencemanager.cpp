#ifdef DATAFORM_TRAINING_ENABLED

#include "ortinferencemanager.h"
#include "tokenizer.h"
#include <QDebug>
#include <QThread>
#include <QMetaObject>
#include <QFileInfo>
#include <algorithm>
#include <numeric>
#include <cmath>

OrtInferenceManager::OrtInferenceManager(QObject *parent)
    : QObject(parent)
    , m_rng(std::random_device{}())
{
}

OrtInferenceManager::~OrtInferenceManager()
{
    cancelGeneration();
    unloadModel();
}

void OrtInferenceManager::setTokenizer(Tokenizer *tokenizer)
{
    m_tokenizer = tokenizer;
}

bool OrtInferenceManager::loadModel(const QString &onnxPath)
{
    if (!QFileInfo::exists(onnxPath)) {
        qWarning() << "[OrtInference] Model file not found:" << onnxPath;
        return false;
    }

    try {
        // Create environment if not yet created
        if (!m_env) {
            m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DataformInference");
        }

        // Session options - limit CPU usage for inference
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load model
        auto newSession = std::make_unique<Ort::Session>(*m_env, onnxPath.toStdWString().c_str(), opts);

        // Swap in the new session (hot-swap safe)
        m_session = std::move(newSession);
        m_modelPath = onnxPath;

        qDebug() << "[OrtInference] Model loaded:" << onnxPath;
        emit isModelLoadedChanged();
        emit modelPathChanged();
        return true;

    } catch (const Ort::Exception &e) {
        qWarning() << "[OrtInference] Failed to load model:" << e.what();
        return false;
    }
}

void OrtInferenceManager::unloadModel()
{
    m_session.reset();
    m_modelPath.clear();
    emit isModelLoadedChanged();
    emit modelPathChanged();
}

void OrtInferenceManager::generate(const QString &systemPrompt, const QJsonArray &messages,
                                    int maxNewTokens, float temperature,
                                    int topK, float topP)
{
    if (!m_session) {
        emit generationError("No model loaded");
        return;
    }

    if (!m_tokenizer || !m_tokenizer->isLoaded()) {
        emit generationError("Tokenizer not loaded");
        return;
    }

    if (m_isGenerating.loadRelaxed()) {
        emit generationError("Generation already in progress");
        return;
    }

    m_cancelRequested.storeRelaxed(0);
    m_isGenerating.storeRelaxed(1);
    emit isGeneratingChanged();

    // Build prompt tokens
    auto promptTokens = buildPromptTokens(systemPrompt, messages);

    // Run on background thread
    QThread *thread = QThread::create([this, promptTokens = std::move(promptTokens),
                                       maxNewTokens, temperature, topK, topP]() mutable {
        generationLoop(std::move(promptTokens), maxNewTokens, temperature, topK, topP);
    });

    connect(thread, &QThread::finished, thread, &QThread::deleteLater);
    thread->start();
}

void OrtInferenceManager::cancelGeneration()
{
    m_cancelRequested.storeRelaxed(1);
}

std::vector<int64_t> OrtInferenceManager::buildPromptTokens(const QString &systemPrompt,
                                                              const QJsonArray &messages)
{
    // Build ChatML-formatted prompt string
    // <|im_start|>system\n{system}<|im_end|>\n
    // <|im_start|>user\n{user}<|im_end|>\n
    // <|im_start|>assistant\n

    QString prompt;

    // System prompt
    if (!systemPrompt.isEmpty()) {
        prompt += "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n";
    }

    // Conversation messages
    for (const QJsonValue &val : messages) {
        QJsonObject msg = val.toObject();
        QString role = msg["role"].toString();
        QString content = msg["content"].toString();
        prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }

    // Start assistant response
    prompt += "<|im_start|>assistant\n";

    auto tokens = m_tokenizer->encode(prompt);

    // Truncate if too long (keep end of context, most relevant)
    int maxPromptLen = MAX_CONTEXT_LENGTH / 2;  // Reserve half for generation
    if (static_cast<int>(tokens.size()) > maxPromptLen) {
        tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - maxPromptLen));
    }

    return tokens;
}

int64_t OrtInferenceManager::sampleToken(const float *logits, int vocabSize,
                                          float temperature, int topK, float topP)
{
    // Create index-probability pairs
    std::vector<std::pair<float, int64_t>> logitPairs(vocabSize);
    for (int i = 0; i < vocabSize; ++i) {
        logitPairs[i] = {logits[i] / temperature, static_cast<int64_t>(i)};
    }

    // Sort by logit descending
    std::partial_sort(logitPairs.begin(),
                      logitPairs.begin() + std::min(topK, vocabSize),
                      logitPairs.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

    // Apply top-k: only keep top-k entries
    int effectiveK = std::min(topK, vocabSize);
    logitPairs.resize(effectiveK);

    // Softmax over remaining logits
    float maxLogit = logitPairs[0].first;
    float sumExp = 0.0f;
    for (auto &[logit, _] : logitPairs) {
        logit = std::exp(logit - maxLogit);
        sumExp += logit;
    }
    for (auto &[prob, _] : logitPairs) {
        prob /= sumExp;
    }

    // Apply top-p (nucleus sampling): accumulate until cumulative prob >= topP
    float cumProb = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < static_cast<int>(logitPairs.size()); ++i) {
        cumProb += logitPairs[i].first;
        cutoff = i + 1;
        if (cumProb >= topP) break;
    }
    logitPairs.resize(cutoff);

    // Re-normalize after top-p cutoff
    float newSum = 0.0f;
    for (const auto &[prob, _] : logitPairs) {
        newSum += prob;
    }
    for (auto &[prob, _] : logitPairs) {
        prob /= newSum;
    }

    // Sample from distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(m_rng);
    float acc = 0.0f;
    for (const auto &[prob, tokenId] : logitPairs) {
        acc += prob;
        if (r <= acc) return tokenId;
    }

    // Fallback: return highest probability token
    return logitPairs[0].second;
}

void OrtInferenceManager::generationLoop(std::vector<int64_t> promptTokens,
                                          int maxNewTokens, float temperature,
                                          int topK, float topP)
{
    QString generatedText;
    std::vector<int64_t> currentTokens = std::move(promptTokens);
    int eosId = m_tokenizer->eosTokenId();
    int imEndId = m_tokenizer->imEndId();

    try {
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input/output names
        size_t numInputs = m_session->GetInputCount();
        size_t numOutputs = m_session->GetOutputCount();

        std::vector<std::string> inputNames;
        std::vector<const char*> inputNamePtrs;
        for (size_t i = 0; i < numInputs; ++i) {
            auto name = m_session->GetInputNameAllocated(i, allocator);
            inputNames.push_back(name.get());
        }
        for (auto &n : inputNames) inputNamePtrs.push_back(n.c_str());

        std::vector<std::string> outputNames;
        std::vector<const char*> outputNamePtrs;
        for (size_t i = 0; i < numOutputs; ++i) {
            auto name = m_session->GetOutputNameAllocated(i, allocator);
            outputNames.push_back(name.get());
        }
        for (auto &n : outputNames) outputNamePtrs.push_back(n.c_str());

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (int step = 0; step < maxNewTokens; ++step) {
            if (m_cancelRequested.loadRelaxed()) {
                qDebug() << "[OrtInference] Generation cancelled at step" << step;
                break;
            }

            // Truncate to max context length if needed
            if (static_cast<int>(currentTokens.size()) > MAX_CONTEXT_LENGTH) {
                currentTokens.erase(currentTokens.begin(),
                                     currentTokens.begin() +
                                     (currentTokens.size() - MAX_CONTEXT_LENGTH));
            }

            int64_t seqLen = static_cast<int64_t>(currentTokens.size());
            std::vector<int64_t> shape = {1, seqLen};

            // Create attention mask (all 1s)
            std::vector<int64_t> attentionMask(seqLen, 1);

            // Create input tensors
            std::vector<Ort::Value> inputTensors;
            inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, currentTokens.data(), currentTokens.size(),
                shape.data(), shape.size()));
            inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, attentionMask.data(), attentionMask.size(),
                shape.data(), shape.size()));

            // Run inference
            auto outputs = m_session->Run(
                Ort::RunOptions{nullptr},
                inputNamePtrs.data(), inputTensors.data(), inputTensors.size(),
                outputNamePtrs.data(), outputNamePtrs.size());

            // Get logits: shape [1, seq_len, vocab_size]
            auto &logitsTensor = outputs[0];
            auto logitsShape = logitsTensor.GetTensorTypeAndShapeInfo().GetShape();
            int vocabSize = static_cast<int>(logitsShape[2]);

            // Get logits for the last token position
            const float *allLogits = logitsTensor.GetTensorData<float>();
            const float *lastLogits = allLogits + (seqLen - 1) * vocabSize;

            // Sample next token
            int64_t nextToken = sampleToken(lastLogits, vocabSize, temperature, topK, topP);

            // Check for stop conditions
            if (nextToken == eosId || nextToken == imEndId) {
                qDebug() << "[OrtInference] EOS reached at step" << step;
                break;
            }

            // Append token to sequence
            currentTokens.push_back(nextToken);

            // Decode and emit token
            std::vector<int64_t> singleToken = {nextToken};
            QString decoded = m_tokenizer->decode(singleToken);
            generatedText += decoded;

            QMetaObject::invokeMethod(this, [this, decoded]() {
                emit tokenGenerated(decoded);
            }, Qt::QueuedConnection);
        }

        // Emit completion on main thread
        QMetaObject::invokeMethod(this, [this, generatedText]() {
            m_isGenerating.storeRelaxed(0);
            emit isGeneratingChanged();
            emit generationComplete(generatedText);
        }, Qt::QueuedConnection);

    } catch (const Ort::Exception &e) {
        QString error = QString("ORT inference error: %1").arg(e.what());
        qWarning() << "[OrtInference]" << error;
        QMetaObject::invokeMethod(this, [this, error]() {
            m_isGenerating.storeRelaxed(0);
            emit isGeneratingChanged();
            emit generationError(error);
        }, Qt::QueuedConnection);
    }
}

#endif // DATAFORM_TRAINING_ENABLED
