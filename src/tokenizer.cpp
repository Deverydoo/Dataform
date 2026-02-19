#ifdef DATAFORM_TRAINING_ENABLED

#include "tokenizer.h"
#include "modelgeneration.h"
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>
#include <QDebug>
#include <QRegularExpression>
#include <algorithm>
#include <set>

Tokenizer::Tokenizer(QObject *parent)
    : QObject(parent)
{
}

Tokenizer::~Tokenizer() = default;

void Tokenizer::configure(const ModelGenerationConfig &config)
{
    m_imStartTokenName = config.imStartToken;
    m_imEndTokenName = config.imEndToken;
    m_eosTokenName = config.eosToken;
    m_systemRole = config.systemRole;
    m_userRole = config.userRole;
    m_assistantRole = config.assistantRole;
    qDebug() << "Tokenizer: configured for" << config.chatTemplate
             << "template (generation" << config.generationId << ")";
}

bool Tokenizer::loadFromFiles(const QString &vocabJsonPath, const QString &mergesPath)
{
    // --- Load vocab.json ---
    QFile vocabFile(vocabJsonPath);
    if (!vocabFile.open(QIODevice::ReadOnly)) {
        qWarning() << "Tokenizer: cannot open vocab file:" << vocabJsonPath;
        return false;
    }

    QJsonDocument vocabDoc = QJsonDocument::fromJson(vocabFile.readAll());
    vocabFile.close();

    if (!vocabDoc.isObject()) {
        qWarning() << "Tokenizer: vocab.json is not a JSON object";
        return false;
    }

    QJsonObject vocabObj = vocabDoc.object();
    for (auto it = vocabObj.begin(); it != vocabObj.end(); ++it) {
        int64_t id = static_cast<int64_t>(it.value().toInt());
        m_tokenToId[it.key()] = id;
        m_idToToken[id] = it.key();
    }
    m_vocabSize = m_tokenToId.size();

    // --- Load merges.txt ---
    QFile mergesFile(mergesPath);
    if (!mergesFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Tokenizer: cannot open merges file:" << mergesPath;
        return false;
    }

    QTextStream stream(&mergesFile);
    int rank = 0;
    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith('#'))
            continue;

        int spaceIdx = line.indexOf(' ');
        if (spaceIdx <= 0) continue;

        QString first = line.left(spaceIdx);
        QString second = line.mid(spaceIdx + 1);
        TokenPair pair(first, second);
        m_merges.append(pair);
        m_mergeRanks[pair] = rank++;
    }
    mergesFile.close();

    // --- Resolve special token IDs using configured token names ---
    auto findToken = [this](const QString &token) -> int64_t {
        auto it = m_tokenToId.find(token);
        return (it != m_tokenToId.end()) ? it.value() : -1;
    };

    m_imStartId = findToken(m_imStartTokenName);
    m_imEndId = findToken(m_imEndTokenName);
    m_eosId = findToken(m_eosTokenName);
    if (m_eosId < 0) m_eosId = findToken("</s>");
    if (m_eosId < 0 && m_imEndId >= 0) m_eosId = m_imEndId;
    m_nlId = findToken("\n");

    // Pad token: use a dedicated one or fall back to eos
    m_padId = findToken("<|padding|>");
    if (m_padId < 0) m_padId = findToken("<pad>");
    if (m_padId < 0) m_padId = 0;

    m_loaded = true;
    emit isLoadedChanged();

    qDebug() << "Tokenizer loaded: vocab=" << m_vocabSize
             << "merges=" << m_merges.size()
             << "imStart=" << m_imStartId
             << "imEnd=" << m_imEndId
             << "eos=" << m_eosId;

    return true;
}

std::vector<QString> Tokenizer::preTokenize(const QString &text) const
{
    // Simple whitespace-aware pre-tokenization
    // Split on whitespace boundaries, keeping leading spaces as part of tokens
    std::vector<QString> words;
    if (text.isEmpty()) return words;

    // Regex-based split similar to GPT-2/Qwen pattern
    // Splits on word boundaries, numbers, punctuation
    static QRegularExpression re(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+)",
        QRegularExpression::CaseInsensitiveOption
    );

    auto matches = re.globalMatch(text);
    while (matches.hasNext()) {
        auto match = matches.next();
        words.push_back(match.captured());
    }

    return words;
}

std::vector<QString> Tokenizer::bpeTokenize(const QString &word) const
{
    if (word.isEmpty()) return {};

    // Start with individual characters
    std::vector<QString> symbols;
    for (int i = 0; i < word.size(); ++i) {
        symbols.push_back(QString(word[i]));
    }

    if (symbols.size() <= 1) return symbols;

    // Iteratively merge the highest-priority pair
    while (symbols.size() > 1) {
        int bestRank = INT_MAX;
        int bestIdx = -1;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            TokenPair pair(symbols[i], symbols[i + 1]);
            auto it = m_mergeRanks.find(pair);
            if (it != m_mergeRanks.end() && it.value() < bestRank) {
                bestRank = it.value();
                bestIdx = static_cast<int>(i);
            }
        }

        if (bestIdx < 0) break;  // No more merges possible

        // Merge the pair at bestIdx
        symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
        symbols.erase(symbols.begin() + bestIdx + 1);
    }

    return symbols;
}

std::vector<int64_t> Tokenizer::textToTokenIds(const QString &text) const
{
    std::vector<int64_t> ids;

    auto words = preTokenize(text);
    for (const QString &word : words) {
        auto tokens = bpeTokenize(word);
        for (const QString &token : tokens) {
            auto it = m_tokenToId.find(token);
            if (it != m_tokenToId.end()) {
                ids.push_back(it.value());
            } else {
                // Unknown token - encode byte-by-byte as fallback
                QByteArray utf8 = token.toUtf8();
                for (char c : utf8) {
                    QString byteToken = QString("<%1>").arg(
                        static_cast<unsigned char>(c), 2, 16, QChar('0'));
                    auto byteIt = m_tokenToId.find(byteToken);
                    if (byteIt != m_tokenToId.end()) {
                        ids.push_back(byteIt.value());
                    }
                    // If even byte fallback fails, skip the character
                }
            }
        }
    }

    return ids;
}

std::vector<int64_t> Tokenizer::encode(const QString &text) const
{
    if (!m_loaded) return {};
    return textToTokenIds(text);
}

QString Tokenizer::decode(const std::vector<int64_t> &tokenIds) const
{
    if (!m_loaded) return {};

    QString result;
    for (int64_t id : tokenIds) {
        if (id == LABEL_IGNORE || id == m_padId) continue;
        auto it = m_idToToken.find(id);
        if (it != m_idToToken.end()) {
            result += it.value();
        }
    }
    return result;
}

std::vector<int64_t> Tokenizer::encodeFixed(const QString &text, int maxLength) const
{
    auto ids = encode(text);

    // Truncate if needed
    if (static_cast<int>(ids.size()) > maxLength) {
        ids.resize(maxLength);
    }

    // Pad if needed
    while (static_cast<int>(ids.size()) < maxLength) {
        ids.push_back(m_padId);
    }

    return ids;
}

Tokenizer::ChatMLResult Tokenizer::encodeChat(
    const QString &systemPrompt,
    const QString &userText,
    const QString &assistantText,
    int maxLength) const
{
    ChatMLResult result;
    if (!m_loaded) return result;

    // Build ChatML sequence:
    // <|im_start|>system\n{system}<|im_end|>\n
    // <|im_start|>user\n{user}<|im_end|>\n
    // <|im_start|>assistant\n{assistant}<|im_end|>

    // Encode each part
    auto sysTokens = textToTokenIds(systemPrompt);
    auto usrTokens = textToTokenIds(userText);
    auto asstTokens = textToTokenIds(assistantText);

    // Build the full sequence with special tokens
    std::vector<int64_t> inputPart;   // Everything before assistant response
    std::vector<int64_t> targetPart;  // Assistant response

    // System turn
    if (m_imStartId >= 0) inputPart.push_back(m_imStartId);
    auto sysTag = textToTokenIds(m_systemRole);
    inputPart.insert(inputPart.end(), sysTag.begin(), sysTag.end());
    if (m_nlId >= 0) inputPart.push_back(m_nlId);
    inputPart.insert(inputPart.end(), sysTokens.begin(), sysTokens.end());
    if (m_imEndId >= 0) inputPart.push_back(m_imEndId);
    if (m_nlId >= 0) inputPart.push_back(m_nlId);

    // User turn
    if (m_imStartId >= 0) inputPart.push_back(m_imStartId);
    auto usrTag = textToTokenIds(m_userRole);
    inputPart.insert(inputPart.end(), usrTag.begin(), usrTag.end());
    if (m_nlId >= 0) inputPart.push_back(m_nlId);
    inputPart.insert(inputPart.end(), usrTokens.begin(), usrTokens.end());
    if (m_imEndId >= 0) inputPart.push_back(m_imEndId);
    if (m_nlId >= 0) inputPart.push_back(m_nlId);

    // Assistant turn prefix (part of input, not target)
    if (m_imStartId >= 0) inputPart.push_back(m_imStartId);
    auto asstTag = textToTokenIds(m_assistantRole);
    inputPart.insert(inputPart.end(), asstTag.begin(), asstTag.end());
    if (m_nlId >= 0) inputPart.push_back(m_nlId);

    // Assistant response (this is the target)
    targetPart = asstTokens;
    if (m_imEndId >= 0) targetPart.push_back(m_imEndId);

    // Combine into full sequence
    int inputLen = static_cast<int>(inputPart.size());
    int targetLen = static_cast<int>(targetPart.size());
    int totalLen = inputLen + targetLen;

    // Truncate if needed (preserve some input context)
    if (totalLen > maxLength) {
        int maxTarget = maxLength / 2;
        if (targetLen > maxTarget) {
            targetPart.resize(maxTarget);
            targetLen = maxTarget;
        }
        int maxInput = maxLength - targetLen;
        if (inputLen > maxInput) {
            // Keep the end of input (closer to assistant response)
            inputPart.erase(inputPart.begin(),
                           inputPart.begin() + (inputLen - maxInput));
            inputLen = maxInput;
        }
        totalLen = inputLen + targetLen;
    }

    // Build final arrays
    result.inputIds.reserve(maxLength);
    result.labels.reserve(maxLength);
    result.attentionMask.reserve(maxLength);

    // Input portion: labels are LABEL_IGNORE (don't compute loss here)
    for (int i = 0; i < inputLen; ++i) {
        result.inputIds.push_back(inputPart[i]);
        result.labels.push_back(LABEL_IGNORE);
        result.attentionMask.push_back(1);
    }

    // Target portion: labels are the actual token IDs
    for (int i = 0; i < targetLen; ++i) {
        result.inputIds.push_back(targetPart[i]);
        result.labels.push_back(targetPart[i]);
        result.attentionMask.push_back(1);
    }

    // Pad to maxLength
    while (static_cast<int>(result.inputIds.size()) < maxLength) {
        result.inputIds.push_back(m_padId);
        result.labels.push_back(LABEL_IGNORE);
        result.attentionMask.push_back(0);
    }

    return result;
}

#endif // DATAFORM_TRAINING_ENABLED
