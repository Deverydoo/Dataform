#ifndef TOKENIZER_H
#define TOKENIZER_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>
#include <QHash>
#include <QPair>
#include <vector>
#include <string>

struct ModelGenerationConfig;

class Tokenizer : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isLoaded READ isLoaded NOTIFY isLoadedChanged)
    Q_PROPERTY(int vocabSize READ vocabSize NOTIFY isLoadedChanged)

public:
    explicit Tokenizer(QObject *parent = nullptr);
    ~Tokenizer();

    void configure(const ModelGenerationConfig &config);
    bool loadFromFiles(const QString &vocabJsonPath, const QString &mergesPath);
    bool isLoaded() const { return m_loaded; }
    int vocabSize() const { return m_vocabSize; }

    // Core operations
    std::vector<int64_t> encode(const QString &text) const;
    QString decode(const std::vector<int64_t> &tokenIds) const;

    // Encode with padding/truncation to fixed length
    std::vector<int64_t> encodeFixed(const QString &text, int maxLength) const;

    // Encode in ChatML format (Qwen2.5):
    // <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{usr}<|im_end|>\n<|im_start|>assistant\n{asst}<|im_end|>
    struct ChatMLResult {
        std::vector<int64_t> inputIds;
        std::vector<int64_t> labels;       // -100 for input portion, real IDs for assistant portion
        std::vector<int64_t> attentionMask;
    };
    ChatMLResult encodeChat(const QString &systemPrompt,
                            const QString &userText,
                            const QString &assistantText,
                            int maxLength = 512) const;

    // Special token IDs
    int eosTokenId() const { return m_eosId; }
    int padTokenId() const { return m_padId; }
    int imStartId() const { return m_imStartId; }
    int imEndId() const { return m_imEndId; }

signals:
    void isLoadedChanged();

private:
    // BPE merge operation
    using TokenPair = QPair<QString, QString>;

    std::vector<QString> bpeTokenize(const QString &word) const;
    std::vector<int64_t> textToTokenIds(const QString &text) const;
    std::vector<QString> preTokenize(const QString &text) const;

    // Vocabulary: token string <-> ID
    QHash<QString, int64_t> m_tokenToId;
    QHash<int64_t, QString> m_idToToken;

    // BPE merges in priority order
    QList<TokenPair> m_merges;
    QHash<TokenPair, int> m_mergeRanks;

    bool m_loaded = false;
    int m_vocabSize = 0;

    // Special token IDs
    int64_t m_eosId = -1;
    int64_t m_padId = 0;
    int64_t m_imStartId = -1;
    int64_t m_imEndId = -1;
    int64_t m_nlId = -1;  // newline token

    // Configurable token/role names (defaults match Qwen2.5 ChatML)
    QString m_imStartTokenName = "<|im_start|>";
    QString m_imEndTokenName = "<|im_end|>";
    QString m_eosTokenName = "<|endoftext|>";
    QString m_systemRole = "system";
    QString m_userRole = "user";
    QString m_assistantRole = "assistant";

    static constexpr int64_t LABEL_IGNORE = -100;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // TOKENIZER_H
