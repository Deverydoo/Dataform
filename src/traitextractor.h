#ifndef TRAITEXTRACTOR_H
#define TRAITEXTRACTOR_H

#include <QObject>
#include <QString>
#include <QList>
#include <QSet>
#include <QJsonArray>
#include <QTimer>
#include <QDateTime>

class LLMProviderManager;
class MemoryStore;
struct EpisodicRecord;

struct ExtractedTrait {
    QString type;       // "value", "preference", "policy", "motivation"
    QString statement;  // Human-readable trait statement
    double confidence;  // 0.0 - 1.0
};

class TraitExtractor : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isExtracting READ isExtracting NOTIFY isExtractingChanged)
    Q_PROPERTY(int extractionCount READ extractionCount NOTIFY extractionCountChanged)
    Q_PROPERTY(QString lastStatus READ lastStatus NOTIFY lastStatusChanged)
    Q_PROPERTY(QString lastResponse READ lastResponse NOTIFY lastResponseChanged)
    Q_PROPERTY(double leanScore READ leanScore NOTIFY leanScoreChanged)
    Q_PROPERTY(QString leanLabel READ leanLabel NOTIFY leanLabelChanged)
    Q_PROPERTY(QString leanAnalyzedTs READ leanAnalyzedTs NOTIFY leanAnalyzedTsChanged)
    Q_PROPERTY(bool isAnalyzingLean READ isAnalyzingLean NOTIFY isAnalyzingLeanChanged)

public:
    explicit TraitExtractor(QObject *parent = nullptr);
    ~TraitExtractor();

    void setLLMProvider(LLMProviderManager *provider);
    void setMemoryStore(MemoryStore *store);

    bool isExtracting() const { return m_isExtracting; }
    int extractionCount() const { return m_extractionCount; }
    QString lastStatus() const { return m_lastStatus; }
    QString lastResponse() const { return m_lastResponse; }

    // Political lean analysis
    double leanScore() const { return m_leanScore; }
    QString leanLabel() const;
    QString leanAnalyzedTs() const { return m_leanAnalyzedTs; }
    bool isAnalyzingLean() const { return m_isAnalyzingLean; }
    Q_INVOKABLE void analyzePoliticalLean();

    // Extract traits from a specific conversation exchange
    Q_INVOKABLE void extractFromEpisode(qint64 episodeId);

    // Extract traits from recent high-signal episodes (batch mode for idle time)
    Q_INVOKABLE void extractFromRecentEpisodes(int limit = 10);

    // Extract traits from a why-engine inquiry response
    void extractFromInquiryResponse(qint64 episodeId,
                                     const QString &inquiry,
                                     const QString &response);

    // Idle-time conversation scanning
    void onIdleWindowOpened();
    void onIdleWindowClosed();

    // Manual scan trigger from QML
    Q_INVOKABLE void scanConversationsForTraits();

signals:
    void isExtractingChanged();
    void extractionCountChanged();
    void lastStatusChanged();
    void lastResponseChanged();
    void traitsExtracted(int count);
    void extractionError(const QString &error);
    void leanScoreChanged();
    void leanLabelChanged();
    void leanAnalyzedTsChanged();
    void isAnalyzingLeanChanged();
    void leanAnalysisComplete(double score);

private:
    void onExtractionResponse(const QString &response);
    void onExtractionError(const QString &error);
    void onLeanAnalysisResponse(const QString &response);

    QString buildExtractionPrompt(const QString &userText,
                                   const QString &assistantText,
                                   const QString &inquiryText = QString()) const;
    QString buildConversationExtractionPrompt(qint64 conversationId) const;
    QString buildEpisodeBatchPrompt(const QList<EpisodicRecord> &episodes) const;
    QList<ExtractedTrait> parseTraitsFromResponse(const QString &response) const;
    void storeExtractedTraits(const QList<ExtractedTrait> &traits, qint64 episodeId);
    double traitSimilarity(const QString &a, const QString &b) const;
    bool isDuplicateTrait(const QString &statement) const;
    void mergeWithExistingTrait(const QString &existingTraitId,
                                 const ExtractedTrait &newTrait,
                                 qint64 episodeId);
    void processNextConversation();
    QString leanScoreToLabel(double score) const;

    LLMProviderManager *m_llmProvider = nullptr;
    MemoryStore *m_memoryStore = nullptr;

    void setStatus(const QString &status);

    bool m_isExtracting = false;
    int m_extractionCount = 0;
    qint64 m_currentExtractionEpisodeId = -1;
    QString m_lastStatus;
    QString m_lastResponse;

    // Idle-time episode scanning
    bool m_idleActive = false;
    QSet<qint64> m_scannedEpisodeIds;     // episodes already trait-scanned this session
    QTimer m_idleScanTimer;               // delay between scans during idle
    static constexpr int IDLE_SCAN_INTERVAL_MS = 30000; // 30s between scans

    // Safety timeout: auto-reset m_isExtracting if response never arrives
    QTimer m_extractionTimeout;
    static constexpr int EXTRACTION_TIMEOUT_MS = 120000; // 2 minutes

    // Political lean analysis state
    double m_leanScore = 0.0;       // -1.0 (Left) to +1.0 (Right)
    QString m_leanAnalyzedTs;
    bool m_isAnalyzingLean = false;
    QDateTime m_lastLeanAnalysis;
};

#endif // TRAITEXTRACTOR_H
