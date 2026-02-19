#ifndef EMBEDDINGMANAGER_H
#define EMBEDDINGMANAGER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QQueue>
#include <QMap>
#include <QDateTime>
#include <QNetworkAccessManager>

class QNetworkReply;
class MemoryStore;
class SettingsManager;

struct SemanticSearchResult {
    QString sourceType;
    qint64 sourceId = -1;
    double similarity = 0.0;
};

class EmbeddingManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isEmbedding READ isEmbedding NOTIFY isEmbeddingChanged)
    Q_PROPERTY(int embeddedCount READ embeddedCount NOTIFY embeddedCountChanged)
    Q_PROPERTY(int pendingCount READ pendingCount NOTIFY pendingCountChanged)
    Q_PROPERTY(QString status READ status NOTIFY statusChanged)

public:
    explicit EmbeddingManager(QObject *parent = nullptr);
    ~EmbeddingManager();

    void setMemoryStore(MemoryStore *store);
    void setSettingsManager(SettingsManager *settings);

    bool isEmbedding() const { return m_isEmbedding; }
    int embeddedCount() const { return m_embeddedCount; }
    int pendingCount() const { return m_batchQueue.size(); }
    QString status() const { return m_status; }

    // Embed text asynchronously — result delivered via embeddingReady signal
    void embedText(const QString &text, const QString &requestTag);

    // Synchronous search against in-memory cache
    QList<SemanticSearchResult> semanticSearch(
        const QVector<float> &queryEmbedding,
        const QStringList &sourceTypes,
        int topK = 5) const;

    // Check if embedding model is reachable
    bool isModelAvailable() const { return m_modelAvailable; }

    // Load all stored embeddings from DB into memory cache
    void loadEmbeddingsFromDb();

    // Check model availability (async)
    void checkModelAvailability();

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();
    void onEpisodeInserted(qint64 episodeId);

signals:
    void isEmbeddingChanged();
    void embeddedCountChanged();
    void pendingCountChanged();
    void statusChanged();
    void embeddingReady(const QString &requestTag, QVector<float> embedding);
    void embeddingError(const QString &requestTag, const QString &error);
    void batchEmbeddingComplete(int count);

private slots:
    void onNetworkReply(QNetworkReply *reply);

private:
    struct EmbeddingRecord {
        QString sourceType;
        qint64 sourceId = -1;
        QVector<float> embedding;
        int dimension = 0;
    };

    struct BatchItem {
        QString sourceType;
        qint64 sourceId;
        QString text;
    };

    // Batch embedding pipeline
    void startBatchEmbedding();
    void processNextBatchItem();

    // API call
    void callEmbeddingApi(const QString &text, const QString &tag);

    // Math
    static double cosineSimilarity(const QVector<float> &a, const QVector<float> &b);

    // DB operations
    void storeEmbedding(const QString &sourceType, qint64 sourceId,
                        const QString &model, const QVector<float> &embedding);
    bool hasEmbedding(const QString &sourceType, qint64 sourceId) const;

    // Serialization
    static QVector<float> blobToVector(const QByteArray &blob);
    static QByteArray vectorToBlob(const QVector<float> &vec);

    // Cache key helper
    static QString cacheKey(const QString &type, qint64 id) { return type + ":" + QString::number(id); }

    MemoryStore *m_memoryStore = nullptr;
    SettingsManager *m_settingsManager = nullptr;
    QNetworkAccessManager *m_networkManager = nullptr;

    // In-memory embedding cache for fast search
    QMap<QString, EmbeddingRecord> m_embeddingCache;

    // State
    bool m_isEmbedding = false;
    bool m_idleWindowOpen = false;
    bool m_modelAvailable = false;
    int m_embeddedCount = 0;
    QString m_status = "Idle";

    // Batch queue
    QQueue<BatchItem> m_batchQueue;

    // Error recovery
    int m_consecutiveErrors = 0;
    static constexpr int MAX_CONSECUTIVE_ERRORS = 3;
    QDateTime m_lastCycleEndTime;
    static constexpr int CYCLE_COOLDOWN_SEC = 60;
};

#endif // EMBEDDINGMANAGER_H
