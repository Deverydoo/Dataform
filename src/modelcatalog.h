#ifndef MODELCATALOG_H
#define MODELCATALOG_H

#include <QObject>
#include <QString>
#include <QVariantList>
#include <QList>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QElapsedTimer>
#include <QProcess>
#include <QFile>

// A single downloadable GGUF build. One catalog entry == one quant of one model,
// because quant choice is the thing that decides whether it fits the user's VRAM.
struct CatalogModel {
    QString id;             // stable slug, used by QML and by catalog_state.json
    QString displayName;
    QString family;         // upstream base architecture
    QString quant;          // "i1-Q4_K_M", "Q8_0", ...
    QString repoId;         // HuggingFace repo, e.g. "mradermacher/Foo-GGUF"
    QString fileName;       // exact .gguf filename inside that repo
    qint64 approxBytes = 0; // from the repo's quant table; real size comes from Content-Length
    double vramGb = 0.0;    // rough weights+KV estimate at the listed context
    int contextLength = 0;
    QString license;
    QString alignment;      // "instruct" | "uncensored" | "roleplay"
    QString role;           // "chat" | "background" | "both"
    QString description;
    QString ollamaTag;      // non-empty when an equivalent exists in the Ollama registry

    bool isUncensored() const { return alignment != QStringLiteral("instruct"); }
};

// Curated download catalog for local GGUF models, plus the downloader that fetches
// them. Files land in models/background_llm/ (background-capable) or models/chat_llm/
// (chat-only) so the existing background-model picker sees them without changes.
class ModelCatalog : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QVariantList models READ models CONSTANT)
    Q_PROPERTY(bool isDownloading READ isDownloading NOTIFY isDownloadingChanged)
    Q_PROPERTY(QString downloadingId READ downloadingId NOTIFY downloadingIdChanged)
    Q_PROPERTY(double progress READ progress NOTIFY progressChanged)
    Q_PROPERTY(qint64 receivedBytes READ receivedBytes NOTIFY progressChanged)
    Q_PROPERTY(qint64 totalBytes READ totalBytes NOTIFY progressChanged)
    Q_PROPERTY(double speedBytesPerSec READ speedBytesPerSec NOTIFY progressChanged)
    Q_PROPERTY(QString statusText READ statusText NOTIFY statusTextChanged)
    Q_PROPERTY(bool uncensoredAcknowledged READ uncensoredAcknowledged NOTIFY uncensoredAcknowledgedChanged)

public:
    explicit ModelCatalog(QObject *parent = nullptr);
    ~ModelCatalog();

    QVariantList models() const;
    bool isDownloading() const { return m_reply != nullptr; }
    QString downloadingId() const { return m_downloadingId; }
    double progress() const { return m_progress; }
    qint64 receivedBytes() const { return m_receivedBytes; }
    qint64 totalBytes() const { return m_totalBytes; }
    double speedBytesPerSec() const { return m_speedBytesPerSec; }
    QString statusText() const { return m_statusText; }
    bool uncensoredAcknowledged() const { return m_uncensoredAcknowledged; }

    // Filtered views for the panel's tabs. role is "chat", "background" or "all".
    Q_INVOKABLE QVariantList modelsForRole(const QString &role) const;
    Q_INVOKABLE QVariantMap modelById(const QString &id) const;

    Q_INVOKABLE bool isInstalled(const QString &id) const;
    Q_INVOKABLE QString installedPath(const QString &id) const;
    // Bytes already on disk in a partial .part file, for "Resume (3.2 GB done)".
    Q_INVOKABLE qint64 partialBytes(const QString &id) const;
    Q_INVOKABLE QString formatBytes(qint64 bytes) const;

    Q_INVOKABLE void startDownload(const QString &id);
    Q_INVOKABLE void cancelDownload();
    Q_INVOKABLE bool removeDownload(const QString &id);

    // Registers a downloaded .gguf with a local Ollama install so it can be used
    // as the chat provider model. Requires the `ollama` CLI on PATH.
    Q_INVOKABLE void installToOllama(const QString &id);

    Q_INVOKABLE void acknowledgeUncensored();

signals:
    void isDownloadingChanged();
    void downloadingIdChanged();
    void progressChanged();
    void statusTextChanged();
    void uncensoredAcknowledgedChanged();
    void downloadFinished(const QString &id, const QString &path);
    void downloadFailed(const QString &id, const QString &error);
    void ollamaInstallFinished(const QString &id, const QString &tag);
    void ollamaInstallFailed(const QString &id, const QString &error);

private slots:
    void onMetaDataChanged();
    void onReadyRead();
    void onDownloadProgress(qint64 received, qint64 total);
    void onFinished();

private:
    void buildCatalog();
    void setStatus(const QString &text);
    void resetTransfer();
    void cleanupReply();
    QString destinationDir(const CatalogModel &m) const;
    QString finalPath(const CatalogModel &m) const;
    QString partPath(const CatalogModel &m) const;
    QString downloadUrl(const CatalogModel &m) const;
    const CatalogModel *find(const QString &id) const;
    QVariantMap toVariant(const CatalogModel &m) const;
    void loadState();
    void saveState();
    QString statePath() const;

    QList<CatalogModel> m_catalog;
    QNetworkAccessManager *m_network;
    QNetworkReply *m_reply = nullptr;
    QFile *m_file = nullptr;
    QProcess *m_ollamaProc = nullptr;
    QString m_ollamaInstallId;

    QString m_downloadingId;
    qint64 m_resumeOffset = 0;
    qint64 m_receivedBytes = 0;
    qint64 m_totalBytes = 0;
    double m_progress = 0.0;
    double m_speedBytesPerSec = 0.0;
    QString m_statusText;
    bool m_uncensoredAcknowledged = false;
    bool m_cancelRequested = false;

    QElapsedTimer m_speedTimer;
    qint64 m_lastSpeedSampleBytes = 0;

    static constexpr int SPEED_SAMPLE_MS = 1000;
};

#endif // MODELCATALOG_H
