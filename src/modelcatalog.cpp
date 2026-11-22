#include "modelcatalog.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkRequest>
#include <QStandardPaths>
#include <QUrl>
#include <QDebug>

namespace {
// Decimal units, matching how HuggingFace reports file sizes, so what the panel
// shows lines up with what the repo page says.
constexpr qint64 KB = 1000LL;
constexpr qint64 MB = 1000LL * 1000LL;
constexpr qint64 GB = 1000LL * 1000LL * 1000LL;
}

ModelCatalog::ModelCatalog(QObject *parent)
    : QObject(parent)
    , m_network(new QNetworkAccessManager(this))
{
    buildCatalog();
    loadState();
    m_statusText = QStringLiteral("Idle");
}

ModelCatalog::~ModelCatalog()
{
    if (m_reply) {
        m_cancelRequested = true;
        m_reply->abort();
    }
    if (m_file) {
        m_file->flush();
        m_file->close();
        delete m_file;
        m_file = nullptr;
    }
}

// --- Catalog definition -----------------------------------------------------

void ModelCatalog::buildCatalog()
{
    // Sizes are the exact Content-Length each URL reports, so the progress bar is
    // accurate before the first byte arrives. VRAM figures are weights plus KV
    // cache at the listed context, rounded up.

    {
        CatalogModel m;
        m.id = "cydonia-24b-v43-heresy-q4km";
        m.displayName = "Cydonia 24B v4.3 (Absolute Heresy)";
        m.family = "Mistral-Small-3.1-24B";
        m.quant = "i1-Q4_K_M";
        m.repoId = "mradermacher/Cydonia-24B-v4.3-absolute-heresy-i1-GGUF";
        m.fileName = "Cydonia-24B-v4.3-absolute-heresy.i1-Q4_K_M.gguf";
        m.approxBytes = 14333911424LL;
        m.vramGb = 16.5;
        m.contextLength = 16384;
        m.license = "Apache-2.0 (Mistral-Small-3.1-24B base)";
        m.alignment = "roleplay";
        m.role = "chat";
        m.description = "Imatrix quant of MuXodious/Cydonia-24B-v4.3-absolute-heresy. "
                        "Decensored roleplay/persona finetune with minimal refusals and strong "
                        "character adherence. Recommended chat brain for a 24 GB card.";
        m_catalog.append(m);
    }

    {
        CatalogModel m;
        m.id = "cydonia-24b-v43-heresy-q5km";
        m.displayName = "Cydonia 24B v4.3 (Absolute Heresy)";
        m.family = "Mistral-Small-3.1-24B";
        m.quant = "i1-Q5_K_M";
        m.repoId = "mradermacher/Cydonia-24B-v4.3-absolute-heresy-i1-GGUF";
        m.fileName = "Cydonia-24B-v4.3-absolute-heresy.i1-Q5_K_M.gguf";
        m.approxBytes = 16763986304LL;
        m.vramGb = 19.0;
        m.contextLength = 12288;
        m.license = "Apache-2.0 (Mistral-Small-3.1-24B base)";
        m.alignment = "roleplay";
        m.role = "chat";
        m.description = "Higher-fidelity quant of the same model. Fits a 24 GB card but leaves "
                        "little headroom — expect a shorter usable context than Q4_K_M.";
        m_catalog.append(m);
    }

    {
        CatalogModel m;
        m.id = "dolphin3-llama31-8b-q8";
        m.displayName = "Dolphin 3.0 (Llama 3.1 8B)";
        m.family = "Llama-3.1-8B";
        m.quant = "Q8_0";
        m.repoId = "dphn/Dolphin3.0-Llama3.1-8B-GGUF";
        m.fileName = "Dolphin3.0-Llama3.1-8B-Q8_0.gguf";
        m.approxBytes = 8540788544LL;
        m.vramGb = 10.5;
        m.contextLength = 8192;
        m.license = "Llama 3.1 Community License";
        m.alignment = "uncensored";
        m.role = "both";
        m.description = "Steerable general-purpose model with no built-in alignment — the system "
                        "prompt sets the policy. Strong at JSON and function calling, which is what "
                        "the background tasks need. Q8_0 keeps extraction accuracy high.";
        m_catalog.append(m);
    }

    {
        CatalogModel m;
        m.id = "dolphin3-llama31-8b-q4km";
        m.displayName = "Dolphin 3.0 (Llama 3.1 8B)";
        m.family = "Llama-3.1-8B";
        m.quant = "Q4_K_M";
        m.repoId = "dphn/Dolphin3.0-Llama3.1-8B-GGUF";
        m.fileName = "Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf";
        m.approxBytes = 4920745856LL;
        m.vramGb = 6.5;
        m.contextLength = 8192;
        m.license = "Llama 3.1 Community License";
        m.alignment = "uncensored";
        m.role = "both";
        m.ollamaTag = "dolphin3:8b";
        m.description = "Same model at half the footprint. Use this one if you want to run it "
                        "alongside a 24B chat model on a single 24 GB card.";
        m_catalog.append(m);
    }

    {
        CatalogModel m;
        m.id = "dolphin3-qwen25-3b-q8";
        m.displayName = "Dolphin 3.0 (Qwen2.5 3B)";
        m.family = "Qwen2.5-3B";
        m.quant = "Q8_0";
        m.repoId = "bartowski/Dolphin3.0-Qwen2.5-3b-GGUF";
        m.fileName = "Dolphin3.0-Qwen2.5-3b-Q8_0.gguf";
        m.approxBytes = 3285479392LL;
        m.vramGb = 4.3;
        m.contextLength = 8192;
        m.license = "Apache-2.0";
        m.alignment = "uncensored";
        m.role = "background";
        m.description = "Drop-in replacement for the default 3B internal brain, uncensored so "
                        "trait extraction and sentiment analysis do not refuse on personal topics. "
                        "Leaves ~20 GB free for the chat model.";
        m_catalog.append(m);
    }
}

const CatalogModel *ModelCatalog::find(const QString &id) const
{
    for (const CatalogModel &m : m_catalog) {
        if (m.id == id) return &m;
    }
    return nullptr;
}

QVariantMap ModelCatalog::toVariant(const CatalogModel &m) const
{
    QVariantMap v;
    v["id"] = m.id;
    v["displayName"] = m.displayName;
    v["family"] = m.family;
    v["quant"] = m.quant;
    v["repoId"] = m.repoId;
    v["fileName"] = m.fileName;
    v["approxBytes"] = m.approxBytes;
    v["approxSize"] = formatBytes(m.approxBytes);
    v["vramGb"] = m.vramGb;
    v["contextLength"] = m.contextLength;
    v["license"] = m.license;
    v["alignment"] = m.alignment;
    v["role"] = m.role;
    v["description"] = m.description;
    v["ollamaTag"] = m.ollamaTag;
    v["uncensored"] = m.isUncensored();
    v["installed"] = isInstalled(m.id);
    v["installedPath"] = installedPath(m.id);
    v["partialBytes"] = partialBytes(m.id);
    v["destinationDir"] = destinationDir(m);
    return v;
}

QVariantList ModelCatalog::models() const
{
    QVariantList list;
    for (const CatalogModel &m : m_catalog) list.append(toVariant(m));
    return list;
}

QVariantList ModelCatalog::modelsForRole(const QString &role) const
{
    QVariantList list;
    for (const CatalogModel &m : m_catalog) {
        if (role == QStringLiteral("all") || m.role == role || m.role == QStringLiteral("both"))
            list.append(toVariant(m));
    }
    return list;
}

QVariantMap ModelCatalog::modelById(const QString &id) const
{
    const CatalogModel *m = find(id);
    return m ? toVariant(*m) : QVariantMap();
}

// --- Paths ------------------------------------------------------------------

QString ModelCatalog::destinationDir(const CatalogModel &m) const
{
    // Background-capable models go where LlamaCppManager already scans, so they
    // appear in the background-model picker the moment the download completes.
    const QString sub = (m.role == QStringLiteral("chat"))
        ? QStringLiteral("models/chat_llm")
        : QStringLiteral("models/background_llm");
    return QCoreApplication::applicationDirPath() + "/" + sub;
}

QString ModelCatalog::finalPath(const CatalogModel &m) const
{
    return destinationDir(m) + "/" + m.fileName;
}

QString ModelCatalog::partPath(const CatalogModel &m) const
{
    return finalPath(m) + ".part";
}

QString ModelCatalog::downloadUrl(const CatalogModel &m) const
{
    return QStringLiteral("https://huggingface.co/%1/resolve/main/%2?download=true")
        .arg(m.repoId, m.fileName);
}

bool ModelCatalog::isInstalled(const QString &id) const
{
    const CatalogModel *m = find(id);
    return m && QFileInfo::exists(finalPath(*m));
}

QString ModelCatalog::installedPath(const QString &id) const
{
    const CatalogModel *m = find(id);
    if (!m) return {};
    const QString p = finalPath(*m);
    return QFileInfo::exists(p) ? p : QString();
}

qint64 ModelCatalog::partialBytes(const QString &id) const
{
    const CatalogModel *m = find(id);
    if (!m) return 0;
    QFileInfo fi(partPath(*m));
    return fi.exists() ? fi.size() : 0;
}

QString ModelCatalog::formatBytes(qint64 bytes) const
{
    if (bytes <= 0) return QStringLiteral("0 B");
    if (bytes >= GB) return QString::number(bytes / double(GB), 'f', 1) + " GB";
    if (bytes >= MB) return QString::number(bytes / double(MB), 'f', 1) + " MB";
    if (bytes >= KB) return QString::number(bytes / double(KB), 'f', 1) + " KB";
    return QString::number(bytes) + " B";
}

// --- Download ---------------------------------------------------------------

void ModelCatalog::startDownload(const QString &id)
{
    if (m_reply) {
        emit downloadFailed(id, QStringLiteral("Another download is already running."));
        return;
    }

    const CatalogModel *m = find(id);
    if (!m) {
        emit downloadFailed(id, QStringLiteral("Unknown model id: %1").arg(id));
        return;
    }

    if (isInstalled(id)) {
        emit downloadFinished(id, finalPath(*m));
        return;
    }

    QDir dir(destinationDir(*m));
    if (!dir.exists() && !dir.mkpath(".")) {
        emit downloadFailed(id, QStringLiteral("Could not create %1").arg(dir.absolutePath()));
        return;
    }

    const QString part = partPath(*m);
    m_resumeOffset = QFileInfo::exists(part) ? QFileInfo(part).size() : 0;

    m_file = new QFile(part);
    const QIODevice::OpenMode mode = m_resumeOffset > 0
        ? (QIODevice::WriteOnly | QIODevice::Append)
        : (QIODevice::WriteOnly | QIODevice::Truncate);
    if (!m_file->open(mode)) {
        const QString err = QStringLiteral("Could not open %1: %2").arg(part, m_file->errorString());
        delete m_file;
        m_file = nullptr;
        emit downloadFailed(id, err);
        return;
    }

    QNetworkRequest req{QUrl(downloadUrl(*m))};
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                     QNetworkRequest::NoLessSafeRedirectPolicy);
    req.setRawHeader("User-Agent", "Dataform/0.6 (+local model catalog)");
    if (m_resumeOffset > 0)
        req.setRawHeader("Range", "bytes=" + QByteArray::number(m_resumeOffset) + "-");

    m_cancelRequested = false;
    m_downloadingId = id;
    m_receivedBytes = m_resumeOffset;
    m_totalBytes = m->approxBytes;
    m_progress = m_totalBytes > 0 ? double(m_receivedBytes) / double(m_totalBytes) : 0.0;
    m_speedBytesPerSec = 0.0;
    m_lastSpeedSampleBytes = m_receivedBytes;
    m_speedTimer.start();

    m_reply = m_network->get(req);
    connect(m_reply, &QNetworkReply::metaDataChanged, this, &ModelCatalog::onMetaDataChanged);
    connect(m_reply, &QNetworkReply::readyRead, this, &ModelCatalog::onReadyRead);
    connect(m_reply, &QNetworkReply::downloadProgress, this, &ModelCatalog::onDownloadProgress);
    connect(m_reply, &QNetworkReply::finished, this, &ModelCatalog::onFinished);

    emit isDownloadingChanged();
    emit downloadingIdChanged();
    emit progressChanged();
    setStatus(m_resumeOffset > 0
        ? QStringLiteral("Resuming %1 at %2…").arg(m->displayName, formatBytes(m_resumeOffset))
        : QStringLiteral("Connecting to huggingface.co…"));
}

void ModelCatalog::onMetaDataChanged()
{
    if (!m_reply || !m_file) return;

    const int status = m_reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();

    // We asked for a byte range but the server sent the whole file anyway. The
    // append-mode handle would corrupt the file, so start over from zero.
    if (m_resumeOffset > 0 && status == 200) {
        qWarning() << "ModelCatalog: server ignored Range request, restarting download";
        m_file->close();
        if (!m_file->open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            setStatus(QStringLiteral("Could not restart download"));
            m_cancelRequested = true;
            m_reply->abort();
            return;
        }
        m_resumeOffset = 0;
        m_receivedBytes = 0;
        m_lastSpeedSampleBytes = 0;
    }
}

void ModelCatalog::onReadyRead()
{
    if (!m_reply || !m_file) return;
    const QByteArray chunk = m_reply->readAll();
    if (chunk.isEmpty()) return;

    if (m_file->write(chunk) != chunk.size()) {
        const QString err = m_file->errorString();
        setStatus(QStringLiteral("Write failed: %1").arg(err));
        m_cancelRequested = true;
        m_reply->abort();
    }
}

void ModelCatalog::onDownloadProgress(qint64 received, qint64 total)
{
    m_receivedBytes = m_resumeOffset + received;
    if (total > 0) m_totalBytes = m_resumeOffset + total;

    m_progress = m_totalBytes > 0
        ? qBound(0.0, double(m_receivedBytes) / double(m_totalBytes), 1.0)
        : 0.0;

    const qint64 elapsed = m_speedTimer.elapsed();
    if (elapsed >= SPEED_SAMPLE_MS) {
        m_speedBytesPerSec = double(m_receivedBytes - m_lastSpeedSampleBytes) * 1000.0 / double(elapsed);
        m_lastSpeedSampleBytes = m_receivedBytes;
        m_speedTimer.restart();

        if (m_speedBytesPerSec > 0 && m_totalBytes > m_receivedBytes) {
            const qint64 secsLeft = qint64((m_totalBytes - m_receivedBytes) / m_speedBytesPerSec);
            setStatus(QStringLiteral("%1 / %2 — %3/s — %4 remaining")
                .arg(formatBytes(m_receivedBytes),
                     formatBytes(m_totalBytes),
                     formatBytes(qint64(m_speedBytesPerSec)),
                     secsLeft >= 60 ? QStringLiteral("%1m %2s").arg(secsLeft / 60).arg(secsLeft % 60)
                                    : QStringLiteral("%1s").arg(secsLeft)));
        }
    }

    emit progressChanged();
}

void ModelCatalog::onFinished()
{
    if (!m_reply) return;

    const QString id = m_downloadingId;
    const CatalogModel *m = find(id);
    const QNetworkReply::NetworkError netErr = m_reply->error();
    const int status = m_reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    const QString netErrString = m_reply->errorString();

    // Drain whatever is still buffered before closing the file.
    if (m_file && m_reply->bytesAvailable() > 0)
        m_file->write(m_reply->readAll());

    if (m_file) {
        m_file->flush();
        m_file->close();
    }

    cleanupReply();

    if (m_cancelRequested) {
        // The .part file is kept on purpose so the download can be resumed.
        resetTransfer();
        setStatus(QStringLiteral("Download cancelled — partial file kept for resume"));
        emit downloadFailed(id, QStringLiteral("Cancelled"));
        return;
    }

    if (netErr != QNetworkReply::NoError || (status != 200 && status != 206 && status != 0)) {
        resetTransfer();
        const QString err = netErr != QNetworkReply::NoError
            ? netErrString
            : QStringLiteral("HTTP %1").arg(status);
        setStatus(QStringLiteral("Download failed: %1").arg(err));
        emit downloadFailed(id, err);
        return;
    }

    if (!m) {
        resetTransfer();
        emit downloadFailed(id, QStringLiteral("Model vanished from catalog"));
        return;
    }

    const QString part = partPath(*m);
    const QString target = finalPath(*m);
    QFile::remove(target);
    if (!QFile::rename(part, target)) {
        resetTransfer();
        const QString err = QStringLiteral("Could not move %1 into place").arg(m->fileName);
        setStatus(err);
        emit downloadFailed(id, err);
        return;
    }

    resetTransfer();
    m_progress = 1.0;
    setStatus(QStringLiteral("%1 %2 ready").arg(m->displayName, m->quant));
    emit progressChanged();
    emit downloadFinished(id, target);
}

void ModelCatalog::cancelDownload()
{
    if (!m_reply) return;
    m_cancelRequested = true;
    setStatus(QStringLiteral("Cancelling…"));
    m_reply->abort();
}

bool ModelCatalog::removeDownload(const QString &id)
{
    const CatalogModel *m = find(id);
    if (!m) return false;
    if (m_reply && m_downloadingId == id) return false;

    bool removed = false;
    if (QFileInfo::exists(finalPath(*m))) removed |= QFile::remove(finalPath(*m));
    if (QFileInfo::exists(partPath(*m))) removed |= QFile::remove(partPath(*m));
    if (removed) setStatus(QStringLiteral("Removed %1").arg(m->fileName));
    return removed;
}

void ModelCatalog::cleanupReply()
{
    if (m_reply) {
        m_reply->deleteLater();
        m_reply = nullptr;
    }
    if (m_file) {
        delete m_file;
        m_file = nullptr;
    }
}

void ModelCatalog::resetTransfer()
{
    m_downloadingId.clear();
    m_resumeOffset = 0;
    m_receivedBytes = 0;
    m_totalBytes = 0;
    m_progress = 0.0;
    m_speedBytesPerSec = 0.0;
    m_cancelRequested = false;
    emit isDownloadingChanged();
    emit downloadingIdChanged();
    emit progressChanged();
}

void ModelCatalog::setStatus(const QString &text)
{
    if (m_statusText == text) return;
    m_statusText = text;
    emit statusTextChanged();
}

// --- Ollama registration ----------------------------------------------------

void ModelCatalog::installToOllama(const QString &id)
{
    const CatalogModel *m = find(id);
    if (!m) {
        emit ollamaInstallFailed(id, QStringLiteral("Unknown model id"));
        return;
    }
    if (m_ollamaProc) {
        emit ollamaInstallFailed(id, QStringLiteral("An Ollama install is already running"));
        return;
    }
    const QString gguf = installedPath(id);
    if (gguf.isEmpty()) {
        emit ollamaInstallFailed(id, QStringLiteral("Download the model first"));
        return;
    }

    const QString ollama = QStandardPaths::findExecutable(QStringLiteral("ollama"));
    if (ollama.isEmpty()) {
        emit ollamaInstallFailed(id, QStringLiteral("The `ollama` command was not found on PATH"));
        return;
    }

    // Ollama reads the chat template out of the GGUF, so FROM is all we need.
    const QString modelfile = destinationDir(*m) + "/" + m->id + ".Modelfile";
    QFile mf(modelfile);
    if (!mf.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        emit ollamaInstallFailed(id, QStringLiteral("Could not write %1").arg(modelfile));
        return;
    }
    mf.write(QStringLiteral("FROM %1\n").arg(QDir::toNativeSeparators(gguf)).toUtf8());
    mf.close();

    m_ollamaInstallId = id;
    m_ollamaProc = new QProcess(this);
    setStatus(QStringLiteral("Registering %1 with Ollama…").arg(m->id));

    connect(m_ollamaProc, &QProcess::finished, this,
            [this, id, tag = m->id, modelfile](int exitCode, QProcess::ExitStatus) {
        // A crashed process emits errorOccurred first, which already tore this down.
        if (!m_ollamaProc) return;
        const QString stderrText = QString::fromUtf8(m_ollamaProc->readAllStandardError()).trimmed();
        m_ollamaProc->deleteLater();
        m_ollamaProc = nullptr;
        m_ollamaInstallId.clear();
        QFile::remove(modelfile);

        if (exitCode == 0) {
            setStatus(QStringLiteral("Ollama model `%1` created").arg(tag));
            emit ollamaInstallFinished(id, tag);
        } else {
            const QString err = stderrText.isEmpty()
                ? QStringLiteral("ollama create exited with code %1").arg(exitCode)
                : stderrText;
            setStatus(QStringLiteral("Ollama install failed"));
            emit ollamaInstallFailed(id, err);
        }
    });

    connect(m_ollamaProc, &QProcess::errorOccurred, this, [this, id](QProcess::ProcessError) {
        if (!m_ollamaProc) return;
        const QString err = m_ollamaProc->errorString();
        m_ollamaProc->deleteLater();
        m_ollamaProc = nullptr;
        m_ollamaInstallId.clear();
        setStatus(QStringLiteral("Ollama install failed"));
        emit ollamaInstallFailed(id, err);
    });

    m_ollamaProc->start(ollama, {QStringLiteral("create"), m->id,
                                 QStringLiteral("-f"), modelfile});
}

// --- Persisted state --------------------------------------------------------

QString ModelCatalog::statePath() const
{
    return QCoreApplication::applicationDirPath() + "/models/catalog_state.json";
}

void ModelCatalog::loadState()
{
    QFile f(statePath());
    if (!f.open(QIODevice::ReadOnly)) return;
    const QJsonObject o = QJsonDocument::fromJson(f.readAll()).object();
    f.close();
    m_uncensoredAcknowledged = o.value("uncensoredAcknowledged").toBool(false);
}

void ModelCatalog::saveState()
{
    QDir dir(QCoreApplication::applicationDirPath() + "/models");
    if (!dir.exists() && !dir.mkpath(".")) {
        qWarning() << "ModelCatalog: could not create" << dir.absolutePath();
        return;
    }
    QJsonObject o;
    o["schemaVersion"] = 1;
    o["uncensoredAcknowledged"] = m_uncensoredAcknowledged;

    QFile f(statePath());
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        qWarning() << "ModelCatalog: could not write" << statePath();
        return;
    }
    f.write(QJsonDocument(o).toJson(QJsonDocument::Indented));
    f.close();
}

void ModelCatalog::acknowledgeUncensored()
{
    if (m_uncensoredAcknowledged) return;
    m_uncensoredAcknowledged = true;
    saveState();
    emit uncensoredAcknowledgedChanged();
}
