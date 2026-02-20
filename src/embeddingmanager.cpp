#include "embeddingmanager.h"
#include "backgroundjobmanager.h"
#include "memorystore.h"
#include "settingsmanager.h"
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QSqlQuery>
#include <QSqlError>
#include <QDebug>
#include <algorithm>
#include <cmath>

EmbeddingManager::EmbeddingManager(QObject *parent)
    : QObject(parent)
    , m_networkManager(new QNetworkAccessManager(this))
{
    connect(m_networkManager, &QNetworkAccessManager::finished,
            this, &EmbeddingManager::onNetworkReply);
}

EmbeddingManager::~EmbeddingManager() = default;

void EmbeddingManager::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

void EmbeddingManager::setSettingsManager(SettingsManager *settings)
{
    m_settingsManager = settings;
}

// --- Public API ---

void EmbeddingManager::embedText(const QString &text, const QString &requestTag)
{
    if (text.trimmed().isEmpty()) {
        emit embeddingError(requestTag, "Empty text");
        return;
    }
    callEmbeddingApi(text.left(2000), requestTag);
}

QList<SemanticSearchResult> EmbeddingManager::semanticSearch(
    const QVector<float> &queryEmbedding,
    const QStringList &sourceTypes,
    int topK) const
{
    QList<SemanticSearchResult> results;
    if (queryEmbedding.isEmpty()) return results;

    for (auto it = m_embeddingCache.constBegin(); it != m_embeddingCache.constEnd(); ++it) {
        if (!sourceTypes.isEmpty() && !sourceTypes.contains(it->sourceType)) {
            continue;
        }

        double sim = cosineSimilarity(queryEmbedding, it->embedding);
        results.append({it->sourceType, it->sourceId, sim});
    }

    // Partial sort for top-K
    int k = qMin(topK, results.size());
    if (k > 0) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
            [](const SemanticSearchResult &a, const SemanticSearchResult &b) {
                return a.similarity > b.similarity;
            });
        results = results.mid(0, k);
    }

    return results;
}

void EmbeddingManager::semanticSearchAsync(
    const QVector<float> &queryEmbedding,
    const QStringList &sourceTypes,
    int topK)
{
    if (queryEmbedding.isEmpty()) {
        emit semanticSearchComplete({});
        return;
    }

    // COW snapshot — QMap implicit sharing makes this O(1).
    // The background thread works on its own detached copy,
    // so main-thread mutations to m_embeddingCache are safe.
    QMap<QString, EmbeddingRecord> cacheSnapshot = m_embeddingCache;

    BackgroundJobManager::instance()->submit<QList<SemanticSearchResult>>(
        "semantic_search",
        [cacheSnapshot = std::move(cacheSnapshot), queryEmbedding, sourceTypes, topK]()
            -> QList<SemanticSearchResult> {
            QList<SemanticSearchResult> results;
            for (auto it = cacheSnapshot.constBegin(); it != cacheSnapshot.constEnd(); ++it) {
                if (!sourceTypes.isEmpty() && !sourceTypes.contains(it->sourceType)) {
                    continue;
                }
                double sim = cosineSimilarity(queryEmbedding, it->embedding);
                results.append({it->sourceType, it->sourceId, sim});
            }

            int k = qMin(topK, results.size());
            if (k > 0) {
                std::partial_sort(results.begin(), results.begin() + k, results.end(),
                    [](const SemanticSearchResult &a, const SemanticSearchResult &b) {
                        return a.similarity > b.similarity;
                    });
                results = results.mid(0, k);
            }
            return results;
        },
        [this](const QList<SemanticSearchResult> &results) {
            emit semanticSearchComplete(results);
        }
    );
}

void EmbeddingManager::loadEmbeddingsFromDb()
{
    if (!m_memoryStore) return;

    m_embeddingCache.clear();

    QSqlQuery q(m_memoryStore->episodicDatabase());
    if (!q.exec("SELECT source_type, source_id, embedding_blob, vector_dimension "
                "FROM embeddings WHERE embedding_blob IS NOT NULL")) {
        qWarning() << "EmbeddingManager: failed to load embeddings:" << q.lastError().text();
        return;
    }

    int loaded = 0;
    while (q.next()) {
        QString sourceType = q.value(0).toString();
        qint64 sourceId = q.value(1).toLongLong();
        QByteArray blob = q.value(2).toByteArray();
        int dimension = q.value(3).toInt();

        QVector<float> vec = blobToVector(blob);
        if (vec.isEmpty()) continue;

        EmbeddingRecord record;
        record.sourceType = sourceType;
        record.sourceId = sourceId;
        record.embedding = vec;
        record.dimension = dimension;

        m_embeddingCache.insert(cacheKey(sourceType, sourceId), record);
        ++loaded;
    }

    m_embeddedCount = loaded;
    emit embeddedCountChanged();
    qDebug() << "EmbeddingManager: loaded" << loaded << "embeddings into memory cache";
}

void EmbeddingManager::checkModelAvailability()
{
    if (!m_settingsManager) return;

    // Send a tiny test embed to check if the model is available
    callEmbeddingApi("test", "availability_check");
}

// --- Idle-time slots ---

void EmbeddingManager::onIdleWindowOpened()
{
    m_idleWindowOpen = true;
    m_consecutiveErrors = 0;
    // Auto-start removed — coordinator calls requestStart()
}

bool EmbeddingManager::canStartCycle() const
{
    if (!m_modelAvailable || m_isEmbedding) return false;
    if (!m_memoryStore) return false;
    if (m_lastCycleEndTime.isValid()
        && m_lastCycleEndTime.secsTo(QDateTime::currentDateTime()) < CYCLE_COOLDOWN_SEC) {
        return false;
    }
    return true;
}

void EmbeddingManager::requestStart()
{
    if (!canStartCycle()) {
        emit cycleFinished();
        return;
    }
    startBatchEmbedding();
    // If nothing to embed, startBatchEmbedding returns without setting m_isEmbedding
    if (!m_isEmbedding) {
        emit cycleFinished();
    }
}

void EmbeddingManager::onIdleWindowClosed()
{
    m_idleWindowOpen = false;
    m_batchQueue.clear();

    if (m_isEmbedding) {
        m_isEmbedding = false;
        emit isEmbeddingChanged();
        m_lastCycleEndTime = QDateTime::currentDateTime();
    }

    m_status = "Idle";
    emit statusChanged();
    emit pendingCountChanged();
}

void EmbeddingManager::onEpisodeInserted(qint64 episodeId)
{
    // Queue this episode for embedding on next idle window
    // (Don't embed immediately to avoid competing with the active LLM request)
    Q_UNUSED(episodeId);
    // The batch embedding pipeline will find un-embedded episodes automatically
}

// --- Network handler ---

void EmbeddingManager::onNetworkReply(QNetworkReply *reply)
{
    reply->deleteLater();
    QString tag = reply->property("requestTag").toString();

    if (reply->error() != QNetworkReply::NoError) {
        m_consecutiveErrors++;

        if (tag == "availability_check") {
            m_modelAvailable = false;
            qDebug() << "EmbeddingManager: model not available:" << reply->errorString();
        } else {
            emit embeddingError(tag, reply->errorString());
        }

        // Continue batch if under error limit
        if (tag.startsWith("batch:") && m_consecutiveErrors < MAX_CONSECUTIVE_ERRORS
            && m_idleWindowOpen) {
            processNextBatchItem();
        } else if (tag.startsWith("batch:")) {
            m_isEmbedding = false;
            emit isEmbeddingChanged();
            m_status = QString("Paused (%1 errors)").arg(m_consecutiveErrors);
            emit statusChanged();
            emit cycleFinished();
        }
        return;
    }

    m_consecutiveErrors = 0;

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    QJsonObject obj = doc.object();

    // Parse response: {"embeddings": [[0.1, 0.2, ...]]}
    QJsonArray embeddingsArray = obj["embeddings"].toArray();
    if (embeddingsArray.isEmpty()) {
        if (tag == "availability_check") {
            m_modelAvailable = false;
        } else {
            emit embeddingError(tag, "Empty embeddings in response");
        }
        if (tag.startsWith("batch:") && m_idleWindowOpen) {
            processNextBatchItem();
        }
        return;
    }

    QJsonArray vectorArray = embeddingsArray[0].toArray();
    QVector<float> embedding;
    embedding.reserve(vectorArray.size());
    for (const QJsonValue &val : vectorArray) {
        embedding.append(static_cast<float>(val.toDouble()));
    }

    if (embedding.isEmpty()) {
        if (tag != "availability_check") {
            emit embeddingError(tag, "Parsed empty embedding vector");
        }
        if (tag.startsWith("batch:") && m_idleWindowOpen) {
            processNextBatchItem();
        }
        return;
    }

    // Handle availability check
    if (tag == "availability_check") {
        m_modelAvailable = true;
        qDebug() << "EmbeddingManager: model available, dimension =" << embedding.size();

        // If idle window is open, start batch embedding now
        if (m_idleWindowOpen && !m_isEmbedding) {
            startBatchEmbedding();
        }
        return;
    }

    // Emit for any listener (e.g., Orchestrator caching user message embedding)
    emit embeddingReady(tag, embedding);

    // Handle batch item: store in DB + cache
    if (tag.startsWith("batch:")) {
        QStringList parts = tag.split(':');
        if (parts.size() >= 3) {
            QString sourceType = parts[1];
            qint64 sourceId = parts[2].toLongLong();
            QString model = m_settingsManager ? m_settingsManager->embeddingModel() : "unknown";

            storeEmbedding(sourceType, sourceId, model, embedding);

            // Update in-memory cache
            EmbeddingRecord record;
            record.sourceType = sourceType;
            record.sourceId = sourceId;
            record.embedding = embedding;
            record.dimension = embedding.size();
            m_embeddingCache.insert(cacheKey(sourceType, sourceId), record);

            m_embeddedCount = m_embeddingCache.size();
            emit embeddedCountChanged();
        }

        // Process next item in queue
        if (m_idleWindowOpen) {
            processNextBatchItem();
        }
    }
}

// --- Batch embedding pipeline ---

void EmbeddingManager::startBatchEmbedding()
{
    if (m_isEmbedding || !m_idleWindowOpen || !m_memoryStore) return;

    // Check cooldown
    if (m_lastCycleEndTime.isValid()
        && m_lastCycleEndTime.secsTo(QDateTime::currentDateTime()) < CYCLE_COOLDOWN_SEC) {
        return;
    }

    m_batchQueue.clear();

    // Find un-embedded episodes (LEFT JOIN to find gaps)
    QSqlQuery q(m_memoryStore->episodicDatabase());
    if (q.exec("SELECT e.id, e.user_text, e.assistant_text FROM episodes e "
               "LEFT JOIN embeddings emb ON emb.source_type = 'episode' AND emb.source_id = e.id "
               "WHERE emb.id IS NULL ORDER BY e.id DESC LIMIT 20")) {
        while (q.next()) {
            qint64 id = q.value(0).toLongLong();
            QString userText = q.value(1).toString();
            QString assistantText = q.value(2).toString();
            QString combined = userText + " " + assistantText;
            if (!combined.trimmed().isEmpty()) {
                m_batchQueue.enqueue({"episode", id, combined.left(2000)});
            }
        }
    }

    // Find un-embedded traits
    if (q.exec("SELECT t.trait_id, t.statement FROM traits t "
               "LEFT JOIN embeddings emb ON emb.source_type = 'trait' AND emb.source_id = CAST(t.trait_id AS INTEGER) "
               "WHERE emb.id IS NULL LIMIT 10")) {
        while (q.next()) {
            QString traitId = q.value(0).toString();
            QString statement = q.value(1).toString();
            if (!statement.trimmed().isEmpty()) {
                m_batchQueue.enqueue({"trait", traitId.toLongLong(), statement.left(2000)});
            }
        }
    }

    // Find un-embedded research findings
    if (q.exec("SELECT rf.id, rf.topic, rf.llm_summary FROM research_findings rf "
               "LEFT JOIN embeddings emb ON emb.source_type = 'research' AND emb.source_id = rf.id "
               "WHERE emb.id IS NULL AND rf.status = 'approved' LIMIT 10")) {
        while (q.next()) {
            qint64 id = q.value(0).toLongLong();
            QString topic = q.value(1).toString();
            QString summary = q.value(2).toString();
            QString combined = topic + ": " + summary;
            if (!combined.trimmed().isEmpty()) {
                m_batchQueue.enqueue({"research", id, combined.left(2000)});
            }
        }
    }

    emit pendingCountChanged();

    if (m_batchQueue.isEmpty()) {
        m_status = "All records embedded";
        emit statusChanged();
        return;
    }

    m_isEmbedding = true;
    emit isEmbeddingChanged();
    m_status = QString("Embedding %1 records...").arg(m_batchQueue.size());
    emit statusChanged();

    processNextBatchItem();
}

void EmbeddingManager::processNextBatchItem()
{
    if (m_batchQueue.isEmpty() || !m_idleWindowOpen) {
        if (m_isEmbedding) {
            m_isEmbedding = false;
            emit isEmbeddingChanged();
            m_lastCycleEndTime = QDateTime::currentDateTime();

            int total = m_embeddingCache.size();
            m_status = QString("Cycle complete (%1 total)").arg(total);
            emit statusChanged();
            emit batchEmbeddingComplete(total);
            emit cycleFinished();
        }
        return;
    }

    BatchItem item = m_batchQueue.dequeue();
    emit pendingCountChanged();

    m_status = QString("Embedding %1:%2 (%3 left)")
        .arg(item.sourceType)
        .arg(item.sourceId)
        .arg(m_batchQueue.size());
    emit statusChanged();

    QString tag = QString("batch:%1:%2").arg(item.sourceType).arg(item.sourceId);
    callEmbeddingApi(item.text, tag);
}

// --- API call ---

void EmbeddingManager::callEmbeddingApi(const QString &text, const QString &tag)
{
    if (!m_settingsManager) {
        emit embeddingError(tag, "No settings manager");
        return;
    }

    QString ollamaUrl = m_settingsManager->ollamaUrl();
    QString model = m_settingsManager->embeddingModel();

    QJsonObject body;
    body["model"] = model;
    body["input"] = text;

    QNetworkRequest request(QUrl(ollamaUrl + "/api/embed"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    request.setTransferTimeout(30000);

    QNetworkReply *reply = m_networkManager->post(request, QJsonDocument(body).toJson());
    reply->setProperty("requestTag", tag);
}

// --- Math ---

double EmbeddingManager::cosineSimilarity(const QVector<float> &a, const QVector<float> &b)
{
    if (a.size() != b.size() || a.isEmpty()) return 0.0;

    double dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        dotProduct += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        normA += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        normB += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }

    double denom = std::sqrt(normA) * std::sqrt(normB);
    return (denom > 0.0) ? (dotProduct / denom) : 0.0;
}

// --- DB operations ---

void EmbeddingManager::storeEmbedding(const QString &sourceType, qint64 sourceId,
                                       const QString &model, const QVector<float> &embedding)
{
    if (!m_memoryStore) return;

    QByteArray blob = vectorToBlob(embedding);

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("INSERT OR REPLACE INTO embeddings "
              "(source_type, source_id, embedding_model, embedding_blob, vector_dimension) "
              "VALUES (:type, :id, :model, :blob, :dim)");
    q.bindValue(":type", sourceType);
    q.bindValue(":id", sourceId);
    q.bindValue(":model", model);
    q.bindValue(":blob", blob);
    q.bindValue(":dim", embedding.size());

    if (!q.exec()) {
        qWarning() << "EmbeddingManager: failed to store embedding:"
                    << q.lastError().text();
    }
}

bool EmbeddingManager::hasEmbedding(const QString &sourceType, qint64 sourceId) const
{
    return m_embeddingCache.contains(cacheKey(sourceType, sourceId));
}

// --- Serialization ---

QVector<float> EmbeddingManager::blobToVector(const QByteArray &blob)
{
    QVector<float> vec;
    if (blob.isEmpty() || blob.size() % sizeof(float) != 0) return vec;

    int count = blob.size() / static_cast<int>(sizeof(float));
    vec.resize(count);
    memcpy(vec.data(), blob.constData(), blob.size());
    return vec;
}

QByteArray EmbeddingManager::vectorToBlob(const QVector<float> &vec)
{
    return QByteArray(reinterpret_cast<const char *>(vec.constData()),
                      vec.size() * static_cast<int>(sizeof(float)));
}
