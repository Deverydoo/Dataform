#include "memorystore.h"
#include "cryptoutil.h"
#include "schemamigrator.h"
#include "settingsmanager.h"
#include <QSqlQuery>
#include <QSqlError>
#include <QUuid>
#include <QJsonDocument>
#include <QJsonArray>
#include <QFile>
#include <QDebug>

MemoryStore::MemoryStore(const QString &profilePath, QObject *parent)
    : QObject(parent)
    , m_profilePath(profilePath)
    , m_memoryPath(profilePath + "/memory")
{
}

MemoryStore::~MemoryStore()
{
    if (m_initialized) {
        close();
    }
}

void MemoryStore::setSettingsManager(SettingsManager *settings)
{
    m_settingsManager = settings;
}

QByteArray MemoryStore::getEncryptionKey() const
{
    QByteArray salt = "DATAFORM_SALT_V1";
    QString passphrase;
    if (m_settingsManager && m_settingsManager->encryptionMode() == "machine_locked") {
        passphrase = CryptoUtil::legacyMachinePassphrase();
    } else {
        passphrase = CryptoUtil::machinePassphrase();
    }
    return CryptoUtil::deriveKey(passphrase, salt);
}

bool MemoryStore::initialize()
{
    if (m_initialized) return true;

    QString episodicDbPath = m_memoryPath + "/episodic.db";
    QString episodicEncPath = m_memoryPath + "/episodic.db.enc";
    QString traitsDbPath = m_memoryPath + "/traits.db";
    QString traitsEncPath = m_memoryPath + "/traits.db.enc";

    // Try to decrypt existing encrypted databases
    QByteArray key = getEncryptionKey();
    if (QFile::exists(episodicEncPath)) {
        decryptAndOpen(episodicEncPath, episodicDbPath);
    }
    if (QFile::exists(traitsEncPath)) {
        decryptAndOpen(traitsEncPath, traitsDbPath);
    }

    // Open databases (creates if not exist)
    if (!openDatabase("dataform_episodic", episodicDbPath, m_episodicDb)) {
        emit databaseError("Failed to open episodic database");
        return false;
    }
    if (!openDatabase("dataform_traits", traitsDbPath, m_traitsDb)) {
        emit databaseError("Failed to open traits database");
        return false;
    }

    // Create schemas (for new databases)
    if (!createEpisodicSchema()) {
        emit databaseError("Failed to create episodic schema");
        return false;
    }
    if (!createTraitsSchema()) {
        emit databaseError("Failed to create traits schema");
        return false;
    }

    // Run schema migrations (safe on new and existing databases)
    if (!migrateEpisodicSchema()) {
        qWarning() << "MemoryStore: Episodic DB migration failed (non-fatal)";
    }
    if (!migrateTraitsSchema()) {
        qWarning() << "MemoryStore: Traits DB migration failed (non-fatal)";
    }

    m_initialized = true;
    qDebug() << "MemoryStore initialized. Episodes:" << episodeCount()
             << "Traits:" << traitCount();
    return true;
}

void MemoryStore::close()
{
    if (!m_initialized) return;

    QString episodicDbPath = m_memoryPath + "/episodic.db";
    QString episodicEncPath = m_memoryPath + "/episodic.db.enc";
    QString traitsDbPath = m_memoryPath + "/traits.db";
    QString traitsEncPath = m_memoryPath + "/traits.db.enc";

    // Close database connections
    {
        QString episodicConnName = m_episodicDb.connectionName();
        QString traitsConnName = m_traitsDb.connectionName();
        m_episodicDb.close();
        m_traitsDb.close();
        m_episodicDb = QSqlDatabase();
        m_traitsDb = QSqlDatabase();
        QSqlDatabase::removeDatabase(episodicConnName);
        QSqlDatabase::removeDatabase(traitsConnName);
    }

    // Encrypt databases
    QByteArray key = getEncryptionKey();
    encryptAndClose(episodicDbPath, episodicEncPath);
    encryptAndClose(traitsDbPath, traitsEncPath);

    m_initialized = false;
    qDebug() << "MemoryStore closed and encrypted";
}

bool MemoryStore::flush()
{
    if (!m_initialized) return false;
    qDebug() << "MemoryStore: flushing (close + re-initialize)";
    close();
    return initialize();
}

bool MemoryStore::openDatabase(const QString &name, const QString &dbFilePath, QSqlDatabase &db)
{
    db = QSqlDatabase::addDatabase("QSQLITE", name);
    db.setDatabaseName(dbFilePath);

    if (!db.open()) {
        qWarning() << "Failed to open database" << name << ":" << db.lastError().text();
        return false;
    }

    // Enable WAL mode for better concurrency
    QSqlQuery query(db);
    query.exec("PRAGMA journal_mode=WAL");
    query.exec("PRAGMA synchronous=NORMAL");

    qDebug() << "Opened database:" << name << "at" << dbFilePath;
    return true;
}

bool MemoryStore::createEpisodicSchema()
{
    QSqlQuery query(m_episodicDb);

    bool ok = query.exec(
        "CREATE TABLE IF NOT EXISTS episodes ("
        "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  timestamp       TEXT NOT NULL DEFAULT (datetime('now')),"
        "  topic           TEXT DEFAULT '',"
        "  user_text       TEXT NOT NULL,"
        "  assistant_text  TEXT NOT NULL,"
        "  inquiry_text    TEXT DEFAULT '',"
        "  user_feedback   INTEGER DEFAULT 0,"
        "  user_edit_text  TEXT DEFAULT '',"
        "  edit_diff       TEXT DEFAULT '',"
        "  reasked         INTEGER DEFAULT 0,"
        "  accepted        INTEGER DEFAULT 1,"
        "  corrected       INTEGER DEFAULT 0,"
        "  tags            TEXT DEFAULT '',"
        "  model_id        TEXT DEFAULT '',"
        "  adapter_version TEXT DEFAULT ''"
        ")"
    );

    if (!ok) {
        qWarning() << "Failed to create episodes table:" << query.lastError().text();
        return false;
    }

    query.exec("CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp)");
    query.exec("CREATE INDEX IF NOT EXISTS idx_episodes_topic ON episodes(topic)");
    query.exec("CREATE INDEX IF NOT EXISTS idx_episodes_feedback ON episodes(user_feedback)");

    return true;
}

bool MemoryStore::createTraitsSchema()
{
    QSqlQuery query(m_traitsDb);

    bool ok = query.exec(
        "CREATE TABLE IF NOT EXISTS traits ("
        "  trait_id            TEXT PRIMARY KEY,"
        "  type                TEXT NOT NULL,"
        "  statement           TEXT NOT NULL,"
        "  confidence          REAL DEFAULT 0.0,"
        "  evidence_episode_ids TEXT DEFAULT '[]',"
        "  last_confirmed_ts   TEXT,"
        "  created_ts          TEXT NOT NULL DEFAULT (datetime('now')),"
        "  updated_ts          TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    );

    if (!ok) {
        qWarning() << "Failed to create traits table:" << query.lastError().text();
        return false;
    }

    query.exec("CREATE INDEX IF NOT EXISTS idx_traits_type ON traits(type)");
    query.exec("CREATE INDEX IF NOT EXISTS idx_traits_confidence ON traits(confidence)");

    return true;
}

void MemoryStore::encryptAndClose(const QString &dbFilePath, const QString &encFilePath)
{
    QByteArray key = getEncryptionKey();
    if (CryptoUtil::encryptFile(dbFilePath, encFilePath, key)) {
        QFile::remove(dbFilePath);
        // Also remove WAL and SHM files
        QFile::remove(dbFilePath + "-wal");
        QFile::remove(dbFilePath + "-shm");
        qDebug() << "Encrypted and removed plaintext:" << dbFilePath;
    } else {
        qWarning() << "Failed to encrypt database:" << dbFilePath;
    }
}

bool MemoryStore::decryptAndOpen(const QString &encFilePath, const QString &dbFilePath)
{
    QByteArray salt = "DATAFORM_SALT_V1";

    // Try current mode's key first
    QByteArray key = getEncryptionKey();
    if (CryptoUtil::decryptFile(encFilePath, dbFilePath, key)) {
        return true;
    }

    // Fallback: try the OTHER key (handles mode switches and legacy migration)
    QString altPassphrase;
    if (m_settingsManager && m_settingsManager->encryptionMode() == "machine_locked") {
        altPassphrase = CryptoUtil::machinePassphrase();  // try portable
    } else {
        altPassphrase = CryptoUtil::legacyMachinePassphrase();  // try machine
    }
    if (!altPassphrase.isEmpty()) {
        QByteArray altKey = CryptoUtil::deriveKey(altPassphrase, salt);
        if (CryptoUtil::decryptFile(encFilePath, dbFilePath, altKey)) {
            qDebug() << "Decrypted with alternate key (will re-encrypt with current mode):" << encFilePath;
            return true;
        }
    }

    return false;
}

// --- Episodic CRUD ---

qint64 MemoryStore::insertEpisode(const QString &topic,
                                    const QString &userText,
                                    const QString &assistantText,
                                    const QString &modelId,
                                    const QString &adapterVersion,
                                    qint64 conversationId)
{
    if (!m_initialized) return -1;

    QSqlQuery query(m_episodicDb);
    if (conversationId >= 0) {
        query.prepare(
            "INSERT INTO episodes (topic, user_text, assistant_text, model_id, adapter_version, conversation_id) "
            "VALUES (:topic, :user_text, :assistant_text, :model_id, :adapter_version, :conv_id)"
        );
        query.bindValue(":conv_id", conversationId);
    } else {
        query.prepare(
            "INSERT INTO episodes (topic, user_text, assistant_text, model_id, adapter_version) "
            "VALUES (:topic, :user_text, :assistant_text, :model_id, :adapter_version)"
        );
    }
    query.bindValue(":topic", topic);
    query.bindValue(":user_text", userText);
    query.bindValue(":assistant_text", assistantText);
    query.bindValue(":model_id", modelId);
    query.bindValue(":adapter_version", adapterVersion);

    if (!query.exec()) {
        qWarning() << "Insert episode failed:" << query.lastError().text();
        emit databaseError("Failed to insert episode: " + query.lastError().text());
        return -1;
    }

    qint64 id = query.lastInsertId().toLongLong();

    // Update conversation metadata
    if (conversationId >= 0) {
        QSqlQuery convUpdate(m_episodicDb);
        convUpdate.prepare(
            "UPDATE conversations SET last_activity_ts = datetime('now'), "
            "message_count = message_count + 1 WHERE id = :id"
        );
        convUpdate.bindValue(":id", conversationId);
        convUpdate.exec();
        emit conversationListChanged();
    }

    emit episodeInserted(id);
    emit episodeCountChanged();
    qDebug() << "Inserted episode" << id << "topic:" << topic;
    return id;
}

bool MemoryStore::updateEpisodeFeedback(qint64 episodeId, int feedback)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_episodicDb);
    query.prepare("UPDATE episodes SET user_feedback = :feedback WHERE id = :id");
    query.bindValue(":feedback", feedback);
    query.bindValue(":id", episodeId);

    if (!query.exec()) {
        qWarning() << "Update feedback failed:" << query.lastError().text();
        return false;
    }
    return query.numRowsAffected() > 0;
}

bool MemoryStore::updateEpisodeEdit(qint64 episodeId,
                                      const QString &editText,
                                      const QString &diff)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_episodicDb);
    query.prepare(
        "UPDATE episodes SET user_edit_text = :edit, edit_diff = :diff, corrected = 1 "
        "WHERE id = :id"
    );
    query.bindValue(":edit", editText);
    query.bindValue(":diff", diff);
    query.bindValue(":id", episodeId);

    if (!query.exec()) {
        qWarning() << "Update edit failed:" << query.lastError().text();
        return false;
    }
    return query.numRowsAffected() > 0;
}

bool MemoryStore::updateEpisodeInquiry(qint64 episodeId, const QString &inquiryText)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_episodicDb);
    query.prepare("UPDATE episodes SET inquiry_text = :inquiry WHERE id = :id");
    query.bindValue(":inquiry", inquiryText);
    query.bindValue(":id", episodeId);

    if (!query.exec()) {
        qWarning() << "Update inquiry failed:" << query.lastError().text();
        return false;
    }
    return query.numRowsAffected() > 0;
}

bool MemoryStore::updateEpisodeOutcome(qint64 episodeId,
                                         bool reasked, bool accepted, bool corrected)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_episodicDb);
    query.prepare(
        "UPDATE episodes SET reasked = :reasked, accepted = :accepted, corrected = :corrected "
        "WHERE id = :id"
    );
    query.bindValue(":reasked", reasked ? 1 : 0);
    query.bindValue(":accepted", accepted ? 1 : 0);
    query.bindValue(":corrected", corrected ? 1 : 0);
    query.bindValue(":id", episodeId);

    if (!query.exec()) {
        qWarning() << "Update outcome failed:" << query.lastError().text();
        return false;
    }
    return query.numRowsAffected() > 0;
}

EpisodicRecord MemoryStore::getEpisode(qint64 id) const
{
    EpisodicRecord record;
    if (!m_initialized) return record;

    QSqlQuery query(m_episodicDb);
    query.prepare("SELECT * FROM episodes WHERE id = :id");
    query.bindValue(":id", id);

    if (query.exec() && query.next()) {
        record.id = query.value("id").toLongLong();
        record.timestamp = QDateTime::fromString(query.value("timestamp").toString(), Qt::ISODate);
        record.topic = query.value("topic").toString();
        record.userText = query.value("user_text").toString();
        record.assistantText = query.value("assistant_text").toString();
        record.inquiryText = query.value("inquiry_text").toString();
        record.userFeedback = query.value("user_feedback").toInt();
        record.userEditText = query.value("user_edit_text").toString();
        record.editDiff = query.value("edit_diff").toString();
        record.reasked = query.value("reasked").toBool();
        record.accepted = query.value("accepted").toBool();
        record.corrected = query.value("corrected").toBool();
        record.tags = query.value("tags").toString();
        record.modelId = query.value("model_id").toString();
        record.adapterVersion = query.value("adapter_version").toString();
    }
    return record;
}

QList<EpisodicRecord> MemoryStore::getRecentEpisodes(int limit) const
{
    QList<EpisodicRecord> records;
    if (!m_initialized) return records;

    QSqlQuery query(m_episodicDb);
    query.prepare("SELECT * FROM episodes ORDER BY timestamp DESC LIMIT :limit");
    query.bindValue(":limit", limit);

    if (query.exec()) {
        while (query.next()) {
            EpisodicRecord r;
            r.id = query.value("id").toLongLong();
            r.timestamp = QDateTime::fromString(query.value("timestamp").toString(), Qt::ISODate);
            r.topic = query.value("topic").toString();
            r.userText = query.value("user_text").toString();
            r.assistantText = query.value("assistant_text").toString();
            r.inquiryText = query.value("inquiry_text").toString();
            r.userFeedback = query.value("user_feedback").toInt();
            r.userEditText = query.value("user_edit_text").toString();
            r.editDiff = query.value("edit_diff").toString();
            r.reasked = query.value("reasked").toBool();
            r.accepted = query.value("accepted").toBool();
            r.corrected = query.value("corrected").toBool();
            r.tags = query.value("tags").toString();
            r.modelId = query.value("model_id").toString();
            r.adapterVersion = query.value("adapter_version").toString();
            records.append(r);
        }
    }
    return records;
}

QList<EpisodicRecord> MemoryStore::searchEpisodes(const QString &query, int limit) const
{
    QList<EpisodicRecord> records;
    if (!m_initialized) return records;

    QSqlQuery q(m_episodicDb);
    q.prepare(
        "SELECT * FROM episodes WHERE user_text LIKE :query OR assistant_text LIKE :query "
        "ORDER BY timestamp DESC LIMIT :limit"
    );
    q.bindValue(":query", "%" + query + "%");
    q.bindValue(":limit", limit);

    if (q.exec()) {
        while (q.next()) {
            EpisodicRecord r;
            r.id = q.value("id").toLongLong();
            r.timestamp = QDateTime::fromString(q.value("timestamp").toString(), Qt::ISODate);
            r.topic = q.value("topic").toString();
            r.userText = q.value("user_text").toString();
            r.assistantText = q.value("assistant_text").toString();
            r.userFeedback = q.value("user_feedback").toInt();
            r.tags = q.value("tags").toString();
            records.append(r);
        }
    }
    return records;
}

QVariantList MemoryStore::searchConversationsForQml(const QString &query, int limit)
{
    QVariantList results;
    if (!m_initialized || query.trimmed().isEmpty()) return results;

    QSqlQuery q(m_episodicDb);
    q.prepare(
        "SELECT e.id, e.timestamp, e.user_text, e.assistant_text, e.conversation_id, e.topic, "
        "snippet(episodes_fts, 0, '<b>', '</b>', '...', 32) as snippet "
        "FROM episodes_fts fts "
        "JOIN episodes e ON e.id = fts.rowid "
        "WHERE episodes_fts MATCH :query "
        "ORDER BY rank LIMIT :limit"
    );
    q.bindValue(":query", query.trimmed());
    q.bindValue(":limit", limit);

    if (q.exec()) {
        while (q.next()) {
            QVariantMap item;
            item["episodeId"] = q.value("id").toLongLong();
            item["timestamp"] = q.value("timestamp").toString();
            item["userText"] = q.value("user_text").toString().left(100);
            item["assistantText"] = q.value("assistant_text").toString().left(100);
            item["conversationId"] = q.value("conversation_id").toLongLong();
            item["topic"] = q.value("topic").toString();
            item["snippet"] = q.value("snippet").toString();
            results.append(item);
        }
    } else {
        qWarning() << "MemoryStore: FTS search failed:" << q.lastError().text();
    }

    return results;
}

QVariantList MemoryStore::getTraitGrowthForQml(int daysPast)
{
    QVariantList results;
    if (!m_initialized) return results;

    QSqlQuery q(m_traitsDb);
    q.prepare(
        "SELECT date(created_ts) as day, COUNT(*) as new_count "
        "FROM traits "
        "WHERE created_ts >= datetime('now', :days) "
        "GROUP BY date(created_ts) "
        "ORDER BY day ASC"
    );
    q.bindValue(":days", QString("-%1 days").arg(daysPast));

    if (q.exec()) {
        int cumulative = 0;
        while (q.next()) {
            cumulative += q.value("new_count").toInt();
            QVariantMap item;
            item["date"] = q.value("day").toString();
            item["count"] = cumulative;
            results.append(item);
        }
    }
    return results;
}

QVariantList MemoryStore::getEpisodeActivityForQml(int daysPast)
{
    QVariantList results;
    if (!m_initialized) return results;

    QSqlQuery q(m_episodicDb);
    q.prepare(
        "SELECT date(timestamp) as day, COUNT(*) as count "
        "FROM episodes "
        "WHERE timestamp >= datetime('now', :days) "
        "GROUP BY date(timestamp) "
        "ORDER BY day ASC"
    );
    q.bindValue(":days", QString("-%1 days").arg(daysPast));

    if (q.exec()) {
        while (q.next()) {
            QVariantMap item;
            item["date"] = q.value("day").toString();
            item["count"] = q.value("count").toInt();
            results.append(item);
        }
    }
    return results;
}

QList<EpisodicRecord> MemoryStore::getHighSignalEpisodes(qint64 afterId, int limit) const
{
    QList<EpisodicRecord> records;
    if (!m_initialized) return records;

    QSqlQuery query(m_episodicDb);
    query.prepare(
        "SELECT * FROM episodes WHERE id > :afterId AND "
        "(user_feedback != 0 OR corrected = 1 OR inquiry_text != '') "
        "ORDER BY "
        "  CASE WHEN corrected = 1 THEN 3 "
        "       WHEN user_feedback != 0 THEN 2 "
        "       WHEN inquiry_text != '' THEN 1 "
        "       ELSE 0 END DESC, "
        "  id DESC "
        "LIMIT :limit"
    );
    query.bindValue(":afterId", afterId);
    query.bindValue(":limit", limit);

    if (query.exec()) {
        while (query.next()) {
            EpisodicRecord r;
            r.id = query.value("id").toLongLong();
            r.timestamp = QDateTime::fromString(query.value("timestamp").toString(), Qt::ISODate);
            r.topic = query.value("topic").toString();
            r.userText = query.value("user_text").toString();
            r.assistantText = query.value("assistant_text").toString();
            r.inquiryText = query.value("inquiry_text").toString();
            r.userFeedback = query.value("user_feedback").toInt();
            r.userEditText = query.value("user_edit_text").toString();
            r.editDiff = query.value("edit_diff").toString();
            r.reasked = query.value("reasked").toBool();
            r.accepted = query.value("accepted").toBool();
            r.corrected = query.value("corrected").toBool();
            r.tags = query.value("tags").toString();
            r.modelId = query.value("model_id").toString();
            r.adapterVersion = query.value("adapter_version").toString();
            records.append(r);
        }
    }
    return records;
}

QVariantList MemoryStore::getRecentEpisodesForQml(int limit) const
{
    QVariantList list;
    for (const EpisodicRecord &r : getRecentEpisodes(limit)) {
        list.append(episodeToVariantMap(r));
    }
    return list;
}

QVariantMap MemoryStore::episodeToVariantMap(const EpisodicRecord &record) const
{
    QVariantMap map;
    map["id"] = record.id;
    map["timestamp"] = record.timestamp.toString("yyyy-MM-dd HH:mm:ss");
    map["topic"] = record.topic;
    map["userText"] = record.userText;
    map["assistantText"] = record.assistantText;
    map["inquiryText"] = record.inquiryText;
    map["userFeedback"] = record.userFeedback;
    map["userEditText"] = record.userEditText;
    map["corrected"] = record.corrected;
    map["modelId"] = record.modelId;
    return map;
}

// --- Traits CRUD ---

QString MemoryStore::insertTrait(const QString &type,
                                   const QString &statement,
                                   double confidence)
{
    if (!m_initialized) return QString();

    QString traitId = QUuid::createUuid().toString(QUuid::WithoutBraces);

    QSqlQuery query(m_traitsDb);
    query.prepare(
        "INSERT INTO traits (trait_id, type, statement, confidence, last_confirmed_ts) "
        "VALUES (:id, :type, :statement, :confidence, datetime('now'))"
    );
    query.bindValue(":id", traitId);
    query.bindValue(":type", type);
    query.bindValue(":statement", statement);
    query.bindValue(":confidence", confidence);

    if (!query.exec()) {
        qWarning() << "Insert trait failed:" << query.lastError().text();
        emit databaseError("Failed to insert trait: " + query.lastError().text());
        return QString();
    }

    emit traitInserted(traitId);
    emit traitCountChanged();
    qDebug() << "Inserted trait" << traitId << ":" << statement;
    return traitId;
}

bool MemoryStore::updateTraitConfidence(const QString &traitId, double confidence)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_traitsDb);
    query.prepare(
        "UPDATE traits SET confidence = :conf, updated_ts = datetime('now') "
        "WHERE trait_id = :id"
    );
    query.bindValue(":conf", confidence);
    query.bindValue(":id", traitId);

    if (!query.exec()) {
        qWarning() << "Update trait confidence failed:" << query.lastError().text();
        return false;
    }
    return query.numRowsAffected() > 0;
}

bool MemoryStore::addTraitEvidence(const QString &traitId, qint64 episodeId)
{
    if (!m_initialized) return false;

    // Read current evidence list
    QSqlQuery query(m_traitsDb);
    query.prepare("SELECT evidence_episode_ids FROM traits WHERE trait_id = :id");
    query.bindValue(":id", traitId);

    if (!query.exec() || !query.next()) return false;

    QString jsonStr = query.value(0).toString();
    QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8());
    QJsonArray arr = doc.array();
    arr.append(episodeId);

    // Update with new evidence
    QSqlQuery update(m_traitsDb);
    update.prepare(
        "UPDATE traits SET evidence_episode_ids = :evidence, "
        "last_confirmed_ts = datetime('now'), updated_ts = datetime('now') "
        "WHERE trait_id = :id"
    );
    update.bindValue(":evidence", QJsonDocument(arr).toJson(QJsonDocument::Compact));
    update.bindValue(":id", traitId);

    if (!update.exec()) {
        qWarning() << "Add trait evidence failed:" << update.lastError().text();
        return false;
    }
    return update.numRowsAffected() > 0;
}

bool MemoryStore::confirmTrait(const QString &traitId)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_traitsDb);
    query.prepare(
        "UPDATE traits SET last_confirmed_ts = datetime('now'), updated_ts = datetime('now') "
        "WHERE trait_id = :id"
    );
    query.bindValue(":id", traitId);
    return query.exec() && query.numRowsAffected() > 0;
}

bool MemoryStore::removeTrait(const QString &traitId)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_traitsDb);
    query.prepare("DELETE FROM traits WHERE trait_id = :id");
    query.bindValue(":id", traitId);

    if (query.exec() && query.numRowsAffected() > 0) {
        emit traitCountChanged();
        return true;
    }
    return false;
}

TraitRecord MemoryStore::getTrait(const QString &traitId) const
{
    TraitRecord record;
    if (!m_initialized) return record;

    QSqlQuery query(m_traitsDb);
    query.prepare("SELECT * FROM traits WHERE trait_id = :id");
    query.bindValue(":id", traitId);

    if (query.exec() && query.next()) {
        record.traitId = query.value("trait_id").toString();
        record.type = query.value("type").toString();
        record.statement = query.value("statement").toString();
        record.confidence = query.value("confidence").toDouble();

        QString evidence = query.value("evidence_episode_ids").toString();
        QJsonDocument doc = QJsonDocument::fromJson(evidence.toUtf8());
        for (const QJsonValue &v : doc.array()) {
            record.evidenceEpisodeIds.append(QString::number(v.toInteger()));
        }

        record.lastConfirmedTs = QDateTime::fromString(
            query.value("last_confirmed_ts").toString(), Qt::ISODate);
        record.createdTs = QDateTime::fromString(
            query.value("created_ts").toString(), Qt::ISODate);
        record.updatedTs = QDateTime::fromString(
            query.value("updated_ts").toString(), Qt::ISODate);
    }
    return record;
}

QList<TraitRecord> MemoryStore::getAllTraits() const
{
    QList<TraitRecord> records;
    if (!m_initialized) return records;

    QSqlQuery query(m_traitsDb);
    if (query.exec("SELECT * FROM traits ORDER BY confidence DESC")) {
        while (query.next()) {
            TraitRecord r;
            r.traitId = query.value("trait_id").toString();
            r.type = query.value("type").toString();
            r.statement = query.value("statement").toString();
            r.confidence = query.value("confidence").toDouble();

            QString evidence = query.value("evidence_episode_ids").toString();
            QJsonDocument doc = QJsonDocument::fromJson(evidence.toUtf8());
            for (const QJsonValue &v : doc.array()) {
                r.evidenceEpisodeIds.append(QString::number(v.toInteger()));
            }

            r.lastConfirmedTs = QDateTime::fromString(
                query.value("last_confirmed_ts").toString(), Qt::ISODate);
            r.createdTs = QDateTime::fromString(
                query.value("created_ts").toString(), Qt::ISODate);
            r.updatedTs = QDateTime::fromString(
                query.value("updated_ts").toString(), Qt::ISODate);
            records.append(r);
        }
    }
    return records;
}

QList<TraitRecord> MemoryStore::getTraitsByType(const QString &type) const
{
    QList<TraitRecord> records;
    if (!m_initialized) return records;

    QSqlQuery query(m_traitsDb);
    query.prepare("SELECT * FROM traits WHERE type = :type ORDER BY confidence DESC");
    query.bindValue(":type", type);

    if (query.exec()) {
        while (query.next()) {
            TraitRecord r;
            r.traitId = query.value("trait_id").toString();
            r.type = query.value("type").toString();
            r.statement = query.value("statement").toString();
            r.confidence = query.value("confidence").toDouble();
            records.append(r);
        }
    }
    return records;
}

QVariantList MemoryStore::getAllTraitsForQml() const
{
    QVariantList list;
    for (const TraitRecord &r : getAllTraits()) {
        list.append(traitToVariantMap(r));
    }
    return list;
}

QVariantMap MemoryStore::traitToVariantMap(const TraitRecord &record) const
{
    QVariantMap map;
    map["traitId"] = record.traitId;
    map["type"] = record.type;
    map["statement"] = record.statement;
    map["confidence"] = record.confidence;
    map["evidenceCount"] = record.evidenceEpisodeIds.size();
    map["lastConfirmed"] = record.lastConfirmedTs.toString("yyyy-MM-dd HH:mm:ss");
    map["created"] = record.createdTs.toString("yyyy-MM-dd HH:mm:ss");
    return map;
}

int MemoryStore::episodeCount() const
{
    if (!m_initialized) return 0;
    QSqlQuery query(m_episodicDb);
    if (query.exec("SELECT COUNT(*) FROM episodes") && query.next()) {
        return query.value(0).toInt();
    }
    return 0;
}

int MemoryStore::traitCount() const
{
    if (!m_initialized) return 0;
    QSqlQuery query(m_traitsDb);
    if (query.exec("SELECT COUNT(*) FROM traits") && query.next()) {
        return query.value(0).toInt();
    }
    return 0;
}

// --- Conversation CRUD ---

qint64 MemoryStore::createConversation(const QString &title)
{
    if (!m_initialized) return -1;

    QSqlQuery query(m_episodicDb);
    query.prepare("INSERT INTO conversations (title) VALUES (:title)");
    query.bindValue(":title", title);

    if (!query.exec()) {
        qWarning() << "Create conversation failed:" << query.lastError().text();
        return -1;
    }

    qint64 id = query.lastInsertId().toLongLong();
    emit conversationCreated(id);
    emit conversationListChanged();
    qDebug() << "Created conversation" << id << ":" << title;
    return id;
}

bool MemoryStore::updateConversationTitle(qint64 id, const QString &title)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_episodicDb);
    query.prepare("UPDATE conversations SET title = :title WHERE id = :id");
    query.bindValue(":title", title);
    query.bindValue(":id", id);

    if (query.exec() && query.numRowsAffected() > 0) {
        emit conversationListChanged();
        return true;
    }
    return false;
}

bool MemoryStore::deleteConversation(qint64 id)
{
    if (!m_initialized) return false;

    // Nullify conversation_id on linked episodes (preserve for training)
    QSqlQuery unlink(m_episodicDb);
    unlink.prepare("UPDATE episodes SET conversation_id = NULL WHERE conversation_id = :id");
    unlink.bindValue(":id", id);
    unlink.exec();

    // Delete conversation row
    QSqlQuery query(m_episodicDb);
    query.prepare("DELETE FROM conversations WHERE id = :id");
    query.bindValue(":id", id);

    if (query.exec() && query.numRowsAffected() > 0) {
        emit conversationListChanged();
        qDebug() << "Deleted conversation" << id;
        return true;
    }
    return false;
}

QVariantList MemoryStore::getConversationsForQml(int limit)
{
    QVariantList list;
    if (!m_initialized) return list;

    QSqlQuery query(m_episodicDb);
    query.prepare(
        "SELECT c.id, c.title, c.created_ts, c.last_activity_ts, c.message_count, "
        "  (SELECT COUNT(*) FROM thoughts t WHERE t.conversation_id = c.id AND t.status = 1) AS is_proactive "
        "FROM conversations c WHERE c.archived = 0 "
        "ORDER BY c.last_activity_ts DESC LIMIT :limit"
    );
    query.bindValue(":limit", limit);

    if (query.exec()) {
        while (query.next()) {
            QVariantMap map;
            map["id"] = query.value("id").toLongLong();
            map["title"] = query.value("title").toString();
            map["createdTs"] = query.value("created_ts").toString();
            map["lastActivityTs"] = query.value("last_activity_ts").toString();
            map["messageCount"] = query.value("message_count").toInt();
            map["isProactive"] = query.value("is_proactive").toInt() > 0;
            list.append(map);
        }
    }
    return list;
}

QList<EpisodicRecord> MemoryStore::getEpisodesForConversation(qint64 id) const
{
    QList<EpisodicRecord> records;
    if (!m_initialized) return records;

    QSqlQuery query(m_episodicDb);
    query.prepare("SELECT * FROM episodes WHERE conversation_id = :id ORDER BY timestamp ASC");
    query.bindValue(":id", id);

    if (query.exec()) {
        while (query.next()) {
            EpisodicRecord r;
            r.id = query.value("id").toLongLong();
            r.timestamp = QDateTime::fromString(query.value("timestamp").toString(), Qt::ISODate);
            r.topic = query.value("topic").toString();
            r.userText = query.value("user_text").toString();
            r.assistantText = query.value("assistant_text").toString();
            r.inquiryText = query.value("inquiry_text").toString();
            r.userFeedback = query.value("user_feedback").toInt();
            r.userEditText = query.value("user_edit_text").toString();
            r.editDiff = query.value("edit_diff").toString();
            r.reasked = query.value("reasked").toBool();
            r.accepted = query.value("accepted").toBool();
            r.corrected = query.value("corrected").toBool();
            r.tags = query.value("tags").toString();
            r.modelId = query.value("model_id").toString();
            r.adapterVersion = query.value("adapter_version").toString();
            records.append(r);
        }
    }
    return records;
}

QVariantList MemoryStore::getEpisodesForConversationQml(qint64 id)
{
    QVariantList list;
    for (const EpisodicRecord &r : getEpisodesForConversation(id)) {
        list.append(episodeToVariantMap(r));
    }
    return list;
}

void MemoryStore::clearAllEpisodes()
{
    if (!m_initialized) return;
    QSqlQuery query(m_episodicDb);
    query.exec("DELETE FROM episodes");
    emit episodeCountChanged();
    qDebug() << "All episodes cleared";
}

void MemoryStore::clearAllTraits()
{
    if (!m_initialized) return;
    QSqlQuery query(m_traitsDb);
    query.exec("DELETE FROM traits");
    emit traitCountChanged();
    qDebug() << "All traits cleared";
}

void MemoryStore::clearAllMemory()
{
    clearAllEpisodes();
    clearAllTraits();
    qDebug() << "All memory cleared";
}

// --- Phase 4: Schema Migrations ---

bool MemoryStore::migrateEpisodicSchema()
{
    QList<SchemaMigrator::DbMigration> migrations;
    migrations.append({0, 1, "Add lifecycle columns (archived, summary, model_generation)",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);
            if (!q.exec("ALTER TABLE episodes ADD COLUMN archived INTEGER DEFAULT 0")) {
                if (!q.lastError().text().contains("duplicate column", Qt::CaseInsensitive))
                    return false;
            }
            if (!q.exec("ALTER TABLE episodes ADD COLUMN summary TEXT DEFAULT ''")) {
                if (!q.lastError().text().contains("duplicate column", Qt::CaseInsensitive))
                    return false;
            }
            if (!q.exec("ALTER TABLE episodes ADD COLUMN model_generation INTEGER DEFAULT 0")) {
                if (!q.lastError().text().contains("duplicate column", Qt::CaseInsensitive))
                    return false;
            }
            q.exec("CREATE INDEX IF NOT EXISTS idx_episodes_archived ON episodes(archived)");
            return true;
        }
    });

    // Phase 5: Create research_findings table
    migrations.append({1, 2, "Create research_findings table for autonomous research",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);
            bool ok = q.exec(
                "CREATE TABLE IF NOT EXISTS research_findings ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  timestamp TEXT NOT NULL DEFAULT (datetime('now')),"
                "  topic TEXT NOT NULL,"
                "  search_query TEXT DEFAULT '',"
                "  source_url TEXT DEFAULT '',"
                "  source_title TEXT DEFAULT '',"
                "  raw_snippet TEXT DEFAULT '',"
                "  llm_summary TEXT DEFAULT '',"
                "  relevance_reason TEXT DEFAULT '',"
                "  status INTEGER DEFAULT 0,"
                "  relevance_score REAL DEFAULT 0.0,"
                "  model_id TEXT DEFAULT ''"
                ")");
            if (!ok) return false;
            q.exec("CREATE INDEX IF NOT EXISTS idx_research_status ON research_findings(status)");
            q.exec("CREATE INDEX IF NOT EXISTS idx_research_topic ON research_findings(topic)");
            q.exec("CREATE INDEX IF NOT EXISTS idx_research_timestamp ON research_findings(timestamp)");
            return true;
        }
    });

    // Conversation history sidebar
    migrations.append({2, 3, "Create conversations table and link episodes",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);
            bool ok = q.exec(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "  id               INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  title            TEXT NOT NULL DEFAULT '',"
                "  created_ts       TEXT NOT NULL DEFAULT (datetime('now')),"
                "  last_activity_ts TEXT NOT NULL DEFAULT (datetime('now')),"
                "  message_count    INTEGER DEFAULT 0,"
                "  archived         INTEGER DEFAULT 0"
                ")");
            if (!ok) return false;
            q.exec("CREATE INDEX IF NOT EXISTS idx_conversations_last_activity ON conversations(last_activity_ts)");

            if (!q.exec("ALTER TABLE episodes ADD COLUMN conversation_id INTEGER DEFAULT NULL")) {
                if (!q.lastError().text().contains("duplicate column", Qt::CaseInsensitive))
                    return false;
            }
            q.exec("CREATE INDEX IF NOT EXISTS idx_episodes_conversation ON episodes(conversation_id)");
            return true;
        }
    });

    // Proactive dialog: thoughts table
    migrations.append({3, 4, "Create thoughts table for proactive dialog",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);
            bool ok = q.exec(
                "CREATE TABLE IF NOT EXISTS thoughts ("
                "  id               INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  timestamp        TEXT NOT NULL DEFAULT (datetime('now')),"
                "  type             TEXT NOT NULL,"
                "  title            TEXT NOT NULL DEFAULT '',"
                "  content          TEXT NOT NULL,"
                "  priority         REAL DEFAULT 0.5,"
                "  source_type      TEXT DEFAULT '',"
                "  source_id        INTEGER DEFAULT -1,"
                "  status           INTEGER DEFAULT 0,"
                "  conversation_id  INTEGER DEFAULT -1,"
                "  generated_by     TEXT DEFAULT ''"
                ")");
            if (!ok) return false;
            q.exec("CREATE INDEX IF NOT EXISTS idx_thoughts_status ON thoughts(status)");
            q.exec("CREATE INDEX IF NOT EXISTS idx_thoughts_priority ON thoughts(priority)");
            q.exec("CREATE INDEX IF NOT EXISTS idx_thoughts_timestamp ON thoughts(timestamp)");
            return true;
        }
    });

    // News Engine: news_seen table for headline dedup
    migrations.append({4, 5, "Create news_seen table for NewsEngine headline dedup",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);
            bool ok = q.exec(
                "CREATE TABLE IF NOT EXISTS news_seen ("
                "  id      INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  url     TEXT NOT NULL UNIQUE,"
                "  seen_ts TEXT NOT NULL DEFAULT (datetime('now'))"
                ")");
            if (!ok) return false;
            q.exec("CREATE INDEX IF NOT EXISTS idx_news_seen_url ON news_seen(url)");
            return true;
        }
    });

    // Agentic features: reminders, goals, sentiment, learning plans
    migrations.append({5, 6, "Create reminders, goals, sentiment_log, learning_plans tables",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);

            if (!q.exec(
                "CREATE TABLE IF NOT EXISTS reminders ("
                "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  content         TEXT NOT NULL,"
                "  due_ts          TEXT,"
                "  created_ts      TEXT NOT NULL DEFAULT (datetime('now')),"
                "  status          INTEGER DEFAULT 0,"
                "  source_episode_id INTEGER DEFAULT -1"
                ")")) return false;
            q.exec("CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status)");
            q.exec("CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_ts)");

            if (!q.exec(
                "CREATE TABLE IF NOT EXISTS goals ("
                "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  title           TEXT NOT NULL,"
                "  description     TEXT DEFAULT '',"
                "  status          INTEGER DEFAULT 0,"
                "  created_ts      TEXT NOT NULL DEFAULT (datetime('now')),"
                "  last_checkin_ts TEXT,"
                "  source_episode_id INTEGER DEFAULT -1,"
                "  checkin_count   INTEGER DEFAULT 0"
                ")")) return false;
            q.exec("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)");

            if (!q.exec(
                "CREATE TABLE IF NOT EXISTS sentiment_log ("
                "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  episode_id      INTEGER NOT NULL,"
                "  sentiment_score REAL DEFAULT 0.0,"
                "  energy_level    REAL DEFAULT 0.5,"
                "  dominant_emotion TEXT DEFAULT '',"
                "  analyzed_ts     TEXT NOT NULL DEFAULT (datetime('now'))"
                ")")) return false;
            q.exec("CREATE INDEX IF NOT EXISTS idx_sentiment_episode ON sentiment_log(episode_id)");

            if (!q.exec(
                "CREATE TABLE IF NOT EXISTS learning_plans ("
                "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  topic           TEXT NOT NULL,"
                "  plan_json       TEXT DEFAULT '{}',"
                "  current_lesson  INTEGER DEFAULT 0,"
                "  total_lessons   INTEGER DEFAULT 0,"
                "  status          INTEGER DEFAULT 0,"
                "  created_ts      TEXT NOT NULL DEFAULT (datetime('now')),"
                "  last_session_ts TEXT"
                ")")) return false;
            q.exec("CREATE INDEX IF NOT EXISTS idx_learning_status ON learning_plans(status)");

            return true;
        }
    });

    // FTS5 full-text search for conversation search
    migrations.append({6, 7, "Create FTS5 virtual table and triggers for episode search",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);

            // FTS5 virtual table
            if (!q.exec(
                "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5("
                "  user_text, assistant_text, topic,"
                "  content=episodes, content_rowid=id"
                ")")) return false;

            // Populate from existing data
            q.exec("INSERT INTO episodes_fts(rowid, user_text, assistant_text, topic) "
                   "SELECT id, user_text, assistant_text, topic FROM episodes");

            // Sync triggers
            if (!q.exec(
                "CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN "
                "  INSERT INTO episodes_fts(rowid, user_text, assistant_text, topic) "
                "  VALUES (new.id, new.user_text, new.assistant_text, new.topic); "
                "END")) return false;

            if (!q.exec(
                "CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN "
                "  INSERT INTO episodes_fts(episodes_fts, rowid, user_text, assistant_text, topic) "
                "  VALUES ('delete', old.id, old.user_text, old.assistant_text, old.topic); "
                "END")) return false;

            if (!q.exec(
                "CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN "
                "  INSERT INTO episodes_fts(episodes_fts, rowid, user_text, assistant_text, topic) "
                "  VALUES ('delete', old.id, old.user_text, old.assistant_text, old.topic); "
                "  INSERT INTO episodes_fts(rowid, user_text, assistant_text, topic) "
                "  VALUES (new.id, new.user_text, new.assistant_text, new.topic); "
                "END")) return false;

            return true;
        }
    });

    // Phase 7: Create embeddings table for semantic vector search
    migrations.append({7, 8, "Create embeddings table for semantic vector search",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);

            if (!q.exec(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  source_type     TEXT NOT NULL,"
                "  source_id       INTEGER NOT NULL,"
                "  embedding_model TEXT NOT NULL DEFAULT '',"
                "  embedding_blob  BLOB,"
                "  vector_dimension INTEGER DEFAULT 0,"
                "  created_ts      TEXT NOT NULL DEFAULT (datetime('now')),"
                "  UNIQUE(source_type, source_id)"
                ")")) return false;

            q.exec("CREATE INDEX IF NOT EXISTS idx_embeddings_source "
                   "ON embeddings(source_type, source_id)");
            q.exec("CREATE INDEX IF NOT EXISTS idx_embeddings_model "
                   "ON embeddings(embedding_model)");

            return true;
        }
    });

    return SchemaMigrator::migrateDatabase(m_episodicDb, migrations,
                                            SchemaMigrator::EPISODIC_DB_VERSION);
}

bool MemoryStore::migrateTraitsSchema()
{
    QList<SchemaMigrator::DbMigration> migrations;
    migrations.append({0, 1, "Add lifecycle columns (decay_rate, archived, source_generation)",
        [](QSqlDatabase &db) -> bool {
            QSqlQuery q(db);
            if (!q.exec("ALTER TABLE traits ADD COLUMN decay_rate REAL DEFAULT 0.0")) {
                if (!q.lastError().text().contains("duplicate column", Qt::CaseInsensitive))
                    return false;
            }
            if (!q.exec("ALTER TABLE traits ADD COLUMN archived INTEGER DEFAULT 0")) {
                if (!q.lastError().text().contains("duplicate column", Qt::CaseInsensitive))
                    return false;
            }
            if (!q.exec("ALTER TABLE traits ADD COLUMN source_generation INTEGER DEFAULT 0")) {
                if (!q.lastError().text().contains("duplicate column", Qt::CaseInsensitive))
                    return false;
            }
            return true;
        }
    });

    return SchemaMigrator::migrateDatabase(m_traitsDb, migrations,
                                            SchemaMigrator::TRAITS_DB_VERSION);
}

// --- Reminders CRUD ---

qint64 MemoryStore::insertReminder(const QString &content, const QString &dueTsStr, qint64 episodeId)
{
    if (!m_initialized) return -1;
    QSqlQuery q(m_episodicDb);
    q.prepare("INSERT INTO reminders (content, due_ts, source_episode_id) VALUES (?, ?, ?)");
    q.addBindValue(content);
    q.addBindValue(dueTsStr.isEmpty() ? QVariant() : dueTsStr);
    q.addBindValue(episodeId);
    if (q.exec()) {
        qint64 id = q.lastInsertId().toLongLong();
        qDebug() << "Reminder inserted:" << id << content.left(50);
        return id;
    }
    qWarning() << "Failed to insert reminder:" << q.lastError().text();
    return -1;
}

QList<ReminderRecord> MemoryStore::getDueReminders() const
{
    QList<ReminderRecord> results;
    if (!m_initialized) return results;
    QSqlQuery q(m_episodicDb);
    q.prepare("SELECT id, content, due_ts, created_ts, status, source_episode_id "
              "FROM reminders WHERE status = 0 AND due_ts IS NOT NULL "
              "AND due_ts <= datetime('now') ORDER BY due_ts ASC");
    if (q.exec()) {
        while (q.next()) {
            ReminderRecord r;
            r.id = q.value(0).toLongLong();
            r.content = q.value(1).toString();
            r.dueTs = QDateTime::fromString(q.value(2).toString(), Qt::ISODate);
            r.createdTs = QDateTime::fromString(q.value(3).toString(), Qt::ISODate);
            r.status = q.value(4).toInt();
            r.sourceEpisodeId = q.value(5).toLongLong();
            results.append(r);
        }
    }
    return results;
}

QList<ReminderRecord> MemoryStore::getPendingReminders() const
{
    QList<ReminderRecord> results;
    if (!m_initialized) return results;
    QSqlQuery q(m_episodicDb);
    q.prepare("SELECT id, content, due_ts, created_ts, status, source_episode_id "
              "FROM reminders WHERE status = 0 ORDER BY created_ts DESC LIMIT 20");
    if (q.exec()) {
        while (q.next()) {
            ReminderRecord r;
            r.id = q.value(0).toLongLong();
            r.content = q.value(1).toString();
            r.dueTs = QDateTime::fromString(q.value(2).toString(), Qt::ISODate);
            r.createdTs = QDateTime::fromString(q.value(3).toString(), Qt::ISODate);
            r.status = q.value(4).toInt();
            r.sourceEpisodeId = q.value(5).toLongLong();
            results.append(r);
        }
    }
    return results;
}

bool MemoryStore::updateReminderStatus(qint64 id, int status)
{
    if (!m_initialized) return false;
    QSqlQuery q(m_episodicDb);
    q.prepare("UPDATE reminders SET status = ? WHERE id = ?");
    q.addBindValue(status);
    q.addBindValue(id);
    return q.exec();
}

QVariantList MemoryStore::getPendingRemindersForQml()
{
    QVariantList list;
    for (const auto &r : getPendingReminders()) {
        QVariantMap map;
        map["id"] = r.id;
        map["content"] = r.content;
        map["dueTs"] = r.dueTs.isValid() ? r.dueTs.toString("yyyy-MM-dd HH:mm") : "No date";
        map["createdTs"] = r.createdTs.toString("yyyy-MM-dd HH:mm");
        map["status"] = r.status;
        list.append(map);
    }
    return list;
}

// --- Goals CRUD ---

qint64 MemoryStore::insertGoal(const QString &title, const QString &description, qint64 episodeId)
{
    if (!m_initialized) return -1;
    QSqlQuery q(m_episodicDb);
    q.prepare("INSERT INTO goals (title, description, source_episode_id) VALUES (?, ?, ?)");
    q.addBindValue(title);
    q.addBindValue(description);
    q.addBindValue(episodeId);
    if (q.exec()) {
        qint64 id = q.lastInsertId().toLongLong();
        qDebug() << "Goal inserted:" << id << title;
        return id;
    }
    qWarning() << "Failed to insert goal:" << q.lastError().text();
    return -1;
}

QList<GoalRecord> MemoryStore::getActiveGoals() const
{
    QList<GoalRecord> results;
    if (!m_initialized) return results;
    QSqlQuery q(m_episodicDb);
    q.prepare("SELECT id, title, description, status, created_ts, last_checkin_ts, "
              "source_episode_id, checkin_count FROM goals WHERE status = 0 "
              "ORDER BY created_ts DESC");
    if (q.exec()) {
        while (q.next()) {
            GoalRecord g;
            g.id = q.value(0).toLongLong();
            g.title = q.value(1).toString();
            g.description = q.value(2).toString();
            g.status = q.value(3).toInt();
            g.createdTs = QDateTime::fromString(q.value(4).toString(), Qt::ISODate);
            g.lastCheckinTs = QDateTime::fromString(q.value(5).toString(), Qt::ISODate);
            g.sourceEpisodeId = q.value(6).toLongLong();
            g.checkinCount = q.value(7).toInt();
            results.append(g);
        }
    }
    return results;
}

bool MemoryStore::updateGoalStatus(qint64 id, int status)
{
    if (!m_initialized) return false;
    QSqlQuery q(m_episodicDb);
    q.prepare("UPDATE goals SET status = ? WHERE id = ?");
    q.addBindValue(status);
    q.addBindValue(id);
    return q.exec();
}

bool MemoryStore::updateGoalCheckin(qint64 id)
{
    if (!m_initialized) return false;
    QSqlQuery q(m_episodicDb);
    q.prepare("UPDATE goals SET last_checkin_ts = datetime('now'), "
              "checkin_count = checkin_count + 1 WHERE id = ?");
    q.addBindValue(id);
    return q.exec();
}

QList<GoalRecord> MemoryStore::getGoalsNeedingCheckin(int daysSince) const
{
    QList<GoalRecord> results;
    if (!m_initialized) return results;
    QSqlQuery q(m_episodicDb);
    q.prepare("SELECT id, title, description, status, created_ts, last_checkin_ts, "
              "source_episode_id, checkin_count FROM goals WHERE status = 0 "
              "AND (last_checkin_ts IS NULL OR "
              "last_checkin_ts < datetime('now', ? || ' days')) "
              "ORDER BY last_checkin_ts ASC LIMIT 5");
    q.addBindValue(QString("-%1").arg(daysSince));
    if (q.exec()) {
        while (q.next()) {
            GoalRecord g;
            g.id = q.value(0).toLongLong();
            g.title = q.value(1).toString();
            g.description = q.value(2).toString();
            g.status = q.value(3).toInt();
            g.createdTs = QDateTime::fromString(q.value(4).toString(), Qt::ISODate);
            g.lastCheckinTs = QDateTime::fromString(q.value(5).toString(), Qt::ISODate);
            g.sourceEpisodeId = q.value(6).toLongLong();
            g.checkinCount = q.value(7).toInt();
            results.append(g);
        }
    }
    return results;
}

QVariantList MemoryStore::getActiveGoalsForQml()
{
    QVariantList list;
    for (const auto &g : getActiveGoals()) {
        QVariantMap map;
        map["id"] = g.id;
        map["title"] = g.title;
        map["description"] = g.description;
        map["status"] = g.status;
        map["createdTs"] = g.createdTs.toString("yyyy-MM-dd");
        map["lastCheckinTs"] = g.lastCheckinTs.isValid() ? g.lastCheckinTs.toString("yyyy-MM-dd") : "Never";
        map["checkinCount"] = g.checkinCount;
        list.append(map);
    }
    return list;
}

// --- Sentiment CRUD ---

qint64 MemoryStore::insertSentiment(qint64 episodeId, double score, double energy, const QString &emotion)
{
    if (!m_initialized) return -1;
    QSqlQuery q(m_episodicDb);
    q.prepare("INSERT INTO sentiment_log (episode_id, sentiment_score, energy_level, dominant_emotion) "
              "VALUES (?, ?, ?, ?)");
    q.addBindValue(episodeId);
    q.addBindValue(score);
    q.addBindValue(energy);
    q.addBindValue(emotion);
    if (q.exec()) return q.lastInsertId().toLongLong();
    return -1;
}

bool MemoryStore::hasEpisodeSentiment(qint64 episodeId) const
{
    if (!m_initialized) return false;
    QSqlQuery q(m_episodicDb);
    q.prepare("SELECT COUNT(*) FROM sentiment_log WHERE episode_id = ?");
    q.addBindValue(episodeId);
    if (q.exec() && q.next()) return q.value(0).toInt() > 0;
    return false;
}

QList<SentimentRecord> MemoryStore::getRecentSentiment(int limit) const
{
    QList<SentimentRecord> results;
    if (!m_initialized) return results;
    QSqlQuery q(m_episodicDb);
    q.prepare("SELECT id, episode_id, sentiment_score, energy_level, dominant_emotion, analyzed_ts "
              "FROM sentiment_log ORDER BY analyzed_ts DESC LIMIT ?");
    q.addBindValue(limit);
    if (q.exec()) {
        while (q.next()) {
            SentimentRecord s;
            s.id = q.value(0).toLongLong();
            s.episodeId = q.value(1).toLongLong();
            s.sentimentScore = q.value(2).toDouble();
            s.energyLevel = q.value(3).toDouble();
            s.dominantEmotion = q.value(4).toString();
            s.analyzedTs = QDateTime::fromString(q.value(5).toString(), Qt::ISODate);
            results.append(s);
        }
    }
    return results;
}

double MemoryStore::getAverageSentiment(int daysPast) const
{
    if (!m_initialized) return 0.0;
    QSqlQuery q(m_episodicDb);
    q.prepare("SELECT AVG(sentiment_score) FROM sentiment_log "
              "WHERE analyzed_ts >= datetime('now', ? || ' days')");
    q.addBindValue(QString("-%1").arg(daysPast));
    if (q.exec() && q.next()) return q.value(0).toDouble();
    return 0.0;
}

// --- Learning Plans CRUD ---

qint64 MemoryStore::insertLearningPlan(const QString &topic, const QString &planJson, int totalLessons)
{
    if (!m_initialized) return -1;
    QSqlQuery q(m_episodicDb);
    q.prepare("INSERT INTO learning_plans (topic, plan_json, total_lessons) VALUES (?, ?, ?)");
    q.addBindValue(topic);
    q.addBindValue(planJson);
    q.addBindValue(totalLessons);
    if (q.exec()) {
        qint64 id = q.lastInsertId().toLongLong();
        qDebug() << "Learning plan created:" << id << topic;
        return id;
    }
    qWarning() << "Failed to insert learning plan:" << q.lastError().text();
    return -1;
}

QList<LearningPlanRecord> MemoryStore::getActiveLearningPlans() const
{
    QList<LearningPlanRecord> results;
    if (!m_initialized) return results;
    QSqlQuery q(m_episodicDb);
    q.prepare("SELECT id, topic, plan_json, current_lesson, total_lessons, status, "
              "created_ts, last_session_ts FROM learning_plans WHERE status = 0 "
              "ORDER BY created_ts DESC");
    if (q.exec()) {
        while (q.next()) {
            LearningPlanRecord lp;
            lp.id = q.value(0).toLongLong();
            lp.topic = q.value(1).toString();
            lp.planJson = q.value(2).toString();
            lp.currentLesson = q.value(3).toInt();
            lp.totalLessons = q.value(4).toInt();
            lp.status = q.value(5).toInt();
            lp.createdTs = QDateTime::fromString(q.value(6).toString(), Qt::ISODate);
            lp.lastSessionTs = QDateTime::fromString(q.value(7).toString(), Qt::ISODate);
            results.append(lp);
        }
    }
    return results;
}

bool MemoryStore::advanceLessonPlan(qint64 id)
{
    if (!m_initialized) return false;
    QSqlQuery q(m_episodicDb);
    q.prepare("UPDATE learning_plans SET current_lesson = current_lesson + 1, "
              "last_session_ts = datetime('now') WHERE id = ?");
    q.addBindValue(id);
    if (!q.exec()) return false;
    // Check if completed
    q.prepare("UPDATE learning_plans SET status = 1 WHERE id = ? "
              "AND current_lesson >= total_lessons");
    q.addBindValue(id);
    q.exec();
    return true;
}

bool MemoryStore::updateLearningPlanStatus(qint64 id, int status)
{
    if (!m_initialized) return false;
    QSqlQuery q(m_episodicDb);
    q.prepare("UPDATE learning_plans SET status = ? WHERE id = ?");
    q.addBindValue(status);
    q.addBindValue(id);
    return q.exec();
}

QVariantList MemoryStore::getActivePlansForQml()
{
    QVariantList list;
    for (const auto &lp : getActiveLearningPlans()) {
        QVariantMap map;
        map["id"] = lp.id;
        map["topic"] = lp.topic;
        map["currentLesson"] = lp.currentLesson;
        map["totalLessons"] = lp.totalLessons;
        map["status"] = lp.status;
        map["createdTs"] = lp.createdTs.toString("yyyy-MM-dd");
        list.append(map);
    }
    return list;
}

// --- Phase 4: Lifecycle Query Methods ---

QList<EpisodicRecord> MemoryStore::getEpisodesOlderThan(int ageDays,
                                                          bool excludeArchived,
                                                          int limit) const
{
    QList<EpisodicRecord> records;
    if (!m_initialized) return records;

    // Note: datetime() modifier must be a string literal, can't use parameter binding.
    // ageDays is always an internal int, not user input, so .arg() is safe here.
    QString sql = "SELECT * FROM episodes WHERE "
                  "timestamp < datetime('now', '-%1 days')";
    if (excludeArchived) {
        sql += " AND archived = 0";
    }
    sql += " ORDER BY timestamp ASC LIMIT :limit";

    QSqlQuery query(m_episodicDb);
    query.prepare(sql.arg(ageDays));
    query.bindValue(":limit", limit);

    if (query.exec()) {
        while (query.next()) {
            EpisodicRecord r;
            r.id = query.value("id").toLongLong();
            r.timestamp = QDateTime::fromString(query.value("timestamp").toString(), Qt::ISODate);
            r.topic = query.value("topic").toString();
            r.userText = query.value("user_text").toString();
            r.assistantText = query.value("assistant_text").toString();
            r.inquiryText = query.value("inquiry_text").toString();
            r.userFeedback = query.value("user_feedback").toInt();
            r.userEditText = query.value("user_edit_text").toString();
            r.editDiff = query.value("edit_diff").toString();
            r.reasked = query.value("reasked").toBool();
            r.accepted = query.value("accepted").toBool();
            r.corrected = query.value("corrected").toBool();
            records.append(r);
        }
    }
    return records;
}

bool MemoryStore::archiveEpisode(qint64 episodeId, const QString &summary)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_episodicDb);
    query.prepare(
        "UPDATE episodes SET archived = 1, summary = :summary, "
        "user_text = '', assistant_text = '', user_edit_text = '', edit_diff = '', "
        "inquiry_text = '' "
        "WHERE id = :id AND archived = 0"
    );
    query.bindValue(":summary", summary);
    query.bindValue(":id", episodeId);

    if (!query.exec()) {
        qWarning() << "Archive episode failed:" << query.lastError().text();
        return false;
    }
    return query.numRowsAffected() > 0;
}

QList<TraitRecord> MemoryStore::getUnconfirmedTraitsSince(int daysSince) const
{
    QList<TraitRecord> records;
    if (!m_initialized) return records;

    QSqlQuery query(m_traitsDb);
    query.prepare(
        "SELECT * FROM traits WHERE archived = 0 AND "
        "(last_confirmed_ts IS NULL OR last_confirmed_ts < datetime('now', :days))"
    );
    query.bindValue(":days", QString("-%1 days").arg(daysSince));

    if (query.exec()) {
        while (query.next()) {
            TraitRecord r;
            r.traitId = query.value("trait_id").toString();
            r.type = query.value("type").toString();
            r.statement = query.value("statement").toString();
            r.confidence = query.value("confidence").toDouble();
            r.lastConfirmedTs = QDateTime::fromString(
                query.value("last_confirmed_ts").toString(), Qt::ISODate);
            r.createdTs = QDateTime::fromString(
                query.value("created_ts").toString(), Qt::ISODate);
            records.append(r);
        }
    }
    return records;
}

bool MemoryStore::updateTraitDecay(const QString &traitId, double newConfidence)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_traitsDb);
    query.prepare(
        "UPDATE traits SET confidence = :conf, updated_ts = datetime('now') "
        "WHERE trait_id = :id"
    );
    query.bindValue(":conf", newConfidence);
    query.bindValue(":id", traitId);

    return query.exec() && query.numRowsAffected() > 0;
}

bool MemoryStore::archiveTrait(const QString &traitId)
{
    if (!m_initialized) return false;

    QSqlQuery query(m_traitsDb);
    query.prepare("UPDATE traits SET archived = 1, updated_ts = datetime('now') WHERE trait_id = :id");
    query.bindValue(":id", traitId);

    if (query.exec() && query.numRowsAffected() > 0) {
        emit traitCountChanged();
        return true;
    }
    return false;
}

int MemoryStore::archivedEpisodeCount() const
{
    if (!m_initialized) return 0;
    QSqlQuery query(m_episodicDb);
    if (query.exec("SELECT COUNT(*) FROM episodes WHERE archived = 1") && query.next()) {
        return query.value(0).toInt();
    }
    return 0;
}

int MemoryStore::archivedTraitCount() const
{
    if (!m_initialized) return 0;
    QSqlQuery query(m_traitsDb);
    if (query.exec("SELECT COUNT(*) FROM traits WHERE archived = 1") && query.next()) {
        return query.value(0).toInt();
    }
    return 0;
}
