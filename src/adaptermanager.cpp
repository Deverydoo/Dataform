#ifdef DATAFORM_TRAINING_ENABLED

#include "adaptermanager.h"
#include "profilemanager.h"
#include "evalsuite.h"
#include "schemamigrator.h"
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDebug>

// --- AdapterMetadata JSON serialization ---

QJsonObject AdapterMetadata::toJson() const
{
    QJsonObject obj;
    obj["version"] = version;
    obj["adapterName"] = adapterName;
    obj["trainingDate"] = trainingDate.toString(Qt::ISODate);
    obj["firstEpisodeId"] = static_cast<qint64>(firstEpisodeId);
    obj["lastEpisodeId"] = static_cast<qint64>(lastEpisodeId);
    obj["trainingSteps"] = trainingSteps;
    obj["finalLoss"] = static_cast<double>(finalLoss);
    obj["evalScore"] = evalScore;
    obj["parentVersion"] = parentVersion;
    obj["status"] = status;
    obj["checkpointPath"] = checkpointPath;
    obj["exportedModelPath"] = exportedModelPath;
    obj["origin"] = origin;
    obj["generationId"] = generationId;
    obj["cycleId"] = cycleId;
    obj["variantIndex"] = variantIndex;
    return obj;
}

AdapterMetadata AdapterMetadata::fromJson(const QJsonObject &json)
{
    AdapterMetadata meta;
    meta.version = json["version"].toInt();
    meta.adapterName = json["adapterName"].toString("communication_style");
    meta.trainingDate = QDateTime::fromString(json["trainingDate"].toString(), Qt::ISODate);
    meta.firstEpisodeId = json["firstEpisodeId"].toInteger();
    meta.lastEpisodeId = json["lastEpisodeId"].toInteger();
    meta.trainingSteps = json["trainingSteps"].toInt();
    meta.finalLoss = static_cast<float>(json["finalLoss"].toDouble());
    meta.evalScore = json["evalScore"].toDouble();
    meta.parentVersion = json["parentVersion"].toInt(-1);
    meta.status = json["status"].toString("unknown");
    meta.checkpointPath = json["checkpointPath"].toString();
    meta.exportedModelPath = json["exportedModelPath"].toString();
    meta.origin = json["origin"].toString();
    meta.generationId = json["generationId"].toInt(0);
    meta.cycleId = json["cycleId"].toInt(-1);
    meta.variantIndex = json["variantIndex"].toInt(-1);
    return meta;
}

// --- AdapterManager ---

AdapterManager::AdapterManager(QObject *parent)
    : QObject(parent)
{
}

AdapterManager::~AdapterManager() = default;

void AdapterManager::setProfileManager(ProfileManager *profileManager)
{
    m_profileManager = profileManager;
}

void AdapterManager::setEvalSuite(EvalSuite *evalSuite)
{
    m_evalSuite = evalSuite;
}

QString AdapterManager::activeAdapterName() const
{
    if (m_activeVersion < 0) return "Base model";
    for (const auto &meta : m_versions) {
        if (meta.version == m_activeVersion) return meta.adapterName;
    }
    return "Unknown";
}

QString AdapterManager::versionDirPath(int version) const
{
    if (!m_profileManager) return {};
    return m_profileManager->adapterVersionsPath() +
           QString("/v%1").arg(version, 3, 10, QChar('0'));
}

QString AdapterManager::activeDirPath() const
{
    if (!m_profileManager) return {};
    return m_profileManager->activeAdaptersPath();
}

int AdapterManager::registerCandidate(const AdapterMetadata &metadata)
{
    AdapterMetadata meta = metadata;
    meta.version = m_nextVersionNumber++;
    meta.status = "candidate";

    // Create version directory
    QString vDir = versionDirPath(meta.version);
    QDir().mkpath(vDir);

    // Save metadata
    QFile metaFile(vDir + "/metadata.json");
    if (metaFile.open(QIODevice::WriteOnly)) {
        metaFile.write(QJsonDocument(meta.toJson()).toJson(QJsonDocument::Indented));
        metaFile.close();
    }

    m_versions.append(meta);
    emit versionCountChanged();
    emit candidateRegistered(meta.version);

    setAdapterStatus(QString("Candidate v%1 registered").arg(meta.version));
    qDebug() << "AdapterManager: registered candidate v" << meta.version;

    saveVersions();
    return meta.version;
}

AdapterMetadata AdapterManager::activeAdapterMetadata() const
{
    for (const auto &meta : m_versions) {
        if (meta.version == m_activeVersion) return meta;
    }
    // Return default (no adapter)
    AdapterMetadata empty;
    empty.version = -1;
    empty.status = "none";
    return empty;
}

AdapterMetadata AdapterManager::getVersionMetadata(int version) const
{
    for (const auto &meta : m_versions) {
        if (meta.version == version) return meta;
    }
    return {};
}

bool AdapterManager::atomicSwapActive(int newVersion)
{
    QString activeDir = activeDirPath();
    QDir().mkpath(activeDir);

    // Write new active pointer via temp file + rename (atomic on same volume)
    QString pointerFile = activeDir + "/active_version.json";
    QString tempFile = activeDir + "/active_version.json.tmp";

    QJsonObject pointer;
    pointer["version"] = newVersion;
    pointer["activated_at"] = QDateTime::currentDateTime().toString(Qt::ISODate);

    QFile temp(tempFile);
    if (!temp.open(QIODevice::WriteOnly)) {
        qWarning() << "AdapterManager: cannot write temp pointer file";
        return false;
    }
    temp.write(QJsonDocument(pointer).toJson());
    temp.close();

    QFile::remove(pointerFile);
    return QFile::rename(tempFile, pointerFile);
}

bool AdapterManager::promoteVersion(int version)
{
    // Find the candidate
    int idx = -1;
    for (int i = 0; i < m_versions.size(); ++i) {
        if (m_versions[i].version == version) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        qWarning() << "AdapterManager: version" << version << "not found";
        return false;
    }

    // Demote current active
    if (m_activeVersion >= 0) {
        for (auto &meta : m_versions) {
            if (meta.version == m_activeVersion) {
                meta.status = "archived";
                break;
            }
        }
    }

    // Promote new version
    m_previousActiveVersion = m_activeVersion;
    m_versions[idx].status = "active";

    if (!atomicSwapActive(version)) {
        qWarning() << "AdapterManager: atomic swap failed for v" << version;
        return false;
    }

    m_activeVersion = version;
    emit activeVersionChanged();

    setAdapterStatus(QString("Active: v%1 (score: %2)")
        .arg(version)
        .arg(m_versions[idx].evalScore, 0, 'f', 2));
    emit adapterPromoted(version);

    qDebug() << "AdapterManager: promoted v" << version;
    saveVersions();
    return true;
}

bool AdapterManager::rollback()
{
    if (m_previousActiveVersion < 0) {
        // Rollback to base model
        if (m_activeVersion >= 0) {
            for (auto &meta : m_versions) {
                if (meta.version == m_activeVersion) {
                    meta.status = "archived";
                    break;
                }
            }
        }
        m_activeVersion = -1;
        atomicSwapActive(-1);
        emit activeVersionChanged();
        setAdapterStatus("Rolled back to base model");
        emit adapterRolledBack(-1);
        saveVersions();
        return true;
    }

    int rollbackTo = m_previousActiveVersion;

    // Demote current active
    if (m_activeVersion >= 0) {
        for (auto &meta : m_versions) {
            if (meta.version == m_activeVersion) {
                meta.status = "rejected";
                break;
            }
        }
    }

    // Restore previous
    for (auto &meta : m_versions) {
        if (meta.version == rollbackTo) {
            meta.status = "active";
            break;
        }
    }

    m_activeVersion = rollbackTo;
    m_previousActiveVersion = -1;
    atomicSwapActive(rollbackTo);
    emit activeVersionChanged();

    setAdapterStatus(QString("Rolled back to v%1").arg(rollbackTo));
    emit adapterRolledBack(rollbackTo);

    qDebug() << "AdapterManager: rolled back to v" << rollbackTo;
    saveVersions();
    return true;
}

void AdapterManager::pruneVersions(int keepCount)
{
    if (m_versions.size() <= keepCount) return;

    // Sort by version number descending
    QList<AdapterMetadata> sorted = m_versions;
    std::sort(sorted.begin(), sorted.end(),
              [](const AdapterMetadata &a, const AdapterMetadata &b) {
                  return a.version > b.version;
              });

    // Keep active + last N
    QList<AdapterMetadata> kept;
    int keptCount = 0;
    for (const auto &meta : sorted) {
        if (meta.version == m_activeVersion || keptCount < keepCount) {
            kept.append(meta);
            if (meta.version != m_activeVersion) keptCount++;
        } else {
            // Archive/delete old version directory
            QString vDir = versionDirPath(meta.version);
            QDir(vDir).removeRecursively();
            qDebug() << "AdapterManager: pruned v" << meta.version;
        }
    }

    m_versions = kept;
    emit versionCountChanged();
    saveVersions();
}

void AdapterManager::loadVersions()
{
    if (!m_profileManager) return;

    QString indexPath = m_profileManager->adaptersPath() + "/versions_index.json";
    QFile indexFile(indexPath);
    if (!indexFile.open(QIODevice::ReadOnly)) {
        qDebug() << "AdapterManager: no versions index found - starting fresh";
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(indexFile.readAll());
    indexFile.close();

    if (!doc.isObject()) return;

    QJsonObject root = doc.object();

    // Run JSON schema migration if needed
    root = SchemaMigrator::migrateJson(root, {}, SchemaMigrator::ADAPTER_INDEX_VERSION);

    m_activeVersion = root["activeVersion"].toInt(-1);
    m_previousActiveVersion = root["previousActiveVersion"].toInt(-1);
    m_nextVersionNumber = root["nextVersionNumber"].toInt(1);

    m_versions.clear();
    QJsonArray versionsArr = root["versions"].toArray();
    for (const auto &val : versionsArr) {
        m_versions.append(AdapterMetadata::fromJson(val.toObject()));
    }

    emit activeVersionChanged();
    emit versionCountChanged();

    if (m_activeVersion >= 0) {
        setAdapterStatus(QString("Active: v%1").arg(m_activeVersion));
    } else {
        setAdapterStatus("No adapter");
    }

    qDebug() << "AdapterManager: loaded" << m_versions.size()
             << "versions, active:" << m_activeVersion;
}

void AdapterManager::saveVersions() const
{
    if (!m_profileManager) return;

    QJsonObject root;
    root["schemaVersion"] = SchemaMigrator::ADAPTER_INDEX_VERSION;
    root["activeVersion"] = m_activeVersion;
    root["previousActiveVersion"] = m_previousActiveVersion;
    root["nextVersionNumber"] = m_nextVersionNumber;

    QJsonArray versionsArr;
    for (const auto &meta : m_versions) {
        versionsArr.append(meta.toJson());
    }
    root["versions"] = versionsArr;

    QString indexPath = m_profileManager->adaptersPath() + "/versions_index.json";
    QFile indexFile(indexPath);
    if (indexFile.open(QIODevice::WriteOnly)) {
        indexFile.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        indexFile.close();
    }
}

QList<AdapterMetadata> AdapterManager::getVersionsByStatus(const QString &status) const
{
    QList<AdapterMetadata> result;
    for (const auto &meta : m_versions) {
        if (meta.status == status) result.append(meta);
    }
    return result;
}

QList<AdapterMetadata> AdapterManager::getTopVersions(int count) const
{
    QList<AdapterMetadata> sorted = m_versions;
    std::sort(sorted.begin(), sorted.end(),
              [](const AdapterMetadata &a, const AdapterMetadata &b) {
                  return a.evalScore > b.evalScore;
              });
    return sorted.mid(0, count);
}

bool AdapterManager::updateVersionMetadata(int version, const AdapterMetadata &updated)
{
    for (int i = 0; i < m_versions.size(); ++i) {
        if (m_versions[i].version == version) {
            m_versions[i] = updated;
            m_versions[i].version = version;

            QString vDir = versionDirPath(version);
            QFile metaFile(vDir + "/metadata.json");
            if (metaFile.open(QIODevice::WriteOnly)) {
                metaFile.write(QJsonDocument(updated.toJson()).toJson(QJsonDocument::Indented));
                metaFile.close();
            }

            saveVersions();
            return true;
        }
    }
    return false;
}

QVariantList AdapterManager::getVersionsForQml() const
{
    QVariantList result;
    for (const auto &meta : m_versions) {
        QVariantMap map;
        map["version"] = meta.version;
        map["adapterName"] = meta.adapterName;
        map["trainingDate"] = meta.trainingDate.toString("yyyy-MM-dd hh:mm");
        map["trainingSteps"] = meta.trainingSteps;
        map["finalLoss"] = static_cast<double>(meta.finalLoss);
        map["evalScore"] = meta.evalScore;
        map["status"] = meta.status;
        result.append(map);
    }
    return result;
}

void AdapterManager::setAdapterStatus(const QString &status)
{
    if (m_adapterStatus != status) {
        m_adapterStatus = status;
        emit adapterStatusChanged();
    }
}

#endif // DATAFORM_TRAINING_ENABLED
