#include "profilehealthmanager.h"
#include "profilemanager.h"
#include "memorystore.h"
#include "settingsmanager.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QDebug>

ProfileHealthManager::ProfileHealthManager(QObject *parent)
    : QObject(parent)
{
}

void ProfileHealthManager::setProfileManager(ProfileManager *pm)
{
    m_profileManager = pm;
}

void ProfileHealthManager::setMemoryStore(MemoryStore *ms)
{
    m_memoryStore = ms;
}

void ProfileHealthManager::setSettingsManager(SettingsManager *sm)
{
    m_settingsManager = sm;
}

QStringList ProfileHealthManager::runStartupCheck()
{
    qDebug() << "ProfileHealthManager: running startup check...";
    QStringList allIssues;
    QStringList repairs;

    // Check directory structure
    auto dirIssues = checkDirectoryStructure();
    allIssues.append(dirIssues);

    // Check settings integrity
    auto settingsIssues = checkSettingsIntegrity();
    allIssues.append(settingsIssues);

    // Check databases
    auto episodicIssues = checkEpisodicDb();
    allIssues.append(episodicIssues);

    auto traitsIssues = checkTraitsDb();
    allIssues.append(traitsIssues);

#ifdef DATAFORM_TRAINING_ENABLED
    auto artifactIssues = checkTrainingArtifacts();
    allIssues.append(artifactIssues);

    auto adapterIssues = checkAdapterIndex();
    allIssues.append(adapterIssues);

    auto evolutionIssues = checkEvolutionState();
    allIssues.append(evolutionIssues);
#endif

    m_lastHealthCheck = QDateTime::currentDateTime();
    bool allPassed = allIssues.isEmpty();
    setHealthStatus(allPassed ? "Healthy" : QString("%1 issue(s) found").arg(allIssues.size()),
                    allPassed);

    emit healthCheckCompleted(allPassed, allIssues);
    qDebug() << "ProfileHealthManager: startup check complete -"
             << (allPassed ? "all passed" : QString("%1 issues").arg(allIssues.size()));
    return allIssues;
}

QStringList ProfileHealthManager::runPeriodicCheck()
{
    qDebug() << "ProfileHealthManager: running periodic check...";
    QStringList issues;

    issues.append(checkDirectoryStructure());
    issues.append(checkEpisodicDb());
    issues.append(checkTraitsDb());

    m_lastHealthCheck = QDateTime::currentDateTime();
    bool allPassed = issues.isEmpty();
    setHealthStatus(allPassed ? "Healthy" : QString("%1 issue(s) found").arg(issues.size()),
                    allPassed);

    emit healthCheckCompleted(allPassed, issues);
    return issues;
}

// --- Individual checks ---

QStringList ProfileHealthManager::checkDirectoryStructure()
{
    QStringList issues;
    if (!m_profileManager) return issues;

    // Ensure directory tree exists; self-heal by creating missing dirs
    m_profileManager->ensureDirectoryTree();

    QStringList requiredDirs = {
        m_profileManager->memoryPath(),
        m_profileManager->adaptersPath(),
        m_profileManager->checkpointsPath(),
        m_profileManager->evalPath(),
        m_profileManager->backupsPath()
    };

    for (const QString &dir : requiredDirs) {
        if (!QDir(dir).exists()) {
            issues.append("Missing directory (could not create): " + dir);
        }
    }

    return issues;
}

QStringList ProfileHealthManager::checkSettingsIntegrity()
{
    QStringList issues;
    if (!m_profileManager) return issues;

    QString settingsPath = m_profileManager->settingsFilePath();
    QFile file(settingsPath);

    if (!file.exists()) {
        // No settings file is fine - defaults will be used
        return issues;
    }

    if (!file.open(QIODevice::ReadOnly)) {
        issues.append("Settings file exists but cannot be read: " + settingsPath);
        return issues;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();

    if (doc.isNull() || !doc.isObject()) {
        issues.append("Settings file is corrupt, resetting to defaults");
        if (m_settingsManager) {
            m_settingsManager->resetToDefaults();
            qDebug() << "ProfileHealthManager: self-healed corrupt settings";
        }
    }

    return issues;
}

QStringList ProfileHealthManager::checkEpisodicDb()
{
    QStringList issues;
    if (!m_profileManager) return issues;

    QString dbPath = m_profileManager->memoryPath() + "/episodic.db";
    if (!QFileInfo::exists(dbPath)) {
        // DB doesn't exist yet - that's fine, MemoryStore will create it
        return issues;
    }

    // Open a temporary connection to run integrity check
    {
        QString connName = "health_episodic_check";
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connName);
        db.setDatabaseName(dbPath);

        if (db.open()) {
            QSqlQuery query(db);
            if (query.exec("PRAGMA integrity_check") && query.next()) {
                QString result = query.value(0).toString();
                if (result != "ok") {
                    issues.append("Episodic DB integrity check failed: " + result);
                }
            }
            db.close();
        } else {
            issues.append("Cannot open episodic DB for integrity check");
        }

        QSqlDatabase::removeDatabase(connName);
    }

    return issues;
}

QStringList ProfileHealthManager::checkTraitsDb()
{
    QStringList issues;
    if (!m_profileManager) return issues;

    QString dbPath = m_profileManager->memoryPath() + "/traits.db";
    if (!QFileInfo::exists(dbPath)) {
        return issues;
    }

    {
        QString connName = "health_traits_check";
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connName);
        db.setDatabaseName(dbPath);

        if (db.open()) {
            QSqlQuery query(db);
            if (query.exec("PRAGMA integrity_check") && query.next()) {
                QString result = query.value(0).toString();
                if (result != "ok") {
                    issues.append("Traits DB integrity check failed: " + result);
                }
            }
            db.close();
        } else {
            issues.append("Cannot open traits DB for integrity check");
        }

        QSqlDatabase::removeDatabase(connName);
    }

    return issues;
}

#ifdef DATAFORM_TRAINING_ENABLED

QStringList ProfileHealthManager::checkTrainingArtifacts()
{
    QStringList issues;
    if (!m_profileManager) return issues;

    QString artifactsDir = m_profileManager->profilePath() + "/training_artifacts";
    if (!QDir(artifactsDir).exists()) {
        issues.append("Training artifacts directory missing: " + artifactsDir);
        return issues;
    }

    QStringList requiredFiles = {
        "training_model.onnx",
        "eval_model.onnx",
        "optimizer_model.onnx",
        "checkpoint",
        "vocab.json",
        "merges.txt"
    };

    for (const QString &file : requiredFiles) {
        if (!QFileInfo::exists(artifactsDir + "/" + file)) {
            issues.append("Missing training artifact: " + file);
        }
    }

    return issues;
}

QStringList ProfileHealthManager::checkAdapterIndex()
{
    QStringList issues;
    if (!m_profileManager) return issues;

    QString indexPath = m_profileManager->adaptersPath() + "/versions_index.json";
    if (!QFileInfo::exists(indexPath)) {
        // No index yet - fine for fresh profile
        return issues;
    }

    QFile file(indexPath);
    if (!file.open(QIODevice::ReadOnly)) {
        issues.append("Adapter index exists but cannot be read");
        return issues;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();

    if (doc.isNull() || !doc.isObject()) {
        issues.append("Adapter index is corrupt - will need rebuild on next load");
    }

    return issues;
}

QStringList ProfileHealthManager::checkEvolutionState()
{
    QStringList issues;
    if (!m_profileManager) return issues;

    QString statePath = m_profileManager->evolutionPath() + "/population_state.json";
    if (!QFileInfo::exists(statePath)) {
        return issues;
    }

    QFile file(statePath);
    if (!file.open(QIODevice::ReadOnly)) {
        issues.append("Evolution state exists but cannot be read");
        return issues;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();

    if (doc.isNull() || !doc.isObject()) {
        issues.append("Evolution state is corrupt - will reset to idle on next load");
    }

    return issues;
}

#endif // DATAFORM_TRAINING_ENABLED

// --- Backup/Restore ---

bool ProfileHealthManager::createBackup()
{
    if (!m_profileManager) {
        emit backupError("No profile manager");
        return false;
    }

    QString backupsDir = m_profileManager->backupsPath();
    QDir().mkpath(backupsDir);

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString backupDir = backupsDir + "/backup_" + timestamp;
    QDir().mkpath(backupDir);

    qDebug() << "ProfileHealthManager: creating backup at" << backupDir;

    // List of lightweight state files to back up (skip large binaries)
    QString profilePath = m_profileManager->profilePath();
    QStringList filesToBackup = {
        "/settings.json",
        "/model_generations.json",
        "/memory/episodic.db",
        "/memory/traits.db",
        "/adapters/versions_index.json",
        "/adapters/active/active_version.json",
        "/evolution/population_state.json",
        "/evolution/evolution_log.jsonl"
    };

    int copied = 0;
    for (const QString &relPath : filesToBackup) {
        QString srcPath = profilePath + relPath;
        if (!QFileInfo::exists(srcPath)) continue;

        QString dstPath = backupDir + relPath;
        QDir().mkpath(QFileInfo(dstPath).absolutePath());

        if (QFile::copy(srcPath, dstPath)) {
            copied++;
        } else {
            qWarning() << "ProfileHealthManager: failed to backup" << relPath;
        }
    }

    // Also back up per-version metadata.json files
    QString versionsDir = m_profileManager->adapterVersionsPath();
    QDir vDir(versionsDir);
    if (vDir.exists()) {
        for (const QString &entry : vDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
            QString metaSrc = versionsDir + "/" + entry + "/metadata.json";
            if (QFileInfo::exists(metaSrc)) {
                QString metaDst = backupDir + "/adapters/versions/" + entry + "/metadata.json";
                QDir().mkpath(QFileInfo(metaDst).absolutePath());
                QFile::copy(metaSrc, metaDst);
            }
        }
    }

    rotateBackups();

    m_lastBackupTime = QDateTime::currentDateTime();
    emit backupCompleted(backupDir);
    qDebug() << "ProfileHealthManager: backup complete -" << copied << "files copied";
    return true;
}

bool ProfileHealthManager::restoreFromBackup(const QString &backupPath)
{
    if (!m_profileManager) {
        emit backupError("No profile manager");
        return false;
    }

    QDir backupDir(backupPath);
    if (!backupDir.exists()) {
        emit backupError("Backup directory does not exist: " + backupPath);
        return false;
    }

    qDebug() << "ProfileHealthManager: restoring from" << backupPath;

    QString profilePath = m_profileManager->profilePath();

    // Copy all files from backup back to profile
    std::function<void(const QString &, const QString &)> copyDir;
    int restored = 0;
    copyDir = [&](const QString &src, const QString &dst) {
        QDir srcDir(src);
        for (const QFileInfo &info : srcDir.entryInfoList(QDir::Files)) {
            QString dstFile = dst + "/" + info.fileName();
            QDir().mkpath(dst);
            QFile::remove(dstFile);
            if (QFile::copy(info.filePath(), dstFile)) {
                restored++;
            }
        }
        for (const QString &subDir : srcDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
            copyDir(src + "/" + subDir, dst + "/" + subDir);
        }
    };

    copyDir(backupPath, profilePath);

    qDebug() << "ProfileHealthManager: restored" << restored << "files";

    // Reload components
    if (m_settingsManager) {
        m_settingsManager->loadSettings();
    }
    if (m_memoryStore) {
        m_memoryStore->close();
        m_memoryStore->initialize();
    }

    return true;
}

void ProfileHealthManager::rotateBackups(int keepCount)
{
    if (!m_profileManager) return;

    QString backupsDir = m_profileManager->backupsPath();
    QDir dir(backupsDir);
    QStringList backups = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);

    while (backups.size() > keepCount) {
        QString oldest = backupsDir + "/" + backups.first();
        QDir(oldest).removeRecursively();
        qDebug() << "ProfileHealthManager: rotated out old backup:" << backups.first();
        backups.removeFirst();
    }
}

void ProfileHealthManager::setHealthStatus(const QString &status, bool healthy)
{
    if (m_healthStatus != status || m_isHealthy != healthy) {
        m_healthStatus = status;
        m_isHealthy = healthy;
        emit healthStatusChanged();
    }
}
