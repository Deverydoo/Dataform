#include "profilemanager.h"
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDebug>

ProfileManager::ProfileManager(QObject *parent)
    : QObject(parent)
    , m_userId("default")
{
    m_basePath = QCoreApplication::applicationDirPath() + "/profiles";
    ensureDirectoryTree();
}

QString ProfileManager::profilePath() const
{
    return m_basePath + "/" + m_userId;
}

QString ProfileManager::memoryPath() const
{
    return profilePath() + "/memory";
}

QString ProfileManager::adaptersPath() const
{
    return profilePath() + "/adapters";
}

QString ProfileManager::activeAdaptersPath() const
{
    return profilePath() + "/adapters/active";
}

QString ProfileManager::adapterVersionsPath() const
{
    return profilePath() + "/adapters/versions";
}

QString ProfileManager::checkpointsPath() const
{
    return profilePath() + "/checkpoints";
}

QString ProfileManager::evalPath() const
{
    return profilePath() + "/eval";
}

QString ProfileManager::evolutionPath() const
{
    return profilePath() + "/evolution";
}

QString ProfileManager::eraSnapshotsPath() const
{
    return profilePath() + "/era_snapshots";
}

QString ProfileManager::backupsPath() const
{
    return profilePath() + "/backups";
}

QString ProfileManager::settingsFilePath() const
{
    return profilePath() + "/settings.json";
}

void ProfileManager::ensureDirectoryTree()
{
    QStringList dirs = {
        memoryPath(),
        activeAdaptersPath(),
        adapterVersionsPath(),
        checkpointsPath(),
        evalPath(),
        evolutionPath(),
        eraSnapshotsPath(),
        backupsPath()
    };

    for (const QString &dir : dirs) {
        QDir d(dir);
        if (!d.exists()) {
            d.mkpath(".");
            qDebug() << "Created directory:" << dir;
        }
    }

    qDebug() << "Profile directory tree ready at:" << profilePath();
}

bool ProfileManager::exportProfile(const QString &targetDir)
{
    if (targetDir.isEmpty()) {
        emit exportComplete(false, "No target directory specified");
        return false;
    }

    QString exportPath = targetDir + "/dataform_profile";
    QDir exportDir(exportPath);

    // Remove existing export at target if present
    if (exportDir.exists()) {
        exportDir.removeRecursively();
    }

    bool ok = recursiveCopy(profilePath(), exportPath);
    if (ok) {
        qDebug() << "ProfileManager: exported profile to" << exportPath;
        emit exportComplete(true, exportPath);
    } else {
        qWarning() << "ProfileManager: export failed";
        emit exportComplete(false, "Failed to copy profile directory");
    }
    return ok;
}

bool ProfileManager::importProfile(const QString &sourceDir)
{
    if (sourceDir.isEmpty()) {
        emit importComplete(false, "No source directory specified");
        return false;
    }

    // Validate source has expected structure (at least memory/ dir)
    QDir srcDir(sourceDir);
    if (!srcDir.exists("memory")) {
        emit importComplete(false, "Invalid profile: missing memory/ directory");
        return false;
    }

    bool ok = recursiveCopy(sourceDir, profilePath());
    if (ok) {
        qDebug() << "ProfileManager: imported profile from" << sourceDir;
        emit importComplete(true, sourceDir);
    } else {
        qWarning() << "ProfileManager: import failed";
        emit importComplete(false, "Failed to copy profile from source");
    }
    return ok;
}

bool ProfileManager::recursiveCopy(const QString &srcDir, const QString &dstDir)
{
    QDir src(srcDir);
    if (!src.exists()) return false;

    QDir dst(dstDir);
    if (!dst.exists()) {
        dst.mkpath(".");
    }

    // Copy files
    for (const QFileInfo &info : src.entryInfoList(QDir::Files)) {
        QString dstFile = dstDir + "/" + info.fileName();
        // Remove existing file at destination
        if (QFile::exists(dstFile)) {
            QFile::remove(dstFile);
        }
        if (!QFile::copy(info.filePath(), dstFile)) {
            qWarning() << "Failed to copy" << info.filePath() << "to" << dstFile;
            return false;
        }
    }

    // Recurse into subdirectories
    for (const QFileInfo &info : src.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        if (!recursiveCopy(info.filePath(), dstDir + "/" + info.fileName())) {
            return false;
        }
    }

    return true;
}
