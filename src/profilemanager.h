#ifndef PROFILEMANAGER_H
#define PROFILEMANAGER_H

#include <QObject>
#include <QString>

class ProfileManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString userId READ userId NOTIFY userIdChanged)
    Q_PROPERTY(QString profilePath READ profilePath NOTIFY profilePathChanged)

public:
    explicit ProfileManager(QObject *parent = nullptr);

    QString userId() const { return m_userId; }
    QString profilePath() const;
    QString memoryPath() const;
    QString adaptersPath() const;
    QString activeAdaptersPath() const;
    QString adapterVersionsPath() const;
    QString checkpointsPath() const;
    QString evalPath() const;
    QString evolutionPath() const;
    QString eraSnapshotsPath() const;
    QString backupsPath() const;
    QString settingsFilePath() const;

    Q_INVOKABLE void ensureDirectoryTree();

    // Profile portability
    Q_INVOKABLE bool exportProfile(const QString &targetDir);
    Q_INVOKABLE bool importProfile(const QString &sourceDir);

signals:
    void userIdChanged();
    void profilePathChanged();
    void exportComplete(bool success, const QString &message);
    void importComplete(bool success, const QString &message);

private:
    bool recursiveCopy(const QString &srcDir, const QString &dstDir);

    QString m_userId;
    QString m_basePath;
};

#endif // PROFILEMANAGER_H
