#ifndef PROFILEHEALTHMANAGER_H
#define PROFILEHEALTHMANAGER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>

class ProfileManager;
class MemoryStore;
class SettingsManager;

class ProfileHealthManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString healthStatus READ healthStatus NOTIFY healthStatusChanged)
    Q_PROPERTY(bool isHealthy READ isHealthy NOTIFY healthStatusChanged)
    Q_PROPERTY(QDateTime lastBackupTime READ lastBackupTime NOTIFY backupCompleted)
    Q_PROPERTY(QDateTime lastHealthCheck READ lastHealthCheck NOTIFY healthCheckCompleted)

public:
    explicit ProfileHealthManager(QObject *parent = nullptr);

    void setProfileManager(ProfileManager *pm);
    void setMemoryStore(MemoryStore *ms);
    void setSettingsManager(SettingsManager *sm);

    QString healthStatus() const { return m_healthStatus; }
    bool isHealthy() const { return m_isHealthy; }
    QDateTime lastBackupTime() const { return m_lastBackupTime; }
    QDateTime lastHealthCheck() const { return m_lastHealthCheck; }

    Q_INVOKABLE QStringList runStartupCheck();
    Q_INVOKABLE QStringList runPeriodicCheck();
    Q_INVOKABLE bool createBackup();
    Q_INVOKABLE bool restoreFromBackup(const QString &backupPath);

signals:
    void healthStatusChanged();
    void healthCheckCompleted(bool allPassed, const QStringList &issues);
    void backupCompleted(const QString &path);
    void backupError(const QString &error);
    void repairCompleted(const QStringList &repairs);

private:
    QStringList checkDirectoryStructure();
    QStringList checkSettingsIntegrity();
    QStringList checkEpisodicDb();
    QStringList checkTraitsDb();
#ifdef DATAFORM_TRAINING_ENABLED
    QStringList checkTrainingArtifacts();
    QStringList checkAdapterIndex();
    QStringList checkEvolutionState();
#endif

    void rotateBackups(int keepCount = 5);
    void setHealthStatus(const QString &status, bool healthy);

    ProfileManager *m_profileManager = nullptr;
    MemoryStore *m_memoryStore = nullptr;
    SettingsManager *m_settingsManager = nullptr;

    QString m_healthStatus = "Not checked";
    bool m_isHealthy = true;
    QDateTime m_lastBackupTime;
    QDateTime m_lastHealthCheck;
};

#endif // PROFILEHEALTHMANAGER_H
