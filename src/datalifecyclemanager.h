#ifndef DATALIFECYCLEMANAGER_H
#define DATALIFECYCLEMANAGER_H

#include <QObject>
#include <QString>
#include <QDateTime>

class MemoryStore;
class ProfileManager;
class SettingsManager;

class DataLifecycleManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(qint64 diskUsageBytes READ diskUsageBytes NOTIFY diskUsageChanged)
    Q_PROPERTY(QString diskUsageFormatted READ diskUsageFormatted NOTIFY diskUsageChanged)
    Q_PROPERTY(int archivedEpisodeCount READ archivedEpisodeCount NOTIFY lifecycleChanged)
    Q_PROPERTY(int decayedTraitCount READ decayedTraitCount NOTIFY lifecycleChanged)

public:
    explicit DataLifecycleManager(QObject *parent = nullptr);

    void setMemoryStore(MemoryStore *ms);
    void setProfileManager(ProfileManager *pm);
    void setSettingsManager(SettingsManager *sm);

    qint64 diskUsageBytes() const { return m_diskUsageBytes; }
    QString diskUsageFormatted() const;
    int archivedEpisodeCount() const { return m_archivedEpisodeCount; }
    int decayedTraitCount() const { return m_decayedTraitCount; }

    Q_INVOKABLE void runLifecycleSweep();
    Q_INVOKABLE void updateDiskUsage();

    // Constants
    static constexpr double TRAIT_DECAY_RATE_PER_MONTH = 0.05;
    static constexpr double TRAIT_MIN_CONFIDENCE = 0.1;
    static constexpr int ERA_SNAPSHOTS_KEEP = 4;
    static constexpr int OLD_CYCLES_KEEP = 2;
    static constexpr int CONSOLIDATED_KEEP = 3;
    static constexpr int SWEEP_INTERVAL_HOURS = 24;

signals:
    void lifecycleSweepComplete(int episodesArchived, int traitsDecayed, qint64 bytesReclaimed);
    void lifecycleError(const QString &error);
    void diskUsageChanged();
    void lifecycleChanged();

private:
    int compactOldEpisodes();
    int decayStaleTraits();
#ifdef DATAFORM_TRAINING_ENABLED
    qint64 pruneCheckpoints();
#endif
    qint64 calculateDirectorySize(const QString &path) const;

    MemoryStore *m_memoryStore = nullptr;
    ProfileManager *m_profileManager = nullptr;
    SettingsManager *m_settingsManager = nullptr;

    qint64 m_diskUsageBytes = 0;
    int m_archivedEpisodeCount = 0;
    int m_decayedTraitCount = 0;
    QDateTime m_lastSweepTime;
};

#endif // DATALIFECYCLEMANAGER_H
