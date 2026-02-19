#include "datalifecyclemanager.h"
#include "memorystore.h"
#include "profilemanager.h"
#include "settingsmanager.h"
#include <QDir>
#include <QFileInfo>
#include <QDirIterator>
#include <QDebug>
#include <algorithm>
#include <cmath>

DataLifecycleManager::DataLifecycleManager(QObject *parent)
    : QObject(parent)
{
}

void DataLifecycleManager::setMemoryStore(MemoryStore *ms) { m_memoryStore = ms; }
void DataLifecycleManager::setProfileManager(ProfileManager *pm) { m_profileManager = pm; }
void DataLifecycleManager::setSettingsManager(SettingsManager *sm) { m_settingsManager = sm; }

QString DataLifecycleManager::diskUsageFormatted() const
{
    if (m_diskUsageBytes < 1024LL)
        return QString("%1 B").arg(m_diskUsageBytes);
    if (m_diskUsageBytes < 1024LL * 1024)
        return QString("%1 KB").arg(m_diskUsageBytes / 1024.0, 0, 'f', 1);
    if (m_diskUsageBytes < 1024LL * 1024 * 1024)
        return QString("%1 MB").arg(m_diskUsageBytes / (1024.0 * 1024.0), 0, 'f', 1);
    return QString("%1 GB").arg(m_diskUsageBytes / (1024.0 * 1024.0 * 1024.0), 0, 'f', 2);
}

void DataLifecycleManager::runLifecycleSweep()
{
    // Throttle: at most once per SWEEP_INTERVAL_HOURS
    if (m_lastSweepTime.isValid()) {
        qint64 hoursSince = m_lastSweepTime.secsTo(QDateTime::currentDateTime()) / 3600;
        if (hoursSince < SWEEP_INTERVAL_HOURS) {
            qDebug() << "DataLifecycleManager: sweep skipped - last run"
                     << hoursSince << "hours ago";
            return;
        }
    }

    qDebug() << "DataLifecycleManager: running lifecycle sweep...";

    qint64 beforeBytes = m_diskUsageBytes;

    // 1. Compact old episodes
    int episodesArchived = compactOldEpisodes();

    // 2. Decay stale traits
    int traitsDecayed = decayStaleTraits();

    // 3. Prune checkpoints (training-only)
    qint64 bytesReclaimed = 0;
#ifdef DATAFORM_TRAINING_ENABLED
    bytesReclaimed = pruneCheckpoints();
#endif

    // 4. Update disk usage
    updateDiskUsage();
    bytesReclaimed += (beforeBytes - m_diskUsageBytes);
    if (bytesReclaimed < 0) bytesReclaimed = 0;

    // 5. Check disk usage against limit
    if (m_settingsManager) {
        qint64 maxBytes = static_cast<qint64>(m_settingsManager->maxDiskUsageGB()) * 1024LL * 1024 * 1024;
        if (m_diskUsageBytes > maxBytes) {
            qWarning() << "DataLifecycleManager: disk usage" << diskUsageFormatted()
                       << "exceeds limit of" << m_settingsManager->maxDiskUsageGB() << "GB";
            // Aggressive pass: compact episodes with shorter retention
            // and prune more aggressively
            // (This is a soft warning - we don't delete user data without consent)
        }
    }

    m_lastSweepTime = QDateTime::currentDateTime();

    emit lifecycleSweepComplete(episodesArchived, traitsDecayed, bytesReclaimed);
    emit lifecycleChanged();

    qDebug() << "DataLifecycleManager: sweep complete -"
             << episodesArchived << "episodes archived,"
             << traitsDecayed << "traits decayed,"
             << (bytesReclaimed / 1024) << "KB reclaimed";
}

void DataLifecycleManager::updateDiskUsage()
{
    if (!m_profileManager) return;

    qint64 totalSize = calculateDirectorySize(m_profileManager->profilePath());

    if (m_diskUsageBytes != totalSize) {
        m_diskUsageBytes = totalSize;
        emit diskUsageChanged();
    }

    // Update archived counts from MemoryStore
    if (m_memoryStore) {
        m_archivedEpisodeCount = m_memoryStore->archivedEpisodeCount();
        m_decayedTraitCount = m_memoryStore->archivedTraitCount();
    }
}

// --- Private ---

int DataLifecycleManager::compactOldEpisodes()
{
    if (!m_memoryStore || !m_settingsManager) return 0;

    int retentionDays = m_settingsManager->episodeRetentionDays();
    int archived = 0;

    // Get old episodes (non-archived)
    auto oldEpisodes = m_memoryStore->getEpisodesOlderThan(retentionDays, true, 100);

    for (const auto &episode : oldEpisodes) {
        // High-signal episodes (with feedback or edits) get 2x retention
        bool isHighSignal = (episode.userFeedback != 0 || episode.corrected || episode.reasked);
        if (isHighSignal) {
            // Check if episode is old enough even with 2x retention
            QDateTime cutoff = QDateTime::currentDateTime().addDays(-retentionDays * 2);
            if (episode.timestamp > cutoff) continue;
        }

        // Create deterministic summary: topic + truncated text + feedback indicator
        QString summary = episode.topic;
        if (!episode.userText.isEmpty()) {
            summary += " | " + episode.userText.left(50);
            if (episode.userText.length() > 50) summary += "...";
        }
        if (episode.userFeedback > 0) summary += " [+]";
        else if (episode.userFeedback < 0) summary += " [-]";
        if (episode.corrected) summary += " [corrected]";

        if (m_memoryStore->archiveEpisode(episode.id, summary)) {
            archived++;
        }
    }

    if (archived > 0) {
        m_archivedEpisodeCount = m_memoryStore->archivedEpisodeCount();
        qDebug() << "DataLifecycleManager: archived" << archived << "old episodes";
    }

    return archived;
}

int DataLifecycleManager::decayStaleTraits()
{
    if (!m_memoryStore) return 0;

    int decayed = 0;

    // Get traits that haven't been confirmed recently
    auto staleTraits = m_memoryStore->getUnconfirmedTraitsSince(30); // 30 days

    for (const auto &trait : staleTraits) {
        // Calculate months since last confirmed
        qint64 daysSince = trait.lastConfirmedTs.daysTo(QDateTime::currentDateTime());
        double monthsSince = daysSince / 30.0;

        // Apply decay formula
        double newConfidence = trait.confidence * (1.0 - TRAIT_DECAY_RATE_PER_MONTH * monthsSince);
        newConfidence = std::max(0.0, newConfidence);

        if (newConfidence < TRAIT_MIN_CONFIDENCE) {
            // Archive the trait
            if (m_memoryStore->archiveTrait(trait.traitId)) {
                decayed++;
            }
        } else if (newConfidence < trait.confidence) {
            // Update confidence
            m_memoryStore->updateTraitDecay(trait.traitId, newConfidence);
            decayed++;
        }
    }

    if (decayed > 0) {
        m_decayedTraitCount = m_memoryStore->archivedTraitCount();
        qDebug() << "DataLifecycleManager: decayed/archived" << decayed << "traits";
    }

    return decayed;
}

#ifdef DATAFORM_TRAINING_ENABLED

qint64 DataLifecycleManager::pruneCheckpoints()
{
    if (!m_profileManager) return 0;

    qint64 bytesReclaimed = 0;

    // Prune era snapshots: keep last ERA_SNAPSHOTS_KEEP
    QString eraDir = m_profileManager->eraSnapshotsPath();
    QDir era(eraDir);
    if (era.exists()) {
        QStringList snapshots = era.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        while (snapshots.size() > ERA_SNAPSHOTS_KEEP) {
            QString oldest = eraDir + "/" + snapshots.first();
            qint64 size = calculateDirectorySize(oldest);
            QDir(oldest).removeRecursively();
            bytesReclaimed += size;
            qDebug() << "DataLifecycleManager: pruned era snapshot:" << snapshots.first();
            snapshots.removeFirst();
        }
    }

    // Prune old evolution cycle directories: keep last OLD_CYCLES_KEEP
    QString evoDir = m_profileManager->evolutionPath();
    QDir evo(evoDir);
    if (evo.exists()) {
        QStringList cycleDirs;
        for (const QString &entry : evo.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
            if (entry.startsWith("cycle_")) {
                cycleDirs.append(entry);
            }
        }
        std::sort(cycleDirs.begin(), cycleDirs.end());
        while (cycleDirs.size() > OLD_CYCLES_KEEP) {
            QString oldest = evoDir + "/" + cycleDirs.first();
            qint64 size = calculateDirectorySize(oldest);
            QDir(oldest).removeRecursively();
            bytesReclaimed += size;
            qDebug() << "DataLifecycleManager: pruned cycle dir:" << cycleDirs.first();
            cycleDirs.removeFirst();
        }
    }

    // Prune old consolidated checkpoints: keep last CONSOLIDATED_KEEP
    QString checkpointsDir = m_profileManager->checkpointsPath();
    QDir cpDir(checkpointsDir);
    if (cpDir.exists()) {
        QStringList consolidatedDirs;
        for (const QString &entry : cpDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
            if (entry.startsWith("consolidated_")) {
                consolidatedDirs.append(entry);
            }
        }
        std::sort(consolidatedDirs.begin(), consolidatedDirs.end());
        while (consolidatedDirs.size() > CONSOLIDATED_KEEP) {
            QString oldest = checkpointsDir + "/" + consolidatedDirs.first();
            qint64 size = calculateDirectorySize(oldest);
            QDir(oldest).removeRecursively();
            bytesReclaimed += size;
            qDebug() << "DataLifecycleManager: pruned consolidated:" << consolidatedDirs.first();
            consolidatedDirs.removeFirst();
        }
    }

    return bytesReclaimed;
}

#endif // DATAFORM_TRAINING_ENABLED

qint64 DataLifecycleManager::calculateDirectorySize(const QString &path) const
{
    qint64 size = 0;
    QDirIterator it(path, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        it.next();
        size += it.fileInfo().size();
    }
    return size;
}
