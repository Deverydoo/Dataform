#ifndef ADAPTERMANAGER_H
#define ADAPTERMANAGER_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>
#include <QList>
#include <QDateTime>
#include <QJsonObject>
#include <QVariantList>

class ProfileManager;
class EvalSuite;

struct AdapterMetadata {
    int version = 0;
    QString adapterName = "communication_style";
    QDateTime trainingDate;
    qint64 firstEpisodeId = 0;
    qint64 lastEpisodeId = 0;
    int trainingSteps = 0;
    float finalLoss = 0.0f;
    double evalScore = 0.0;
    int parentVersion = -1;
    QString status;                 // "candidate", "active", "archived", "rejected"
    QString checkpointPath;
    QString exportedModelPath;
    // Phase 4: Lineage/generation tracking
    QString origin;                 // "evolution", "consolidation", "reflection", "migration"
    int generationId = 0;           // Which model generation produced this adapter
    int cycleId = -1;               // Evolution cycle that created this (-1 if not from evolution)
    int variantIndex = -1;          // Variant index within cycle (-1 if not from evolution)

    QJsonObject toJson() const;
    static AdapterMetadata fromJson(const QJsonObject &json);
};

class AdapterManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int activeVersion READ activeVersion NOTIFY activeVersionChanged)
    Q_PROPERTY(QString activeAdapterName READ activeAdapterName NOTIFY activeVersionChanged)
    Q_PROPERTY(int versionCount READ versionCount NOTIFY versionCountChanged)
    Q_PROPERTY(QString adapterStatus READ adapterStatus NOTIFY adapterStatusChanged)

public:
    explicit AdapterManager(QObject *parent = nullptr);
    ~AdapterManager();

    void setProfileManager(ProfileManager *profileManager);
    void setEvalSuite(EvalSuite *evalSuite);

    int activeVersion() const { return m_activeVersion; }
    QString activeAdapterName() const;
    int versionCount() const { return m_versions.size(); }
    QString adapterStatus() const { return m_adapterStatus; }

    // Register a newly trained candidate adapter
    int registerCandidate(const AdapterMetadata &metadata);

    // Get metadata for the current active adapter
    AdapterMetadata activeAdapterMetadata() const;

    // Get metadata by version
    AdapterMetadata getVersionMetadata(int version) const;

    // Atomic promotion: switch active adapter pointer
    bool promoteVersion(int version);

    // Rollback to previous active version
    Q_INVOKABLE bool rollback();

    // Pruning: keep last N versions + active
    void pruneVersions(int keepCount = 3);

    // Persistence
    void loadVersions();
    void saveVersions() const;

    // Query helpers (Phase 3)
    QList<AdapterMetadata> getVersionsByStatus(const QString &status) const;
    QList<AdapterMetadata> getTopVersions(int count) const;
    bool updateVersionMetadata(int version, const AdapterMetadata &updated);

    // QML-friendly version list
    Q_INVOKABLE QVariantList getVersionsForQml() const;

signals:
    void activeVersionChanged();
    void versionCountChanged();
    void adapterStatusChanged();
    void candidateRegistered(int version);
    void adapterPromoted(int version);
    void adapterRolledBack(int previousVersion);

private:
    QString versionDirPath(int version) const;
    QString activeDirPath() const;
    bool atomicSwapActive(int newVersion);
    void setAdapterStatus(const QString &status);

    ProfileManager *m_profileManager = nullptr;
    EvalSuite *m_evalSuite = nullptr;

    int m_activeVersion = -1;       // -1 = base model only
    int m_previousActiveVersion = -1;
    int m_nextVersionNumber = 1;
    QString m_adapterStatus = "No adapter";
    QList<AdapterMetadata> m_versions;

    static constexpr int MAX_KEPT_VERSIONS = 5;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // ADAPTERMANAGER_H
