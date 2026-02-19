#ifndef LINEAGETRACKER_H
#define LINEAGETRACKER_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>
#include <QList>
#include <QDateTime>
#include <QJsonObject>
#include <QVariantList>
#include <QVariantMap>

class ProfileManager;
class AdapterManager;
class MemoryStore;

struct LineageNode {
    int version = 0;
    int parentVersion = -1;
    QString adapterName;
    QDateTime trainingDate;
    double evalScore = 0.0;
    float finalLoss = 0.0f;
    int trainingSteps = 0;
    QString status;                 // "candidate", "active", "archived", "rejected"
    QString origin;                 // "evolution", "consolidation", "reflection", "migration"
    int generationId = 0;
    int cycleId = -1;
    int variantIndex = -1;
    QList<int> childVersions;

    QJsonObject toJson() const;
    static LineageNode fromJson(const QJsonObject &json);
};

struct EraComparison {
    QString eraLabel;
    double evalScore = 0.0;
    int adapterVersion = -1;
    int totalEpisodes = 0;
    int totalTraits = 0;
    int cyclesCompleted = 0;
    QDateTime timestamp;
};

class LineageTracker : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int totalNodes READ totalNodes NOTIFY lineageChanged)
    Q_PROPERTY(int currentDepth READ currentDepth NOTIFY lineageChanged)
    Q_PROPERTY(int totalGenerations READ totalGenerations NOTIFY lineageChanged)

public:
    explicit LineageTracker(QObject *parent = nullptr);

    void setProfileManager(ProfileManager *pm);
    void setAdapterManager(AdapterManager *am);
    void setMemoryStore(MemoryStore *ms);

    int totalNodes() const { return m_nodes.size(); }
    int currentDepth() const;
    int totalGenerations() const;

    void buildLineage();
    void recordNode(const LineageNode &node);

    QList<LineageNode> getAncestors(int version) const;
    QList<LineageNode> getDescendants(int version) const;
    QList<EraComparison> getEraTimeline() const;

    Q_INVOKABLE QVariantList getLineageForQml() const;
    Q_INVOKABLE QVariantList getEraTimelineForQml() const;
    Q_INVOKABLE QVariantMap getNodeForQml(int version) const;

signals:
    void lineageChanged();

private:
    void loadLineage();
    void saveLineage() const;
    LineageNode* findNode(int version);
    const LineageNode* findNode(int version) const;

    ProfileManager *m_profileManager = nullptr;
    AdapterManager *m_adapterManager = nullptr;
    MemoryStore *m_memoryStore = nullptr;

    QList<LineageNode> m_nodes;
    QString m_lineageFilePath;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // LINEAGETRACKER_H
