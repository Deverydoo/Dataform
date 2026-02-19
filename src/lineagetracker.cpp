#ifdef DATAFORM_TRAINING_ENABLED

#include "lineagetracker.h"
#include "profilemanager.h"
#include "adaptermanager.h"
#include "memorystore.h"
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDebug>
#include <algorithm>

// --- LineageNode ---

QJsonObject LineageNode::toJson() const
{
    QJsonObject obj;
    obj["version"] = version;
    obj["parentVersion"] = parentVersion;
    obj["adapterName"] = adapterName;
    obj["trainingDate"] = trainingDate.toString(Qt::ISODate);
    obj["evalScore"] = evalScore;
    obj["finalLoss"] = static_cast<double>(finalLoss);
    obj["trainingSteps"] = trainingSteps;
    obj["status"] = status;
    obj["origin"] = origin;
    obj["generationId"] = generationId;
    obj["cycleId"] = cycleId;
    obj["variantIndex"] = variantIndex;

    QJsonArray children;
    for (int child : childVersions) {
        children.append(child);
    }
    obj["childVersions"] = children;
    return obj;
}

LineageNode LineageNode::fromJson(const QJsonObject &json)
{
    LineageNode node;
    node.version = json["version"].toInt();
    node.parentVersion = json["parentVersion"].toInt(-1);
    node.adapterName = json["adapterName"].toString();
    node.trainingDate = QDateTime::fromString(json["trainingDate"].toString(), Qt::ISODate);
    node.evalScore = json["evalScore"].toDouble();
    node.finalLoss = static_cast<float>(json["finalLoss"].toDouble());
    node.trainingSteps = json["trainingSteps"].toInt();
    node.status = json["status"].toString();
    node.origin = json["origin"].toString();
    node.generationId = json["generationId"].toInt(0);
    node.cycleId = json["cycleId"].toInt(-1);
    node.variantIndex = json["variantIndex"].toInt(-1);

    QJsonArray children = json["childVersions"].toArray();
    for (const auto &val : children) {
        node.childVersions.append(val.toInt());
    }
    return node;
}

// --- LineageTracker ---

LineageTracker::LineageTracker(QObject *parent)
    : QObject(parent)
{
}

void LineageTracker::setProfileManager(ProfileManager *pm)
{
    m_profileManager = pm;
    if (pm) {
        m_lineageFilePath = pm->adaptersPath() + "/lineage.json";
    }
}

void LineageTracker::setAdapterManager(AdapterManager *am) { m_adapterManager = am; }
void LineageTracker::setMemoryStore(MemoryStore *ms) { m_memoryStore = ms; }

int LineageTracker::currentDepth() const
{
    if (!m_adapterManager || m_nodes.isEmpty()) return 0;

    int activeVersion = m_adapterManager->activeVersion();
    int depth = 0;
    int current = activeVersion;

    while (current >= 0) {
        const LineageNode *node = findNode(current);
        if (!node) break;
        depth++;
        current = node->parentVersion;
    }

    return depth;
}

int LineageTracker::totalGenerations() const
{
    QSet<int> gens;
    for (const auto &node : m_nodes) {
        gens.insert(node.generationId);
    }
    return gens.size();
}

void LineageTracker::buildLineage()
{
    // First try loading from persisted file
    loadLineage();

    if (!m_nodes.isEmpty()) {
        qDebug() << "LineageTracker: loaded" << m_nodes.size() << "nodes from lineage file";
        emit lineageChanged();
        return;
    }

    // Rebuild from AdapterManager's version list
    if (!m_adapterManager) return;

    qDebug() << "LineageTracker: rebuilding lineage from adapter versions...";

    auto versions = m_adapterManager->getVersionsForQml();
    for (const auto &v : versions) {
        QVariantMap map = v.toMap();
        int ver = map["version"].toInt();

        // Check if we already have this node
        if (findNode(ver)) continue;

        // Get full metadata
        AdapterMetadata meta = m_adapterManager->getVersionMetadata(ver);

        LineageNode node;
        node.version = meta.version;
        node.parentVersion = meta.parentVersion;
        node.adapterName = meta.adapterName;
        node.trainingDate = meta.trainingDate;
        node.evalScore = meta.evalScore;
        node.finalLoss = meta.finalLoss;
        node.trainingSteps = meta.trainingSteps;
        node.status = meta.status;
        node.origin = meta.origin;
        node.generationId = meta.generationId;
        node.cycleId = meta.cycleId;
        node.variantIndex = meta.variantIndex;

        m_nodes.append(node);
    }

    // Rebuild child relationships
    for (auto &node : m_nodes) {
        node.childVersions.clear();
    }
    for (const auto &node : m_nodes) {
        if (node.parentVersion >= 0) {
            LineageNode *parent = findNode(node.parentVersion);
            if (parent && !parent->childVersions.contains(node.version)) {
                parent->childVersions.append(node.version);
            }
        }
    }

    saveLineage();
    emit lineageChanged();

    qDebug() << "LineageTracker: rebuilt" << m_nodes.size() << "lineage nodes";
}

void LineageTracker::recordNode(const LineageNode &node)
{
    // Remove existing node with same version if any
    for (int i = 0; i < m_nodes.size(); ++i) {
        if (m_nodes[i].version == node.version) {
            m_nodes.removeAt(i);
            break;
        }
    }

    m_nodes.append(node);

    // Update parent's child list
    if (node.parentVersion >= 0) {
        LineageNode *parent = findNode(node.parentVersion);
        if (parent && !parent->childVersions.contains(node.version)) {
            parent->childVersions.append(node.version);
        }
    }

    saveLineage();
    emit lineageChanged();
}

QList<LineageNode> LineageTracker::getAncestors(int version) const
{
    QList<LineageNode> ancestors;
    int current = version;

    while (current >= 0) {
        const LineageNode *node = findNode(current);
        if (!node) break;
        if (current != version) ancestors.append(*node);
        current = node->parentVersion;
    }

    return ancestors;
}

QList<LineageNode> LineageTracker::getDescendants(int version) const
{
    QList<LineageNode> descendants;
    QList<int> queue;

    const LineageNode *root = findNode(version);
    if (!root) return descendants;

    queue.append(root->childVersions);

    while (!queue.isEmpty()) {
        int ver = queue.takeFirst();
        const LineageNode *node = findNode(ver);
        if (!node) continue;
        descendants.append(*node);
        queue.append(node->childVersions);
    }

    return descendants;
}

QList<EraComparison> LineageTracker::getEraTimeline() const
{
    QList<EraComparison> timeline;

    // Group nodes by generation, find best of each era
    QMap<int, QList<const LineageNode*>> byGeneration;
    for (const auto &node : m_nodes) {
        byGeneration[node.generationId].append(&node);
    }

    for (auto it = byGeneration.begin(); it != byGeneration.end(); ++it) {
        int genId = it.key();
        const auto &nodes = it.value();

        // Find best scoring active/archived adapter in this generation
        const LineageNode *best = nullptr;
        int cyclesInGen = 0;

        QSet<int> seenCycles;
        for (const auto *node : nodes) {
            if (!best || node->evalScore > best->evalScore) {
                best = node;
            }
            if (node->cycleId >= 0) {
                seenCycles.insert(node->cycleId);
            }
        }
        cyclesInGen = seenCycles.size();

        if (best) {
            EraComparison era;
            era.eraLabel = QString("Gen %1").arg(genId);
            era.evalScore = best->evalScore;
            era.adapterVersion = best->version;
            era.cyclesCompleted = cyclesInGen;
            era.timestamp = best->trainingDate;

            if (m_memoryStore) {
                era.totalEpisodes = m_memoryStore->episodeCount();
                era.totalTraits = m_memoryStore->traitCount();
            }

            timeline.append(era);
        }
    }

    return timeline;
}

// --- QML interfaces ---

QVariantList LineageTracker::getLineageForQml() const
{
    QVariantList result;
    // Sort by version ascending
    QList<LineageNode> sorted = m_nodes;
    std::sort(sorted.begin(), sorted.end(),
              [](const LineageNode &a, const LineageNode &b) { return a.version < b.version; });

    for (const auto &node : sorted) {
        QVariantMap map;
        map["version"] = node.version;
        map["parentVersion"] = node.parentVersion;
        map["adapterName"] = node.adapterName;
        map["trainingDate"] = node.trainingDate.toString("yyyy-MM-dd hh:mm");
        map["evalScore"] = node.evalScore;
        map["finalLoss"] = static_cast<double>(node.finalLoss);
        map["trainingSteps"] = node.trainingSteps;
        map["status"] = node.status;
        map["origin"] = node.origin;
        map["generationId"] = node.generationId;
        map["cycleId"] = node.cycleId;
        map["variantIndex"] = node.variantIndex;
        map["childCount"] = node.childVersions.size();
        result.append(map);
    }
    return result;
}

QVariantList LineageTracker::getEraTimelineForQml() const
{
    QVariantList result;
    auto timeline = getEraTimeline();
    for (const auto &era : timeline) {
        QVariantMap map;
        map["eraLabel"] = era.eraLabel;
        map["evalScore"] = era.evalScore;
        map["adapterVersion"] = era.adapterVersion;
        map["totalEpisodes"] = era.totalEpisodes;
        map["totalTraits"] = era.totalTraits;
        map["cyclesCompleted"] = era.cyclesCompleted;
        map["timestamp"] = era.timestamp.toString("yyyy-MM-dd");
        result.append(map);
    }
    return result;
}

QVariantMap LineageTracker::getNodeForQml(int version) const
{
    const LineageNode *node = findNode(version);
    if (!node) return {};

    QVariantMap map;
    map["version"] = node->version;
    map["parentVersion"] = node->parentVersion;
    map["adapterName"] = node->adapterName;
    map["trainingDate"] = node->trainingDate.toString("yyyy-MM-dd hh:mm");
    map["evalScore"] = node->evalScore;
    map["finalLoss"] = static_cast<double>(node->finalLoss);
    map["trainingSteps"] = node->trainingSteps;
    map["status"] = node->status;
    map["origin"] = node->origin;
    map["generationId"] = node->generationId;
    map["cycleId"] = node->cycleId;
    map["variantIndex"] = node->variantIndex;

    QVariantList children;
    for (int child : node->childVersions) {
        children.append(child);
    }
    map["childVersions"] = children;

    // Ancestor chain
    auto ancestors = getAncestors(version);
    QVariantList ancestorVersions;
    for (const auto &a : ancestors) {
        ancestorVersions.append(a.version);
    }
    map["ancestors"] = ancestorVersions;
    map["depth"] = ancestors.size();

    return map;
}

// --- Private ---

void LineageTracker::loadLineage()
{
    if (m_lineageFilePath.isEmpty()) return;

    QFile file(m_lineageFilePath);
    if (!file.open(QIODevice::ReadOnly)) return;

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();

    if (!doc.isObject()) return;

    QJsonObject root = doc.object();
    QJsonArray nodesArr = root["nodes"].toArray();

    m_nodes.clear();
    for (const auto &val : nodesArr) {
        m_nodes.append(LineageNode::fromJson(val.toObject()));
    }
}

void LineageTracker::saveLineage() const
{
    if (m_lineageFilePath.isEmpty()) return;

    QJsonObject root;
    QJsonArray nodesArr;
    for (const auto &node : m_nodes) {
        nodesArr.append(node.toJson());
    }
    root["nodes"] = nodesArr;

    QFile file(m_lineageFilePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        file.close();
    }
}

LineageNode* LineageTracker::findNode(int version)
{
    for (auto &node : m_nodes) {
        if (node.version == version) return &node;
    }
    return nullptr;
}

const LineageNode* LineageTracker::findNode(int version) const
{
    for (const auto &node : m_nodes) {
        if (node.version == version) return &node;
    }
    return nullptr;
}

#endif // DATAFORM_TRAINING_ENABLED
