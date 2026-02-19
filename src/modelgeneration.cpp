#include "modelgeneration.h"
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDir>
#include <QDebug>

// --- ModelGenerationConfig ---

QJsonObject ModelGenerationConfig::toJson() const
{
    QJsonObject obj;
    obj["generationId"] = generationId;
    obj["modelFamily"] = modelFamily;
    obj["modelVariant"] = modelVariant;
    obj["modelFullName"] = modelFullName;
    obj["chatTemplate"] = chatTemplate;
    obj["maxSequenceLength"] = maxSequenceLength;
    obj["loraRank"] = loraRank;
    obj["imStartToken"] = imStartToken;
    obj["imEndToken"] = imEndToken;
    obj["eosToken"] = eosToken;
    obj["systemRole"] = systemRole;
    obj["userRole"] = userRole;
    obj["assistantRole"] = assistantRole;
    obj["status"] = status;
    obj["createdAt"] = createdAt.toString(Qt::ISODate);
    obj["artifactsSubdir"] = artifactsSubdir;
    return obj;
}

ModelGenerationConfig ModelGenerationConfig::fromJson(const QJsonObject &json)
{
    ModelGenerationConfig config;
    config.generationId = json["generationId"].toInt(0);
    config.modelFamily = json["modelFamily"].toString("qwen2.5");
    config.modelVariant = json["modelVariant"].toString("1.5B");
    config.modelFullName = json["modelFullName"].toString("Qwen/Qwen2.5-1.5B");
    config.chatTemplate = json["chatTemplate"].toString("chatml");
    config.maxSequenceLength = json["maxSequenceLength"].toInt(512);
    config.loraRank = json["loraRank"].toInt(8);
    config.imStartToken = json["imStartToken"].toString("<|im_start|>");
    config.imEndToken = json["imEndToken"].toString("<|im_end|>");
    config.eosToken = json["eosToken"].toString("<|endoftext|>");
    config.systemRole = json["systemRole"].toString("system");
    config.userRole = json["userRole"].toString("user");
    config.assistantRole = json["assistantRole"].toString("assistant");
    config.status = json["status"].toString("active");
    config.createdAt = QDateTime::fromString(json["createdAt"].toString(), Qt::ISODate);
    config.artifactsSubdir = json["artifactsSubdir"].toString("training_artifacts");
    return config;
}

// --- ModelGenerationManager ---

ModelGenerationManager::ModelGenerationManager(const QString &profilePath, QObject *parent)
    : QObject(parent)
    , m_profilePath(profilePath)
    , m_generationsFilePath(profilePath + "/model_generations.json")
{
}

void ModelGenerationManager::loadGenerations()
{
    QFile file(m_generationsFilePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "ModelGenerationManager: No generations file found, creating default";
        createDefaultGeneration();
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();

    if (!doc.isObject()) {
        qWarning() << "ModelGenerationManager: Invalid generations file, creating default";
        createDefaultGeneration();
        return;
    }

    QJsonObject root = doc.object();
    m_activeGenerationId = root["activeGenerationId"].toInt(0);

    m_generations.clear();
    QJsonArray arr = root["generations"].toArray();
    for (const auto &val : arr) {
        m_generations.append(ModelGenerationConfig::fromJson(val.toObject()));
    }

    if (m_generations.isEmpty()) {
        createDefaultGeneration();
        return;
    }

    emit currentGenerationChanged();
    emit generationsLoaded();
    qDebug() << "ModelGenerationManager: loaded" << m_generations.size()
             << "generations, active:" << m_activeGenerationId;
}

ModelGenerationConfig ModelGenerationManager::currentGeneration() const
{
    for (const auto &gen : m_generations) {
        if (gen.generationId == m_activeGenerationId) return gen;
    }
    // Fallback: return default
    ModelGenerationConfig fallback;
    fallback.generationId = 0;
    return fallback;
}

int ModelGenerationManager::currentGenerationId() const
{
    return m_activeGenerationId;
}

QString ModelGenerationManager::modelFamily() const
{
    return currentGeneration().modelFamily;
}

QString ModelGenerationManager::modelVariant() const
{
    return currentGeneration().modelVariant;
}

bool ModelGenerationManager::archiveGeneration(int id)
{
    for (auto &gen : m_generations) {
        if (gen.generationId == id) {
            gen.status = "archived";
            saveGenerations();
            qDebug() << "ModelGenerationManager: archived generation" << id;
            return true;
        }
    }
    return false;
}

int ModelGenerationManager::createGeneration(const ModelGenerationConfig &config)
{
    // Determine next ID
    int maxId = -1;
    for (const auto &gen : m_generations) {
        if (gen.generationId > maxId) maxId = gen.generationId;
    }

    ModelGenerationConfig newGen = config;
    newGen.generationId = maxId + 1;
    newGen.status = "active";
    if (!newGen.createdAt.isValid()) {
        newGen.createdAt = QDateTime::currentDateTime();
    }

    // Archive current active generation
    for (auto &gen : m_generations) {
        if (gen.status == "active") {
            gen.status = "archived";
        }
    }

    m_generations.append(newGen);
    m_activeGenerationId = newGen.generationId;

    saveGenerations();
    emit currentGenerationChanged();

    qDebug() << "ModelGenerationManager: created generation" << newGen.generationId
             << "(" << newGen.modelFullName << ")";
    return newGen.generationId;
}

QString ModelGenerationManager::artifactsPath() const
{
    return m_profilePath + "/" + currentGeneration().artifactsSubdir;
}

QVariantMap ModelGenerationManager::currentGenerationForQml() const
{
    auto gen = currentGeneration();
    QVariantMap map;
    map["generationId"] = gen.generationId;
    map["modelFamily"] = gen.modelFamily;
    map["modelVariant"] = gen.modelVariant;
    map["modelFullName"] = gen.modelFullName;
    map["chatTemplate"] = gen.chatTemplate;
    map["maxSequenceLength"] = gen.maxSequenceLength;
    map["loraRank"] = gen.loraRank;
    map["status"] = gen.status;
    map["createdAt"] = gen.createdAt.toString("yyyy-MM-dd");
    return map;
}

void ModelGenerationManager::saveGenerations() const
{
    QJsonObject root;
    root["activeGenerationId"] = m_activeGenerationId;

    QJsonArray arr;
    for (const auto &gen : m_generations) {
        arr.append(gen.toJson());
    }
    root["generations"] = arr;

    QFile file(m_generationsFilePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        file.close();
    }
}

void ModelGenerationManager::createDefaultGeneration()
{
    m_generations.clear();

    ModelGenerationConfig gen0;
    gen0.generationId = 0;
    gen0.modelFamily = "qwen2.5";
    gen0.modelVariant = "1.5B";
    gen0.modelFullName = "Qwen/Qwen2.5-1.5B";
    gen0.chatTemplate = "chatml";
    gen0.maxSequenceLength = 512;
    gen0.loraRank = 8;
    gen0.status = "active";
    gen0.createdAt = QDateTime::currentDateTime();

    m_generations.append(gen0);
    m_activeGenerationId = 0;

    saveGenerations();
    emit currentGenerationChanged();
    emit generationsLoaded();

    qDebug() << "ModelGenerationManager: created default generation 0 (Qwen2.5-1.5B)";
}
