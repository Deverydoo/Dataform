#ifndef MODELGENERATION_H
#define MODELGENERATION_H

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QJsonObject>
#include <QList>

struct ModelGenerationConfig {
    int generationId = 0;
    QString modelFamily = "qwen2.5";
    QString modelVariant = "1.5B";
    QString modelFullName = "Qwen/Qwen2.5-1.5B";
    QString chatTemplate = "chatml";
    int maxSequenceLength = 512;
    int loraRank = 8;

    // Special token names
    QString imStartToken = "<|im_start|>";
    QString imEndToken = "<|im_end|>";
    QString eosToken = "<|endoftext|>";

    // Role names
    QString systemRole = "system";
    QString userRole = "user";
    QString assistantRole = "assistant";

    QString status = "active";          // "active" or "archived"
    QDateTime createdAt;
    QString artifactsSubdir = "training_artifacts";

    QJsonObject toJson() const;
    static ModelGenerationConfig fromJson(const QJsonObject &json);
};

class ModelGenerationManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int currentGenerationId READ currentGenerationId NOTIFY currentGenerationChanged)
    Q_PROPERTY(QString modelFamily READ modelFamily NOTIFY currentGenerationChanged)
    Q_PROPERTY(QString modelVariant READ modelVariant NOTIFY currentGenerationChanged)

public:
    explicit ModelGenerationManager(const QString &profilePath, QObject *parent = nullptr);

    void loadGenerations();

    ModelGenerationConfig currentGeneration() const;
    int currentGenerationId() const;
    QString modelFamily() const;
    QString modelVariant() const;

    bool archiveGeneration(int id);
    int createGeneration(const ModelGenerationConfig &config);

    QString artifactsPath() const;

    Q_INVOKABLE QVariantMap currentGenerationForQml() const;

signals:
    void currentGenerationChanged();
    void generationsLoaded();

private:
    void saveGenerations() const;
    void createDefaultGeneration();

    QString m_profilePath;
    QString m_generationsFilePath;
    QList<ModelGenerationConfig> m_generations;
    int m_activeGenerationId = 0;
};

#endif // MODELGENERATION_H
