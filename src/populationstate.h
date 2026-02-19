#ifndef POPULATIONSTATE_H
#define POPULATIONSTATE_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QString>
#include <QList>
#include <QDateTime>
#include <QJsonObject>
#include <QJsonArray>

struct VariantSpec {
    int variantIndex = 0;
    float learningRate = 1e-4f;
    quint32 dataSeed = 0;
    float dataSubsampleRatio = 1.0f;

    // Progress tracking
    QString status = "pending";     // pending, training, trained, evaluated, selected, rejected
    QString checkpointPath;
    QString exportedModelPath;
    int trainingSteps = 0;
    float finalLoss = 0.0f;
    double evalScore = 0.0;
    int adapterVersion = -1;

    QJsonObject toJson() const;
    static VariantSpec fromJson(const QJsonObject &json);
};

struct PopulationState {
    int cycleId = 0;
    int populationSize = 4;
    QString cycleStage = "idle";    // idle, seeding, training, evaluating, selecting, consolidating
    int currentVariantIndex = 0;
    QList<VariantSpec> variants;
    QDateTime cycleStartTime;
    int championVersion = -1;
    double championScore = 0.0;
    bool variantTrainingPaused = false;

    QJsonObject toJson() const;
    static PopulationState fromJson(const QJsonObject &json);
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // POPULATIONSTATE_H
