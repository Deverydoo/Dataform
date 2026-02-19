#ifdef DATAFORM_TRAINING_ENABLED

#include "populationstate.h"
#include "schemamigrator.h"

// --- VariantSpec ---

QJsonObject VariantSpec::toJson() const
{
    QJsonObject obj;
    obj["variantIndex"] = variantIndex;
    obj["learningRate"] = static_cast<double>(learningRate);
    obj["dataSeed"] = static_cast<qint64>(dataSeed);
    obj["dataSubsampleRatio"] = static_cast<double>(dataSubsampleRatio);
    obj["status"] = status;
    obj["checkpointPath"] = checkpointPath;
    obj["exportedModelPath"] = exportedModelPath;
    obj["trainingSteps"] = trainingSteps;
    obj["finalLoss"] = static_cast<double>(finalLoss);
    obj["evalScore"] = evalScore;
    obj["adapterVersion"] = adapterVersion;
    return obj;
}

VariantSpec VariantSpec::fromJson(const QJsonObject &json)
{
    VariantSpec spec;
    spec.variantIndex = json["variantIndex"].toInt();
    spec.learningRate = static_cast<float>(json["learningRate"].toDouble(1e-4));
    spec.dataSeed = static_cast<quint32>(json["dataSeed"].toInteger());
    spec.dataSubsampleRatio = static_cast<float>(json["dataSubsampleRatio"].toDouble(1.0));
    spec.status = json["status"].toString("pending");
    spec.checkpointPath = json["checkpointPath"].toString();
    spec.exportedModelPath = json["exportedModelPath"].toString();
    spec.trainingSteps = json["trainingSteps"].toInt();
    spec.finalLoss = static_cast<float>(json["finalLoss"].toDouble());
    spec.evalScore = json["evalScore"].toDouble();
    spec.adapterVersion = json["adapterVersion"].toInt(-1);
    return spec;
}

// --- PopulationState ---

QJsonObject PopulationState::toJson() const
{
    QJsonObject obj;
    obj["schemaVersion"] = SchemaMigrator::POPULATION_STATE_VERSION;
    obj["cycleId"] = cycleId;
    obj["populationSize"] = populationSize;
    obj["cycleStage"] = cycleStage;
    obj["currentVariantIndex"] = currentVariantIndex;
    obj["cycleStartTime"] = cycleStartTime.toString(Qt::ISODate);
    obj["championVersion"] = championVersion;
    obj["championScore"] = championScore;
    obj["variantTrainingPaused"] = variantTrainingPaused;

    QJsonArray variantsArray;
    for (const auto &v : variants) {
        variantsArray.append(v.toJson());
    }
    obj["variants"] = variantsArray;

    return obj;
}

PopulationState PopulationState::fromJson(const QJsonObject &json)
{
    // Run JSON schema migration if needed
    QJsonObject migrated = SchemaMigrator::migrateJson(json, {}, SchemaMigrator::POPULATION_STATE_VERSION);

    PopulationState state;
    state.cycleId = migrated["cycleId"].toInt();
    state.populationSize = migrated["populationSize"].toInt(4);
    state.cycleStage = migrated["cycleStage"].toString("idle");
    state.currentVariantIndex = migrated["currentVariantIndex"].toInt();
    state.cycleStartTime = QDateTime::fromString(migrated["cycleStartTime"].toString(), Qt::ISODate);
    state.championVersion = migrated["championVersion"].toInt(-1);
    state.championScore = migrated["championScore"].toDouble();
    state.variantTrainingPaused = migrated["variantTrainingPaused"].toBool();

    QJsonArray variantsArray = migrated["variants"].toArray();
    for (const auto &val : variantsArray) {
        state.variants.append(VariantSpec::fromJson(val.toObject()));
    }

    return state;
}

#endif // DATAFORM_TRAINING_ENABLED
