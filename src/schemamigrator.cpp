#include "schemamigrator.h"
#include <QSqlQuery>
#include <QSqlError>
#include <QDebug>

int SchemaMigrator::getDatabaseVersion(QSqlDatabase &db)
{
    QSqlQuery query(db);
    if (!query.exec("PRAGMA user_version")) {
        qWarning() << "SchemaMigrator: Failed to read PRAGMA user_version:"
                   << query.lastError().text();
        return 0;
    }
    if (query.next()) {
        return query.value(0).toInt();
    }
    return 0;
}

bool SchemaMigrator::setDatabaseVersion(QSqlDatabase &db, int version)
{
    QSqlQuery query(db);
    if (!query.exec(QString("PRAGMA user_version = %1").arg(version))) {
        qWarning() << "SchemaMigrator: Failed to set PRAGMA user_version:"
                   << query.lastError().text();
        return false;
    }
    return true;
}

bool SchemaMigrator::migrateDatabase(QSqlDatabase &db,
                                      const QList<DbMigration> &migrations,
                                      int targetVersion)
{
    int currentVersion = getDatabaseVersion(db);

    if (currentVersion >= targetVersion) {
        return true; // Already up to date
    }

    qDebug() << "SchemaMigrator: Migrating database from version"
             << currentVersion << "to" << targetVersion;

    for (const auto &migration : migrations) {
        if (migration.fromVersion < currentVersion) {
            continue; // Already applied
        }
        if (migration.fromVersion >= targetVersion) {
            break; // Beyond target
        }

        qDebug() << "SchemaMigrator: Applying migration:" << migration.description
                 << "(" << migration.fromVersion << "->" << migration.toVersion << ")";

        // Run migration in a transaction
        if (!db.transaction()) {
            qWarning() << "SchemaMigrator: Failed to begin transaction";
            return false;
        }

        if (!migration.migrate(db)) {
            qWarning() << "SchemaMigrator: Migration failed:" << migration.description;
            db.rollback();
            return false;
        }

        if (!db.commit()) {
            qWarning() << "SchemaMigrator: Failed to commit migration";
            db.rollback();
            return false;
        }

        currentVersion = migration.toVersion;
    }

    // Set the final version
    if (!setDatabaseVersion(db, targetVersion)) {
        return false;
    }

    qDebug() << "SchemaMigrator: Database migration complete, now at version" << targetVersion;
    return true;
}

int SchemaMigrator::getJsonVersion(const QJsonObject &json)
{
    return json.value("schemaVersion").toInt(0);
}

QJsonObject SchemaMigrator::setJsonVersion(const QJsonObject &json, int version)
{
    QJsonObject result = json;
    result["schemaVersion"] = version;
    return result;
}

QJsonObject SchemaMigrator::migrateJson(const QJsonObject &json,
                                          const QList<JsonMigration> &migrations,
                                          int targetVersion)
{
    int currentVersion = getJsonVersion(json);

    if (currentVersion >= targetVersion) {
        return json; // Already up to date
    }

    qDebug() << "SchemaMigrator: Migrating JSON from version"
             << currentVersion << "to" << targetVersion;

    QJsonObject result = json;

    for (const auto &migration : migrations) {
        if (migration.fromVersion < currentVersion) {
            continue; // Already applied
        }
        if (migration.fromVersion >= targetVersion) {
            break;
        }

        qDebug() << "SchemaMigrator: Applying JSON migration:" << migration.description;
        result = migration.migrate(result);
    }

    // Stamp the final version
    result["schemaVersion"] = targetVersion;

    qDebug() << "SchemaMigrator: JSON migration complete, now at version" << targetVersion;
    return result;
}
