#ifndef SCHEMAMIGRATOR_H
#define SCHEMAMIGRATOR_H

#include <QString>
#include <QList>
#include <QJsonObject>
#include <QSqlDatabase>
#include <functional>

class SchemaMigrator
{
public:
    // --- Database migration ---

    struct DbMigration {
        int fromVersion;
        int toVersion;
        QString description;
        std::function<bool(QSqlDatabase &db)> migrate;
    };

    static int getDatabaseVersion(QSqlDatabase &db);
    static bool setDatabaseVersion(QSqlDatabase &db, int version);
    static bool migrateDatabase(QSqlDatabase &db,
                                const QList<DbMigration> &migrations,
                                int targetVersion);

    // --- JSON config migration ---

    struct JsonMigration {
        int fromVersion;
        int toVersion;
        QString description;
        std::function<QJsonObject(const QJsonObject &)> migrate;
    };

    static int getJsonVersion(const QJsonObject &json);
    static QJsonObject setJsonVersion(const QJsonObject &json, int version);
    static QJsonObject migrateJson(const QJsonObject &json,
                                    const QList<JsonMigration> &migrations,
                                    int targetVersion);

    // --- Current schema version constants ---
    static constexpr int EPISODIC_DB_VERSION = 10;
    static constexpr int TRAITS_DB_VERSION = 2;
    static constexpr int SETTINGS_VERSION = 2;
    static constexpr int ADAPTER_INDEX_VERSION = 1;
    static constexpr int ADAPTER_METADATA_VERSION = 1;
    static constexpr int POPULATION_STATE_VERSION = 1;
};

#endif // SCHEMAMIGRATOR_H
