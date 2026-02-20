#ifndef TOOLREGISTRY_H
#define TOOLREGISTRY_H

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QMap>
#include "websearchengine.h"

class MemoryStore;
class ResearchStore;
class ReminderEngine;
class EmbeddingManager;

struct ToolDefinition {
    QString name;
    QString description;
    QJsonObject parameterSchema;
    bool isAsync = false;
};

struct ToolResult {
    QString toolName;
    bool success = false;
    QString resultText;
    QString errorMessage;
};

class ToolRegistry : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int toolCount READ toolCount CONSTANT)

public:
    explicit ToolRegistry(QObject *parent = nullptr);
    ~ToolRegistry();

    void setWebSearchEngine(WebSearchEngine *engine);
    void setMemoryStore(MemoryStore *store);
    void setResearchStore(ResearchStore *store);
    void setReminderEngine(ReminderEngine *engine);
    void setEmbeddingManager(EmbeddingManager *manager);

    void registerTool(const ToolDefinition &def);
    QString buildToolSchemaPrompt() const;
    void executeTool(const QString &name, const QJsonObject &params);
    bool hasTool(const QString &name) const;
    int toolCount() const { return m_tools.size(); }

signals:
    void toolExecutionComplete(const ToolResult &result);

private:
    void registerBuiltinTools();

    // Built-in tool implementations
    void executeWebSearch(const QJsonObject &params);
    ToolResult executeMemoryRecall(const QJsonObject &params);
    ToolResult executeResearchLookup(const QJsonObject &params);
    ToolResult executeSaveNote(const QJsonObject &params);
    ToolResult executeSetReminder(const QJsonObject &params);

    // Async completion handlers
    void onSearchResultsReady(const QList<SearchResult> &results);
    void onSearchError(const QString &error);

    QMap<QString, ToolDefinition> m_tools;

    WebSearchEngine *m_webSearch = nullptr;
    MemoryStore *m_memoryStore = nullptr;
    ResearchStore *m_researchStore = nullptr;
    ReminderEngine *m_reminderEngine = nullptr;
    EmbeddingManager *m_embeddingManager = nullptr;

    bool m_awaitingSearchResults = false;
};

#endif // TOOLREGISTRY_H
