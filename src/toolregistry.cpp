#include "toolregistry.h"
#include "memorystore.h"
#include "researchstore.h"
#include "reminderengine.h"
#include "embeddingmanager.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonArray>

ToolRegistry::ToolRegistry(QObject *parent)
    : QObject(parent)
{
    registerBuiltinTools();
}

ToolRegistry::~ToolRegistry() = default;

void ToolRegistry::setWebSearchEngine(WebSearchEngine *engine)
{
    if (m_webSearch) {
        disconnect(m_webSearch, nullptr, this, nullptr);
    }
    m_webSearch = engine;
    if (m_webSearch) {
        connect(m_webSearch, &WebSearchEngine::searchResultsReady,
                this, &ToolRegistry::onSearchResultsReady);
        connect(m_webSearch, &WebSearchEngine::searchError,
                this, &ToolRegistry::onSearchError);
    }
}

void ToolRegistry::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void ToolRegistry::setResearchStore(ResearchStore *store) { m_researchStore = store; }
void ToolRegistry::setReminderEngine(ReminderEngine *engine) { m_reminderEngine = engine; }
void ToolRegistry::setEmbeddingManager(EmbeddingManager *manager) { m_embeddingManager = manager; }

void ToolRegistry::registerTool(const ToolDefinition &def)
{
    m_tools.insert(def.name, def);
    qDebug() << "ToolRegistry: registered tool:" << def.name;
}

bool ToolRegistry::hasTool(const QString &name) const
{
    return m_tools.contains(name);
}

void ToolRegistry::registerBuiltinTools()
{
    // web_search
    {
        ToolDefinition def;
        def.name = "web_search";
        def.description = "Search the web for current information. Use when the user asks about recent events, facts you're unsure about, or anything that benefits from up-to-date data.";
        def.isAsync = true;
        QJsonObject params;
        QJsonObject query;
        query["type"] = "string";
        query["description"] = "Search query";
        query["required"] = true;
        params["query"] = query;
        def.parameterSchema = params;
        registerTool(def);
    }

    // memory_recall
    {
        ToolDefinition def;
        def.name = "memory_recall";
        def.description = "Search past conversations with the user. Use when they reference something discussed before or you need context from previous interactions.";
        def.isAsync = false;
        QJsonObject params;
        QJsonObject query;
        query["type"] = "string";
        query["description"] = "Search keywords for past conversations";
        query["required"] = true;
        params["query"] = query;
        def.parameterSchema = params;
        registerTool(def);
    }

    // research_lookup
    {
        ToolDefinition def;
        def.name = "research_lookup";
        def.description = "Look up previously researched topics from DATAFORM's research library.";
        def.isAsync = false;
        QJsonObject params;
        QJsonObject topic;
        topic["type"] = "string";
        topic["description"] = "Topic to look up";
        topic["required"] = true;
        params["topic"] = topic;
        def.parameterSchema = params;
        registerTool(def);
    }

    // save_note
    {
        ToolDefinition def;
        def.name = "save_note";
        def.description = "Save an important piece of information the user wants to remember.";
        def.isAsync = false;
        QJsonObject params;
        QJsonObject content;
        content["type"] = "string";
        content["description"] = "The note to save";
        content["required"] = true;
        params["content"] = content;
        def.parameterSchema = params;
        registerTool(def);
    }

    // set_reminder
    {
        ToolDefinition def;
        def.name = "set_reminder";
        def.description = "Set a reminder for the user at a specific time or after a delay.";
        def.isAsync = false;
        QJsonObject params;
        QJsonObject content;
        content["type"] = "string";
        content["description"] = "What to remind about";
        content["required"] = true;
        params["content"] = content;
        QJsonObject when;
        when["type"] = "string";
        when["description"] = "When to remind, e.g. 'in 30 minutes', 'tomorrow at 3pm'";
        when["required"] = true;
        params["when"] = when;
        def.parameterSchema = params;
        registerTool(def);
    }
}

QString ToolRegistry::buildToolSchemaPrompt() const
{
    if (m_tools.isEmpty()) return QString();

    QString prompt = QStringLiteral(
        "\nAVAILABLE TOOLS:\n"
        "You can use tools by including a <tool_call> block in your response. Format:\n"
        "<tool_call>\n"
        "{\"tool\": \"tool_name\", \"params\": {\"param1\": \"value1\"}}\n"
        "</tool_call>\n\n"
        "You may include text before or after a tool call. "
        "After a tool is executed, you will receive the result and can respond to the user.\n\n"
        "Tools:\n");

    for (auto it = m_tools.constBegin(); it != m_tools.constEnd(); ++it) {
        const ToolDefinition &def = it.value();
        prompt += QStringLiteral("- %1: %2\n").arg(def.name, def.description);

        // List parameters
        QStringList paramDescs;
        const QJsonObject &params = def.parameterSchema;
        for (auto pit = params.constBegin(); pit != params.constEnd(); ++pit) {
            QJsonObject paramDef = pit.value().toObject();
            QString desc = QStringLiteral("  %1 (%2)")
                .arg(pit.key(), paramDef.value("type").toString("string"));
            QString paramDesc = paramDef.value("description").toString();
            if (!paramDesc.isEmpty()) desc += QStringLiteral(" - %1").arg(paramDesc);
            paramDescs.append(desc);
        }
        if (!paramDescs.isEmpty()) {
            prompt += QStringLiteral("  Parameters:\n");
            for (const QString &pd : paramDescs) {
                prompt += QStringLiteral("  %1\n").arg(pd);
            }
        }
    }

    prompt += QStringLiteral(
        "\nIf you don't need a tool, just respond normally. "
        "Only use tools when they would genuinely help answer the user's question.\n");

    return prompt;
}

void ToolRegistry::executeTool(const QString &name, const QJsonObject &params)
{
    if (!m_tools.contains(name)) {
        ToolResult result;
        result.toolName = name;
        result.success = false;
        result.errorMessage = QStringLiteral("Unknown tool: %1").arg(name);
        emit toolExecutionComplete(result);
        return;
    }

    qDebug() << "ToolRegistry: executing tool:" << name << "params:" << params;

    if (name == QLatin1String("web_search")) {
        executeWebSearch(params);
        return;  // Async — result comes via signal
    }

    ToolResult result;
    if (name == QLatin1String("memory_recall")) {
        result = executeMemoryRecall(params);
    } else if (name == QLatin1String("research_lookup")) {
        result = executeResearchLookup(params);
    } else if (name == QLatin1String("save_note")) {
        result = executeSaveNote(params);
    } else if (name == QLatin1String("set_reminder")) {
        result = executeSetReminder(params);
    } else {
        result.toolName = name;
        result.success = false;
        result.errorMessage = QStringLiteral("Tool '%1' has no implementation").arg(name);
    }

    emit toolExecutionComplete(result);
}

// --- Built-in tool implementations ---

void ToolRegistry::executeWebSearch(const QJsonObject &params)
{
    QString query = params.value("query").toString().trimmed();
    if (query.isEmpty()) {
        ToolResult result;
        result.toolName = "web_search";
        result.success = false;
        result.errorMessage = "Missing required parameter: query";
        emit toolExecutionComplete(result);
        return;
    }

    if (!m_webSearch) {
        ToolResult result;
        result.toolName = "web_search";
        result.success = false;
        result.errorMessage = "Web search is not available";
        emit toolExecutionComplete(result);
        return;
    }

    if (!m_webSearch->canSearch()) {
        ToolResult result;
        result.toolName = "web_search";
        result.success = false;
        result.errorMessage = "Search rate limited. Please wait a moment before searching again.";
        emit toolExecutionComplete(result);
        return;
    }

    m_awaitingSearchResults = true;
    m_webSearch->search(query, 5);
}

ToolResult ToolRegistry::executeMemoryRecall(const QJsonObject &params)
{
    ToolResult result;
    result.toolName = "memory_recall";

    QString query = params.value("query").toString().trimmed();
    if (query.isEmpty()) {
        result.success = false;
        result.errorMessage = "Missing required parameter: query";
        return result;
    }

    if (!m_memoryStore) {
        result.success = false;
        result.errorMessage = "Memory store is not available";
        return result;
    }

    QList<EpisodicRecord> episodes = m_memoryStore->searchEpisodes(query, 5);
    if (episodes.isEmpty()) {
        result.success = true;
        result.resultText = "No past conversations found matching that query.";
        return result;
    }

    QString text;
    for (int i = 0; i < episodes.size(); ++i) {
        const EpisodicRecord &ep = episodes[i];
        text += QStringLiteral("%1. [%2] Topic: %3\n")
            .arg(i + 1)
            .arg(ep.timestamp.toString("yyyy-MM-dd"))
            .arg(ep.topic.isEmpty() ? "general" : ep.topic);
        if (!ep.userText.isEmpty())
            text += QStringLiteral("   User: %1\n").arg(ep.userText.left(200));
        if (!ep.assistantText.isEmpty())
            text += QStringLiteral("   Assistant: %1\n").arg(ep.assistantText.left(200));
        text += "\n";
    }

    result.success = true;
    result.resultText = QStringLiteral("Found %1 past conversation(s):\n\n%2")
        .arg(episodes.size()).arg(text);
    return result;
}

ToolResult ToolRegistry::executeResearchLookup(const QJsonObject &params)
{
    ToolResult result;
    result.toolName = "research_lookup";

    QString topic = params.value("topic").toString().trimmed();
    if (topic.isEmpty()) {
        result.success = false;
        result.errorMessage = "Missing required parameter: topic";
        return result;
    }

    if (!m_researchStore) {
        result.success = false;
        result.errorMessage = "Research store is not available";
        return result;
    }

    QList<ResearchFinding> findings = m_researchStore->searchFindings(topic, 5);
    if (findings.isEmpty()) {
        result.success = true;
        result.resultText = "No research findings found for that topic.";
        return result;
    }

    QString text;
    for (int i = 0; i < findings.size(); ++i) {
        const ResearchFinding &f = findings[i];
        text += QStringLiteral("%1. %2\n").arg(i + 1).arg(f.sourceTitle.isEmpty() ? f.topic : f.sourceTitle);
        if (!f.llmSummary.isEmpty())
            text += QStringLiteral("   Summary: %1\n").arg(f.llmSummary.left(300));
        if (!f.sourceUrl.isEmpty())
            text += QStringLiteral("   Source: %1\n").arg(f.sourceUrl);
        text += "\n";
    }

    result.success = true;
    result.resultText = QStringLiteral("Found %1 research finding(s):\n\n%2")
        .arg(findings.size()).arg(text);
    return result;
}

ToolResult ToolRegistry::executeSaveNote(const QJsonObject &params)
{
    ToolResult result;
    result.toolName = "save_note";

    QString content = params.value("content").toString().trimmed();
    if (content.isEmpty()) {
        result.success = false;
        result.errorMessage = "Missing required parameter: content";
        return result;
    }

    if (!m_memoryStore) {
        result.success = false;
        result.errorMessage = "Memory store is not available";
        return result;
    }

    qint64 id = m_memoryStore->insertEpisode(
        "saved_note", content, "Note saved by DATAFORM.", "", "");
    if (id >= 0) {
        result.success = true;
        result.resultText = QStringLiteral("Note saved successfully (id: %1).").arg(id);
    } else {
        result.success = false;
        result.errorMessage = "Failed to save note to memory store.";
    }
    return result;
}

ToolResult ToolRegistry::executeSetReminder(const QJsonObject &params)
{
    ToolResult result;
    result.toolName = "set_reminder";

    QString content = params.value("content").toString().trimmed();
    QString when = params.value("when").toString().trimmed();
    if (content.isEmpty()) {
        result.success = false;
        result.errorMessage = "Missing required parameter: content";
        return result;
    }

    if (!m_reminderEngine) {
        result.success = false;
        result.errorMessage = "Reminder engine is not available";
        return result;
    }

    // Compose the reminder text so ReminderEngine can parse the time expression
    QString reminderText = content;
    if (!when.isEmpty()) {
        reminderText = QStringLiteral("remind me %1 to %2").arg(when, content);
    }
    m_reminderEngine->detectAndStore(reminderText, -1);

    result.success = true;
    result.resultText = QStringLiteral("Reminder set: %1").arg(content);
    if (!when.isEmpty()) result.resultText += QStringLiteral(" (%1)").arg(when);
    return result;
}

// --- Async completion handlers ---

void ToolRegistry::onSearchResultsReady(const QList<SearchResult> &results)
{
    if (!m_awaitingSearchResults) return;
    m_awaitingSearchResults = false;

    ToolResult result;
    result.toolName = "web_search";
    result.success = true;

    if (results.isEmpty()) {
        result.resultText = "No search results found.";
    } else {
        QString text;
        for (int i = 0; i < results.size(); ++i) {
            const SearchResult &sr = results[i];
            text += QStringLiteral("%1. %2\n").arg(i + 1).arg(sr.title);
            if (!sr.snippet.isEmpty())
                text += QStringLiteral("   %1\n").arg(sr.snippet);
            if (!sr.url.isEmpty())
                text += QStringLiteral("   URL: %1\n").arg(sr.url);
            text += "\n";
        }
        result.resultText = QStringLiteral("Search results:\n\n%1").arg(text);
    }

    emit toolExecutionComplete(result);
}

void ToolRegistry::onSearchError(const QString &error)
{
    if (!m_awaitingSearchResults) return;
    m_awaitingSearchResults = false;

    ToolResult result;
    result.toolName = "web_search";
    result.success = false;
    result.errorMessage = QStringLiteral("Web search failed: %1").arg(error);
    emit toolExecutionComplete(result);
}
