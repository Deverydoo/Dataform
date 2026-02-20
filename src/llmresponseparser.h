#ifndef LLMRESPONSEPARSER_H
#define LLMRESPONSEPARSER_H

#include <QString>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QList>

struct ToolCallRequest {
    QString toolName;
    QJsonObject params;
};

struct ParsedLLMResponse {
    QString textContent;
    QList<ToolCallRequest> toolCalls;
    bool hasToolCalls() const { return !toolCalls.isEmpty(); }
};

class LLMResponseParser
{
public:
    // Strip <think>...</think> blocks (closed and unclosed)
    static QString stripThinkTags(const QString &text);

    // Strip markdown code fences (```json\n...\n```)
    static QString stripMarkdownFences(const QString &text);

    // Full cleaning: stripThinkTags + stripMarkdownFences + trim
    static QString cleanResponse(const QString &text);

    // Extract a JSON object from mixed text. Returns null doc on failure.
    static QJsonDocument extractJsonObject(const QString &text,
                                           const QString &callerContext = QString());

    // Extract a JSON array from mixed text. Returns null doc on failure.
    static QJsonDocument extractJsonArray(const QString &text,
                                          const QString &callerContext = QString());

    // Parse tool calls from LLM response: <tool_call>{...}</tool_call>
    // Returns text with tool_call blocks removed + list of parsed calls.
    static ParsedLLMResponse parseToolCalls(const QString &response);

private:
    LLMResponseParser() = delete;
};

#endif // LLMRESPONSEPARSER_H
