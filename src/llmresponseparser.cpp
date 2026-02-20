#include "llmresponseparser.h"
#include <QRegularExpression>
#include <QJsonParseError>
#include <QDebug>

QString LLMResponseParser::stripThinkTags(const QString &text)
{
    QString result = text;

    // 1. Strip closed <think>...</think> pairs
    static const QRegularExpression closedThink(
        QStringLiteral("<think>.*?</think>"),
        QRegularExpression::DotMatchesEverythingOption);
    result.remove(closedThink);

    // 2. Strip unclosed <think> tags (everything from <think> to end)
    static const QRegularExpression unclosedThink(
        QStringLiteral("<think>.*$"),
        QRegularExpression::DotMatchesEverythingOption);
    result.remove(unclosedThink);

    return result.trimmed();
}

QString LLMResponseParser::stripMarkdownFences(const QString &text)
{
    QString result = text.trimmed();
    if (!result.startsWith(QLatin1String("```")))
        return result;

    int firstNewline = result.indexOf('\n');
    int lastFence = result.lastIndexOf(QLatin1String("```"));
    if (firstNewline >= 0 && lastFence > firstNewline) {
        result = result.mid(firstNewline + 1, lastFence - firstNewline - 1).trimmed();
    }
    return result;
}

QString LLMResponseParser::cleanResponse(const QString &text)
{
    QString result = stripThinkTags(text);
    result = stripMarkdownFences(result);
    return result.trimmed();
}

QJsonDocument LLMResponseParser::extractJsonObject(const QString &text,
                                                   const QString &callerContext)
{
    QString cleaned = cleanResponse(text);

    // Find outermost { ... } boundaries
    int start = cleaned.indexOf('{');
    int end = cleaned.lastIndexOf('}');
    if (start < 0 || end < 0 || end <= start) {
        if (!callerContext.isEmpty())
            qWarning() << callerContext << ": no JSON object boundaries found in response";
        return QJsonDocument();
    }

    QString jsonStr = cleaned.mid(start, end - start + 1);

    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8(), &parseError);

    if (parseError.error != QJsonParseError::NoError) {
        if (!callerContext.isEmpty())
            qWarning() << callerContext << ": JSON parse error:" << parseError.errorString()
                       << "at offset:" << parseError.offset
                       << "in:" << jsonStr.left(500);
        return QJsonDocument();
    }

    if (!doc.isObject()) {
        if (!callerContext.isEmpty())
            qWarning() << callerContext << ": expected JSON object but got array";
        return QJsonDocument();
    }

    return doc;
}

QJsonDocument LLMResponseParser::extractJsonArray(const QString &text,
                                                  const QString &callerContext)
{
    QString cleaned = cleanResponse(text);

    // Find outermost [ ... ] boundaries
    int start = cleaned.indexOf('[');
    int end = cleaned.lastIndexOf(']');
    if (start < 0 || end < 0 || end <= start) {
        if (!callerContext.isEmpty())
            qWarning() << callerContext << ": no JSON array boundaries found in response";
        return QJsonDocument();
    }

    QString jsonStr = cleaned.mid(start, end - start + 1);

    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8(), &parseError);

    if (parseError.error != QJsonParseError::NoError) {
        if (!callerContext.isEmpty())
            qWarning() << callerContext << ": JSON parse error:" << parseError.errorString()
                       << "at offset:" << parseError.offset
                       << "in:" << jsonStr.left(500);
        return QJsonDocument();
    }

    if (!doc.isArray()) {
        if (!callerContext.isEmpty())
            qWarning() << callerContext << ": expected JSON array but got object";
        return QJsonDocument();
    }

    return doc;
}

ParsedLLMResponse LLMResponseParser::parseToolCalls(const QString &response)
{
    ParsedLLMResponse result;

    // Strip think tags first (but NOT markdown fences — the text content is conversational)
    QString cleaned = stripThinkTags(response);

    // Extract all <tool_call>...</tool_call> blocks
    static const QRegularExpression toolCallRegex(
        QStringLiteral("<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>"),
        QRegularExpression::DotMatchesEverythingOption);

    QRegularExpressionMatchIterator it = toolCallRegex.globalMatch(cleaned);

    // Remove tool_call blocks from text to get the conversational content
    QString textContent = cleaned;
    while (it.hasNext()) {
        QRegularExpressionMatch match = it.next();
        QString jsonStr = match.captured(1);

        QJsonParseError parseError;
        QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8(), &parseError);

        if (parseError.error != QJsonParseError::NoError) {
            qWarning() << "LLMResponseParser: malformed tool_call JSON:"
                       << parseError.errorString() << "in:" << jsonStr.left(300);
            continue;
        }

        if (!doc.isObject()) {
            qWarning() << "LLMResponseParser: tool_call is not a JSON object";
            continue;
        }

        QJsonObject obj = doc.object();
        QString toolName = obj.value(QLatin1String("tool")).toString();
        QJsonObject params = obj.value(QLatin1String("params")).toObject();

        if (toolName.isEmpty()) {
            qWarning() << "LLMResponseParser: tool_call missing 'tool' field";
            continue;
        }

        result.toolCalls.append({toolName, params});

        // Remove this match from the text content
        textContent.replace(match.captured(0), QString());
    }

    // Clean up excess whitespace from removed blocks
    static const QRegularExpression multiNewlines(QStringLiteral("\\n{3,}"));
    textContent.replace(multiNewlines, QStringLiteral("\n\n"));
    result.textContent = textContent.trimmed();

    return result;
}
