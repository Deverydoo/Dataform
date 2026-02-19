#include "websearchengine.h"
#include <QNetworkRequest>
#include <QUrlQuery>
#include <QUrl>
#include <QRegularExpression>
#include <QDebug>

WebSearchEngine::WebSearchEngine(QObject *parent)
    : QObject(parent)
    , m_networkManager(new QNetworkAccessManager(this))
{
}

WebSearchEngine::~WebSearchEngine()
{
}

bool WebSearchEngine::canSearch() const
{
    if (m_isSearching) return false;
    if (!m_lastSearchTime.isValid()) return true;
    return m_lastSearchTime.secsTo(QDateTime::currentDateTime()) >= MIN_SEARCH_INTERVAL_SEC;
}

void WebSearchEngine::search(const QString &query, int maxResults)
{
    if (!canSearch()) {
        emit searchError("Search rate limited. Please wait.");
        return;
    }

    m_isSearching = true;
    m_pendingMaxResults = maxResults;
    emit isSearchingChanged();

    QUrl url("https://html.duckduckgo.com/html/");
    QUrlQuery params;
    params.addQueryItem("q", query);
    url.setQuery(params);

    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::UserAgentHeader,
                      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");
    request.setTransferTimeout(REQUEST_TIMEOUT_MS);

    QNetworkReply *reply = m_networkManager->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        onSearchReply(reply);
    });

    m_lastSearchTime = QDateTime::currentDateTime();
    qDebug() << "WebSearchEngine: searching DuckDuckGo for:" << query;
}

void WebSearchEngine::fetchPage(const QString &urlStr)
{
    QUrl url(urlStr);
    if (!url.isValid()) {
        emit searchError("Invalid URL: " + urlStr);
        return;
    }

    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::UserAgentHeader,
                      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");
    request.setTransferTimeout(REQUEST_TIMEOUT_MS);
    request.setMaximumRedirectsAllowed(3);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::NoLessSafeRedirectPolicy);

    QNetworkReply *reply = m_networkManager->get(request);
    reply->setProperty("sourceUrl", urlStr);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        onPageReply(reply);
    });

    qDebug() << "WebSearchEngine: fetching page:" << urlStr;
}

void WebSearchEngine::onSearchReply(QNetworkReply *reply)
{
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError) {
        qWarning() << "WebSearchEngine: search error:" << reply->errorString();
        emit searchError(reply->errorString());
        m_isSearching = false;
        emit isSearchingChanged();
        return;
    }

    QByteArray html = reply->readAll();
    QList<SearchResult> results = parseDuckDuckGoHtml(html, m_pendingMaxResults);

    m_searchCount++;
    m_isSearching = false;
    emit searchCountChanged();
    emit isSearchingChanged();
    emit searchResultsReady(results);

    qDebug() << "WebSearchEngine: got" << results.size() << "results";
}

void WebSearchEngine::onPageReply(QNetworkReply *reply)
{
    reply->deleteLater();

    QString sourceUrl = reply->property("sourceUrl").toString();

    if (reply->error() != QNetworkReply::NoError) {
        qWarning() << "WebSearchEngine: page fetch error:" << reply->errorString();
        emit searchError("Failed to fetch " + sourceUrl + ": " + reply->errorString());
        return;
    }

    // Check content size
    QByteArray html = reply->read(MAX_PAGE_SIZE_BYTES);

    PageContent content;
    content.url = sourceUrl;
    content.title = extractTitle(html);
    content.plainText = htmlToPlainText(html);

    // Count words
    content.wordCount = content.plainText.split(QRegularExpression("\\s+"),
                                                 Qt::SkipEmptyParts).size();

    // Truncate to ~2000 words
    if (content.wordCount > 2000) {
        QStringList words = content.plainText.split(QRegularExpression("\\s+"),
                                                     Qt::SkipEmptyParts);
        content.plainText = words.mid(0, 2000).join(' ');
        content.wordCount = 2000;
    }

    emit pageContentReady(content);
    qDebug() << "WebSearchEngine: fetched page" << sourceUrl
             << "(" << content.wordCount << "words)";
}

QList<SearchResult> WebSearchEngine::parseDuckDuckGoHtml(const QByteArray &html,
                                                          int maxResults) const
{
    QList<SearchResult> results;
    QString htmlStr = QString::fromUtf8(html);

    // DuckDuckGo HTML results are in elements with class="result__a" for links
    // and class="result__snippet" for snippets

    // Split by result boundaries
    static QRegularExpression resultSplitter("class=\"result__body\"");
    QStringList blocks = htmlStr.split(resultSplitter);

    // Skip first block (before first result)
    for (int i = 1; i < blocks.size() && results.size() < maxResults; ++i) {
        const QString &block = blocks[i];

        SearchResult result;

        // Extract link: <a class="result__a" href="...">Title</a>
        static QRegularExpression linkRe(
            "class=\"result__a\"[^>]*href=\"([^\"]*)\"[^>]*>([^<]*(?:<[^>]*>[^<]*)*)</a>");
        QRegularExpressionMatch linkMatch = linkRe.match(block);
        if (linkMatch.hasMatch()) {
            QString href = decodeHtmlEntities(linkMatch.captured(1));
            result.url = decodeDdgUrl(href);
            // Strip any inner HTML tags from the title
            QString rawTitle = linkMatch.captured(2);
            rawTitle.remove(QRegularExpression("<[^>]*>"));
            result.title = decodeHtmlEntities(rawTitle).trimmed();
        }

        // Extract snippet: <a class="result__snippet" ...>text</a>
        static QRegularExpression snippetRe(
            "class=\"result__snippet\"[^>]*>([^<]*(?:<[^>]*>[^<]*)*)</a>");
        QRegularExpressionMatch snippetMatch = snippetRe.match(block);
        if (snippetMatch.hasMatch()) {
            QString rawSnippet = snippetMatch.captured(1);
            rawSnippet.remove(QRegularExpression("<[^>]*>"));
            result.snippet = decodeHtmlEntities(rawSnippet).trimmed();
        }

        // Only add if we have a valid URL
        if (!result.url.isEmpty() && result.url.startsWith("http")) {
            results.append(result);
        }
    }

    return results;
}

QString WebSearchEngine::htmlToPlainText(const QByteArray &html) const
{
    QString text = QString::fromUtf8(html);

    // Remove script blocks
    text.remove(QRegularExpression("<script[^>]*>[\\s\\S]*?</script>",
                                    QRegularExpression::CaseInsensitiveOption));
    // Remove style blocks
    text.remove(QRegularExpression("<style[^>]*>[\\s\\S]*?</style>",
                                    QRegularExpression::CaseInsensitiveOption));
    // Remove HTML comments
    text.remove(QRegularExpression("<!--[\\s\\S]*?-->"));

    // Convert common block elements to newlines
    text.replace(QRegularExpression("<br[^>]*>", QRegularExpression::CaseInsensitiveOption), "\n");
    text.replace(QRegularExpression("</p>", QRegularExpression::CaseInsensitiveOption), "\n");
    text.replace(QRegularExpression("</div>", QRegularExpression::CaseInsensitiveOption), "\n");
    text.replace(QRegularExpression("</h[1-6]>", QRegularExpression::CaseInsensitiveOption), "\n");
    text.replace(QRegularExpression("</li>", QRegularExpression::CaseInsensitiveOption), "\n");

    // Strip all remaining tags
    text.remove(QRegularExpression("<[^>]*>"));

    // Decode HTML entities
    text = decodeHtmlEntities(text);

    // Normalize whitespace
    text.replace(QRegularExpression("[ \\t]+"), " ");
    text.replace(QRegularExpression("\\n\\s*\\n+"), "\n\n");
    text = text.trimmed();

    return text;
}

QString WebSearchEngine::extractTitle(const QByteArray &html) const
{
    QString text = QString::fromUtf8(html);
    static QRegularExpression titleRe("<title[^>]*>([^<]*)</title>",
                                       QRegularExpression::CaseInsensitiveOption);
    QRegularExpressionMatch match = titleRe.match(text);
    if (match.hasMatch()) {
        return decodeHtmlEntities(match.captured(1)).trimmed();
    }
    return QString();
}

QString WebSearchEngine::decodeHtmlEntities(const QString &text) const
{
    QString decoded = text;
    decoded.replace("&amp;", "&");
    decoded.replace("&lt;", "<");
    decoded.replace("&gt;", ">");
    decoded.replace("&quot;", "\"");
    decoded.replace("&#39;", "'");
    decoded.replace("&apos;", "'");
    decoded.replace("&nbsp;", " ");
    decoded.replace("&#x27;", "'");
    decoded.replace("&#x2F;", "/");

    // Numeric entities: &#NNN;
    static QRegularExpression numericRe("&#(\\d+);");
    QRegularExpressionMatchIterator it = numericRe.globalMatch(decoded);
    while (it.hasNext()) {
        QRegularExpressionMatch m = it.next();
        int code = m.captured(1).toInt();
        if (code > 0 && code < 0x10FFFF) {
            decoded.replace(m.captured(0), QChar(code));
        }
    }

    return decoded;
}

QString WebSearchEngine::decodeDdgUrl(const QString &href) const
{
    // DuckDuckGo wraps URLs in redirects like:
    // //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...
    if (href.contains("uddg=")) {
        QUrl ddgUrl("https:" + href);
        QUrlQuery query(ddgUrl);
        QString decoded = query.queryItemValue("uddg", QUrl::FullyDecoded);
        if (!decoded.isEmpty()) return decoded;
    }

    // Some results have direct URLs
    if (href.startsWith("http")) return href;
    if (href.startsWith("//")) return "https:" + href;

    return href;
}
