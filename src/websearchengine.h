#ifndef WEBSEARCHENGINE_H
#define WEBSEARCHENGINE_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QList>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QDateTime>

struct SearchResult {
    QString title;
    QString snippet;
    QString url;
};

struct PageContent {
    QString url;
    QString title;
    QString plainText;
    int wordCount = 0;
};

class WebSearchEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isSearching READ isSearching NOTIFY isSearchingChanged)
    Q_PROPERTY(int searchCount READ searchCount NOTIFY searchCountChanged)

public:
    explicit WebSearchEngine(QObject *parent = nullptr);
    ~WebSearchEngine();

    bool isSearching() const { return m_isSearching; }
    int searchCount() const { return m_searchCount; }

    void search(const QString &query, int maxResults = 5);
    void fetchPage(const QString &url);
    bool canSearch() const;

signals:
    void isSearchingChanged();
    void searchCountChanged();
    void searchResultsReady(const QList<SearchResult> &results);
    void pageContentReady(const PageContent &content);
    void searchError(const QString &error);

private slots:
    void onSearchReply(QNetworkReply *reply);
    void onPageReply(QNetworkReply *reply);

private:
    QList<SearchResult> parseDuckDuckGoHtml(const QByteArray &html, int maxResults) const;
    QString htmlToPlainText(const QByteArray &html) const;
    QString extractTitle(const QByteArray &html) const;
    QString decodeHtmlEntities(const QString &text) const;
    QString decodeDdgUrl(const QString &href) const;

    QNetworkAccessManager *m_networkManager;
    bool m_isSearching = false;
    int m_searchCount = 0;
    int m_pendingMaxResults = 5;
    QDateTime m_lastSearchTime;

    static constexpr int MIN_SEARCH_INTERVAL_SEC = 10;
    static constexpr int MAX_PAGE_SIZE_BYTES = 512 * 1024;
    static constexpr int REQUEST_TIMEOUT_MS = 15000;
};

#endif // WEBSEARCHENGINE_H
