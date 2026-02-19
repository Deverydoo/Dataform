#ifndef NEWSENGINE_H
#define NEWSENGINE_H

#include <QObject>
#include <QString>
#include <QList>
#include <QSet>
#include <QDateTime>
#include <QDate>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QRegularExpression>
#include <QXmlStreamReader>

class MemoryStore;
class LLMProviderManager;
class ThoughtEngine;
class SettingsManager;

struct NewsHeadline {
    QString title;
    QString url;
    QString description;
    QDateTime pubDate;
};

class NewsEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isFetchingNews READ isFetchingNews NOTIFY isFetchingNewsChanged)
    Q_PROPERTY(QString currentPhase READ currentPhase NOTIFY currentPhaseChanged)
    Q_PROPERTY(int cyclesCompletedToday READ cyclesCompletedToday NOTIFY cyclesCompletedTodayChanged)

public:
    explicit NewsEngine(QObject *parent = nullptr);
    ~NewsEngine();

    void setMemoryStore(MemoryStore *store);
    void setLLMProvider(LLMProviderManager *provider);
    void setThoughtEngine(ThoughtEngine *engine);
    void setSettingsManager(SettingsManager *settings);

    bool isFetchingNews() const { return m_isFetchingNews; }
    QString currentPhase() const { return m_currentPhase; }
    int cyclesCompletedToday() const { return m_cyclesCompletedToday; }

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();

signals:
    void isFetchingNewsChanged();
    void currentPhaseChanged();
    void cyclesCompletedTodayChanged();
    void newsCycleComplete(const QString &headline, const QString &url);
    void newsError(const QString &error);

private slots:
    void onLLMResponse(const QString &response);
    void onLLMError(const QString &error);

private:
    enum Phase {
        Idle,
        FetchHeadlines,
        FilterNew,
        SelectInteresting,
        GenerateThought,
        Store
    };

    void startCycle();
    void advancePhase();
    void setPhase(Phase phase);

    // Phase implementations
    void phaseFetchHeadlines();
    void phaseFilterNew();
    void phaseSelectInteresting();
    void phaseGenerateThought();
    void phaseStore();

    // Helpers
    bool canStartCycle() const;
    void resetDailyCounterIfNeeded();
    QList<NewsHeadline> parseRssFeed(const QByteArray &xml) const;
    bool isHeadlineSeen(const QString &url) const;
    void markHeadlineSeen(const QString &url);
    void pruneOldHeadlines();

    MemoryStore *m_memoryStore = nullptr;
    LLMProviderManager *m_llmProvider = nullptr;
    ThoughtEngine *m_thoughtEngine = nullptr;
    SettingsManager *m_settingsManager = nullptr;
    QNetworkAccessManager *m_networkManager = nullptr;

    bool m_isFetchingNews = false;
    bool m_idleWindowOpen = false;
    bool m_paused = false;
    Phase m_phase = Idle;
    QString m_currentPhase;

    // Current cycle state
    QList<NewsHeadline> m_allHeadlines;
    QList<NewsHeadline> m_newHeadlines;
    QList<NewsHeadline> m_selectedHeadlines;
    int m_currentHeadlineIndex = 0;

    // Daily tracking
    int m_cyclesCompletedToday = 0;
    QDate m_lastCycleDate;

    // Cooldown
    QDateTime m_lastCycleEndTime;
    static constexpr int CYCLE_COOLDOWN_SEC = 600;     // 10 minutes between cycles
    static constexpr int MAX_HEADLINES_TO_PARSE = 50;

    int m_currentFeedIndex = 0;

    // Error recovery
    int m_consecutiveErrors = 0;
    static constexpr int MAX_CONSECUTIVE_ERRORS = 3;
};

#endif // NEWSENGINE_H
