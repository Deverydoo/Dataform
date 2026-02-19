#ifndef RESEARCHENGINE_H
#define RESEARCHENGINE_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QTimer>
#include "websearchengine.h"
#include "researchstore.h"

class MemoryStore;
class LLMProviderManager;
class WhyEngine;
class SettingsManager;

class ResearchEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isResearching READ isResearching NOTIFY isResearchingChanged)
    Q_PROPERTY(QString currentPhase READ currentPhase NOTIFY currentPhaseChanged)
    Q_PROPERTY(QString currentTopic READ currentTopic NOTIFY currentTopicChanged)
    Q_PROPERTY(int cyclesCompletedToday READ cyclesCompletedToday NOTIFY cyclesCompletedTodayChanged)
    Q_PROPERTY(int topicQueueSize READ topicQueueSize NOTIFY topicQueueChanged)

public:
    explicit ResearchEngine(QObject *parent = nullptr);
    ~ResearchEngine();

    void setWebSearchEngine(WebSearchEngine *engine);
    void setMemoryStore(MemoryStore *store);
    void setResearchStore(ResearchStore *store);
    void setLLMProvider(LLMProviderManager *provider);
    void setWhyEngine(WhyEngine *engine);
    void setSettingsManager(SettingsManager *settings);

    bool isResearching() const { return m_isResearching; }
    QString currentPhase() const { return m_currentPhase; }
    QString currentTopic() const { return m_currentTopic; }
    int cyclesCompletedToday() const { return m_cyclesCompletedToday; }
    int topicQueueSize() const { return m_topicQueue.size(); }

    // Build context string from approved findings for conversation
    QString buildResearchContext(const QString &topic, int maxFindings = 3) const;

    // Queue a topic for research (from user request or WhyEngine)
    Q_INVOKABLE void queueTopic(const QString &topic, double priority = 0.5);

    // Pause current research (safe mid-cycle stop)
    void pauseResearch();

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();

signals:
    void isResearchingChanged();
    void currentPhaseChanged();
    void currentTopicChanged();
    void cyclesCompletedTodayChanged();
    void topicQueueChanged();
    void researchCycleComplete(const QString &topic, int findingsCount);
    void researchError(const QString &error);

private slots:
    void onSearchResultsReady(const QList<SearchResult> &results);
    void onPageContentReady(const PageContent &content);
    void onLLMResponse(const QString &response);
    void onLLMError(const QString &error);

private:
    enum Phase {
        Idle,
        SelectTopic,
        GenerateQuery,
        ExecuteSearch,
        FetchAndSummarize,
        Store
    };

    struct TopicEntry {
        QString topic;
        double priority;
    };

    void startCycle();
    void advancePhase();
    void setPhase(Phase phase);

    // Phase implementations
    void phaseSelectTopic();
    void phaseGenerateQuery();
    void phaseExecuteSearch();
    void phaseFetchAndSummarize();
    void phaseStore();

    // Helpers
    bool canStartCycle() const;
    void resetDailyCounterIfNeeded();
    void buildTopicQueue();

    WebSearchEngine *m_webSearch = nullptr;
    MemoryStore *m_memoryStore = nullptr;
    ResearchStore *m_researchStore = nullptr;
    LLMProviderManager *m_llmProvider = nullptr;
    WhyEngine *m_whyEngine = nullptr;
    SettingsManager *m_settingsManager = nullptr;

    bool m_isResearching = false;
    bool m_idleWindowOpen = false;
    bool m_paused = false;
    Phase m_phase = Idle;
    QString m_currentPhase;
    QString m_currentTopic;

    // Topic queue
    QList<TopicEntry> m_topicQueue;

    // Current cycle state
    QList<SearchResult> m_currentResults;
    QList<PageContent> m_fetchedPages;
    QStringList m_currentQueries;
    int m_currentQueryIndex = 0;
    int m_currentPageIndex = 0;
    int m_currentFindingsCount = 0;

    // Daily tracking
    int m_cyclesCompletedToday = 0;
    QDate m_lastCycleDate;

    // Cooldown
    QDateTime m_lastCycleEndTime;
    static constexpr int CYCLE_COOLDOWN_SEC = 300;  // 5 minutes between cycles

    // Error recovery
    int m_consecutiveErrors = 0;
    static constexpr int MAX_CONSECUTIVE_ERRORS = 3;
};

#endif // RESEARCHENGINE_H
