#ifndef THOUGHTENGINE_H
#define THOUGHTENGINE_H

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QVariantList>
#include <QVariantMap>
#include <QDate>

class MemoryStore;
class ResearchStore;
class ResearchEngine;
class WhyEngine;
class LLMProviderManager;
class SettingsManager;

struct ThoughtRecord {
    qint64 id = -1;
    QDateTime timestamp;
    QString type;           // research_proposal, self_improvement, curiosity_overflow,
                            // evolution_observation, training_observation
    QString title;
    QString content;
    double priority = 0.5;
    QString sourceType;     // "research", "curiosity", "evolution", "reflection"
    qint64 sourceId = -1;
    int status = 0;         // 0=pending, 1=discussed, 2=dismissed
    qint64 conversationId = -1;
    QString generatedBy;
};

class ThoughtEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int pendingCount READ pendingCount NOTIFY pendingCountChanged)
    Q_PROPERTY(bool isGenerating READ isGenerating NOTIFY isGeneratingChanged)

public:
    explicit ThoughtEngine(QObject *parent = nullptr);
    ~ThoughtEngine();

    void setMemoryStore(MemoryStore *store);
    void setResearchStore(ResearchStore *store);
    void setResearchEngine(ResearchEngine *engine);
    void setWhyEngine(WhyEngine *engine);
    void setLLMProvider(LLMProviderManager *provider);
    void setSettingsManager(SettingsManager *settings);

    bool initialize();

    int pendingCount() const { return m_pendingCount; }
    bool isGenerating() const { return m_isGenerating; }

    // Thought CRUD
    qint64 insertThought(const ThoughtRecord &record);
    bool updateThoughtStatus(qint64 id, int status, qint64 conversationId = -1);
    QList<ThoughtRecord> getPendingThoughts(int limit = 10) const;
    Q_INVOKABLE QVariantList getPendingThoughtsForQml(int limit = 10);
    Q_INVOKABLE QVariantMap getTopThoughtForQml();
    Q_INVOKABLE void dismissThought(qint64 id);
    Q_INVOKABLE void dismissAllThoughts();

    // Build opening message for a proactive conversation
    QString buildOpeningMessage(qint64 thoughtId);
    QString getThoughtTitle(qint64 thoughtId) const;
    qint64 getThoughtForConversation(qint64 conversationId) const;

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();
    void onResearchCycleComplete(const QString &topic, int findingsCount);
    void onNewsCycleComplete(const QString &headline, const QString &url);
#ifdef DATAFORM_TRAINING_ENABLED
    void onEvolutionCycleComplete(bool adapterPromoted, int winnerVersion, double winnerScore);
    void onReflectionComplete(bool promoted, double score);
#endif

signals:
    void pendingCountChanged();
    void isGeneratingChanged();
    void thoughtGenerated(qint64 thoughtId);

private:
    void refreshPendingCount();
    void generateThoughtsFromResearch();
    void generateThoughtsFromCuriosity();
    void generateDigest();
    void formatThoughtViaLLM(ThoughtRecord &record);
    double thoughtReadiness() const;
    bool canGenerateMore() const;
    bool isDuplicateTopic(const QString &type, const QString &title) const;
    QDateTime getOldestPendingTime() const;
    void resetDailyCounterIfNeeded();
    QVariantMap thoughtToVariantMap(const ThoughtRecord &record) const;
    int countThoughtsSince(const QString &type, int hoursPast) const;
    int countNewTraitsSince(int hoursPast) const;

    MemoryStore *m_memoryStore = nullptr;
    ResearchStore *m_researchStore = nullptr;
    ResearchEngine *m_researchEngine = nullptr;
    WhyEngine *m_whyEngine = nullptr;
    LLMProviderManager *m_llmProvider = nullptr;
    SettingsManager *m_settingsManager = nullptr;

    bool m_initialized = false;
    bool m_isGenerating = false;
    bool m_idleWindowOpen = false;
    int m_pendingCount = 0;

    // Daily generation tracking
    int m_generatedToday = 0;
    QDate m_lastGenerationDate;
    QDate m_lastDigestDate;

    // Interaction-based rate limiting
    QDateTime m_lastInteractionTime;  // When user last responded/dismissed

    static constexpr int MAX_PENDING_THOUGHTS = 10;  // database safety cap
};

#endif // THOUGHTENGINE_H
