#ifndef SENTIMENTTRACKER_H
#define SENTIMENTTRACKER_H

#include <QObject>
#include <QString>
#include <QDateTime>

class MemoryStore;
class LLMProviderManager;
class ThoughtEngine;
class SettingsManager;

class SentimentTracker : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString currentMood READ currentMood NOTIFY currentMoodChanged)
    Q_PROPERTY(QString moodTrend READ moodTrend NOTIFY moodTrendChanged)

public:
    explicit SentimentTracker(QObject *parent = nullptr);
    ~SentimentTracker();

    void setMemoryStore(MemoryStore *store);
    void setLLMProvider(LLMProviderManager *provider);
    void setThoughtEngine(ThoughtEngine *engine);
    void setSettingsManager(SettingsManager *settings);

    QString currentMood() const { return m_currentMood; }
    QString moodTrend() const { return m_moodTrend; }
    QString getMoodHint() const;
    Q_INVOKABLE QVariantList getSentimentHistoryForQml(int daysPast = 30);

public slots:
    void onEpisodeStored(qint64 episodeId);
    void onIdleWindowOpened();

signals:
    void currentMoodChanged();
    void moodTrendChanged();
    void sentimentAnalyzed(qint64 episodeId, double score);

private slots:
    void onLLMResponse(const QString &response);
    void onLLMError(const QString &error);

private:
    void analyzeSentiment(qint64 episodeId);
    void detectPatterns();
    void updateMoodFromRecent();

    MemoryStore *m_memoryStore = nullptr;
    LLMProviderManager *m_llmProvider = nullptr;
    ThoughtEngine *m_thoughtEngine = nullptr;
    SettingsManager *m_settingsManager = nullptr;

    QString m_currentMood;
    QString m_moodTrend;
    int m_episodeCounter = 0;
    qint64 m_pendingEpisodeId = -1;
    bool m_isAnalyzing = false;
    bool m_patternDetectedToday = false;
    QDateTime m_lastPatternCheck;

    // Error recovery
    int m_consecutiveErrors = 0;
    static constexpr int MAX_CONSECUTIVE_ERRORS = 3;
};

#endif // SENTIMENTTRACKER_H
