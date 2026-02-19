#include "sentimenttracker.h"
#include "fuzzylogic.h"
#include "memorystore.h"
#include "llmprovider.h"
#include "thoughtengine.h"
#include "settingsmanager.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QRegularExpression>
#include <QDate>
#include <QSqlQuery>
#include <QSqlDatabase>
#include <QRandomGenerator>

SentimentTracker::SentimentTracker(QObject *parent)
    : QObject(parent)
{
}

SentimentTracker::~SentimentTracker()
{
}

void SentimentTracker::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void SentimentTracker::setLLMProvider(LLMProviderManager *provider) {
    m_llmProvider = provider;
    if (provider) {
        connect(provider, &LLMProviderManager::backgroundResponseReceived,
                this, [this](const QString &owner, const QString &response) {
            if (owner == "SentimentTracker") onLLMResponse(response);
        });
        connect(provider, &LLMProviderManager::backgroundErrorOccurred,
                this, [this](const QString &owner, const QString &error) {
            if (owner == "SentimentTracker") onLLMError(error);
        });
    }
}
void SentimentTracker::setThoughtEngine(ThoughtEngine *engine) { m_thoughtEngine = engine; }
void SentimentTracker::setSettingsManager(SettingsManager *settings) { m_settingsManager = settings; }

void SentimentTracker::onEpisodeStored(qint64 episodeId)
{
    if (!m_settingsManager || !m_settingsManager->sentimentTrackingEnabled()) return;
    if (m_isAnalyzing) return;

    m_episodeCounter++;
    // Only analyze every 3rd episode to avoid LLM contention
    if (m_episodeCounter % 3 != 0) return;

    analyzeSentiment(episodeId);
}

void SentimentTracker::onIdleWindowOpened()
{
    m_consecutiveErrors = 0;  // Fresh idle window gets a fresh chance
    updateMoodFromRecent();
    detectPatterns();
}

void SentimentTracker::analyzeSentiment(qint64 episodeId)
{
    if (!m_memoryStore || !m_llmProvider) return;
    if (m_settingsManager && m_settingsManager->privacyLevel() == "minimal") return;
    // Error recovery: skip analysis after too many consecutive errors
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) return;
    if (m_memoryStore->hasEpisodeSentiment(episodeId)) return;

    EpisodicRecord ep = m_memoryStore->getEpisode(episodeId);
    if (ep.userText.isEmpty()) return;

    m_isAnalyzing = true;
    m_pendingEpisodeId = episodeId;

    QString prompt = QString(
        "Rate the sentiment of this message:\n\"%1\"\n\n"
        "Output a JSON object. Example: {\"sentiment\":0.3,\"energy\":0.6,\"emotion\":\"curious\"}\n"
        "sentiment: -1.0 to 1.0, energy: 0.0 to 1.0, emotion: one word.\n"
        "IMPORTANT: Output ONLY the JSON object starting with { and ending with }. No other text."
    ).arg(ep.userText.left(500));

    QJsonArray msgs;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    msgs.append(msg);
    m_llmProvider->sendBackgroundRequest("SentimentTracker",
        "You are a JSON-only API. You output valid JSON and nothing else. "
        "Never include explanation, analysis, or commentary. Only JSON.",
        msgs, LLMProviderManager::PriorityHigh);
}

void SentimentTracker::onLLMResponse(const QString &response)
{
    m_isAnalyzing = false;
    m_consecutiveErrors = 0;  // Reset on any successful LLM response

    if (m_pendingEpisodeId < 0) return;

    QString cleaned = response.trimmed();
    // Strip <think>...</think> reasoning blocks (qwen3 and other reasoning models)
    static const QRegularExpression thinkRegex(
        "<think>.*?</think>",
        QRegularExpression::DotMatchesEverythingOption);
    cleaned.remove(thinkRegex);
    cleaned = cleaned.trimmed();
    if (cleaned.startsWith("```")) {
        int start = cleaned.indexOf('\n') + 1;
        int end = cleaned.lastIndexOf("```");
        if (end > start) cleaned = cleaned.mid(start, end - start).trimmed();
    }

    QJsonDocument doc = QJsonDocument::fromJson(cleaned.toUtf8());
    if (!doc.isObject()) {
        qDebug() << "SentimentTracker: failed to parse LLM response:" << cleaned.left(200);
        return;
    }

    QJsonObject obj = doc.object();
    double sentiment = qBound(-1.0, obj.value("sentiment").toDouble(0.0), 1.0);
    double energy = qBound(0.0, obj.value("energy").toDouble(0.5), 1.0);
    QString emotion = obj.value("emotion").toString("neutral");

    m_memoryStore->insertSentiment(m_pendingEpisodeId, sentiment, energy, emotion);
    emit sentimentAnalyzed(m_pendingEpisodeId, sentiment);

    qDebug() << "SentimentTracker: episode" << m_pendingEpisodeId
             << "sentiment=" << sentiment << "energy=" << energy << "emotion=" << emotion;

    m_pendingEpisodeId = -1;
    updateMoodFromRecent();
}

void SentimentTracker::onLLMError(const QString &error)
{
    m_isAnalyzing = false;
    m_pendingEpisodeId = -1;
    m_consecutiveErrors++;
    qDebug() << "SentimentTracker: LLM error:" << error;
    if (m_consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        qWarning() << "SentimentTracker: pausing after" << m_consecutiveErrors << "consecutive errors";
    }
}

void SentimentTracker::detectPatterns()
{
    if (!m_memoryStore || !m_thoughtEngine) return;
    if (!m_settingsManager || !m_settingsManager->sentimentTrackingEnabled()) return;

    // Only detect patterns once per day
    if (m_patternDetectedToday && m_lastPatternCheck.date() == QDate::currentDate()) return;

    double recentAvg = m_memoryStore->getAverageSentiment(7);
    double longtermAvg = m_memoryStore->getAverageSentiment(30);

    // Need sufficient data
    QList<SentimentRecord> recent = m_memoryStore->getRecentSentiment(5);
    if (recent.size() < 3) return;

    double delta = recentAvg - longtermAvg;

    // Soft pattern detection: larger deltas are more likely to trigger
    double patternStrength = Fuzzy::sigmoid(qAbs(delta), 0.3, 0.08);
    double roll = QRandomGenerator::global()->generateDouble();
    if (roll < patternStrength) {
        QString trend = delta > 0 ? "more positive and upbeat" : "more stressed or down";
        ThoughtRecord thought;
        thought.type = "mood_pattern";
        thought.title = "I noticed a shift in our conversations";
        thought.content = QString("Your recent messages seem %1 compared to usual. "
                                  "Just wanted to check in — how are things going?").arg(trend);
        thought.priority = 0.5 + patternStrength * 0.3; // priority scales with signal strength
        thought.sourceType = "sentiment";
        thought.generatedBy = "sentiment_tracker";
        m_thoughtEngine->insertThought(thought);

        m_patternDetectedToday = true;
        m_lastPatternCheck = QDateTime::currentDateTime();
        qDebug() << "SentimentTracker: mood pattern detected, delta=" << delta
                 << "strength=" << patternStrength;
    }
}

QString SentimentTracker::getMoodHint() const
{
    if (m_currentMood.isEmpty()) return QString();

    if (m_moodTrend == "declining" || m_moodTrend == "slightly declining") {
        return QString("Context: the user's recent tone has been %1. Don't comment on this or try to cheer them up -- just be aware of it.").arg(m_currentMood);
    } else if (m_moodTrend == "improving" || m_moodTrend == "slightly improving") {
        return QString("Context: the user's recent tone has been %1.").arg(m_currentMood);
    }
    return QString("Context: the user's general tone has been %1.").arg(m_currentMood);
}

void SentimentTracker::updateMoodFromRecent()
{
    if (!m_memoryStore) return;

    QList<SentimentRecord> recent = m_memoryStore->getRecentSentiment(5);
    if (recent.isEmpty()) return;

    double avgSentiment = 0.0;
    for (const auto &s : recent) avgSentiment += s.sentimentScore;
    avgSentiment /= recent.size();

    // Gaussian mood blending: each mood is a fuzzy set with a center and spread
    struct MoodDef { const char *name; double center; double sigma; };
    static const MoodDef moods[] = {
        {"very positive",  0.65,  0.25},
        {"positive",       0.35,  0.20},
        {"neutral",        0.0,   0.25},
        {"somewhat low",  -0.35,  0.20},
        {"stressed",      -0.65,  0.25}
    };

    double bestScore = -1.0;
    QString bestMood;
    double secondScore = -1.0;
    QString secondMood;

    for (const auto &m : moods) {
        double membership = Fuzzy::gaussian(avgSentiment, m.center, m.sigma);
        if (membership > bestScore) {
            secondScore = bestScore;
            secondMood = bestMood;
            bestScore = membership;
            bestMood = m.name;
        } else if (membership > secondScore) {
            secondScore = membership;
            secondMood = m.name;
        }
    }

    // Blend if two moods are close (second > 70% of best)
    QString mood;
    if (secondScore > bestScore * 0.7 && !secondMood.isEmpty()) {
        mood = QString("between %1 and %2").arg(bestMood, secondMood);
    } else {
        mood = bestMood;
    }

    if (m_currentMood != mood) {
        m_currentMood = mood;
        emit currentMoodChanged();
    }

    // Trend: sigmoid-based with 5 levels instead of 3
    double recentAvg = m_memoryStore->getAverageSentiment(7);
    double longtermAvg = m_memoryStore->getAverageSentiment(30);
    double delta = recentAvg - longtermAvg;

    double improvingMembership = Fuzzy::sigmoid(delta, 0.15, 0.05);
    double decliningMembership = Fuzzy::sigmoid(-delta, 0.15, 0.05);

    QString trend;
    if (improvingMembership > 0.7) trend = "improving";
    else if (decliningMembership > 0.7) trend = "declining";
    else if (improvingMembership > 0.3) trend = "slightly improving";
    else if (decliningMembership > 0.3) trend = "slightly declining";
    else trend = "stable";

    if (m_moodTrend != trend) {
        m_moodTrend = trend;
        emit moodTrendChanged();
    }
}

QVariantList SentimentTracker::getSentimentHistoryForQml(int daysPast)
{
    QVariantList results;
    if (!m_memoryStore) return results;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare(
        "SELECT date(analyzed_ts) as day, "
        "AVG(sentiment_score) as avg_score, "
        "AVG(energy_level) as avg_energy, "
        "COUNT(*) as count "
        "FROM sentiment_log "
        "WHERE analyzed_ts >= datetime('now', :days) "
        "GROUP BY date(analyzed_ts) "
        "ORDER BY day ASC"
    );
    q.bindValue(":days", QString("-%1 days").arg(daysPast));

    if (q.exec()) {
        while (q.next()) {
            QVariantMap item;
            item["date"] = q.value("day").toString();
            item["avgScore"] = q.value("avg_score").toDouble();
            item["avgEnergy"] = q.value("avg_energy").toDouble();
            item["count"] = q.value("count").toInt();
            results.append(item);
        }
    }
    return results;
}

