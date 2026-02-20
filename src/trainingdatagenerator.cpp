#ifdef DATAFORM_TRAINING_ENABLED

#include "trainingdatagenerator.h"
#include "memorystore.h"
#include "tokenizer.h"
#include "researchstore.h"
#include <QDebug>
#include <QSqlQuery>
#include <QSqlError>
#include <QMap>
#include <algorithm>
#include <cmath>

TrainingDataGenerator::TrainingDataGenerator(QObject *parent)
    : QObject(parent)
{
}

TrainingDataGenerator::~TrainingDataGenerator() = default;

void TrainingDataGenerator::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

void TrainingDataGenerator::setTokenizer(Tokenizer *tokenizer)
{
    m_tokenizer = tokenizer;
}

void TrainingDataGenerator::setResearchStore(ResearchStore *store)
{
    m_researchStore = store;
}

double TrainingDataGenerator::computeSignalStrength(int feedback, bool corrected, bool hasInquiry) const
{
    double strength = 0.0;
    if (corrected) strength += 3.0;        // Edits are strongest
    if (feedback > 0) strength += 2.0;      // Thumbs up
    if (feedback < 0) strength += 1.0;      // Thumbs down (still useful)
    if (hasInquiry) strength += 1.5;        // Inquiry response
    if (strength == 0.0) strength = 0.1;    // Baseline for no-signal episodes
    return strength;
}

QList<qint64> TrainingDataGenerator::selectHighSignalEpisodes(int maxCount)
{
    QList<qint64> result;
    if (!m_memoryStore) return result;

    // Get high-signal episodes after the last processed one
    auto episodes = m_memoryStore->getHighSignalEpisodes(m_lastProcessedEpisodeId, maxCount);

    for (const auto &ep : episodes) {
        result.append(ep.id);
    }

    qDebug() << "TrainingDataGenerator: selected" << result.size()
             << "high-signal episodes after id" << m_lastProcessedEpisodeId;
    return result;
}

QString TrainingDataGenerator::buildTrainingSystemPrompt(const QString &identityContext) const
{
    QString prompt = "You are DATAFORM, a learning AI companion. "
                     "Respond naturally and helpfully, adapting to the user's communication style.";

    if (!identityContext.isEmpty()) {
        prompt += "\n\nWhat you know about the user:\n" + identityContext;
    }

    return prompt;
}

TrainingExample TrainingDataGenerator::episodeToExample(qint64 episodeId, const QString &identityContext)
{
    TrainingExample example;
    example.sourceEpisodeId = episodeId;

    if (!m_memoryStore || !m_tokenizer || !m_tokenizer->isLoaded()) {
        return example;
    }

    EpisodicRecord episode = m_memoryStore->getEpisode(episodeId);
    if (episode.id < 0) return example;

    // Determine the target text:
    // - If user edited the response, use the edited version (strongest signal)
    // - If positive feedback, use the original response (confirmed good)
    // - If negative feedback, still use original but with low weight
    QString targetText;
    if (episode.corrected && !episode.userEditText.isEmpty()) {
        targetText = episode.userEditText;
        example.weight = EDIT_WEIGHT;
    } else if (episode.userFeedback > 0) {
        targetText = episode.assistantText;
        example.weight = POSITIVE_WEIGHT;
    } else if (episode.userFeedback < 0) {
        targetText = episode.assistantText;
        example.weight = NEGATIVE_WEIGHT;
    } else if (!episode.inquiryText.isEmpty()) {
        targetText = episode.assistantText;
        example.weight = INQUIRY_WEIGHT;
    } else {
        targetText = episode.assistantText;
        example.weight = NEUTRAL_WEIGHT;
    }

    // Build system prompt with identity context
    QString systemPrompt = buildTrainingSystemPrompt(identityContext);

    // Tokenize as ChatML
    auto chatResult = m_tokenizer->encodeChat(
        systemPrompt, episode.userText, targetText, MAX_SEQUENCE_LENGTH);

    example.inputIds = std::move(chatResult.inputIds);
    example.labels = std::move(chatResult.labels);
    example.attentionMask = std::move(chatResult.attentionMask);

    return example;
}

void TrainingDataGenerator::generateTrainingBatch(int maxExamples)
{
    m_examples.clear();

    if (!m_memoryStore || !m_tokenizer || !m_tokenizer->isLoaded()) {
        emit generationError("Missing dependencies (memory store or tokenizer)");
        return;
    }

    // Build identity context from current traits
    QString identityContext;
    auto traits = m_memoryStore->getAllTraits();
    QStringList traitStatements;
    for (const auto &trait : traits) {
        if (trait.confidence >= 0.3) {
            traitStatements.append(QString("- %1 (confidence: %2)")
                .arg(trait.statement)
                .arg(trait.confidence, 0, 'f', 2));
        }
        if (traitStatements.size() >= 7) break;
    }
    identityContext = traitStatements.join('\n');

    // Check if distillation pairs are available (Phase 8)
    int unusedDistillationCount = 0;
    auto unusedPairs = m_memoryStore->getUnusedDistillationPairs(maxExamples);
    unusedDistillationCount = unusedPairs.size();

    // Budget allocation:
    //   With distillation (>=5 pairs): 45/10/15/10/10/5/5
    //   Without distillation:          55/0/15/10/10/5/5
    int distillBudget;
    int episodeBudget;
    if (unusedDistillationCount >= 5) {
        distillBudget     = static_cast<int>(maxExamples * 0.10);
        episodeBudget     = static_cast<int>(maxExamples * 0.45);
    } else {
        distillBudget     = 0;
        episodeBudget     = static_cast<int>(maxExamples * 0.55);
    }
    int researchBudget    = static_cast<int>(maxExamples * 0.15);
    int traitBudget       = static_cast<int>(maxExamples * 0.10);
    int goalLearnBudget   = static_cast<int>(maxExamples * 0.10);
    int newsThoughtBudget = static_cast<int>(maxExamples * 0.05);
    int sentimentBudget   = maxExamples - episodeBudget - distillBudget - researchBudget
                            - traitBudget - goalLearnBudget - newsThoughtBudget;

    // 1. Episodic examples (core data source)
    generateEpisodicExamples(m_examples, episodeBudget, identityContext);

    // 2. Distillation examples (Phase 8 — teacher knowledge transfer)
    if (distillBudget > 0)
        generateDistillationExamples(m_examples, distillBudget, identityContext);

    // 3. Research finding examples
    generateResearchExamples(m_examples, researchBudget, identityContext);

    // 4. Trait reinforcement examples
    generateTraitExamples(m_examples, traitBudget, identityContext);

    // 5. Goals + Learning plan examples (Phase 7)
    int goalBudget = goalLearnBudget / 2;
    int learnBudget = goalLearnBudget - goalBudget;
    generateGoalExamples(m_examples, goalBudget, identityContext);
    generateLearningExamples(m_examples, learnBudget, identityContext);

    // 6. News insights + Discussed thoughts examples (Phase 7)
    int newsBudget = newsThoughtBudget / 2;
    int thoughtBudget = newsThoughtBudget - newsBudget;
    generateNewsExamples(m_examples, newsBudget, identityContext);
    generateThoughtExamples(m_examples, thoughtBudget, identityContext);

    // 7. Sentiment pattern examples (Phase 7)
    generateSentimentExamples(m_examples, sentimentBudget, identityContext);

    // Shuffle to prevent ordering bias
    std::shuffle(m_examples.begin(), m_examples.end(), m_rng);

    emit exampleCountChanged();
    emit generationComplete(m_examples.size());

    qDebug() << "TrainingDataGenerator: generated" << m_examples.size()
             << "training examples (episodes + research + traits)";
}

void TrainingDataGenerator::generateEpisodicExamples(QList<TrainingExample> &examples,
                                                      int maxCount,
                                                      const QString &identityContext)
{
    auto episodeIds = selectHighSignalEpisodes(maxCount * 2);

    if (episodeIds.isEmpty()) {
        qDebug() << "TrainingDataGenerator: no high-signal episodes available";
        return;
    }

    int added = 0;
    qint64 maxEpisodeId = m_lastProcessedEpisodeId;
    for (qint64 episodeId : episodeIds) {
        if (added >= maxCount) break;

        auto example = episodeToExample(episodeId, identityContext);
        if (example.inputIds.empty()) continue;
        if (example.weight <= NEGATIVE_WEIGHT) continue;

        examples.append(std::move(example));
        if (episodeId > maxEpisodeId) maxEpisodeId = episodeId;
        ++added;
    }

    m_lastProcessedEpisodeId = maxEpisodeId;
    emit lastProcessedEpisodeIdChanged();
}

void TrainingDataGenerator::generateResearchExamples(QList<TrainingExample> &examples,
                                                      int maxCount,
                                                      const QString &identityContext)
{
    if (!m_researchStore || maxCount <= 0) return;

    auto findings = m_researchStore->getApprovedFindings();
    if (findings.isEmpty()) return;

    QString systemPrompt = buildTrainingSystemPrompt(identityContext);
    int added = 0;

    for (const auto &finding : findings) {
        if (added >= maxCount) break;
        if (finding.findingId <= m_lastProcessedFindingId) continue;
        if (finding.llmSummary.isEmpty()) continue;

        // Create synthetic Q&A: "What do you know about [topic]?" -> [summary]
        QString userText = "What do you know about " + finding.topic + "?";
        QString assistantText = finding.llmSummary;
        if (!finding.sourceTitle.isEmpty()) {
            assistantText += " (Based on research from " + finding.sourceTitle + ")";
        }

        auto chatResult = m_tokenizer->encodeChat(
            systemPrompt, userText, assistantText, MAX_SEQUENCE_LENGTH);

        if (chatResult.inputIds.empty()) continue;

        TrainingExample example;
        example.inputIds = std::move(chatResult.inputIds);
        example.labels = std::move(chatResult.labels);
        example.attentionMask = std::move(chatResult.attentionMask);
        example.weight = RESEARCH_WEIGHT;
        example.sourceEpisodeId = -1;

        examples.append(std::move(example));
        if (finding.findingId > m_lastProcessedFindingId) {
            m_lastProcessedFindingId = finding.findingId;
        }
        ++added;
    }

    if (added > 0) {
        qDebug() << "TrainingDataGenerator: generated" << added << "research examples";
    }
}

void TrainingDataGenerator::generateTraitExamples(QList<TrainingExample> &examples,
                                                    int maxCount,
                                                    const QString &identityContext)
{
    if (!m_memoryStore || maxCount <= 0) return;

    auto traits = m_memoryStore->getAllTraits();
    if (traits.isEmpty()) return;

    QString systemPrompt = buildTrainingSystemPrompt(identityContext);

    // Collect high-confidence traits with sufficient evidence
    QList<TraitRecord> qualifiedTraits;
    for (const auto &trait : traits) {
        if (trait.confidence >= 0.5 && trait.evidenceEpisodeIds.size() >= 3) {
            qualifiedTraits.append(trait);
        }
    }

    if (qualifiedTraits.isEmpty()) return;

    // Shuffle to vary which traits get reinforced each cycle
    std::shuffle(qualifiedTraits.begin(), qualifiedTraits.end(), m_rng);

    int added = 0;
    // Vary the question templates for diversity
    QStringList questionTemplates = {
        "How would you describe my preferences?",
        "What have you learned about me?",
        "What do you know about my interests?",
        "Can you tell me what you've observed about me?",
        "What patterns have you noticed in our conversations?"
    };

    for (const auto &trait : qualifiedTraits) {
        if (added >= maxCount) break;

        // Pick a question template
        QString userText = questionTemplates[added % questionTemplates.size()];

        QString assistantText = QString("Based on what I've learned, %1. "
            "This is something I've observed with %2% confidence "
            "across %3 interactions.")
            .arg(trait.statement)
            .arg(static_cast<int>(trait.confidence * 100))
            .arg(trait.evidenceEpisodeIds.size());

        auto chatResult = m_tokenizer->encodeChat(
            systemPrompt, userText, assistantText, MAX_SEQUENCE_LENGTH);

        if (chatResult.inputIds.empty()) continue;

        TrainingExample example;
        example.inputIds = std::move(chatResult.inputIds);
        example.labels = std::move(chatResult.labels);
        example.attentionMask = std::move(chatResult.attentionMask);
        example.weight = TRAIT_WEIGHT;
        example.sourceEpisodeId = -1;

        examples.append(std::move(example));
        ++added;
    }

    if (added > 0) {
        qDebug() << "TrainingDataGenerator: generated" << added << "trait reinforcement examples";
    }
}

// --- Phase 7: Expanded training data generators ---

void TrainingDataGenerator::generateGoalExamples(QList<TrainingExample> &examples,
                                                  int maxCount,
                                                  const QString &identityContext)
{
    if (!m_memoryStore || !m_tokenizer || maxCount <= 0) return;

    auto goals = m_memoryStore->getActiveGoals();
    if (goals.isEmpty()) return;

    QString systemPrompt = buildTrainingSystemPrompt(identityContext);

    QStringList questionTemplates = {
        "What goals have I mentioned?",
        "What am I working towards?",
        "Do you remember my goals?",
        "What have I said I want to achieve?"
    };

    int added = 0;
    for (const auto &goal : goals) {
        if (added >= maxCount) break;

        QString userText = questionTemplates[added % questionTemplates.size()];
        QString targetText = QString("You mentioned wanting to %1.")
            .arg(goal.description.isEmpty() ? goal.title : goal.description);
        if (goal.checkinCount > 0) {
            targetText += QString(" We've talked about this %1 time%2 so far.")
                .arg(goal.checkinCount)
                .arg(goal.checkinCount > 1 ? "s" : "");
        }

        auto chatResult = m_tokenizer->encodeChat(
            systemPrompt, userText, targetText, MAX_SEQUENCE_LENGTH);
        if (chatResult.inputIds.empty()) continue;

        TrainingExample example;
        example.inputIds = std::move(chatResult.inputIds);
        example.labels = std::move(chatResult.labels);
        example.attentionMask = std::move(chatResult.attentionMask);
        example.weight = GOAL_WEIGHT;
        example.sourceEpisodeId = -1;
        examples.append(std::move(example));
        ++added;
    }

    if (added > 0) {
        qDebug() << "TrainingDataGenerator: generated" << added << "goal examples";
    }
}

void TrainingDataGenerator::generateLearningExamples(QList<TrainingExample> &examples,
                                                      int maxCount,
                                                      const QString &identityContext)
{
    if (!m_memoryStore || !m_tokenizer || maxCount <= 0) return;

    auto plans = m_memoryStore->getActiveLearningPlans();
    if (plans.isEmpty()) return;

    QString systemPrompt = buildTrainingSystemPrompt(identityContext);

    QStringList questionTemplates = {
        "What am I learning about?",
        "What topics are we studying together?",
        "How is my learning going?",
        "What are my current learning plans?"
    };

    int added = 0;
    for (const auto &plan : plans) {
        if (added >= maxCount) break;

        QString userText = questionTemplates[added % questionTemplates.size()];
        QString progress = (plan.totalLessons > 0)
            ? QString(" You're on lesson %1 of %2.").arg(plan.currentLesson).arg(plan.totalLessons)
            : "";
        QString targetText = QString("You're learning about %1.%2")
            .arg(plan.topic, progress);

        auto chatResult = m_tokenizer->encodeChat(
            systemPrompt, userText, targetText, MAX_SEQUENCE_LENGTH);
        if (chatResult.inputIds.empty()) continue;

        TrainingExample example;
        example.inputIds = std::move(chatResult.inputIds);
        example.labels = std::move(chatResult.labels);
        example.attentionMask = std::move(chatResult.attentionMask);
        example.weight = LEARNING_WEIGHT;
        example.sourceEpisodeId = -1;
        examples.append(std::move(example));
        ++added;
    }

    if (added > 0) {
        qDebug() << "TrainingDataGenerator: generated" << added << "learning plan examples";
    }
}

void TrainingDataGenerator::generateNewsExamples(QList<TrainingExample> &examples,
                                                  int maxCount,
                                                  const QString &identityContext)
{
    if (!m_memoryStore || !m_tokenizer || maxCount <= 0) return;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("SELECT title, content FROM thoughts WHERE type = 'news_insight' "
              "ORDER BY timestamp DESC LIMIT ?");
    q.addBindValue(maxCount * 2);
    if (!q.exec()) return;

    QString systemPrompt = buildTrainingSystemPrompt(identityContext);

    QStringList questionTemplates = {
        "What news topics interest me?",
        "What have you found interesting in the news lately?",
        "Have you seen anything newsworthy that relates to my interests?"
    };

    int added = 0;
    while (q.next() && added < maxCount) {
        QString title = q.value(0).toString();
        QString content = q.value(1).toString();
        if (title.isEmpty() || content.isEmpty()) continue;

        QString userText = questionTemplates[added % questionTemplates.size()];
        QString targetText = QString("I noticed this: %1. %2").arg(title, content.left(300));

        auto chatResult = m_tokenizer->encodeChat(
            systemPrompt, userText, targetText, MAX_SEQUENCE_LENGTH);
        if (chatResult.inputIds.empty()) continue;

        TrainingExample example;
        example.inputIds = std::move(chatResult.inputIds);
        example.labels = std::move(chatResult.labels);
        example.attentionMask = std::move(chatResult.attentionMask);
        example.weight = NEWS_WEIGHT;
        example.sourceEpisodeId = -1;
        examples.append(std::move(example));
        ++added;
    }

    if (added > 0) {
        qDebug() << "TrainingDataGenerator: generated" << added << "news insight examples";
    }
}

void TrainingDataGenerator::generateThoughtExamples(QList<TrainingExample> &examples,
                                                     int maxCount,
                                                     const QString &identityContext)
{
    if (!m_memoryStore || !m_tokenizer || maxCount <= 0) return;

    QSqlQuery q(m_memoryStore->episodicDatabase());
    q.prepare("SELECT type, title, content FROM thoughts WHERE status = 1 "
              "AND type != 'news_insight' ORDER BY timestamp DESC LIMIT ?");
    q.addBindValue(maxCount * 2);
    if (!q.exec()) return;

    QString systemPrompt = buildTrainingSystemPrompt(identityContext);

    QStringList questionTemplates = {
        "What topics have come up between us proactively?",
        "What have you brought up that we discussed?",
        "What ideas have you shared with me?"
    };

    int added = 0;
    while (q.next() && added < maxCount) {
        QString title = q.value(1).toString();
        QString content = q.value(2).toString();
        if (content.isEmpty()) continue;

        QString userText = questionTemplates[added % questionTemplates.size()];
        QString targetText = QString("I brought up %1 -- %2").arg(title, content.left(300));

        auto chatResult = m_tokenizer->encodeChat(
            systemPrompt, userText, targetText, MAX_SEQUENCE_LENGTH);
        if (chatResult.inputIds.empty()) continue;

        TrainingExample example;
        example.inputIds = std::move(chatResult.inputIds);
        example.labels = std::move(chatResult.labels);
        example.attentionMask = std::move(chatResult.attentionMask);
        example.weight = THOUGHT_WEIGHT;
        example.sourceEpisodeId = -1;
        examples.append(std::move(example));
        ++added;
    }

    if (added > 0) {
        qDebug() << "TrainingDataGenerator: generated" << added << "discussed thought examples";
    }
}

void TrainingDataGenerator::generateSentimentExamples(QList<TrainingExample> &examples,
                                                       int maxCount,
                                                       const QString &identityContext)
{
    if (!m_memoryStore || !m_tokenizer || maxCount <= 0) return;

    auto sentiments = m_memoryStore->getRecentSentiment(50);
    if (sentiments.size() < 5) return;  // Need enough data for meaningful patterns

    QString systemPrompt = buildTrainingSystemPrompt(identityContext);

    // Compute aggregate stats
    double avgScore = 0.0, avgEnergy = 0.0;
    QMap<QString, int> emotionCounts;
    for (const auto &s : sentiments) {
        avgScore += s.sentimentScore;
        avgEnergy += s.energyLevel;
        if (!s.dominantEmotion.isEmpty()) {
            emotionCounts[s.dominantEmotion]++;
        }
    }
    avgScore /= sentiments.size();
    avgEnergy /= sentiments.size();

    // Find top emotions
    QStringList topEmotions;
    QList<QPair<QString, int>> emotionList;
    for (auto it = emotionCounts.begin(); it != emotionCounts.end(); ++it) {
        emotionList.append({it.key(), it.value()});
    }
    std::sort(emotionList.begin(), emotionList.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    for (int i = 0; i < qMin(3, static_cast<int>(emotionList.size())); ++i) {
        topEmotions.append(emotionList[i].first);
    }

    QStringList questionTemplates = {
        "How do I usually feel in our conversations?",
        "What's my typical mood?",
        "Have you noticed any emotional patterns?"
    };

    QString moodWord = (avgScore > 0.3) ? "generally positive"
                     : (avgScore < -0.3) ? "often more reflective"
                     : "fairly balanced";
    QString energyWord = (avgEnergy > 0.65) ? "high-energy"
                       : (avgEnergy < 0.35) ? "low-key"
                       : "moderate";

    int added = 0;
    for (int i = 0; i < maxCount && i < 2; ++i) {
        QString userText = questionTemplates[i % questionTemplates.size()];
        QString targetText = QString("Your overall mood tends to be %1 with %2 energy. "
            "The emotions I see most often are %3.")
            .arg(moodWord, energyWord, topEmotions.join(", "));

        auto chatResult = m_tokenizer->encodeChat(
            systemPrompt, userText, targetText, MAX_SEQUENCE_LENGTH);
        if (chatResult.inputIds.empty()) continue;

        TrainingExample example;
        example.inputIds = std::move(chatResult.inputIds);
        example.labels = std::move(chatResult.labels);
        example.attentionMask = std::move(chatResult.attentionMask);
        example.weight = SENTIMENT_WEIGHT;
        example.sourceEpisodeId = -1;
        examples.append(std::move(example));
        ++added;
    }

    if (added > 0) {
        qDebug() << "TrainingDataGenerator: generated" << added << "sentiment pattern examples";
    }
}

void TrainingDataGenerator::generateDistillationExamples(QList<TrainingExample> &examples,
                                                          int maxCount,
                                                          const QString &identityContext)
{
    if (!m_memoryStore || !m_tokenizer || maxCount <= 0) return;

    auto pairs = m_memoryStore->getUnusedDistillationPairs(maxCount);
    if (pairs.isEmpty()) return;

    int added = 0;
    for (const auto &pair : pairs) {
        // Build system prompt from stored context or identity
        QString systemPrompt = pair.systemContext.isEmpty()
            ? buildTrainingSystemPrompt(identityContext)
            : pair.systemContext;

        auto chatResult = m_tokenizer->encodeChat(
            systemPrompt, pair.userPrompt, pair.teacherResponse, MAX_SEQUENCE_LENGTH);
        if (chatResult.inputIds.empty()) continue;

        // Validate quality score before using as weight multiplier
        double qs = pair.qualityScore;
        if (std::isnan(qs) || std::isinf(qs) || qs < 0.0 || qs > 1.0)
            qs = 0.5;

        TrainingExample example;
        example.inputIds = std::move(chatResult.inputIds);
        example.labels = std::move(chatResult.labels);
        example.attentionMask = std::move(chatResult.attentionMask);
        example.weight = DISTILLATION_WEIGHT * static_cast<float>(qs);
        example.sourceEpisodeId = pair.sourceEpisodeId;
        examples.append(std::move(example));

        // Mark as used in training
        if (!m_memoryStore->markDistillationPairUsed(pair.id)) {
            qWarning() << "TrainingDataGenerator: failed to mark distillation pair" << pair.id << "as used";
        }
        ++added;
    }

    if (added > 0) {
        qDebug() << "TrainingDataGenerator: generated" << added << "distillation examples";
    }
}

void TrainingDataGenerator::clear()
{
    m_examples.clear();
    emit exampleCountChanged();
}

#endif // DATAFORM_TRAINING_ENABLED
