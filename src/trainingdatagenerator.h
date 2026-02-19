#ifndef TRAININGDATAGENERATOR_H
#define TRAININGDATAGENERATOR_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>
#include <QList>
#include <vector>
#include <cstdint>
#include <random>

class MemoryStore;
class Tokenizer;
class ResearchStore;

struct TrainingExample {
    std::vector<int64_t> inputIds;
    std::vector<int64_t> labels;        // -100 for input portion, real IDs for target
    std::vector<int64_t> attentionMask;
    float weight = 1.0f;               // Signal strength multiplier
    qint64 sourceEpisodeId = -1;
};

class TrainingDataGenerator : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int exampleCount READ exampleCount NOTIFY exampleCountChanged)
    Q_PROPERTY(qint64 lastProcessedEpisodeId READ lastProcessedEpisodeId NOTIFY lastProcessedEpisodeIdChanged)

public:
    explicit TrainingDataGenerator(QObject *parent = nullptr);
    ~TrainingDataGenerator();

    void setMemoryStore(MemoryStore *store);
    void setTokenizer(Tokenizer *tokenizer);
    void setResearchStore(ResearchStore *store);

    int exampleCount() const { return static_cast<int>(m_examples.size()); }
    qint64 lastProcessedEpisodeId() const { return m_lastProcessedEpisodeId; }
    void setLastProcessedEpisodeId(qint64 id) { m_lastProcessedEpisodeId = id; }

    // Select high-signal episodes for training (edits > thumbs-up > inquiry > thumbs-down)
    QList<qint64> selectHighSignalEpisodes(int maxCount = 50);

    // Generate training batch from selected episodes
    void generateTrainingBatch(int maxExamples = 32);

    // Access generated examples
    const QList<TrainingExample> &examples() const { return m_examples; }
    void clear();

signals:
    void exampleCountChanged();
    void lastProcessedEpisodeIdChanged();
    void generationComplete(int count);
    void generationError(const QString &error);

private:
    TrainingExample episodeToExample(qint64 episodeId, const QString &identityContext);
    QString buildTrainingSystemPrompt(const QString &identityContext) const;
    double computeSignalStrength(int feedback, bool corrected, bool hasInquiry) const;

    // Phase 6: Generate examples from research findings and traits
    void generateEpisodicExamples(QList<TrainingExample> &examples,
                                   int maxCount, const QString &identityContext);
    void generateResearchExamples(QList<TrainingExample> &examples,
                                   int maxCount, const QString &identityContext);
    void generateTraitExamples(QList<TrainingExample> &examples,
                                int maxCount, const QString &identityContext);

    // Phase 7: Expanded training data sources
    void generateGoalExamples(QList<TrainingExample> &examples,
                               int maxCount, const QString &identityContext);
    void generateLearningExamples(QList<TrainingExample> &examples,
                                   int maxCount, const QString &identityContext);
    void generateNewsExamples(QList<TrainingExample> &examples,
                               int maxCount, const QString &identityContext);
    void generateThoughtExamples(QList<TrainingExample> &examples,
                                  int maxCount, const QString &identityContext);
    void generateSentimentExamples(QList<TrainingExample> &examples,
                                    int maxCount, const QString &identityContext);

    MemoryStore *m_memoryStore = nullptr;
    Tokenizer *m_tokenizer = nullptr;
    ResearchStore *m_researchStore = nullptr;
    QList<TrainingExample> m_examples;
    qint64 m_lastProcessedEpisodeId = 0;
    qint64 m_lastProcessedFindingId = 0;
    std::mt19937 m_rng{std::random_device{}()};

    static constexpr int MAX_SEQUENCE_LENGTH = 512;
    static constexpr float EDIT_WEIGHT = 2.0f;
    static constexpr float POSITIVE_WEIGHT = 1.5f;
    static constexpr float INQUIRY_WEIGHT = 1.2f;
    static constexpr float NEUTRAL_WEIGHT = 1.0f;
    static constexpr float NEGATIVE_WEIGHT = 0.3f;
    static constexpr float RESEARCH_WEIGHT = 1.0f;
    static constexpr float TRAIT_WEIGHT = 1.3f;

    // Phase 7: Expanded data source weights
    static constexpr float GOAL_WEIGHT = 1.1f;
    static constexpr float LEARNING_WEIGHT = 1.0f;
    static constexpr float NEWS_WEIGHT = 0.8f;
    static constexpr float THOUGHT_WEIGHT = 0.9f;
    static constexpr float SENTIMENT_WEIGHT = 0.7f;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // TRAININGDATAGENERATOR_H
