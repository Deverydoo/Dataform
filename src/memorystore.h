#ifndef MEMORYSTORE_H
#define MEMORYSTORE_H

#include <QObject>
#include <QString>
#include <QSqlDatabase>
#include <QDateTime>
#include <QVariantList>
#include <QVariantMap>

class SettingsManager;

struct EpisodicRecord {
    qint64 id = -1;
    QDateTime timestamp;
    QString topic;
    QString userText;
    QString assistantText;
    QString inquiryText;
    int userFeedback = 0;       // -1 = thumbs down, 0 = none, 1 = thumbs up
    QString userEditText;
    QString editDiff;
    bool reasked = false;
    bool accepted = true;
    bool corrected = false;
    QString tags;
    QString modelId;
    QString adapterVersion;
};

struct ConversationRecord {
    qint64 id = -1;
    QString title;
    QDateTime createdTs;
    QDateTime lastActivityTs;
    int messageCount = 0;
};

struct TraitRecord {
    QString traitId;
    QString type;               // "value", "preference", "policy", "motivation"
    QString statement;
    double confidence = 0.0;
    QStringList evidenceEpisodeIds;
    QDateTime lastConfirmedTs;
    QDateTime createdTs;
    QDateTime updatedTs;
};

struct ReminderRecord {
    qint64 id = -1;
    QString content;
    QDateTime dueTs;
    QDateTime createdTs;
    int status = 0;         // 0=pending, 1=triggered, 2=dismissed
    qint64 sourceEpisodeId = -1;
};

struct GoalRecord {
    qint64 id = -1;
    QString title;
    QString description;
    int status = 0;         // 0=active, 1=achieved, 2=abandoned
    QDateTime createdTs;
    QDateTime lastCheckinTs;
    qint64 sourceEpisodeId = -1;
    int checkinCount = 0;
};

struct SentimentRecord {
    qint64 id = -1;
    qint64 episodeId = -1;
    double sentimentScore = 0.0;    // -1.0 to 1.0
    double energyLevel = 0.5;       // 0.0 to 1.0
    QString dominantEmotion;
    QDateTime analyzedTs;
};

struct LearningPlanRecord {
    qint64 id = -1;
    QString topic;
    QString planJson;
    int currentLesson = 0;
    int totalLessons = 0;
    int status = 0;         // 0=active, 1=completed, 2=paused
    QDateTime createdTs;
    QDateTime lastSessionTs;
};

struct DistillationRecord {
    qint64 id = -1;
    QString userPrompt;
    QString teacherResponse;
    QString systemContext;
    qint64 sourceEpisodeId = -1;
    QString sourceType;         // "episode", "synthetic", "trait_qa"
    double qualityScore = 0.0;
    bool usedInTraining = false;
    QDateTime createdTs;
};

class MemoryStore : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int episodeCount READ episodeCount NOTIFY episodeCountChanged)
    Q_PROPERTY(int traitCount READ traitCount NOTIFY traitCountChanged)

public:
    explicit MemoryStore(const QString &profilePath, QObject *parent = nullptr);
    ~MemoryStore();

    void setSettingsManager(SettingsManager *settings);
    bool initialize();
    void close();
    bool flush();  // Close + re-initialize (syncs .enc files for export)

    // --- Conversation CRUD ---
    Q_INVOKABLE qint64 createConversation(const QString &title);
    Q_INVOKABLE bool updateConversationTitle(qint64 id, const QString &title);
    Q_INVOKABLE bool deleteConversation(qint64 id);
    Q_INVOKABLE QVariantList getConversationsForQml(int limit = 50);
    QList<EpisodicRecord> getEpisodesForConversation(qint64 id) const;
    Q_INVOKABLE QVariantList getEpisodesForConversationQml(qint64 id);

    // --- Episodic CRUD ---
    Q_INVOKABLE qint64 insertEpisode(const QString &topic,
                                      const QString &userText,
                                      const QString &assistantText,
                                      const QString &modelId,
                                      const QString &adapterVersion,
                                      qint64 conversationId = -1);

    Q_INVOKABLE bool updateEpisodeFeedback(qint64 episodeId, int feedback);
    Q_INVOKABLE bool updateEpisodeEdit(qint64 episodeId,
                                        const QString &editText,
                                        const QString &diff);
    Q_INVOKABLE bool updateEpisodeInquiry(qint64 episodeId,
                                           const QString &inquiryText);
    Q_INVOKABLE bool updateEpisodeOutcome(qint64 episodeId,
                                           bool reasked, bool accepted, bool corrected);
    Q_INVOKABLE bool updateEpisodeTags(qint64 episodeId, const QString &tags);

    EpisodicRecord getEpisode(qint64 id) const;
    QList<EpisodicRecord> getRecentEpisodes(int limit = 50) const;
    QList<EpisodicRecord> searchEpisodes(const QString &query, int limit = 20) const;
    QList<EpisodicRecord> getHighSignalEpisodes(qint64 afterId = 0, int limit = 50) const;
    Q_INVOKABLE QVariantList getRecentEpisodesForQml(int limit = 50) const;

    // FTS search
    Q_INVOKABLE QVariantList searchConversationsForQml(const QString &query, int limit = 30);

    // Analytics queries
    Q_INVOKABLE QVariantList getTraitGrowthForQml(int daysPast = 90);
    Q_INVOKABLE QVariantList getEpisodeActivityForQml(int daysPast = 30);

    // --- Traits CRUD ---
    Q_INVOKABLE QString insertTrait(const QString &type,
                                     const QString &statement,
                                     double confidence);
    Q_INVOKABLE bool updateTraitConfidence(const QString &traitId,
                                            double confidence);
    Q_INVOKABLE bool addTraitEvidence(const QString &traitId,
                                       qint64 episodeId);
    Q_INVOKABLE bool confirmTrait(const QString &traitId);
    Q_INVOKABLE bool removeTrait(const QString &traitId);

    TraitRecord getTrait(const QString &traitId) const;
    QList<TraitRecord> getAllTraits() const;
    QList<TraitRecord> getTraitsByType(const QString &type) const;
    Q_INVOKABLE QVariantList getAllTraitsForQml() const;

    // Counts
    int episodeCount() const;
    int traitCount() const;

    // Bulk operations
    Q_INVOKABLE void clearAllEpisodes();
    Q_INVOKABLE void clearAllTraits();
    Q_INVOKABLE void clearAllMemory();

    // Phase 4: Lifecycle query methods
    QList<EpisodicRecord> getEpisodesOlderThan(int ageDays,
                                                 bool excludeArchived = true,
                                                 int limit = 100) const;
    bool archiveEpisode(qint64 episodeId, const QString &summary);
    QList<TraitRecord> getUnconfirmedTraitsSince(int daysSince) const;
    bool updateTraitDecay(const QString &traitId, double newConfidence);
    bool archiveTrait(const QString &traitId);
    int archivedEpisodeCount() const;
    int archivedTraitCount() const;

    // --- Reminders CRUD ---
    qint64 insertReminder(const QString &content, const QString &dueTsStr = QString(), qint64 episodeId = -1);
    QList<ReminderRecord> getDueReminders() const;
    QList<ReminderRecord> getPendingReminders() const;
    bool updateReminderStatus(qint64 id, int status);
    Q_INVOKABLE QVariantList getPendingRemindersForQml();

    // --- Goals CRUD ---
    qint64 insertGoal(const QString &title, const QString &description = QString(), qint64 episodeId = -1);
    QList<GoalRecord> getActiveGoals() const;
    bool updateGoalStatus(qint64 id, int status);
    bool updateGoalCheckin(qint64 id);
    QList<GoalRecord> getGoalsNeedingCheckin(int daysSince = 7) const;
    Q_INVOKABLE QVariantList getActiveGoalsForQml();

    // --- Sentiment CRUD ---
    qint64 insertSentiment(qint64 episodeId, double score, double energy, const QString &emotion);
    bool hasEpisodeSentiment(qint64 episodeId) const;
    QList<SentimentRecord> getRecentSentiment(int limit = 20) const;
    double getAverageSentiment(int daysPast = 7) const;

    // --- Learning Plans CRUD ---
    qint64 insertLearningPlan(const QString &topic, const QString &planJson, int totalLessons);
    QList<LearningPlanRecord> getActiveLearningPlans() const;
    bool advanceLessonPlan(qint64 id);
    bool updateLearningPlanStatus(qint64 id, int status);
    Q_INVOKABLE QVariantList getActivePlansForQml();

    // --- Political Lean CRUD ---
    qint64 insertLeanAnalysis(double leanScore, const QStringList &contributingTraitIds);
    Q_INVOKABLE QVariantMap getLatestLeanForQml();
    Q_INVOKABLE QVariantList getValueAndPolicyTraitsForQml();

    // --- Distillation CRUD ---
    qint64 insertDistillationPair(const QString &userPrompt, const QString &teacherResponse,
                                   const QString &systemContext, qint64 sourceEpisodeId = -1,
                                   const QString &sourceType = "episode", double qualityScore = 0.0);
    QList<DistillationRecord> getUnusedDistillationPairs(int limit = 20) const;
    bool markDistillationPairUsed(qint64 id);
    int distillationPairCount() const;
    int usedDistillationPairCount() const;
    QList<qint64> getDistillationSourceEpisodeIds() const;

    // Distillation eval CRUD
    qint64 insertDistillationEval(const QString &testPrompt, const QString &teacherResp,
                                   const QString &studentResp, double similarityScore);
    double getAverageDistillationScore(int recentCount = 20) const;
    Q_INVOKABLE QVariantList getDistillationStatsForQml();

    // Phase 5: Database accessor for ResearchStore
    QSqlDatabase &episodicDatabase() { return m_episodicDb; }

signals:
    void episodeCountChanged();
    void traitCountChanged();
    void episodeInserted(qint64 id);
    void traitInserted(const QString &traitId);
    void databaseError(const QString &error);
    void conversationListChanged();
    void conversationCreated(qint64 id);

private:
    bool openDatabase(const QString &name, const QString &dbFilePath, QSqlDatabase &db);
    bool createEpisodicSchema();
    bool createTraitsSchema();
    bool migrateEpisodicSchema();
    bool migrateTraitsSchema();
    void encryptAndClose(const QString &dbFilePath, const QString &encFilePath);
    bool decryptAndOpen(const QString &encFilePath, const QString &dbFilePath);
    QByteArray getEncryptionKey() const;

    QVariantMap episodeToVariantMap(const EpisodicRecord &record) const;
    QVariantMap traitToVariantMap(const TraitRecord &record) const;

    QString m_profilePath;
    QString m_memoryPath;
    QSqlDatabase m_episodicDb;
    QSqlDatabase m_traitsDb;
    bool m_initialized = false;
    SettingsManager *m_settingsManager = nullptr;
};

#endif // MEMORYSTORE_H
