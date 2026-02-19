#ifndef SETTINGSMANAGER_H
#define SETTINGSMANAGER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QJsonObject>
#include <QDir>

class SettingsManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString userName READ userName WRITE setUserName NOTIFY userNameChanged)
    Q_PROPERTY(QString provider READ provider WRITE setProvider NOTIFY providerChanged)
    Q_PROPERTY(QString model READ model WRITE setModel NOTIFY modelChanged)
    Q_PROPERTY(QString openAIKey READ openAIKey WRITE setOpenAIKey NOTIFY openAIKeyChanged)
    Q_PROPERTY(QString anthropicKey READ anthropicKey WRITE setAnthropicKey NOTIFY anthropicKeyChanged)
    Q_PROPERTY(QString lmStudioUrl READ lmStudioUrl WRITE setLmStudioUrl NOTIFY lmStudioUrlChanged)
    Q_PROPERTY(QString ollamaUrl READ ollamaUrl WRITE setOllamaUrl NOTIFY ollamaUrlChanged)
    // DATAFORM-specific settings
    Q_PROPERTY(bool idleLearningEnabled READ idleLearningEnabled WRITE setIdleLearningEnabled NOTIFY idleLearningEnabledChanged)
    Q_PROPERTY(int computeBudgetPercent READ computeBudgetPercent WRITE setComputeBudgetPercent NOTIFY computeBudgetPercentChanged)
    Q_PROPERTY(QString privacyLevel READ privacyLevel WRITE setPrivacyLevel NOTIFY privacyLevelChanged)
    Q_PROPERTY(int populationSize READ populationSize WRITE setPopulationSize NOTIFY populationSizeChanged)
    Q_PROPERTY(bool evolutionEnabled READ evolutionEnabled WRITE setEvolutionEnabled NOTIFY evolutionEnabledChanged)
    // Phase 4: Lifecycle settings
    Q_PROPERTY(int episodeRetentionDays READ episodeRetentionDays WRITE setEpisodeRetentionDays NOTIFY episodeRetentionDaysChanged)
    Q_PROPERTY(bool autoBackupEnabled READ autoBackupEnabled WRITE setAutoBackupEnabled NOTIFY autoBackupEnabledChanged)
    Q_PROPERTY(int maxDiskUsageGB READ maxDiskUsageGB WRITE setMaxDiskUsageGB NOTIFY maxDiskUsageGBChanged)
    // Phase 5: Research settings
    Q_PROPERTY(bool researchEnabled READ researchEnabled WRITE setResearchEnabled NOTIFY researchEnabledChanged)
    Q_PROPERTY(int maxResearchPerDay READ maxResearchPerDay WRITE setMaxResearchPerDay NOTIFY maxResearchPerDayChanged)
    // Phase 6: Local model path
    Q_PROPERTY(QString localModelPath READ localModelPath WRITE setLocalModelPath NOTIFY localModelPathChanged)
    // News settings
    Q_PROPERTY(bool newsEnabled READ newsEnabled WRITE setNewsEnabled NOTIFY newsEnabledChanged)
    Q_PROPERTY(int maxNewsPerDay READ maxNewsPerDay WRITE setMaxNewsPerDay NOTIFY maxNewsPerDayChanged)
    Q_PROPERTY(QStringList newsFeeds READ newsFeeds WRITE setNewsFeeds NOTIFY newsFeedsChanged)
    // Agentic features settings
    Q_PROPERTY(bool goalsEnabled READ goalsEnabled WRITE setGoalsEnabled NOTIFY goalsEnabledChanged)
    Q_PROPERTY(bool sentimentTrackingEnabled READ sentimentTrackingEnabled WRITE setSentimentTrackingEnabled NOTIFY sentimentTrackingEnabledChanged)
    Q_PROPERTY(bool teachMeEnabled READ teachMeEnabled WRITE setTeachMeEnabled NOTIFY teachMeEnabledChanged)
    // Proactive Dialog settings
    Q_PROPERTY(bool proactiveDialogsEnabled READ proactiveDialogsEnabled WRITE setProactiveDialogsEnabled NOTIFY proactiveDialogsEnabledChanged)
    Q_PROPERTY(int maxThoughtsPerDay READ maxThoughtsPerDay WRITE setMaxThoughtsPerDay NOTIFY maxThoughtsPerDayChanged)
    // Model context settings
    Q_PROPERTY(int modelContextLength READ modelContextLength WRITE setModelContextLength NOTIFY modelContextLengthChanged)
    // Phase 7: Semantic Memory settings
    Q_PROPERTY(QString embeddingModel READ embeddingModel WRITE setEmbeddingModel NOTIFY embeddingModelChanged)
    Q_PROPERTY(bool semanticSearchEnabled READ semanticSearchEnabled WRITE setSemanticSearchEnabled NOTIFY semanticSearchEnabledChanged)
    // Security settings
    Q_PROPERTY(QString encryptionMode READ encryptionMode WRITE setEncryptionMode NOTIFY encryptionModeChanged)

public:
    explicit SettingsManager(const QString &profilePath, QObject *parent = nullptr);
    ~SettingsManager();

    // LLM settings
    QString userName() const { return m_userName; }
    void setUserName(const QString &userName);

    QString provider() const { return m_provider; }
    void setProvider(const QString &provider);

    QString model() const { return m_model; }
    void setModel(const QString &model);

    QString openAIKey() const { return m_openAIKey; }
    void setOpenAIKey(const QString &key);

    QString anthropicKey() const { return m_anthropicKey; }
    void setAnthropicKey(const QString &key);

    QString lmStudioUrl() const { return m_lmStudioUrl; }
    void setLmStudioUrl(const QString &url);

    QString ollamaUrl() const { return m_ollamaUrl; }
    void setOllamaUrl(const QString &url);

    // DATAFORM settings
    bool idleLearningEnabled() const { return m_idleLearningEnabled; }
    void setIdleLearningEnabled(bool enabled);

    int computeBudgetPercent() const { return m_computeBudgetPercent; }
    void setComputeBudgetPercent(int percent);

    QString privacyLevel() const { return m_privacyLevel; }
    void setPrivacyLevel(const QString &level);

    // Evolution settings (Phase 3)
    int populationSize() const { return m_populationSize; }
    void setPopulationSize(int size);

    bool evolutionEnabled() const { return m_evolutionEnabled; }
    void setEvolutionEnabled(bool enabled);

    // Phase 4: Lifecycle settings
    int episodeRetentionDays() const { return m_episodeRetentionDays; }
    void setEpisodeRetentionDays(int days);

    bool autoBackupEnabled() const { return m_autoBackupEnabled; }
    void setAutoBackupEnabled(bool enabled);

    int maxDiskUsageGB() const { return m_maxDiskUsageGB; }
    void setMaxDiskUsageGB(int gb);

    // Phase 5: Research settings
    bool researchEnabled() const { return m_researchEnabled; }
    void setResearchEnabled(bool enabled);

    int maxResearchPerDay() const { return m_maxResearchPerDay; }
    void setMaxResearchPerDay(int max);

    // Phase 6: Local model
    QString localModelPath() const { return m_localModelPath; }
    void setLocalModelPath(const QString &path);

    // News
    bool newsEnabled() const { return m_newsEnabled; }
    void setNewsEnabled(bool enabled);
    int maxNewsPerDay() const { return m_maxNewsPerDay; }
    void setMaxNewsPerDay(int max);
    QStringList newsFeeds() const { return m_newsFeeds; }
    void setNewsFeeds(const QStringList &feeds);
    Q_INVOKABLE void addNewsFeed(const QString &url);
    Q_INVOKABLE void removeNewsFeed(int index);

    // Agentic features
    bool goalsEnabled() const { return m_goalsEnabled; }
    void setGoalsEnabled(bool enabled);
    bool sentimentTrackingEnabled() const { return m_sentimentTrackingEnabled; }
    void setSentimentTrackingEnabled(bool enabled);
    bool teachMeEnabled() const { return m_teachMeEnabled; }
    void setTeachMeEnabled(bool enabled);

    // Proactive Dialog
    bool proactiveDialogsEnabled() const { return m_proactiveDialogsEnabled; }
    void setProactiveDialogsEnabled(bool enabled);
    int maxThoughtsPerDay() const { return m_maxThoughtsPerDay; }
    void setMaxThoughtsPerDay(int max);

    // Model context
    int modelContextLength() const { return m_modelContextLength; }
    void setModelContextLength(int tokens);

    // Phase 7: Semantic Memory
    QString embeddingModel() const { return m_embeddingModel; }
    void setEmbeddingModel(const QString &model);
    bool semanticSearchEnabled() const { return m_semanticSearchEnabled; }
    void setSemanticSearchEnabled(bool enabled);

    // Security
    QString encryptionMode() const { return m_encryptionMode; }
    void setEncryptionMode(const QString &mode);

    Q_INVOKABLE void saveSettings();
    Q_INVOKABLE void loadSettings();
    Q_INVOKABLE void resetToDefaults();

signals:
    void userNameChanged();
    void providerChanged();
    void modelChanged();
    void openAIKeyChanged();
    void anthropicKeyChanged();
    void lmStudioUrlChanged();
    void ollamaUrlChanged();
    void idleLearningEnabledChanged();
    void computeBudgetPercentChanged();
    void privacyLevelChanged();
    void populationSizeChanged();
    void evolutionEnabledChanged();
    void episodeRetentionDaysChanged();
    void autoBackupEnabledChanged();
    void maxDiskUsageGBChanged();
    void researchEnabledChanged();
    void maxResearchPerDayChanged();
    void localModelPathChanged();
    void newsEnabledChanged();
    void maxNewsPerDayChanged();
    void newsFeedsChanged();
    void goalsEnabledChanged();
    void sentimentTrackingEnabledChanged();
    void teachMeEnabledChanged();
    void proactiveDialogsEnabledChanged();
    void maxThoughtsPerDayChanged();
    void modelContextLengthChanged();
    void embeddingModelChanged();
    void semanticSearchEnabledChanged();
    void encryptionModeChanged();
    void settingsSaved();
    void settingsLoaded();

private:
    QString getConfigFilePath() const;

    QString m_profilePath;
    QString m_userName;
    QString m_provider;
    QString m_model;
    QString m_openAIKey;
    QString m_anthropicKey;
    QString m_lmStudioUrl;
    QString m_ollamaUrl;
    bool m_idleLearningEnabled;
    int m_computeBudgetPercent;
    QString m_privacyLevel;
    int m_populationSize;
    bool m_evolutionEnabled;
    int m_episodeRetentionDays;
    bool m_autoBackupEnabled;
    int m_maxDiskUsageGB;
    bool m_researchEnabled = true;
    int m_maxResearchPerDay = 10;
    QString m_localModelPath;
    bool m_newsEnabled = true;
    int m_maxNewsPerDay = 3;
    QStringList m_newsFeeds = {"https://www.allsides.com/rss/news"};
    bool m_goalsEnabled = true;
    bool m_sentimentTrackingEnabled = true;
    bool m_teachMeEnabled = true;
    bool m_proactiveDialogsEnabled = true;
    int m_maxThoughtsPerDay = 5;
    int m_modelContextLength = 8192;
    QString m_embeddingModel = "nomic-embed-text";
    bool m_semanticSearchEnabled = true;
    QString m_encryptionMode = "portable";
};

#endif // SETTINGSMANAGER_H
