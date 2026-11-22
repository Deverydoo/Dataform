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
    Q_PROPERTY(int backgroundBudgetPercent READ backgroundBudgetPercent WRITE setBackgroundBudgetPercent NOTIFY backgroundBudgetPercentChanged)
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
    Q_PROPERTY(QStringList siteBlacklist READ siteBlacklist WRITE setSiteBlacklist NOTIFY siteBlacklistChanged)
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
    // Background model (llama.cpp)
    Q_PROPERTY(bool backgroundModelEnabled READ backgroundModelEnabled WRITE setBackgroundModelEnabled NOTIFY backgroundModelEnabledChanged)
    Q_PROPERTY(QString backgroundModelPath READ backgroundModelPath WRITE setBackgroundModelPath NOTIFY backgroundModelPathChanged)
    // Phase 8: Distillation settings
    Q_PROPERTY(bool distillationEnabled READ distillationEnabled WRITE setDistillationEnabled NOTIFY distillationEnabledChanged)
    Q_PROPERTY(int distillationDailyCycles READ distillationDailyCycles WRITE setDistillationDailyCycles NOTIFY distillationDailyCyclesChanged)
    // Security settings
    Q_PROPERTY(QString encryptionMode READ encryptionMode WRITE setEncryptionMode NOTIFY encryptionModeChanged)
    // Voice settings (spoken assistant)
    Q_PROPERTY(bool voiceEnabled READ voiceEnabled WRITE setVoiceEnabled NOTIFY voiceEnabledChanged)
    Q_PROPERTY(bool speakReplies READ speakReplies WRITE setSpeakReplies NOTIFY speakRepliesChanged)
    Q_PROPERTY(QString sttModelPath READ sttModelPath WRITE setSttModelPath NOTIFY sttModelPathChanged)
    Q_PROPERTY(QString sttDevice READ sttDevice WRITE setSttDevice NOTIFY sttDeviceChanged)
    Q_PROPERTY(QString sttComputeType READ sttComputeType WRITE setSttComputeType NOTIFY sttComputeTypeChanged)
    Q_PROPERTY(QString sttLanguage READ sttLanguage WRITE setSttLanguage NOTIFY sttLanguageChanged)
    Q_PROPERTY(QString sttVadModelPath READ sttVadModelPath WRITE setSttVadModelPath NOTIFY sttVadModelPathChanged)
    Q_PROPERTY(QString micDeviceName READ micDeviceName WRITE setMicDeviceName NOTIFY micDeviceNameChanged)
    Q_PROPERTY(QString ttsBackend READ ttsBackend WRITE setTtsBackend NOTIFY ttsBackendChanged)
    Q_PROPERTY(QString piperExePath READ piperExePath WRITE setPiperExePath NOTIFY piperExePathChanged)
    Q_PROPERTY(QString piperVoicePath READ piperVoicePath WRITE setPiperVoicePath NOTIFY piperVoicePathChanged)
    Q_PROPERTY(QString sapiVoiceName READ sapiVoiceName WRITE setSapiVoiceName NOTIFY sapiVoiceNameChanged)
    Q_PROPERTY(double ttsSpeed READ ttsSpeed WRITE setTtsSpeed NOTIFY ttsSpeedChanged)
    Q_PROPERTY(QString assistantName READ assistantName WRITE setAssistantName NOTIFY assistantNameChanged)

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

    int backgroundBudgetPercent() const { return m_backgroundBudgetPercent; }
    void setBackgroundBudgetPercent(int percent);

    QString privacyLevel() const { return m_privacyLevel; }
    void setPrivacyLevel(const QString &level);

    // True when the user has asked DATAFORM to stop deriving new inferences from
    // their data. Every idle engine that analyses episodes must consult this
    // before starting a cycle — it is the single definition of what "minimal"
    // means, so the check cannot drift between engines.
    Q_INVOKABLE bool privacyBlocksInference() const { return m_privacyLevel == QLatin1String("minimal"); }

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

    // Site blacklist
    QStringList siteBlacklist() const { return m_siteBlacklist; }
    void setSiteBlacklist(const QStringList &list);
    Q_INVOKABLE void addBlacklistedSite(const QString &domain);
    Q_INVOKABLE void removeBlacklistedSite(const QString &domain);

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

    // Background model (llama.cpp)
    bool backgroundModelEnabled() const { return m_backgroundModelEnabled; }
    void setBackgroundModelEnabled(bool enabled);
    QString backgroundModelPath() const { return m_backgroundModelPath; }
    void setBackgroundModelPath(const QString &path);

    // Phase 8: Distillation
    bool distillationEnabled() const { return m_distillationEnabled; }
    void setDistillationEnabled(bool enabled);
    int distillationDailyCycles() const { return m_distillationDailyCycles; }
    void setDistillationDailyCycles(int cycles);

    // Security
    QString encryptionMode() const { return m_encryptionMode; }
    void setEncryptionMode(const QString &mode);

    // Voice (spoken assistant)
    bool voiceEnabled() const { return m_voiceEnabled; }
    void setVoiceEnabled(bool enabled);

    bool speakReplies() const { return m_speakReplies; }
    void setSpeakReplies(bool enabled);

    QString sttModelPath() const { return m_sttModelPath; }
    void setSttModelPath(const QString &path);

    QString sttDevice() const { return m_sttDevice; }
    void setSttDevice(const QString &device);

    QString sttComputeType() const { return m_sttComputeType; }
    void setSttComputeType(const QString &computeType);

    QString sttLanguage() const { return m_sttLanguage; }
    void setSttLanguage(const QString &language);

    QString sttVadModelPath() const { return m_sttVadModelPath; }
    void setSttVadModelPath(const QString &path);

    QString micDeviceName() const { return m_micDeviceName; }
    void setMicDeviceName(const QString &name);

    QString ttsBackend() const { return m_ttsBackend; }
    void setTtsBackend(const QString &backend);

    QString piperExePath() const { return m_piperExePath; }
    void setPiperExePath(const QString &path);

    QString piperVoicePath() const { return m_piperVoicePath; }
    void setPiperVoicePath(const QString &path);

    QString sapiVoiceName() const { return m_sapiVoiceName; }
    void setSapiVoiceName(const QString &name);

    double ttsSpeed() const { return m_ttsSpeed; }
    void setTtsSpeed(double speed);

    QString assistantName() const { return m_assistantName; }
    void setAssistantName(const QString &name);

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
    void backgroundBudgetPercentChanged();
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
    void siteBlacklistChanged();
    void goalsEnabledChanged();
    void sentimentTrackingEnabledChanged();
    void teachMeEnabledChanged();
    void proactiveDialogsEnabledChanged();
    void maxThoughtsPerDayChanged();
    void modelContextLengthChanged();
    void embeddingModelChanged();
    void semanticSearchEnabledChanged();
    void backgroundModelEnabledChanged();
    void backgroundModelPathChanged();
    void distillationEnabledChanged();
    void distillationDailyCyclesChanged();
    void encryptionModeChanged();
    void voiceEnabledChanged();
    void speakRepliesChanged();
    void sttModelPathChanged();
    void sttDeviceChanged();
    void sttComputeTypeChanged();
    void sttLanguageChanged();
    void sttVadModelPathChanged();
    void micDeviceNameChanged();
    void ttsBackendChanged();
    void piperExePathChanged();
    void piperVoicePathChanged();
    void sapiVoiceNameChanged();
    void ttsSpeedChanged();
    void assistantNameChanged();
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
    int m_backgroundBudgetPercent = 5;
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
    QStringList m_siteBlacklist;
    bool m_goalsEnabled = true;
    bool m_sentimentTrackingEnabled = true;
    bool m_teachMeEnabled = true;
    bool m_proactiveDialogsEnabled = true;
    int m_maxThoughtsPerDay = 5;
    int m_modelContextLength = 8192;
    QString m_embeddingModel = "nomic-embed-text";
    bool m_semanticSearchEnabled = true;
    bool m_backgroundModelEnabled = false;
    QString m_backgroundModelPath = "models/background_llm";
    bool m_distillationEnabled = false;
    int m_distillationDailyCycles = 3;
    QString m_encryptionMode = "portable";

    // Voice. Off by default: the speech model and Piper are separate downloads,
    // and a first run should not fail because they are missing.
    bool m_voiceEnabled = false;
    bool m_speakReplies = true;
    QString m_sttModelPath;
    QString m_sttDevice = "cuda";
    QString m_sttComputeType = "float16";
    QString m_sttLanguage = "en";
    QString m_sttVadModelPath;   // empty = look beside the Whisper model
    QString m_micDeviceName;
    QString m_ttsBackend = "piper";
    QString m_piperExePath;
    QString m_piperVoicePath;
    QString m_sapiVoiceName;
    double m_ttsSpeed = 1.0;
    QString m_assistantName = "Dataform";
};

#endif // SETTINGSMANAGER_H
