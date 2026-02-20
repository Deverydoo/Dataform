#include "settingsmanager.h"
#include "schemamigrator.h"
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

SettingsManager::SettingsManager(const QString &profilePath, QObject *parent)
    : QObject(parent)
    , m_profilePath(profilePath)
    , m_userName("User")
    , m_provider("Ollama")
    , m_lmStudioUrl("http://localhost:1234")
    , m_ollamaUrl("http://localhost:11434")
    , m_idleLearningEnabled(true)
    , m_computeBudgetPercent(10)
    , m_privacyLevel("standard")
    , m_populationSize(4)
    , m_evolutionEnabled(true)
    , m_episodeRetentionDays(365)
    , m_autoBackupEnabled(true)
    , m_maxDiskUsageGB(10)
{
}

SettingsManager::~SettingsManager()
{
}

// --- Property setters ---

void SettingsManager::setUserName(const QString &userName)
{
    if (m_userName != userName) {
        m_userName = userName;
        emit userNameChanged();
        saveSettings();
    }
}

void SettingsManager::setProvider(const QString &provider)
{
    if (m_provider != provider) {
        m_provider = provider;
        emit providerChanged();
        saveSettings();
    }
}

void SettingsManager::setModel(const QString &model)
{
    if (m_model != model) {
        m_model = model;
        emit modelChanged();
        saveSettings();
    }
}

void SettingsManager::setOpenAIKey(const QString &key)
{
    if (m_openAIKey != key) {
        m_openAIKey = key;
        emit openAIKeyChanged();
        saveSettings();
    }
}

void SettingsManager::setAnthropicKey(const QString &key)
{
    if (m_anthropicKey != key) {
        m_anthropicKey = key;
        emit anthropicKeyChanged();
        saveSettings();
    }
}

void SettingsManager::setLmStudioUrl(const QString &url)
{
    if (m_lmStudioUrl != url) {
        m_lmStudioUrl = url;
        emit lmStudioUrlChanged();
        saveSettings();
    }
}

void SettingsManager::setOllamaUrl(const QString &url)
{
    if (m_ollamaUrl != url) {
        m_ollamaUrl = url;
        emit ollamaUrlChanged();
        saveSettings();
    }
}

void SettingsManager::setIdleLearningEnabled(bool enabled)
{
    if (m_idleLearningEnabled != enabled) {
        m_idleLearningEnabled = enabled;
        emit idleLearningEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setComputeBudgetPercent(int percent)
{
    percent = qBound(1, percent, 50);
    if (m_computeBudgetPercent != percent) {
        m_computeBudgetPercent = percent;
        emit computeBudgetPercentChanged();
        saveSettings();
    }
}

void SettingsManager::setPrivacyLevel(const QString &level)
{
    if (m_privacyLevel != level) {
        m_privacyLevel = level;
        emit privacyLevelChanged();
        saveSettings();
    }
}

void SettingsManager::setPopulationSize(int size)
{
    size = qBound(1, size, 8);
    if (m_populationSize != size) {
        m_populationSize = size;
        emit populationSizeChanged();
        saveSettings();
    }
}

void SettingsManager::setEvolutionEnabled(bool enabled)
{
    if (m_evolutionEnabled != enabled) {
        m_evolutionEnabled = enabled;
        emit evolutionEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setEpisodeRetentionDays(int days)
{
    days = qBound(90, days, 3650);
    if (m_episodeRetentionDays != days) {
        m_episodeRetentionDays = days;
        emit episodeRetentionDaysChanged();
        saveSettings();
    }
}

void SettingsManager::setAutoBackupEnabled(bool enabled)
{
    if (m_autoBackupEnabled != enabled) {
        m_autoBackupEnabled = enabled;
        emit autoBackupEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setMaxDiskUsageGB(int gb)
{
    gb = qBound(1, gb, 100);
    if (m_maxDiskUsageGB != gb) {
        m_maxDiskUsageGB = gb;
        emit maxDiskUsageGBChanged();
        saveSettings();
    }
}

void SettingsManager::setResearchEnabled(bool enabled)
{
    if (m_researchEnabled != enabled) {
        m_researchEnabled = enabled;
        emit researchEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setMaxResearchPerDay(int max)
{
    max = qBound(1, max, 20);
    if (m_maxResearchPerDay != max) {
        m_maxResearchPerDay = max;
        emit maxResearchPerDayChanged();
        saveSettings();
    }
}

void SettingsManager::setLocalModelPath(const QString &path)
{
    if (m_localModelPath != path) {
        m_localModelPath = path;
        emit localModelPathChanged();
        saveSettings();
    }
}

void SettingsManager::setNewsEnabled(bool enabled)
{
    if (m_newsEnabled != enabled) {
        m_newsEnabled = enabled;
        emit newsEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setMaxNewsPerDay(int max)
{
    max = qBound(1, max, 10);
    if (m_maxNewsPerDay != max) {
        m_maxNewsPerDay = max;
        emit maxNewsPerDayChanged();
        saveSettings();
    }
}

void SettingsManager::setNewsFeeds(const QStringList &feeds)
{
    if (m_newsFeeds != feeds) {
        m_newsFeeds = feeds;
        emit newsFeedsChanged();
        saveSettings();
    }
}

void SettingsManager::addNewsFeed(const QString &url)
{
    QString trimmed = url.trimmed();
    if (trimmed.isEmpty()) return;
    if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) return;
    if (m_newsFeeds.contains(trimmed)) return;

    m_newsFeeds.append(trimmed);
    emit newsFeedsChanged();
    saveSettings();
}

void SettingsManager::removeNewsFeed(int index)
{
    if (index < 0 || index >= m_newsFeeds.size()) return;
    m_newsFeeds.removeAt(index);
    // Ensure at least one feed remains
    if (m_newsFeeds.isEmpty()) {
        m_newsFeeds.append("https://www.allsides.com/rss/news");
    }
    emit newsFeedsChanged();
    saveSettings();
}

void SettingsManager::setSiteBlacklist(const QStringList &list)
{
    if (m_siteBlacklist != list) {
        m_siteBlacklist = list;
        emit siteBlacklistChanged();
        saveSettings();
    }
}

void SettingsManager::addBlacklistedSite(const QString &domain)
{
    QString d = domain.toLower().trimmed();
    if (d.isEmpty() || m_siteBlacklist.contains(d)) return;
    m_siteBlacklist.append(d);
    emit siteBlacklistChanged();
    saveSettings();
}

void SettingsManager::removeBlacklistedSite(const QString &domain)
{
    if (m_siteBlacklist.removeAll(domain.toLower().trimmed()) > 0) {
        emit siteBlacklistChanged();
        saveSettings();
    }
}

void SettingsManager::setGoalsEnabled(bool enabled)
{
    if (m_goalsEnabled != enabled) {
        m_goalsEnabled = enabled;
        emit goalsEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setSentimentTrackingEnabled(bool enabled)
{
    if (m_sentimentTrackingEnabled != enabled) {
        m_sentimentTrackingEnabled = enabled;
        emit sentimentTrackingEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setTeachMeEnabled(bool enabled)
{
    if (m_teachMeEnabled != enabled) {
        m_teachMeEnabled = enabled;
        emit teachMeEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setProactiveDialogsEnabled(bool enabled)
{
    if (m_proactiveDialogsEnabled != enabled) {
        m_proactiveDialogsEnabled = enabled;
        emit proactiveDialogsEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setMaxThoughtsPerDay(int max)
{
    max = qBound(1, max, 15);
    if (m_maxThoughtsPerDay != max) {
        m_maxThoughtsPerDay = max;
        emit maxThoughtsPerDayChanged();
        saveSettings();
    }
}

void SettingsManager::setModelContextLength(int tokens)
{
    tokens = qBound(2048, tokens, 131072);
    if (m_modelContextLength != tokens) {
        m_modelContextLength = tokens;
        emit modelContextLengthChanged();
        saveSettings();
    }
}

void SettingsManager::setEmbeddingModel(const QString &model)
{
    if (m_embeddingModel != model) {
        m_embeddingModel = model;
        emit embeddingModelChanged();
        saveSettings();
    }
}

void SettingsManager::setSemanticSearchEnabled(bool enabled)
{
    if (m_semanticSearchEnabled != enabled) {
        m_semanticSearchEnabled = enabled;
        emit semanticSearchEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setBackgroundModelEnabled(bool enabled)
{
    if (m_backgroundModelEnabled != enabled) {
        m_backgroundModelEnabled = enabled;
        emit backgroundModelEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setBackgroundModelPath(const QString &path)
{
    if (m_backgroundModelPath != path) {
        m_backgroundModelPath = path;
        emit backgroundModelPathChanged();
        saveSettings();
    }
}

void SettingsManager::setDistillationEnabled(bool enabled)
{
    if (m_distillationEnabled != enabled) {
        m_distillationEnabled = enabled;
        emit distillationEnabledChanged();
        saveSettings();
    }
}

void SettingsManager::setDistillationDailyCycles(int cycles)
{
    cycles = qBound(1, cycles, 10);
    if (m_distillationDailyCycles != cycles) {
        m_distillationDailyCycles = cycles;
        emit distillationDailyCyclesChanged();
        saveSettings();
    }
}

void SettingsManager::setEncryptionMode(const QString &mode)
{
    QString m = mode;
    if (m != "portable" && m != "machine_locked") m = "portable";
    if (m_encryptionMode != m) {
        m_encryptionMode = m;
        emit encryptionModeChanged();
        saveSettings();
    }
}

// --- Persistence ---

QString SettingsManager::getConfigFilePath() const
{
    return m_profilePath + "/settings.json";
}

void SettingsManager::saveSettings()
{
    qDebug() << "Saving settings to:" << getConfigFilePath();

    QJsonObject json;
    json["schemaVersion"] = SchemaMigrator::SETTINGS_VERSION;
    // LLM settings
    json["userName"] = m_userName;
    json["provider"] = m_provider;
    json["model"] = m_model;
    json["openAIKey"] = m_openAIKey;
    json["anthropicKey"] = m_anthropicKey;
    json["lmStudioUrl"] = m_lmStudioUrl;
    json["ollamaUrl"] = m_ollamaUrl;
    // DATAFORM settings
    json["idleLearningEnabled"] = m_idleLearningEnabled;
    json["computeBudgetPercent"] = m_computeBudgetPercent;
    json["privacyLevel"] = m_privacyLevel;
    // Evolution settings (Phase 3)
    json["populationSize"] = m_populationSize;
    json["evolutionEnabled"] = m_evolutionEnabled;
    // Lifecycle settings (Phase 4)
    json["episodeRetentionDays"] = m_episodeRetentionDays;
    json["autoBackupEnabled"] = m_autoBackupEnabled;
    json["maxDiskUsageGB"] = m_maxDiskUsageGB;
    // Research settings (Phase 5)
    json["researchEnabled"] = m_researchEnabled;
    json["maxResearchPerDay"] = m_maxResearchPerDay;
    // Local model (Phase 6)
    json["localModelPath"] = m_localModelPath;
    // News settings
    json["newsEnabled"] = m_newsEnabled;
    json["maxNewsPerDay"] = m_maxNewsPerDay;
    QJsonArray feedsArray;
    for (const QString &feed : m_newsFeeds) feedsArray.append(feed);
    json["newsFeeds"] = feedsArray;
    // Site blacklist
    QJsonArray blacklistArray;
    for (const QString &site : m_siteBlacklist) blacklistArray.append(site);
    json["siteBlacklist"] = blacklistArray;
    // Agentic features
    json["goalsEnabled"] = m_goalsEnabled;
    json["sentimentTrackingEnabled"] = m_sentimentTrackingEnabled;
    json["teachMeEnabled"] = m_teachMeEnabled;
    // Proactive Dialog
    json["proactiveDialogsEnabled"] = m_proactiveDialogsEnabled;
    json["maxThoughtsPerDay"] = m_maxThoughtsPerDay;
    // Model context
    json["modelContextLength"] = m_modelContextLength;
    // Phase 7: Semantic Memory
    json["embeddingModel"] = m_embeddingModel;
    json["semanticSearchEnabled"] = m_semanticSearchEnabled;
    // Background model (llama.cpp)
    json["backgroundModelEnabled"] = m_backgroundModelEnabled;
    json["backgroundModelPath"] = m_backgroundModelPath;
    // Phase 8: Distillation
    json["distillationEnabled"] = m_distillationEnabled;
    json["distillationDailyCycles"] = m_distillationDailyCycles;
    // Security
    json["encryptionMode"] = m_encryptionMode;

    QJsonDocument doc(json);
    QFile file(getConfigFilePath());

    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson(QJsonDocument::Indented));
        file.close();
        qDebug() << "Settings saved successfully";
        emit settingsSaved();
    } else {
        qDebug() << "Failed to save settings:" << file.errorString();
    }
}

void SettingsManager::loadSettings()
{
    QString configFile = getConfigFilePath();
    qDebug() << "Loading settings from:" << configFile;

    QFile file(configFile);
    if (file.open(QIODevice::ReadOnly)) {
        QByteArray data = file.readAll();
        file.close();

        QJsonDocument doc = QJsonDocument::fromJson(data);
        if (!doc.isNull() && doc.isObject()) {
            QJsonObject json = doc.object();

            m_userName = json.value("userName").toString("User");
            m_provider = json.value("provider").toString("Ollama");
            m_model = json.value("model").toString("");
            m_openAIKey = json.value("openAIKey").toString("");
            m_anthropicKey = json.value("anthropicKey").toString("");
            m_lmStudioUrl = json.value("lmStudioUrl").toString("http://localhost:1234");
            m_ollamaUrl = json.value("ollamaUrl").toString("http://localhost:11434");
            m_idleLearningEnabled = json.value("idleLearningEnabled").toBool(true);
            m_computeBudgetPercent = json.value("computeBudgetPercent").toInt(10);
            m_privacyLevel = json.value("privacyLevel").toString("standard");
            m_populationSize = json.value("populationSize").toInt(4);
            m_evolutionEnabled = json.value("evolutionEnabled").toBool(true);
            m_episodeRetentionDays = json.value("episodeRetentionDays").toInt(365);
            m_autoBackupEnabled = json.value("autoBackupEnabled").toBool(true);
            m_maxDiskUsageGB = json.value("maxDiskUsageGB").toInt(10);
            m_researchEnabled = json.value("researchEnabled").toBool(true);
            m_maxResearchPerDay = json.value("maxResearchPerDay").toInt(10);
            m_localModelPath = json.value("localModelPath").toString("");
            m_newsEnabled = json.value("newsEnabled").toBool(true);
            m_maxNewsPerDay = json.value("maxNewsPerDay").toInt(3);
            if (json.contains("newsFeeds")) {
                m_newsFeeds.clear();
                QJsonArray arr = json["newsFeeds"].toArray();
                for (const QJsonValue &v : arr) {
                    QString feed = v.toString().trimmed();
                    if (!feed.isEmpty()) m_newsFeeds.append(feed);
                }
                if (m_newsFeeds.isEmpty()) {
                    m_newsFeeds.append("https://www.allsides.com/rss/news");
                }
            }
            if (json.contains("siteBlacklist")) {
                m_siteBlacklist.clear();
                QJsonArray arr = json["siteBlacklist"].toArray();
                for (const QJsonValue &v : arr) {
                    QString site = v.toString().toLower().trimmed();
                    if (!site.isEmpty()) m_siteBlacklist.append(site);
                }
            }
            m_goalsEnabled = json.value("goalsEnabled").toBool(true);
            m_sentimentTrackingEnabled = json.value("sentimentTrackingEnabled").toBool(true);
            m_teachMeEnabled = json.value("teachMeEnabled").toBool(true);
            m_proactiveDialogsEnabled = json.value("proactiveDialogsEnabled").toBool(true);
            m_maxThoughtsPerDay = json.value("maxThoughtsPerDay").toInt(5);
            m_modelContextLength = json.value("modelContextLength").toInt(8192);
            m_embeddingModel = json.value("embeddingModel").toString("nomic-embed-text");
            m_semanticSearchEnabled = json.value("semanticSearchEnabled").toBool(true);
            m_backgroundModelEnabled = json.value("backgroundModelEnabled").toBool(false);
            m_backgroundModelPath = json.value("backgroundModelPath").toString("models/background_llm");
            m_distillationEnabled = json.value("distillationEnabled").toBool(false);
            m_distillationDailyCycles = json.value("distillationDailyCycles").toInt(3);
            m_encryptionMode = json.value("encryptionMode").toString("portable");
            if (m_encryptionMode != "portable" && m_encryptionMode != "machine_locked")
                m_encryptionMode = "portable";

            qDebug() << "Settings loaded: Provider =" << m_provider << ", Model =" << m_model;
        } else {
            qDebug() << "No valid settings file found, using defaults";
        }
    } else {
        qDebug() << "Settings file not found, using defaults";
    }

    // Emit all signals to sync UI
    emit userNameChanged();
    emit providerChanged();
    emit modelChanged();
    emit openAIKeyChanged();
    emit anthropicKeyChanged();
    emit lmStudioUrlChanged();
    emit ollamaUrlChanged();
    emit idleLearningEnabledChanged();
    emit computeBudgetPercentChanged();
    emit privacyLevelChanged();
    emit populationSizeChanged();
    emit evolutionEnabledChanged();
    emit episodeRetentionDaysChanged();
    emit autoBackupEnabledChanged();
    emit maxDiskUsageGBChanged();
    emit researchEnabledChanged();
    emit maxResearchPerDayChanged();
    emit localModelPathChanged();
    emit newsEnabledChanged();
    emit maxNewsPerDayChanged();
    emit newsFeedsChanged();
    emit siteBlacklistChanged();
    emit goalsEnabledChanged();
    emit sentimentTrackingEnabledChanged();
    emit teachMeEnabledChanged();
    emit proactiveDialogsEnabledChanged();
    emit maxThoughtsPerDayChanged();
    emit modelContextLengthChanged();
    emit embeddingModelChanged();
    emit semanticSearchEnabledChanged();
    emit backgroundModelEnabledChanged();
    emit backgroundModelPathChanged();
    emit distillationEnabledChanged();
    emit distillationDailyCyclesChanged();
    emit encryptionModeChanged();
    emit settingsLoaded();
}

void SettingsManager::resetToDefaults()
{
    qDebug() << "Resetting settings to defaults...";

    m_userName = "User";
    m_provider = "Ollama";
    m_model = "";
    m_openAIKey = "";
    m_anthropicKey = "";
    m_lmStudioUrl = "http://localhost:1234";
    m_ollamaUrl = "http://localhost:11434";
    m_idleLearningEnabled = true;
    m_computeBudgetPercent = 10;
    m_privacyLevel = "standard";
    m_populationSize = 4;
    m_evolutionEnabled = true;
    m_episodeRetentionDays = 365;
    m_autoBackupEnabled = true;
    m_maxDiskUsageGB = 10;
    m_researchEnabled = true;
    m_maxResearchPerDay = 10;
    m_localModelPath = "";
    m_newsEnabled = true;
    m_maxNewsPerDay = 3;
    m_newsFeeds = {"https://www.allsides.com/rss/news"};
    m_siteBlacklist.clear();
    m_goalsEnabled = true;
    m_sentimentTrackingEnabled = true;
    m_teachMeEnabled = true;
    m_proactiveDialogsEnabled = true;
    m_maxThoughtsPerDay = 5;
    m_modelContextLength = 8192;
    m_embeddingModel = "nomic-embed-text";
    m_semanticSearchEnabled = true;
    m_backgroundModelEnabled = false;
    m_backgroundModelPath = "models/background_llm";
    m_distillationEnabled = false;
    m_distillationDailyCycles = 3;
    m_encryptionMode = "portable";

    emit userNameChanged();
    emit providerChanged();
    emit modelChanged();
    emit openAIKeyChanged();
    emit anthropicKeyChanged();
    emit lmStudioUrlChanged();
    emit ollamaUrlChanged();
    emit idleLearningEnabledChanged();
    emit computeBudgetPercentChanged();
    emit privacyLevelChanged();
    emit populationSizeChanged();
    emit evolutionEnabledChanged();
    emit episodeRetentionDaysChanged();
    emit autoBackupEnabledChanged();
    emit maxDiskUsageGBChanged();
    emit researchEnabledChanged();
    emit maxResearchPerDayChanged();
    emit localModelPathChanged();
    emit newsEnabledChanged();
    emit maxNewsPerDayChanged();
    emit newsFeedsChanged();
    emit siteBlacklistChanged();
    emit goalsEnabledChanged();
    emit sentimentTrackingEnabledChanged();
    emit teachMeEnabledChanged();
    emit proactiveDialogsEnabledChanged();
    emit maxThoughtsPerDayChanged();
    emit modelContextLengthChanged();
    emit embeddingModelChanged();
    emit semanticSearchEnabledChanged();
    emit backgroundModelEnabledChanged();
    emit backgroundModelPathChanged();
    emit distillationEnabledChanged();
    emit distillationDailyCyclesChanged();
    emit encryptionModeChanged();

    saveSettings();
}
