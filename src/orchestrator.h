#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <QObject>
#include <QString>
#include <QJsonArray>
#include <QJsonObject>
#include <QList>
#include <QPair>
#include <QVector>
#include <QTimer>

struct SemanticSearchResult;
struct ToolCallRequest;
struct ToolResult;

class LLMProviderManager;
class MemoryStore;
class SettingsManager;
class WhyEngine;
class TraitExtractor;
class ResearchEngine;
class ThoughtEngine;
class ReminderEngine;
class GoalTracker;
class SentimentTracker;
class LearningEngine;
class ProfileManager;
class EmbeddingManager;
class ToolRegistry;

class Orchestrator : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isProcessing READ isProcessing NOTIFY isProcessingChanged)
    Q_PROPERTY(int conversationLength READ conversationLength NOTIFY conversationLengthChanged)
    Q_PROPERTY(QString currentTopic READ currentTopic WRITE setTopic NOTIFY currentTopicChanged)
    Q_PROPERTY(double curiosityLevel READ curiosityLevel NOTIFY curiosityLevelChanged)
    Q_PROPERTY(bool lastResponseHadInquiry READ lastResponseHadInquiry NOTIFY lastResponseHadInquiryChanged)
    Q_PROPERTY(qint64 currentConversationId READ currentConversationId NOTIFY currentConversationIdChanged)

public:
    explicit Orchestrator(QObject *parent = nullptr);
    ~Orchestrator();

    void setLLMProvider(LLMProviderManager *provider);
    void setMemoryStore(MemoryStore *store);
    void setSettingsManager(SettingsManager *settings);
    void setWhyEngine(WhyEngine *engine);
    void setTraitExtractor(TraitExtractor *extractor);
    void setResearchEngine(ResearchEngine *engine);
    void setThoughtEngine(ThoughtEngine *engine);
    void setReminderEngine(ReminderEngine *engine);
    void setGoalTracker(GoalTracker *tracker);
    void setSentimentTracker(SentimentTracker *tracker);
    void setLearningEngine(LearningEngine *engine);
    void setProfileManager(ProfileManager *manager);
    void setEmbeddingManager(EmbeddingManager *manager);
    void setToolRegistry(ToolRegistry *registry);

    bool isProcessing() const { return m_isProcessing; }
    int conversationLength() const { return m_conversationHistory.size(); }
    QString currentTopic() const { return m_currentTopic; }
    double curiosityLevel() const;
    bool lastResponseHadInquiry() const { return m_lastResponseHadInquiry; }
    qint64 currentConversationId() const { return m_currentConversationId; }

    // Primary entry point: user sends a message
    Q_INVOKABLE void handleUserMessage(const QString &message);
    Q_INVOKABLE void handleUserMessageWithImages(const QString &message, const QStringList &imagesBase64);

    // Feedback on the most recent response
    Q_INVOKABLE void submitFeedback(int feedback);
    Q_INVOKABLE void submitEdit(const QString &editedText);

    // Conversation management
    Q_INVOKABLE void clearConversation();
    Q_INVOKABLE void setTopic(const QString &topic);
    Q_INVOKABLE void startNewConversation();
    Q_INVOKABLE void loadConversation(qint64 id);
    Q_INVOKABLE void startProactiveConversation(qint64 thoughtId);

    // Profile portability
    Q_INVOKABLE void exportProfile(const QString &targetDir);
    Q_INVOKABLE void importProfile(const QString &sourceDir);

signals:
    void isProcessingChanged();
    void conversationLengthChanged();
    void currentTopicChanged();
    void curiosityLevelChanged();
    void lastResponseHadInquiryChanged();

    // UI signals
    void assistantResponseReady(const QString &response);
    void errorOccurred(const QString &error);
    void processingStarted();
    void processingFinished();

    // Learning signals
    void learningSignalQueued(qint64 episodeId, const QString &signalType);

    // Conversation signals
    void currentConversationIdChanged();
    void conversationLoaded(qint64 id, QVariantList messages);

    // Episode storage signal (for SentimentTracker)
    void episodeStored(qint64 episodeId);

private slots:
    void onLLMResponse(const QString &response);
    void onLLMError(const QString &error);
    void onToolExecutionComplete(const ToolResult &result);
    void onAgentLoopTimeout();

private:
    struct ConversationTurn {
        QString role;
        QString content;
        QStringList imageData;  // Base64-encoded images (empty for text-only)
    };

    QString buildSystemPrompt(const QString &curiosityDirective, int &tokenBudgetUsed) const;
    QJsonArray buildMessagesArray(int remainingTokenBudget) const;
    QString retrieveIdentityContext() const;
    static int estimateTokens(const QString &text);
    void storeEpisode(const QString &userText, const QString &assistantText);
    bool detectInquiryInResponse(const QString &response) const;
    void triggerTraitExtraction(qint64 episodeId);
    QString computeEditDiff(const QString &original, const QString &edited) const;
    void updateTopicKeywords(const QString &message);
    QString extractSearchKeywords(const QString &message) const;
    void sendToLLM();
    void finalizeAgentResponse(const QString &finalText);
    void resetAgentLoop();

    LLMProviderManager *m_llmProvider = nullptr;
    MemoryStore *m_memoryStore = nullptr;
    SettingsManager *m_settingsManager = nullptr;
    WhyEngine *m_whyEngine = nullptr;
    TraitExtractor *m_traitExtractor = nullptr;
    ResearchEngine *m_researchEngine = nullptr;
    ThoughtEngine *m_thoughtEngine = nullptr;
    ReminderEngine *m_reminderEngine = nullptr;
    GoalTracker *m_goalTracker = nullptr;
    SentimentTracker *m_sentimentTracker = nullptr;
    LearningEngine *m_learningEngine = nullptr;
    ProfileManager *m_profileManager = nullptr;
    EmbeddingManager *m_embeddingManager = nullptr;
    ToolRegistry *m_toolRegistry = nullptr;

    bool m_isProcessing = false;
    QString m_currentTopic;
    qint64 m_currentEpisodeId = -1;
    qint64 m_currentConversationId = -1;
    QString m_lastAssistantResponse;
    bool m_lastResponseHadInquiry = false;
    QString m_pendingCuriosityDirective;

    // Deferred inquiry extraction (avoid concurrent LLM requests)
    bool m_pendingInquiryExtraction = false;
    QString m_pendingInquiry;
    QString m_pendingInquiryUserResponse;

    QList<ConversationTurn> m_conversationHistory;
    QStringList m_pendingImages;  // Temp storage for current message images
    QVector<float> m_lastUserEmbedding;  // Cached embedding for semantic recall
    QList<SemanticSearchResult> m_pendingSemanticResults;
    bool m_awaitingSemanticSearch = false;
    static constexpr int MAX_HISTORY_TURNS = 20;

    // Agentic tool-calling loop state
    struct AgentLoopState {
        int iterationCount = 0;
        bool inAgentLoop = false;
        QList<ToolCallRequest> pendingToolCalls;
        int currentToolIndex = 0;
        QJsonArray toolResultMessages;
        QString accumulatedTextContent;
        QList<QPair<QString, QString>> toolInvocations;  // (toolName, resultSummary)
    };
    static constexpr int MAX_AGENT_ITERATIONS = 5;
    AgentLoopState m_agentLoop;
    QTimer m_agentLoopTimeout;
};

#endif // ORCHESTRATOR_H
