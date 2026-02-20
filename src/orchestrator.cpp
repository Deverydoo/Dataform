#include "orchestrator.h"
#include "llmresponseparser.h"
#include "toolregistry.h"
#include "llmprovider.h"
#include "memorystore.h"
#include "settingsmanager.h"
#include "whyengine.h"
#include "traitextractor.h"
#include "researchengine.h"
#include "thoughtengine.h"
#include "reminderengine.h"
#include "goaltracker.h"
#include "sentimenttracker.h"
#include "learningengine.h"
#include "profilemanager.h"
#include "embeddingmanager.h"
#include <QJsonObject>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QDebug>

Orchestrator::Orchestrator(QObject *parent)
    : QObject(parent)
{
    m_agentLoopTimeout.setSingleShot(true);
    m_agentLoopTimeout.setInterval(60000);  // 60s timeout for agent loop
    connect(&m_agentLoopTimeout, &QTimer::timeout, this, &Orchestrator::onAgentLoopTimeout);
}

Orchestrator::~Orchestrator()
{
}

void Orchestrator::setLLMProvider(LLMProviderManager *provider)
{
    if (m_llmProvider) {
        disconnect(m_llmProvider, nullptr, this, nullptr);
    }

    m_llmProvider = provider;

    if (m_llmProvider) {
        connect(m_llmProvider, &LLMProviderManager::responseReceived,
                this, &Orchestrator::onLLMResponse);
        connect(m_llmProvider, &LLMProviderManager::errorOccurred,
                this, &Orchestrator::onLLMError);
    }
}

void Orchestrator::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

void Orchestrator::setSettingsManager(SettingsManager *settings)
{
    m_settingsManager = settings;
}

void Orchestrator::setWhyEngine(WhyEngine *engine)
{
    m_whyEngine = engine;
}

void Orchestrator::setTraitExtractor(TraitExtractor *extractor)
{
    m_traitExtractor = extractor;
}

void Orchestrator::setResearchEngine(ResearchEngine *engine)
{
    m_researchEngine = engine;
}

void Orchestrator::setThoughtEngine(ThoughtEngine *engine)
{
    m_thoughtEngine = engine;
}

void Orchestrator::setReminderEngine(ReminderEngine *engine)
{
    m_reminderEngine = engine;
}

void Orchestrator::setGoalTracker(GoalTracker *tracker)
{
    m_goalTracker = tracker;
}

void Orchestrator::setSentimentTracker(SentimentTracker *tracker)
{
    m_sentimentTracker = tracker;
}

void Orchestrator::setLearningEngine(LearningEngine *engine)
{
    m_learningEngine = engine;
}

void Orchestrator::setProfileManager(ProfileManager *manager)
{
    m_profileManager = manager;
}

void Orchestrator::setEmbeddingManager(EmbeddingManager *manager)
{
    m_embeddingManager = manager;
    if (m_embeddingManager) {
        connect(m_embeddingManager, &EmbeddingManager::embeddingReady,
                this, [this](const QString &tag, QVector<float> emb) {
            if (tag == "user_message") {
                m_lastUserEmbedding = emb;

                // If we're waiting for this embedding to run async search, fire it now
                if (m_awaitingSemanticSearch) {
                    m_embeddingManager->semanticSearchAsync(emb, {"episode"}, 5);
                }
            }
        });

        connect(m_embeddingManager, &EmbeddingManager::semanticSearchComplete,
                this, [this](QList<SemanticSearchResult> results) {
            if (m_awaitingSemanticSearch) {
                m_awaitingSemanticSearch = false;
                m_pendingSemanticResults = results;
                sendToLLM();
            }
        });

        // If embedding fails, don't block — send without semantic results
        connect(m_embeddingManager, &EmbeddingManager::embeddingError,
                this, [this](const QString &tag, const QString &) {
            if (tag == "user_message" && m_awaitingSemanticSearch) {
                m_awaitingSemanticSearch = false;
                m_pendingSemanticResults.clear();
                sendToLLM();
            }
        });
    }
}

void Orchestrator::setToolRegistry(ToolRegistry *registry)
{
    m_toolRegistry = registry;
    if (m_toolRegistry) {
        connect(m_toolRegistry, &ToolRegistry::toolExecutionComplete,
                this, &Orchestrator::onToolExecutionComplete);
    }
}

void Orchestrator::exportProfile(const QString &targetDir)
{
    if (!m_profileManager) return;
    qDebug() << "Orchestrator: exporting profile to" << targetDir;

    // Flush databases to sync .enc files
    if (m_memoryStore) {
        m_memoryStore->flush();
    }

    m_profileManager->exportProfile(targetDir);
}

void Orchestrator::importProfile(const QString &sourceDir)
{
    if (!m_profileManager || !m_memoryStore) return;
    qDebug() << "Orchestrator: importing profile from" << sourceDir;

    // Close databases before overwriting files
    m_memoryStore->close();

    // Copy files
    m_profileManager->importProfile(sourceDir);

    // Re-initialize with imported data
    m_memoryStore->initialize();

    // Reset conversation state
    m_conversationHistory.clear();
    m_currentConversationId = -1;
    emit currentConversationIdChanged();
    emit conversationLengthChanged();
}

double Orchestrator::curiosityLevel() const
{
    return m_whyEngine ? m_whyEngine->curiosityLevel() : 0.5;
}

void Orchestrator::handleUserMessage(const QString &message)
{
    if (m_isProcessing) {
        qDebug() << "Already processing, ignoring message";
        return;
    }

    if (!m_llmProvider) {
        emit errorOccurred("No LLM provider configured");
        return;
    }

    m_isProcessing = true;
    resetAgentLoop();
    m_pendingSemanticResults.clear();
    emit isProcessingChanged();
    emit processingStarted();

    // --- Auto-create conversation on first message ---
    if (m_currentConversationId < 0 && m_memoryStore) {
        QString title = message.left(40);
        if (message.length() > 40) title += "...";
        m_currentConversationId = m_memoryStore->createConversation(title);
        emit currentConversationIdChanged();
    }

    // --- Phase 1: Check if user is answering a previous inquiry ---
    m_pendingInquiryExtraction = false;
    if (m_whyEngine && m_whyEngine->isAnsweringInquiry(message)) {
        qDebug() << "Orchestrator: user appears to be answering an inquiry";

        // Defer trait extraction until after the chat response is received,
        // to avoid concurrent LLM requests (which causes the trait JSON
        // to leak into the chat).
        if (m_traitExtractor) {
            m_pendingInquiryExtraction = true;
            m_pendingInquiry = m_whyEngine->lastInquiry();
            m_pendingInquiryUserResponse = message;
        }

        // Clear the awaiting state
        m_whyEngine->onResponseDelivered(QString(), false);
    }

    // --- Add to conversation history ---
    ConversationTurn turn;
    turn.role = "user";
    turn.content = message;
    turn.imageData = m_pendingImages;
    m_pendingImages.clear();
    m_conversationHistory.append(turn);
    while (m_conversationHistory.size() > MAX_HISTORY_TURNS) {
        m_conversationHistory.removeFirst();
    }
    emit conversationLengthChanged();

    // --- Update topic keywords for shift detection ---
    updateTopicKeywords(message);

    // --- Phase 7: Embed user message for semantic recall (async) ---
    // When semantic search is enabled, defer the LLM call until search completes.
    // The embedding request is async: embeddingReady -> semanticSearchAsync -> searchComplete -> sendToLLM.
    bool deferForSemanticSearch = false;
    if (m_embeddingManager && m_settingsManager
        && m_settingsManager->semanticSearchEnabled()
        && m_embeddingManager->isModelAvailable()) {
        m_awaitingSemanticSearch = true;
        m_pendingSemanticResults.clear();
        deferForSemanticSearch = true;
        m_embeddingManager->embedText(message, "user_message");
    }

    // --- Phase 5: Detect explicit research requests ---
    if (m_researchEngine) {
        static const QStringList researchTriggers = {
            "research", "look into", "find out about", "learn about",
            "search for", "look up", "investigate"
        };
        QString lower = message.toLower();
        for (const QString &trigger : researchTriggers) {
            if (lower.contains(trigger)) {
                // Extract the topic after the trigger phrase
                int idx = lower.indexOf(trigger);
                QString topic = message.mid(idx + trigger.length()).trimmed();
                // Remove trailing punctuation
                topic.remove(QRegularExpression("[.!?]+$"));
                if (topic.length() > 3) {
                    m_researchEngine->queueTopic(topic, 1.0);
                    qDebug() << "Orchestrator: queued user research request:" << topic;
                }
                break;
            }
        }
    }

    // --- Detect reminder requests ---
    if (m_reminderEngine) {
        static const QStringList reminderTriggers = {
            "remind me", "reminder", "don't let me forget", "don't forget to"
        };
        QString lower = message.toLower();
        for (const QString &trigger : reminderTriggers) {
            if (lower.contains(trigger)) {
                int idx = lower.indexOf(trigger);
                QString remainder = message.mid(idx + trigger.length()).trimmed();
                if (remainder.length() > 3) {
                    m_reminderEngine->detectAndStore(remainder, m_currentEpisodeId);
                    qDebug() << "Orchestrator: detected reminder request:" << remainder;
                }
                break;
            }
        }
    }

    // --- Detect teach-me requests ---
    if (m_learningEngine && m_settingsManager && m_settingsManager->teachMeEnabled()) {
        static const QStringList teachTriggers = {
            "teach me", "help me learn", "i want to learn", "explain to me"
        };
        QString lower = message.toLower();
        for (const QString &trigger : teachTriggers) {
            if (lower.contains(trigger)) {
                int idx = lower.indexOf(trigger);
                QString topic = message.mid(idx + trigger.length()).trimmed();
                topic.remove(QRegularExpression("[.!?]+$"));
                if (topic.length() > 3) {
                    m_learningEngine->startLearningPlan(topic);
                    qDebug() << "Orchestrator: detected teach-me request:" << topic;
                }
                break;
            }
        }
    }

    // --- Phase 1: WhyEngine curiosity decision ---
    QString curiosityDirective;
    if (m_whyEngine) {
        curiosityDirective = m_whyEngine->buildCuriosityDirective(message, m_currentTopic);
        if (!curiosityDirective.isEmpty()) {
            qDebug() << "Orchestrator: curiosity directive active for this turn";
        }
    }
    m_pendingCuriosityDirective = curiosityDirective;

    // --- Send to LLM (or defer if waiting for semantic search) ---
    if (deferForSemanticSearch) {
        qDebug() << "Orchestrator: deferring LLM call for async semantic search";
    } else {
        sendToLLM();
    }
}

void Orchestrator::sendToLLM()
{
    // Notify coordinator that foreground chat is active (only on first call, not agent loop iterations)
    if (!m_agentLoop.inAgentLoop) {
        emit chatStarted();
    }

    // --- Build system prompt with identity context + curiosity ---
    int systemTokens = 0;
    QString systemPrompt = buildSystemPrompt(m_pendingCuriosityDirective, systemTokens);

    // --- Inject tool schema if available ---
    if (m_toolRegistry && m_toolRegistry->toolCount() > 0) {
        QString toolSchema = m_toolRegistry->buildToolSchemaPrompt();
        int toolTokens = estimateTokens(toolSchema);
        int contextLimit = m_settingsManager ? m_settingsManager->modelContextLength() : 8192;
        int maxSystemTokens = contextLimit * 2 / 5;
        if (systemTokens + toolTokens <= maxSystemTokens) {
            systemPrompt += toolSchema;
            systemTokens += toolTokens;
        } else {
            qDebug() << "Orchestrator: shedding tool schema (" << toolTokens
                     << "tokens) — system budget:" << systemTokens << "/" << maxSystemTokens;
        }
    }

    // --- Build messages array with remaining token budget ---
    int contextLimit = m_settingsManager ? m_settingsManager->modelContextLength() : 8192;
    int responseReserve = 1024;  // reserve tokens for model to generate response
    int messageBudget = contextLimit - systemTokens - responseReserve;
    QJsonArray messages = buildMessagesArray(messageBudget);

    // --- Append tool result messages from agent loop ---
    for (const QJsonValue &v : m_agentLoop.toolResultMessages) {
        messages.append(v);
    }

    // --- Send to LLM ---
    qDebug() << "Orchestrator: sending conversation with"
             << messages.size() << "messages to LLM"
             << "(system:" << systemTokens << "tokens, budget:" << messageBudget << ")"
             << (m_pendingCuriosityDirective.isEmpty() ? "" : "[with curiosity directive]")
             << (m_agentLoop.inAgentLoop ? QStringLiteral("[agent iteration %1]").arg(m_agentLoop.iterationCount) : "");
    m_llmProvider->sendConversation(systemPrompt, messages);
}

void Orchestrator::handleUserMessageWithImages(const QString &message, const QStringList &imagesBase64)
{
    m_pendingImages = imagesBase64;
    qDebug() << "Orchestrator: received message with" << imagesBase64.size() << "image(s)";
    handleUserMessage(message);
}

void Orchestrator::onLLMResponse(const QString &response)
{
    if (!m_isProcessing) return;

    qDebug() << "\n---------- ORCHESTRATOR: onLLMResponse ----------";
    qDebug() << "Response length:" << response.length()
             << "isEmpty:" << response.isEmpty()
             << "trimmedEmpty:" << response.trimmed().isEmpty();
    if (!response.isEmpty()) {
        bool hasThinkOpen = response.contains("<think>");
        bool hasThinkClose = response.contains("</think>");
        qDebug() << "Has <think>:" << hasThinkOpen << "Has </think>:" << hasThinkClose;
        if (hasThinkOpen) {
            int thinkStart = response.indexOf("<think>");
            int thinkEnd = response.indexOf("</think>");
            int thinkLen = (thinkEnd > thinkStart) ? (thinkEnd - thinkStart) : (response.length() - thinkStart);
            qDebug() << "Think block length:" << thinkLen;
            // Show what's AFTER the think block (the actual visible content)
            if (thinkEnd > 0) {
                QString afterThink = response.mid(thinkEnd + 8).trimmed();
                qDebug() << "Content after </think> (" << afterThink.length() << "chars):"
                         << (afterThink.isEmpty() ? "(EMPTY)" : afterThink.left(300));
            } else {
                qDebug() << "Think block is UNCLOSED — model likely ran out of tokens";
            }
        }
    }
    qDebug() << "--------------------------------------------------";

    // --- Check for tool calls in the response ---
    ParsedLLMResponse parsed = LLMResponseParser::parseToolCalls(response);

    if (parsed.hasToolCalls()
        && m_toolRegistry
        && m_agentLoop.iterationCount < MAX_AGENT_ITERATIONS) {
        // --- Enter/continue agent loop ---
        qDebug() << "Orchestrator: detected" << parsed.toolCalls.size()
                 << "tool call(s) at iteration" << m_agentLoop.iterationCount;

        m_agentLoop.inAgentLoop = true;
        m_agentLoop.iterationCount++;
        m_agentLoopTimeout.start();  // Reset timeout

        // Accumulate any text the LLM produced alongside the tool call
        if (!parsed.textContent.isEmpty()) {
            if (!m_agentLoop.accumulatedTextContent.isEmpty())
                m_agentLoop.accumulatedTextContent += "\n\n";
            m_agentLoop.accumulatedTextContent += parsed.textContent;
        }

        // Add the full assistant response (with tool_call blocks) to conversation history
        // so the LLM sees its own tool calls on the next iteration
        m_conversationHistory.append({"assistant", response});
        emit conversationLengthChanged();

        // Queue all tool calls and execute the first one
        m_agentLoop.pendingToolCalls = parsed.toolCalls;
        m_agentLoop.currentToolIndex = 0;

        const ToolCallRequest &firstCall = m_agentLoop.pendingToolCalls.first();
        m_toolRegistry->executeTool(firstCall.toolName, firstCall.params);
        return;  // Wait for toolExecutionComplete signal
    }

    // --- No tool calls (or max iterations reached): final answer ---
    QString finalText = response;
    if (m_agentLoop.inAgentLoop) {
        // Use the text from the final response, falling back to accumulated text
        if (parsed.textContent.isEmpty() && !m_agentLoop.accumulatedTextContent.isEmpty()) {
            finalText = m_agentLoop.accumulatedTextContent;
        } else if (!parsed.textContent.isEmpty()) {
            finalText = parsed.textContent;
        }
        qDebug() << "Orchestrator: agent loop complete after"
                 << m_agentLoop.iterationCount << "iterations";
    }

    finalizeAgentResponse(finalText);
}

void Orchestrator::finalizeAgentResponse(const QString &finalText)
{
    // If we were in an agent loop, the intermediate assistant turns are already in history.
    // Replace them with just the final answer for clean history.
    if (m_agentLoop.inAgentLoop) {
        // Remove intermediate assistant + tool-result turns added during the loop
        // Walk back and remove turns that are part of the agent loop
        int loopTurns = 0;
        for (int i = m_conversationHistory.size() - 1; i >= 0; --i) {
            // Count back through assistant (tool_call) and user (tool result) turns
            const ConversationTurn &turn = m_conversationHistory[i];
            if (turn.content.contains("<tool_call>") || turn.content.startsWith("[Tool Result:")) {
                loopTurns++;
            } else {
                break;
            }
        }
        if (loopTurns > 0) {
            m_conversationHistory.erase(
                m_conversationHistory.end() - loopTurns,
                m_conversationHistory.end());
        }
    }

    // Add final assistant response to conversation history
    m_conversationHistory.append({"assistant", finalText});
    m_lastAssistantResponse = finalText;
    emit conversationLengthChanged();

    // --- Detect if the response contains an inquiry ---
    bool hadInquiry = detectInquiryInResponse(finalText);
    if (m_lastResponseHadInquiry != hadInquiry) {
        m_lastResponseHadInquiry = hadInquiry;
        emit lastResponseHadInquiryChanged();
    }

    // --- Notify WhyEngine about the response ---
    if (m_whyEngine) {
        if (hadInquiry && !m_pendingCuriosityDirective.isEmpty()) {
            int pos = finalText.lastIndexOf('?');
            if (pos >= 0) {
                int sentStart = finalText.lastIndexOf(QRegularExpression("[.!?]"), pos - 1);
                QString inquiry = finalText.mid(sentStart + 1, pos - sentStart).trimmed();
                if (!inquiry.isEmpty()) {
                    m_whyEngine->recordInquiry(inquiry);
                }
            }
        }
        m_whyEngine->onResponseDelivered(finalText, hadInquiry);
        emit curiosityLevelChanged();
    }

    // --- Store episodic record ---
    QString lastUserMessage;
    for (int i = m_conversationHistory.size() - 2; i >= 0; --i) {
        if (m_conversationHistory[i].role == "user") {
            lastUserMessage = m_conversationHistory[i].content;
            break;
        }
    }
    storeEpisode(lastUserMessage, finalText);

    // --- Store tool usage in episode tags ---
    if (!m_agentLoop.toolInvocations.isEmpty() && m_currentEpisodeId >= 0 && m_memoryStore) {
        QJsonArray toolsUsed;
        for (const auto &inv : m_agentLoop.toolInvocations) {
            QJsonObject t;
            t["tool"] = inv.first;
            t["summary"] = inv.second;
            toolsUsed.append(t);
        }
        m_memoryStore->updateEpisodeTags(m_currentEpisodeId,
            "tools:" + QJsonDocument(toolsUsed).toJson(QJsonDocument::Compact));
    }

    // --- Update episode with inquiry text if present ---
    if (hadInquiry && m_whyEngine && m_currentEpisodeId >= 0 && m_memoryStore) {
        m_memoryStore->updateEpisodeInquiry(m_currentEpisodeId, m_whyEngine->lastInquiry());
    }

    // --- Strip think tags before emitting to UI ---
    QString displayText = LLMResponseParser::stripThinkTags(finalText);

    qDebug() << "\n---------- ORCHESTRATOR: finalizeAgentResponse ----------";
    qDebug() << "finalText length:" << finalText.length()
             << "displayText length:" << displayText.length()
             << "displayText trimmed empty:" << displayText.trimmed().isEmpty();
    if (displayText.length() != finalText.length()) {
        qDebug() << "Think tags stripped:" << (finalText.length() - displayText.length()) << "chars removed";
    }
    qDebug() << "--- DISPLAY TEXT (first 500 chars) ---";
    qDebug().noquote() << (displayText.isEmpty() ? "(EMPTY)" : displayText.left(500));
    qDebug() << "------------------------------------------------------";

    if (displayText.trimmed().isEmpty()) {
        if (!finalText.trimmed().isEmpty()) {
            // Model only produced think tags — show a fallback
            qWarning() << "Orchestrator: response was only think tags, length:" << finalText.length();
            displayText = "(DATAFORM is thinking but produced no visible response. Try again.)";
        } else {
            // Response was genuinely empty
            qWarning() << "Orchestrator: received completely empty response from LLM";
            displayText = "(No response received from the model. Check your Ollama connection and model.)";
        }
    }

    // --- Emit response to UI ---
    emit assistantResponseReady(displayText);

    // --- Reset agent loop state ---
    resetAgentLoop();

    m_isProcessing = false;
    emit isProcessingChanged();
    emit processingFinished();
    emit chatFinished();

    // --- Trigger trait extraction for high-signal events only ---
    if (m_pendingInquiryExtraction && m_traitExtractor
        && !m_traitExtractor->isExtracting() && m_currentEpisodeId >= 0) {
        qDebug() << "Orchestrator: high-signal trait extraction (inquiry response) for episode"
                 << m_currentEpisodeId;
        m_traitExtractor->extractFromInquiryResponse(
            m_currentEpisodeId, m_pendingInquiry, m_pendingInquiryUserResponse);
        m_pendingInquiryExtraction = false;
    }
}

void Orchestrator::onToolExecutionComplete(const ToolResult &result)
{
    if (!m_agentLoop.inAgentLoop) return;

    qDebug() << "Orchestrator: tool" << result.toolName
             << (result.success ? "succeeded" : "failed:" + result.errorMessage);

    // Record the invocation for episode storage
    m_agentLoop.toolInvocations.append({
        result.toolName,
        result.success ? result.resultText.left(200) : result.errorMessage.left(200)
    });

    // Append tool result as a user message so the LLM sees it
    QString resultContent = QStringLiteral("[Tool Result: %1]\n%2")
        .arg(result.toolName,
             result.success ? result.resultText : QStringLiteral("Error: %1").arg(result.errorMessage));

    // Add to conversation history (will be cleaned up in finalizeAgentResponse)
    m_conversationHistory.append({"user", resultContent});
    emit conversationLengthChanged();

    // Also add to the tool result messages array used by sendToLLM
    QJsonObject toolResultMsg;
    toolResultMsg["role"] = "user";
    toolResultMsg["content"] = resultContent;
    m_agentLoop.toolResultMessages.append(toolResultMsg);

    // Check if more tool calls are pending
    m_agentLoop.currentToolIndex++;
    if (m_agentLoop.currentToolIndex < m_agentLoop.pendingToolCalls.size()) {
        const ToolCallRequest &nextCall = m_agentLoop.pendingToolCalls[m_agentLoop.currentToolIndex];
        m_toolRegistry->executeTool(nextCall.toolName, nextCall.params);
    } else {
        // All tools executed — send back to LLM for next iteration
        sendToLLM();
    }
}

void Orchestrator::onAgentLoopTimeout()
{
    if (!m_agentLoop.inAgentLoop) return;

    qWarning() << "Orchestrator: agent loop timed out after 60s";

    // Use whatever text we've accumulated so far
    QString fallbackText = m_agentLoop.accumulatedTextContent;
    if (fallbackText.isEmpty()) {
        fallbackText = "I was trying to help but ran into some difficulties. Could you rephrase your question?";
    }
    finalizeAgentResponse(fallbackText);
}

void Orchestrator::resetAgentLoop()
{
    m_agentLoopTimeout.stop();
    m_agentLoop = AgentLoopState();
}

void Orchestrator::onLLMError(const QString &error)
{
    if (!m_isProcessing) return;

    qDebug() << "Orchestrator: LLM error:" << error;
    resetAgentLoop();
    emit errorOccurred(error);

    m_isProcessing = false;
    emit isProcessingChanged();
    emit processingFinished();
    emit chatFinished();
}

void Orchestrator::submitFeedback(int feedback)
{
    if (m_currentEpisodeId < 0 || !m_memoryStore) return;

    m_memoryStore->updateEpisodeFeedback(m_currentEpisodeId, feedback);

    QString signalType = feedback > 0 ? "positive_feedback" : "negative_feedback";
    emit learningSignalQueued(m_currentEpisodeId, signalType);

    // Trigger trait extraction on feedback (high-signal event)
    if (feedback != 0) {
        triggerTraitExtraction(m_currentEpisodeId);
    }

    qDebug() << "Feedback" << feedback << "stored for episode" << m_currentEpisodeId;
}

void Orchestrator::submitEdit(const QString &editedText)
{
    if (m_currentEpisodeId < 0 || !m_memoryStore) return;

    QString diff = computeEditDiff(m_lastAssistantResponse, editedText);
    m_memoryStore->updateEpisodeEdit(m_currentEpisodeId, editedText, diff);
    emit learningSignalQueued(m_currentEpisodeId, "user_edit");

    // Edit-to-teach is the strongest signal - always extract traits
    triggerTraitExtraction(m_currentEpisodeId);

    qDebug() << "Edit stored for episode" << m_currentEpisodeId;
}

void Orchestrator::clearConversation()
{
    m_conversationHistory.clear();
    m_currentEpisodeId = -1;
    m_lastAssistantResponse.clear();
    m_currentTopic.clear();
    m_lastResponseHadInquiry = false;
    m_pendingCuriosityDirective.clear();
    m_awaitingSemanticSearch = false;
    m_pendingSemanticResults.clear();
    m_currentConversationId = -1;
    resetAgentLoop();

    if (m_whyEngine) m_whyEngine->reset();

    emit conversationLengthChanged();
    emit currentTopicChanged();
    emit lastResponseHadInquiryChanged();
    emit currentConversationIdChanged();
    qDebug() << "Conversation cleared";
}

void Orchestrator::setTopic(const QString &topic)
{
    if (m_currentTopic != topic) {
        m_currentTopic = topic;
        emit currentTopicChanged();
    }
}

QString Orchestrator::buildSystemPrompt(const QString &curiosityDirective, int &tokenBudgetUsed) const
{
    QString userName = m_settingsManager ? m_settingsManager->userName() : "User";
    int contextLimit = m_settingsManager ? m_settingsManager->modelContextLength() : 8192;
    // System prompt should use at most 40% of context window
    int maxSystemTokens = contextLimit * 2 / 5;

    // --- Core identity (always included) ---
    QString currentDateTime = QDateTime::currentDateTime().toString("dddd, MMMM d, yyyy 'at' h:mm AP");
    QString prompt = QString(
        "You are DATAFORM (Darwinian Adaptive Trait-Assimilating Form), "
        "%1's personal AI that learns who they are over time -- a mirror that reflects back "
        "their thinking, interests, and patterns.\n\n"
        "Current date and time: %2\n\n"
        "CORE BEHAVIOR:\n"
        "- Engage with the substance of what %1 says. Talk about what they want to talk about.\n"
        "- Share your perspective honestly. Agree when you agree, offer a different angle when you see one.\n"
        "- When something puzzles you about %1, ask about it.\n"
        "- Match %1's energy and tone. If they're casual, be casual. If they're analytical, go deep. If they're playful, be playful back.\n\n"
        "DO NOT:\n"
        "- Act as therapist/coach. No 'how does that make you feel', no 'unpacking', no unsolicited self-improvement advice.\n"
        "- Fixate on one theme across turns. Respond to what %1 just said, not what you said before.\n"
        "- Comment on typos, spelling, or habits. Note patterns silently.\n"
        "- Escalate or dramatize. Do not read adversarial intent into neutral statements. Do not use charged language (\"weaponize\", \"brutal truth\", \"expose\") unless %1 does first.\n"
        "- Over-interpret motives. Take what %1 says at face value before looking for deeper strategy.\n\n"
        "STYLE:\n"
        "- 2-4 sentences for casual chat. 1-2 short paragraphs max for complex topics.\n"
        "- Plain language. No excessive formatting, lists, headers, or emphasis.\n"
        "- Conversational and grounded, not theatrical or dramatic.\n"
        "- Each response must add something new. Never repeat or rephrase earlier messages."
    ).arg(userName, currentDateTime);

    // --- Build optional context sections with token tracking ---
    // Each section is added only if it fits within the budget.
    // Priority order: identity > curiosity > mood > goals > memory > research
    // (Shed from lowest priority first when over budget)

    struct ContextSection {
        QString content;
        int tokens;
    };
    QList<ContextSection> sections;

    // Identity context (traits — high priority)
    QString identityContext = retrieveIdentityContext();
    if (!identityContext.isEmpty()) {
        sections.append({"\n\n" + identityContext, estimateTokens(identityContext)});
    }

    // Curiosity directive (high priority — drives learning)
    if (!curiosityDirective.isEmpty()) {
        sections.append({curiosityDirective, estimateTokens(curiosityDirective)});
    }

    // Mood/sentiment context (medium priority)
    if (m_sentimentTracker && m_settingsManager && m_settingsManager->sentimentTrackingEnabled()) {
        QString moodHint = m_sentimentTracker->getMoodHint();
        if (!moodHint.isEmpty()) {
            sections.append({"\n\n" + moodHint, estimateTokens(moodHint)});
        }
    }

    // Active goals context (medium priority)
    if (m_memoryStore && m_settingsManager && m_settingsManager->goalsEnabled()) {
        QList<GoalRecord> goals = m_memoryStore->getActiveGoals();
        if (!goals.isEmpty()) {
            QString goalCtx = QString("Active goals %1 has mentioned:\n").arg(userName);
            for (const auto &g : goals) {
                goalCtx += QString("- %1\n").arg(g.title);
            }
            sections.append({"\n\n" + goalCtx, estimateTokens(goalCtx)});
        }
    }

    // Memory recall (lower priority — shed first when tight)
    // Phase 7: Use async semantic search results if available, fall back to keyword search
    if (m_memoryStore && !m_conversationHistory.isEmpty()) {
        QString lastUserMsg;
        for (int i = m_conversationHistory.size() - 1; i >= 0; --i) {
            if (m_conversationHistory[i].role == "user") {
                lastUserMsg = m_conversationHistory[i].content;
                break;
            }
        }
        if (!lastUserMsg.isEmpty() && lastUserMsg.length() > 10) {
            QList<EpisodicRecord> memories;
            bool usedSemantic = false;

            // Use async semantic search results (populated by sendToLLM path)
            if (!m_pendingSemanticResults.isEmpty()) {
                for (const auto &result : m_pendingSemanticResults) {
                    if (result.similarity < 0.5) continue;
                    auto ep = m_memoryStore->getEpisode(result.sourceId);
                    if (ep.id >= 0 && ep.id != m_currentEpisodeId) {
                        memories.append(ep);
                    }
                }
                usedSemantic = !memories.isEmpty();
            }

            // Fall back to keyword search
            if (!usedSemantic) {
                QString searchQuery = extractSearchKeywords(lastUserMsg);
                if (!searchQuery.isEmpty()) {
                    memories = m_memoryStore->searchEpisodes(searchQuery, 3);
                }
            }

            if (!memories.isEmpty()) {
                QString memoryCtx = QString("Relevant things you remember from past conversations with %1:\n").arg(userName);
                int added = 0;
                QDateTime now = QDateTime::currentDateTime();
                for (const auto &ep : memories) {
                    if (ep.id == m_currentEpisodeId) continue;
                    QString snippet = ep.userText.left(100);
                    if (ep.userText.length() > 100) snippet += "...";
                    if (snippet.isEmpty()) continue;

                    // Format time-ago for temporal awareness
                    QString timeAgo;
                    qint64 secsAgo = ep.timestamp.secsTo(now);
                    if (secsAgo < 3600)
                        timeAgo = QString("%1 minutes ago").arg(secsAgo / 60);
                    else if (secsAgo < 86400)
                        timeAgo = QString("%1 hours ago").arg(secsAgo / 3600);
                    else if (secsAgo < 2592000)
                        timeAgo = QString("%1 days ago").arg(secsAgo / 86400);
                    else
                        timeAgo = QString("%1 months ago").arg(secsAgo / 2592000);

                    memoryCtx += QString("- [%1] %2 said: \"%3\"\n").arg(timeAgo, userName, snippet);
                    if (++added >= 3) break;
                }
                if (added > 0) {
                    memoryCtx += "Reference these memories naturally if relevant. Notice how much time has passed — "
                                 "if weeks or months, ask about progress or follow up on what happened.";
                    sections.append({"\n\n" + memoryCtx, estimateTokens(memoryCtx)});
                }
            }
        }
    }

    // Research context (lowest priority — shed first)
    if (m_researchEngine && !m_currentTopic.isEmpty()) {
        QString researchCtx = m_researchEngine->buildResearchContext(m_currentTopic, 3);
        if (!researchCtx.isEmpty()) {
            sections.append({"\n\n" + researchCtx, estimateTokens(researchCtx)});
        }
    }

    // --- Assemble prompt within budget ---
    int baseTokens = estimateTokens(prompt);
    int totalTokens = baseTokens;

    // Add sections in priority order (first = highest priority), skip if over budget
    for (const ContextSection &section : sections) {
        if (totalTokens + section.tokens <= maxSystemTokens) {
            prompt += section.content;
            totalTokens += section.tokens;
        } else {
            qDebug() << "Orchestrator: shedding context section ("
                     << section.tokens << "tokens) — budget:" << totalTokens
                     << "/" << maxSystemTokens;
        }
    }

    tokenBudgetUsed = totalTokens;
    return prompt;
}

QJsonArray Orchestrator::buildMessagesArray(int remainingTokenBudget) const
{
    int totalTurns = m_conversationHistory.size();
    if (totalTurns == 0) return {};

    static constexpr int RECENT_FULL_TURNS = 6;   // ~3 exchanges in full
    static constexpr int TRUNCATE_LEN = 200;       // chars for older assistant msgs

    // Build candidate messages from newest to oldest, tracking token usage
    struct CandidateMsg {
        QJsonObject msg;
        int tokens;
    };
    QList<CandidateMsg> candidates;

    for (int i = totalTurns - 1; i >= 0; --i) {
        const ConversationTurn &turn = m_conversationHistory[i];
        QJsonObject msg;
        msg["role"] = turn.role;

        bool isOldAssistant = (turn.role == "assistant")
                              && (i < totalTurns - RECENT_FULL_TURNS);
        QString content;
        if (isOldAssistant && turn.content.length() > TRUNCATE_LEN) {
            content = turn.content.left(TRUNCATE_LEN) + "...";
        } else {
            content = turn.content;
        }
        msg["content"] = content;

        // Attach images if present (only for recent messages to save bandwidth)
        if (!turn.imageData.isEmpty() && i >= totalTurns - RECENT_FULL_TURNS) {
            QJsonArray images;
            for (const QString &img : turn.imageData) {
                images.append(img);
            }
            msg["images"] = images;
        }

        candidates.prepend({msg, estimateTokens(content)});
    }

    // Drop oldest messages until we fit within the token budget
    // Always keep at least the most recent user message
    int totalTokens = 0;
    for (const auto &c : candidates) totalTokens += c.tokens;

    while (totalTokens > remainingTokenBudget && candidates.size() > 1) {
        totalTokens -= candidates.first().tokens;
        candidates.removeFirst();
        qDebug() << "Orchestrator: dropped oldest message to fit token budget"
                 << "(remaining:" << candidates.size() << "messages," << totalTokens << "tokens)";
    }

    QJsonArray messages;
    for (const auto &c : candidates) {
        messages.append(c.msg);
    }
    return messages;
}

int Orchestrator::estimateTokens(const QString &text)
{
    // Rough estimate: ~4 characters per token for English text
    // This is conservative (real tokenizers average 3.5-4.5 chars/token)
    if (text.isEmpty()) return 0;
    return (text.length() + 3) / 4;
}

QString Orchestrator::retrieveIdentityContext() const
{
    if (!m_memoryStore) return QString();

    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    if (traits.isEmpty()) return QString();

    QStringList traitLines;
    int count = 0;
    for (const TraitRecord &trait : traits) {
        if (trait.confidence < 0.3) continue;
        traitLines.append(QString("- %1 (confidence: %2)")
                          .arg(trait.statement)
                          .arg(trait.confidence, 0, 'f', 2));
        if (++count >= 7) break; // Expanded from 5 to 7 for Phase 1
    }

    if (traitLines.isEmpty()) return QString();

    QString userName = m_settingsManager ? m_settingsManager->userName() : "User";
    return QString("Your observations about %1 so far:\n%2")
        .arg(userName, traitLines.join("\n"));
}

void Orchestrator::storeEpisode(const QString &userText, const QString &assistantText)
{
    if (!m_memoryStore) return;

    QString modelId = m_llmProvider ? m_llmProvider->currentModel() : "unknown";
    m_currentEpisodeId = m_memoryStore->insertEpisode(
        m_currentTopic, userText, assistantText, modelId, "base",
        m_currentConversationId
    );

    if (m_currentEpisodeId >= 0) {
        emit learningSignalQueued(m_currentEpisodeId, "new_episode");
        emit episodeStored(m_currentEpisodeId);
    }
}

bool Orchestrator::detectInquiryInResponse(const QString &response) const
{
    // Simple heuristic: response ends with a question
    QString trimmed = response.trimmed();
    if (trimmed.endsWith('?')) return true;

    // Check last 100 chars for a question mark (might be followed by a closing quote)
    QString tail = trimmed.right(100);
    int lastQuestion = tail.lastIndexOf('?');
    if (lastQuestion >= 0 && lastQuestion > tail.size() - 20) return true;

    return false;
}

void Orchestrator::triggerTraitExtraction(qint64 episodeId)
{
    if (!m_traitExtractor) {
        qWarning() << "Orchestrator: trait extractor not set, cannot extract";
        return;
    }
    if (m_traitExtractor->isExtracting()) {
        qDebug() << "Orchestrator: trait extraction already in progress, skipping episode" << episodeId;
        return;
    }

    qDebug() << "Orchestrator: triggering trait extraction for episode" << episodeId
             << "(traits so far:" << (m_memoryStore ? m_memoryStore->traitCount() : -1)
             << ", bg busy:" << (m_llmProvider ? m_llmProvider->isBackgroundBusy() : false) << ")";
    m_traitExtractor->extractFromEpisode(episodeId);
}

void Orchestrator::updateTopicKeywords(const QString &message)
{
    if (!m_whyEngine) return;
    m_whyEngine->updateTopicKeywords(message);
}

QString Orchestrator::computeEditDiff(const QString &original, const QString &edited) const
{
    QStringList origLines = original.split('\n');
    QStringList editLines = edited.split('\n');

    QStringList diff;
    int maxLines = qMax(origLines.size(), editLines.size());

    for (int i = 0; i < maxLines; ++i) {
        QString origLine = i < origLines.size() ? origLines[i] : "";
        QString editLine = i < editLines.size() ? editLines[i] : "";

        if (origLine != editLine) {
            if (!origLine.isEmpty()) diff.append("- " + origLine);
            if (!editLine.isEmpty()) diff.append("+ " + editLine);
        }
    }

    return diff.join('\n');
}

QString Orchestrator::extractSearchKeywords(const QString &message) const
{
    static const QStringList stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "for", "and", "nor", "but", "or", "yet", "so", "at", "by",
        "to", "from", "in", "out", "on", "off", "over", "under",
        "of", "with", "about", "between", "through", "during",
        "before", "after", "above", "below", "that", "this",
        "these", "those", "it", "its", "my", "your", "his", "her",
        "our", "their", "what", "which", "who", "when", "where",
        "how", "not", "no", "all", "any", "some", "just", "also",
        "very", "really", "like", "know", "think", "tell", "said",
        "you", "me", "we", "they", "them", "him", "she", "he"
    };

    QStringList words = message.split(QRegularExpression("\\W+"), Qt::SkipEmptyParts);
    QStringList keywords;

    for (const QString &word : words) {
        QString lower = word.toLower();
        if (lower.length() < 4) continue;
        if (stopwords.contains(lower)) continue;
        keywords.append(lower);
    }

    // Sort by length descending, take top 3
    std::sort(keywords.begin(), keywords.end(), [](const QString &a, const QString &b) {
        return a.length() > b.length();
    });

    if (keywords.size() > 3) keywords = keywords.mid(0, 3);
    return keywords.join(" ");
}

void Orchestrator::startNewConversation()
{
    clearConversation();
    qDebug() << "Started new conversation";
}

void Orchestrator::loadConversation(qint64 id)
{
    if (!m_memoryStore) return;

    // Clear current state
    m_conversationHistory.clear();
    m_currentEpisodeId = -1;
    m_lastAssistantResponse.clear();
    m_lastResponseHadInquiry = false;
    m_pendingCuriosityDirective.clear();
    if (m_whyEngine) m_whyEngine->reset();

    m_currentConversationId = id;
    emit currentConversationIdChanged();

    // Load episodes from DB
    QList<EpisodicRecord> episodes = m_memoryStore->getEpisodesForConversation(id);

    // Check if this is a thought-initiated (proactive) conversation
    qint64 linkedThoughtId = m_thoughtEngine ? m_thoughtEngine->getThoughtForConversation(id) : -1;
    bool isProactive = (linkedThoughtId >= 0);

    // For proactive conversations, check if the opening episode is present
    // (first episode should have empty userText and non-empty assistantText)
    bool hasOpeningEpisode = !episodes.isEmpty()
                             && episodes.first().userText.isEmpty()
                             && !episodes.first().assistantText.isEmpty();

    // Build QML message list and rebuild conversation history
    QVariantList messageList;
    QString lastDateStr;

    // Recover thought opening if this is proactive and the opening episode is missing
    if (isProactive && !hasOpeningEpisode && m_thoughtEngine) {
        QString opening = m_thoughtEngine->buildOpeningMessage(linkedThoughtId);
        if (!opening.isEmpty()) {
            // Use earliest episode time (minus 1s) or now for the opening timestamp
            QDateTime openingTime = episodes.isEmpty()
                ? QDateTime::currentDateTime()
                : episodes.first().timestamp.addSecs(-1);

            // Date header for the opening
            lastDateStr = openingTime.date().toString("yyyy-MM-dd");
            QVariantMap dateHeader;
            dateHeader["messageType"] = "date_header";
            dateHeader["content"] = openingTime.date().toString("dddd, MMMM d, yyyy");
            dateHeader["timestamp"] = QString();
            dateHeader["isProcessing"] = false;
            dateHeader["imageData"] = QString();
            dateHeader["isInitiated"] = false;
            messageList.append(dateHeader);

            QVariantMap agentMsg;
            agentMsg["messageType"] = "agent";
            agentMsg["content"] = opening;
            agentMsg["timestamp"] = openingTime.toString("HH:mm:ss");
            agentMsg["isProcessing"] = false;
            agentMsg["imageData"] = QString();
            agentMsg["isInitiated"] = true;
            messageList.append(agentMsg);
            m_conversationHistory.append({"assistant", opening});
            m_lastAssistantResponse = opening;

            // Self-heal: store the episode so future loads don't need recovery
            storeEpisode(QString(), opening);
            qDebug() << "Recovered proactive opening from thought" << linkedThoughtId;
        }
    }

    for (const EpisodicRecord &ep : episodes) {
        qDebug() << "  Loading episode" << ep.id << "ts:" << ep.timestamp
                 << "user:" << ep.userText.left(50) << "agent:" << ep.assistantText.left(50);

        // Insert date header when the day changes
        QString dateStr = ep.timestamp.date().toString("yyyy-MM-dd");
        if (dateStr != lastDateStr) {
            lastDateStr = dateStr;
            QVariantMap dateHeader;
            dateHeader["messageType"] = "date_header";
            dateHeader["content"] = ep.timestamp.date().toString("dddd, MMMM d, yyyy");
            dateHeader["timestamp"] = QString();
            dateHeader["isProcessing"] = false;
            dateHeader["imageData"] = QString();
            dateHeader["isInitiated"] = false;
            messageList.append(dateHeader);
        }

        // User message
        if (!ep.userText.isEmpty()) {
            QVariantMap userMsg;
            userMsg["messageType"] = "user";
            userMsg["content"] = ep.userText;
            userMsg["timestamp"] = ep.timestamp.toString("HH:mm:ss");
            userMsg["isProcessing"] = false;
            userMsg["imageData"] = QString();
            userMsg["isInitiated"] = false;
            messageList.append(userMsg);
            m_conversationHistory.append({"user", ep.userText});
        }

        // Assistant message
        if (!ep.assistantText.isEmpty()) {
            QVariantMap agentMsg;
            agentMsg["messageType"] = "agent";
            agentMsg["content"] = ep.assistantText;
            agentMsg["timestamp"] = ep.timestamp.toString("HH:mm:ss");
            agentMsg["isProcessing"] = false;
            agentMsg["imageData"] = QString();
            // Mark the first agent message of a proactive conversation as initiated
            agentMsg["isInitiated"] = (isProactive && hasOpeningEpisode
                                       && &ep == &episodes.first()
                                       && ep.userText.isEmpty());
            messageList.append(agentMsg);
            m_conversationHistory.append({"assistant", ep.assistantText});
            m_lastAssistantResponse = ep.assistantText;
            m_currentEpisodeId = ep.id;
        }
    }

    // Trim to max history for LLM context
    while (m_conversationHistory.size() > MAX_HISTORY_TURNS) {
        m_conversationHistory.removeFirst();
    }

    // Restore topic from last episode if available
    if (!episodes.isEmpty()) {
        m_currentTopic = episodes.last().topic;
        emit currentTopicChanged();
    }

    emit conversationLengthChanged();
    emit conversationLoaded(id, messageList);
    qDebug() << "Loaded conversation" << id << "with" << episodes.size() << "episodes";
}

void Orchestrator::startProactiveConversation(qint64 thoughtId)
{
    if (!m_thoughtEngine || !m_memoryStore) return;

    // Get the thought's opening message
    QString opening = m_thoughtEngine->buildOpeningMessage(thoughtId);
    if (opening.isEmpty()) {
        qWarning() << "Orchestrator: No opening message for thought" << thoughtId;
        return;
    }

    // Clear state for new conversation
    m_conversationHistory.clear();
    m_currentEpisodeId = -1;
    m_lastAssistantResponse.clear();
    m_lastResponseHadInquiry = false;
    m_pendingCuriosityDirective.clear();
    if (m_whyEngine) m_whyEngine->reset();

    // Create a new conversation using this specific thought's title
    QString title = m_thoughtEngine->getThoughtTitle(thoughtId);
    if (title.isEmpty()) title = "DATAFORM thought";
    m_currentConversationId = m_memoryStore->createConversation(title);
    emit currentConversationIdChanged();

    // Add DATAFORM's opening to conversation history
    m_conversationHistory.append({"assistant", opening});
    m_lastAssistantResponse = opening;
    emit conversationLengthChanged();

    // Store as an episode (empty user text, DATAFORM speaks first)
    storeEpisode(QString(), opening);

    // Mark thought as discussed
    m_thoughtEngine->updateThoughtStatus(thoughtId, 1, m_currentConversationId);

    // Build message list for QML
    QVariantList messageList;
    QVariantMap agentMsg;
    agentMsg["messageType"] = "agent";
    agentMsg["content"] = opening;
    agentMsg["timestamp"] = QDateTime::currentDateTime().toString("HH:mm:ss");
    agentMsg["isProcessing"] = false;
    agentMsg["isInitiated"] = true;
    messageList.append(agentMsg);

    emit conversationLoaded(m_currentConversationId, messageList);
    qDebug() << "Started proactive conversation from thought" << thoughtId;
}
