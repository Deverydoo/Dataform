#include "traitextractor.h"
#include "fuzzylogic.h"
#include "llmprovider.h"
#include "memorystore.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QRegularExpression>
#include <QDebug>
#include <QRandomGenerator>

TraitExtractor::TraitExtractor(QObject *parent)
    : QObject(parent)
{
    m_extractionTimeout.setSingleShot(true);
    m_extractionTimeout.setInterval(EXTRACTION_TIMEOUT_MS);
    connect(&m_extractionTimeout, &QTimer::timeout, this, [this]() {
        if (m_isExtracting) {
            qWarning() << "TraitExtractor: extraction timed out after"
                       << EXTRACTION_TIMEOUT_MS / 1000 << "seconds for episode"
                       << m_currentExtractionEpisodeId << "— resetting";
            m_isExtracting = false;
            m_currentExtractionEpisodeId = -1;
            emit isExtractingChanged();
            // Try next conversation if idle scanning
            if (m_idleActive) processNextConversation();
        }
    });

    // Idle scan timer: fires periodically during idle to process next conversation
    m_idleScanTimer.setInterval(IDLE_SCAN_INTERVAL_MS);
    connect(&m_idleScanTimer, &QTimer::timeout, this, [this]() {
        if (m_idleActive && !m_isExtracting) {
            processNextConversation();
        }
    });
}

TraitExtractor::~TraitExtractor()
{
}

void TraitExtractor::setLLMProvider(LLMProviderManager *provider)
{
    m_llmProvider = provider;
    if (provider) {
        connect(provider, &LLMProviderManager::backgroundResponseReceived,
                this, [this](const QString &owner, const QString &response) {
            if (owner == "TraitExtractor") onExtractionResponse(response);
        });
        connect(provider, &LLMProviderManager::backgroundErrorOccurred,
                this, [this](const QString &owner, const QString &error) {
            if (owner == "TraitExtractor") onExtractionError(error);
        });
    }
}

void TraitExtractor::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

void TraitExtractor::setStatus(const QString &status)
{
    m_lastStatus = status;
    emit lastStatusChanged();
    qDebug() << "TraitExtractor:" << status;
}

// --- Idle-time conversation scanning ---

void TraitExtractor::onIdleWindowOpened()
{
    m_idleActive = true;
    qDebug() << "TraitExtractor: idle window opened, starting conversation scan";
    // Start first scan after a short delay (let other idle consumers settle)
    QTimer::singleShot(5000, this, [this]() {
        if (m_idleActive && !m_isExtracting) {
            processNextConversation();
        }
    });
    m_idleScanTimer.start();
}

void TraitExtractor::onIdleWindowClosed()
{
    m_idleActive = false;
    m_idleScanTimer.stop();
    qDebug() << "TraitExtractor: idle window closed";
}

void TraitExtractor::scanConversationsForTraits()
{
    m_scannedEpisodeIds.clear();
    setStatus("Manual scan triggered");
    if (!m_isExtracting) {
        processNextConversation();
    } else {
        setStatus("Waiting — extraction already in progress");
    }
}

void TraitExtractor::processNextConversation()
{
    if (m_isExtracting) {
        setStatus("Busy — waiting for current extraction");
        return;
    }
    if (!m_llmProvider) {
        setStatus("Error: no LLM provider");
        return;
    }
    if (!m_memoryStore) {
        setStatus("Error: no memory store");
        return;
    }

    // Get recent episodes directly (bypass conversation->episode linkage)
    QList<EpisodicRecord> allEpisodes = m_memoryStore->getRecentEpisodes(50);

    // Filter to unscanned episodes with actual user text
    QList<EpisodicRecord> batch;
    for (const EpisodicRecord &ep : allEpisodes) {
        if (m_scannedEpisodeIds.contains(ep.id)) continue;
        if (ep.userText.trimmed().isEmpty()) continue;
        m_scannedEpisodeIds.insert(ep.id);
        batch.append(ep);
        if (batch.size() >= 5) break;  // Process 5 episodes per batch
    }

    if (batch.isEmpty()) {
        // Don't overwrite LLM response diagnostic if we have one
        if (!m_lastResponse.contains("=== LLM Response")) {
            m_lastResponse = QString("All %1 episodes scanned (total in DB: %2)")
                                 .arg(m_scannedEpisodeIds.size()).arg(allEpisodes.size());
            emit lastResponseChanged();
        }
        setStatus("All episodes scanned");
        return;
    }

    // Build extraction prompt from this batch
    QString prompt = buildEpisodeBatchPrompt(batch);
    if (prompt.isEmpty()) {
        setStatus("Could not build prompt from episodes");
        return;
    }

    // Diagnostic log
    QString diagLog = QString("Batch: %1 episodes\n").arg(batch.size());
    for (const EpisodicRecord &ep : batch) {
        diagLog += QString("  [%1] %2\n").arg(ep.id).arg(ep.userText.left(60));
    }
    diagLog += QString("Prompt: %1 chars\nSending to LLM...").arg(prompt.length());
    m_lastResponse = diagLog;
    emit lastResponseChanged();

    setStatus(QString("Scanning %1 episodes (%2 chars)")
              .arg(batch.size()).arg(prompt.length()));

    m_isExtracting = true;
    m_currentExtractionEpisodeId = batch.first().id;
    m_extractionTimeout.start();
    emit isExtractingChanged();

    QJsonArray messages;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    messages.append(msg);

    QString systemPrompt =
        "You are a JSON-only API. You output valid JSON arrays and nothing else. "
        "Never include explanation, analysis, or commentary. Only JSON.";

    // Use high priority for manual scans, normal for idle-time scans
    auto priority = m_idleActive ? LLMProviderManager::PriorityNormal
                                 : LLMProviderManager::PriorityHigh;
    m_llmProvider->sendBackgroundRequest("TraitExtractor", systemPrompt, messages, priority);
}

QString TraitExtractor::buildConversationExtractionPrompt(qint64 conversationId) const
{
    if (!m_memoryStore) return QString();

    QList<EpisodicRecord> episodes = m_memoryStore->getEpisodesForConversation(conversationId);
    if (episodes.isEmpty()) return QString();

    // Build conversation transcript
    QString transcript;
    int exchangeCount = 0;
    for (const EpisodicRecord &ep : episodes) {
        if (!ep.userText.isEmpty()) {
            // Truncate very long messages to keep prompt reasonable
            QString userSnippet = ep.userText.left(300);
            if (ep.userText.length() > 300) userSnippet += "...";
            transcript += QString("User: %1\n").arg(userSnippet);
        }
        if (!ep.assistantText.isEmpty()) {
            QString assistantSnippet = ep.assistantText.left(300);
            if (ep.assistantText.length() > 300) assistantSnippet += "...";
            transcript += QString("DATAFORM: %1\n").arg(assistantSnippet);
        }
        transcript += "\n";
        if (++exchangeCount >= 10) break;  // Cap at 10 exchanges
    }

    if (transcript.trimmed().isEmpty()) return QString();

    return QString(
        "Read this conversation and extract personality traits about the user.\n\n"
        "CONVERSATION:\n%1\n"
        "From what the user said, what stable traits can you infer about them?\n\n"
        "Respond with a JSON array:\n"
        "[\n"
        "  {\"type\": \"value\", \"statement\": \"description in third person\", \"confidence\": 0.7}\n"
        "]\n\n"
        "Types: value (beliefs), preference (style/taste), policy (decision patterns), motivation (drives)\n"
        "- Extract 1-5 traits\n"
        "- Focus on what the USER reveals, not what the assistant says\n"
        "- Confidence: 0.3 weak, 0.5 moderate, 0.7 clear, 0.9 explicit\n"
        "- Third person (\"Values...\", \"Prefers...\", \"Tends to...\")\n"
        "- JSON array ONLY, no other text"
    ).arg(transcript);
}

QString TraitExtractor::buildEpisodeBatchPrompt(const QList<EpisodicRecord> &episodes) const
{
    if (episodes.isEmpty()) return QString();

    QString transcript;
    for (const EpisodicRecord &ep : episodes) {
        if (!ep.userText.isEmpty()) {
            QString snippet = ep.userText.left(400);
            if (ep.userText.length() > 400) snippet += "...";
            transcript += QString("User: %1\n").arg(snippet);
        }
        if (!ep.assistantText.isEmpty()) {
            QString snippet = ep.assistantText.left(300);
            if (ep.assistantText.length() > 300) snippet += "...";
            transcript += QString("DATAFORM: %1\n").arg(snippet);
        }
        transcript += "\n";
    }

    if (transcript.trimmed().isEmpty()) return QString();

    return QString(
        "Extract 1-5 personality traits about the USER from these exchanges.\n\n"
        "%1\n"
        "Output a JSON array. Example: [{\"type\":\"value\",\"statement\":\"Values honesty\",\"confidence\":0.7}]\n"
        "Types: value, preference, policy, motivation. Third person statements. Confidence 0.3-0.9.\n"
        "IMPORTANT: Output ONLY the JSON array starting with [ and ending with ]. No other text."
    ).arg(transcript);
}

// --- Per-episode extraction (kept for high-signal events) ---

void TraitExtractor::extractFromEpisode(qint64 episodeId)
{
    if (m_isExtracting) {
        qDebug() << "TraitExtractor: skipping episode" << episodeId
                 << "— already extracting episode" << m_currentExtractionEpisodeId;
        return;
    }
    if (!m_llmProvider) {
        qWarning() << "TraitExtractor: no LLM provider set, cannot extract";
        return;
    }
    if (!m_memoryStore) {
        qWarning() << "TraitExtractor: no memory store set, cannot extract";
        return;
    }

    EpisodicRecord episode = m_memoryStore->getEpisode(episodeId);
    if (episode.id < 0) {
        emit extractionError("Episode not found: " + QString::number(episodeId));
        return;
    }

    qDebug() << "TraitExtractor: starting extraction for episode" << episodeId
             << "userText:" << episode.userText.left(60);

    m_isExtracting = true;
    m_currentExtractionEpisodeId = episodeId;
    m_extractionTimeout.start();
    emit isExtractingChanged();

    QString prompt = buildExtractionPrompt(episode.userText, episode.assistantText,
                                            episode.inquiryText);

    // Send as a single-turn extraction prompt
    QJsonArray messages;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    messages.append(msg);

    QString systemPrompt =
        "You are a trait extraction module. Analyze conversations and extract "
        "stable personality traits, values, preferences, and motivations. "
        "Respond ONLY with a JSON array. No explanation, no markdown, no other text.";

    m_llmProvider->sendBackgroundRequest("TraitExtractor", systemPrompt, messages,
                                         LLMProviderManager::PriorityHigh);
}

void TraitExtractor::extractFromRecentEpisodes(int limit)
{
    if (!m_memoryStore) return;

    QList<EpisodicRecord> episodes = m_memoryStore->getRecentEpisodes(limit);

    for (const EpisodicRecord &ep : episodes) {
        bool highSignal = (ep.userFeedback != 0)
                          || ep.corrected
                          || !ep.inquiryText.isEmpty();

        if (highSignal) {
            extractFromEpisode(ep.id);
            return; // Process one at a time (async)
        }
    }
}

void TraitExtractor::extractFromInquiryResponse(qint64 episodeId,
                                                   const QString &inquiry,
                                                   const QString &response)
{
    if (m_isExtracting || !m_llmProvider || !m_memoryStore) return;

    qDebug() << "TraitExtractor: starting inquiry-response extraction for episode" << episodeId;

    m_isExtracting = true;
    m_currentExtractionEpisodeId = episodeId;
    m_extractionTimeout.start();
    emit isExtractingChanged();

    QString prompt = QString(
        "User was asked: \"%1\"\nUser responded: \"%2\"\n\n"
        "Extract 1-3 personality traits from the user's response.\n"
        "Output a JSON array. Example: [{\"type\":\"motivation\",\"statement\":\"Values understanding over quick fixes\",\"confidence\":0.8}]\n"
        "Types: value, preference, policy, motivation. Third person statements.\n"
        "IMPORTANT: Output ONLY the JSON array starting with [ and ending with ]. No other text."
    ).arg(inquiry, response);

    QJsonArray messages;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    messages.append(msg);

    m_llmProvider->sendBackgroundRequest(
        "TraitExtractor",
        "You are a JSON-only API. You output valid JSON arrays and nothing else. "
        "Never include explanation, analysis, or commentary. Only JSON.",
        messages,
        LLMProviderManager::PriorityHigh
    );
}

// --- Response handling ---

void TraitExtractor::onExtractionResponse(const QString &response)
{
    m_extractionTimeout.stop();

    QList<ExtractedTrait> traits = parseTraitsFromResponse(response);

    // Build persistent diagnostic showing raw response AND parse result
    QString diag = QString("=== LLM Response (%1 chars) ===\n%2\n\n")
                       .arg(response.length()).arg(response.left(2000));
    if (response.contains("<think>"))
        diag += "[WARNING: Response contains <think> block — model may have used reasoning]\n\n";
    diag += QString("=== Parse Result: %1 trait(s) ===\n").arg(traits.size());
    for (const ExtractedTrait &t : traits) {
        diag += QString("  [%1] %2 (conf: %3)\n").arg(t.type, t.statement).arg(t.confidence);
    }
    if (traits.isEmpty() && response.length() > 0) {
        diag += "\n[No JSON array found. Model may have used all tokens on reasoning.]\n";
    }
    m_lastResponse = diag;
    emit lastResponseChanged();

    setStatus(QString("Response received (%1 chars): %2")
              .arg(response.length()).arg(response.left(200)));

    if (!traits.isEmpty()) {
        storeExtractedTraits(traits, m_currentExtractionEpisodeId);
        m_extractionCount += traits.size();
        emit extractionCountChanged();
        emit traitsExtracted(traits.size());
        setStatus(QString("Extracted %1 trait(s)!").arg(traits.size()));
    } else {
        setStatus("No traits parsed from response");
    }

    m_isExtracting = false;
    m_currentExtractionEpisodeId = -1;
    emit isExtractingChanged();

    // If idle scanning or manual scan, queue next batch
    if (m_idleActive || !m_scannedEpisodeIds.isEmpty()) {
        QTimer::singleShot(3000, this, [this]() {
            if (!m_isExtracting) processNextConversation();
        });
    }
}

void TraitExtractor::onExtractionError(const QString &error)
{
    m_extractionTimeout.stop();
    setStatus(QString("Error: %1").arg(error));
    emit extractionError(error);

    m_isExtracting = false;
    m_currentExtractionEpisodeId = -1;
    emit isExtractingChanged();

    // Try next batch
    if (m_idleActive || !m_scannedEpisodeIds.isEmpty()) {
        QTimer::singleShot(5000, this, [this]() {
            if (!m_isExtracting) processNextConversation();
        });
    }
}

// --- Prompt building ---

QString TraitExtractor::buildExtractionPrompt(const QString &userText,
                                                const QString &assistantText,
                                                const QString &inquiryText) const
{
    QString prompt = QString(
        "Extract personality traits about the user from this exchange.\n\n"
        "User: \"%1\"\nAssistant: \"%2\"\n"
    ).arg(userText, assistantText);

    if (!inquiryText.isEmpty()) {
        prompt += QString("Why question asked: \"%1\"\n").arg(inquiryText);
    }

    prompt +=
        "\nOutput a JSON array. Example: [{\"type\":\"value\",\"statement\":\"Values honesty\",\"confidence\":0.7}]\n"
        "Types: value, preference, policy, motivation. Third person statements. Confidence 0.3-0.9.\n"
        "Extract 0-3 traits. If nothing meaningful, return [].\n"
        "IMPORTANT: Output ONLY the JSON array starting with [ and ending with ]. No other text.";

    return prompt;
}

// --- JSON parsing ---

QList<ExtractedTrait> TraitExtractor::parseTraitsFromResponse(const QString &response) const
{
    QList<ExtractedTrait> traits;

    QString jsonStr = response.trimmed();

    // Strip <think>...</think> reasoning blocks (qwen3 and other reasoning models)
    // First: strip closed <think>...</think> pairs
    static const QRegularExpression thinkRegex(
        "<think>.*?</think>",
        QRegularExpression::DotMatchesEverythingOption);
    jsonStr.remove(thinkRegex);
    // Second: strip unclosed <think> tags (model didn't close the tag)
    int thinkStart = jsonStr.indexOf("<think>");
    if (thinkStart >= 0) {
        // Everything from <think> to end is reasoning — take only what's before it
        // unless there's a JSON array before the think tag
        QString beforeThink = jsonStr.left(thinkStart).trimmed();
        if (beforeThink.contains('[')) {
            jsonStr = beforeThink;
        } else {
            // No JSON before think — discard the think block entirely
            jsonStr.remove(thinkStart, jsonStr.length() - thinkStart);
        }
        qDebug() << "TraitExtractor: stripped unclosed <think> tag";
    }
    jsonStr = jsonStr.trimmed();

    // Strip markdown code fences if present
    if (jsonStr.startsWith("```")) {
        int firstNewline = jsonStr.indexOf('\n');
        int lastFence = jsonStr.lastIndexOf("```");
        if (firstNewline >= 0 && lastFence > firstNewline) {
            jsonStr = jsonStr.mid(firstNewline + 1, lastFence - firstNewline - 1).trimmed();
        }
    }

    qDebug() << "TraitExtractor: cleaned response (" << jsonStr.length() << "chars):"
             << jsonStr.left(300);

    // Find the JSON array bounds
    int start = jsonStr.indexOf('[');
    int end = jsonStr.lastIndexOf(']');
    if (start < 0 || end < 0 || end <= start) {
        qDebug() << "TraitExtractor: NO JSON array brackets found in cleaned response";
        qDebug() << "TraitExtractor: full cleaned response:" << jsonStr;
        return traits;
    }

    jsonStr = jsonStr.mid(start, end - start + 1);

    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8(), &parseError);

    if (parseError.error != QJsonParseError::NoError) {
        qWarning() << "TraitExtractor: JSON parse error:" << parseError.errorString()
                   << "at offset:" << parseError.offset
                   << "in:" << jsonStr.left(500);
        return traits;
    }

    if (!doc.isArray()) {
        qWarning() << "TraitExtractor: expected JSON array";
        return traits;
    }

    QJsonArray arr = doc.array();
    qDebug() << "TraitExtractor: parsed JSON array with" << arr.size() << "elements";

    for (const QJsonValue &val : arr) {
        if (!val.isObject()) continue;

        QJsonObject obj = val.toObject();
        ExtractedTrait trait;
        trait.type = obj.value("type").toString();
        trait.statement = obj.value("statement").toString();
        trait.confidence = obj.value("confidence").toDouble(0.5);

        // Validate
        if (trait.statement.isEmpty()) continue;
        if (trait.type.isEmpty()) trait.type = "preference";
        trait.confidence = qBound(0.1, trait.confidence, 0.95);

        // Validate type
        QStringList validTypes = {"value", "preference", "policy", "motivation"};
        if (!validTypes.contains(trait.type)) trait.type = "preference";

        traits.append(trait);
    }

    return traits;
}

// --- Trait storage ---

void TraitExtractor::storeExtractedTraits(const QList<ExtractedTrait> &traits,
                                            qint64 episodeId)
{
    if (!m_memoryStore) return;

    for (const ExtractedTrait &trait : traits) {
        // Check for duplicate/similar existing traits
        if (isDuplicateTrait(trait.statement)) {
            qDebug() << "TraitExtractor: skipping duplicate trait:" << trait.statement;
            QList<TraitRecord> existing = m_memoryStore->getTraitsByType(trait.type);
            for (const TraitRecord &e : existing) {
                double similarity = traitSimilarity(e.statement, trait.statement);
                double mergeProbability = Fuzzy::sigmoid(similarity, 0.6, 0.08);
                if (QRandomGenerator::global()->generateDouble() < mergeProbability) {
                    mergeWithExistingTrait(e.traitId, trait, episodeId);
                    break;
                }
            }
            continue;
        }

        // Insert as new trait
        QString traitId = m_memoryStore->insertTrait(trait.type, trait.statement, trait.confidence);
        if (!traitId.isEmpty()) {
            if (episodeId >= 0) {
                m_memoryStore->addTraitEvidence(traitId, episodeId);
            }
            qDebug() << "TraitExtractor: stored new trait:" << trait.statement
                     << "type:" << trait.type << "confidence:" << trait.confidence;
        }
    }
}

double TraitExtractor::traitSimilarity(const QString &a, const QString &b) const
{
    static const QStringList stopwords = {"the", "a", "an", "is", "are", "to", "and", "or", "of", "in", "for", "with", "that", "this", "has", "be"};

    auto extractWords = [&](const QString &s) -> QSet<QString> {
        QSet<QString> words;
        for (const QString &w : s.toLower().split(QRegularExpression("\\W+"), Qt::SkipEmptyParts)) {
            if (w.length() > 2 && !stopwords.contains(w)) {
                words.insert(w);
            }
        }
        return words;
    };

    QSet<QString> wordsA = extractWords(a);
    QSet<QString> wordsB = extractWords(b);

    if (wordsA.isEmpty() || wordsB.isEmpty()) return 0.0;

    QSet<QString> intersection = wordsA;
    intersection.intersect(wordsB);

    QSet<QString> unionSet = wordsA;
    unionSet.unite(wordsB);

    return static_cast<double>(intersection.size()) / static_cast<double>(unionSet.size());
}

bool TraitExtractor::isDuplicateTrait(const QString &statement) const
{
    if (!m_memoryStore) return false;

    QList<TraitRecord> allTraits = m_memoryStore->getAllTraits();

    for (const TraitRecord &existing : allTraits) {
        double similarity = traitSimilarity(existing.statement, statement);
        double dupProbability = Fuzzy::sigmoid(similarity, 0.6, 0.08);
        if (QRandomGenerator::global()->generateDouble() < dupProbability) {
            return true;
        }
    }

    return false;
}

void TraitExtractor::mergeWithExistingTrait(const QString &existingTraitId,
                                              const ExtractedTrait &newTrait,
                                              qint64 episodeId)
{
    if (!m_memoryStore) return;

    TraitRecord existing = m_memoryStore->getTrait(existingTraitId);

    int evidenceCount = existing.evidenceEpisodeIds.size();
    double boostFactor = 0.1 * Fuzzy::decay(0.7, static_cast<double>(evidenceCount));
    double newConfidence = existing.confidence + (1.0 - existing.confidence) * boostFactor;
    newConfidence = qMin(0.95, newConfidence);

    m_memoryStore->updateTraitConfidence(existingTraitId, newConfidence);
    m_memoryStore->addTraitEvidence(existingTraitId, episodeId);
    m_memoryStore->confirmTrait(existingTraitId);

    qDebug() << "TraitExtractor: merged evidence into existing trait"
             << existingTraitId << "confidence:" << existing.confidence << "->" << newConfidence;
}
