#include "traitextractor.h"
#include "fuzzylogic.h"
#include "llmresponseparser.h"
#include "llmprovider.h"
#include "memorystore.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QRegularExpression>
#include <QDebug>

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
            else if (owner == "LeanAnalyzer") onLeanAnalysisResponse(response);
        });
        connect(provider, &LLMProviderManager::backgroundErrorOccurred,
                this, [this](const QString &owner, const QString &error) {
            if (owner == "TraitExtractor") onExtractionError(error);
            else if (owner == "LeanAnalyzer") {
                qWarning() << "LeanAnalyzer: LLM error:" << error;
                m_isAnalyzingLean = false;
                emit isAnalyzingLeanChanged();
            }
        });
    }
}

void TraitExtractor::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
    if (store) {
        QVariantMap latest = store->getLatestLeanForQml();
        if (!latest.isEmpty()) {
            m_leanScore = latest["leanScore"].toDouble();
            m_leanAnalyzedTs = latest["analyzedTs"].toString();
            emit leanScoreChanged();
            emit leanLabelChanged();
            emit leanAnalyzedTsChanged();
        }
    }
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
    qDebug() << "TraitExtractor: idle window opened";
    // Auto-start removed — coordinator calls requestStart()
}

bool TraitExtractor::canStartCycle() const
{
    if (!m_llmProvider || !m_memoryStore) return false;
    if (m_isExtracting) return false;
    return true;
}

void TraitExtractor::requestStart()
{
    if (!canStartCycle()) {
        emit cycleFinished();
        return;
    }
    // Process one batch — cycleFinished emitted from response/error handler
    m_idleActive = true;  // Ensure idle flag is set for priority selection
    processNextConversation();
    // If processNextConversation returned without starting extraction (no episodes),
    // we need to signal done
    if (!m_isExtracting) {
        emit cycleFinished();
    }
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
        "What does the user reveal about their personality, interests, opinions, or communication style?\n\n"
        "GOOD trait examples: \"Thinks critically about government motives\", \"Prefers natural remedies over pharmaceuticals\", "
        "\"Enjoys building complex software systems\", \"Communicates directly without softening\", \"Interested in herbalism and organic farming\"\n\n"
        "BAD — do NOT extract:\n"
        "- Things OTHER people think/say/feel that the user merely reports\n"
        "- Fleeting emotional reactions (\"feels frustrated right now\")\n\n"
        "Respond with a JSON array:\n"
        "[{\"type\": \"value\", \"statement\": \"description in third person\", \"confidence\": 0.7}]\n\n"
        "Types: value (beliefs), preference (style/taste), policy (decision patterns), motivation (drives)\n"
        "- Extract 1-5 traits. Be specific — capture what makes THIS user distinct.\n"
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
        "Extract personality traits about the USER from these exchanges.\n\n"
        "%1\n"
        "What does the user reveal about their personality, interests, opinions, or communication style?\n\n"
        "GOOD examples: \"Thinks critically about government motives\", \"Prefers hands-on building over theory\", "
        "\"Interested in herbalism and organic farming\"\n"
        "BAD — do NOT extract: things OTHER people think/say that the user merely reports, or fleeting emotions.\n\n"
        "Output a JSON array. Example: [{\"type\":\"value\",\"statement\":\"Values honesty\",\"confidence\":0.7}]\n"
        "Types: value, preference, policy, motivation. Third person. Confidence 0.3-0.9.\n"
        "Extract 1-5 traits. Be specific — capture what makes THIS user distinct.\n"
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
        "What does this response reveal about the user's personality, interests, or opinions?\n"
        "Do NOT extract things other people think/say that the user merely reports.\n\n"
        "Output a JSON array. Example: [{\"type\":\"motivation\",\"statement\":\"Values understanding over quick fixes\",\"confidence\":0.8}]\n"
        "Types: value, preference, policy, motivation. Third person. Confidence 0.3-0.9.\n"
        "Extract 1-3 traits. Be specific.\n"
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

    // Single-batch mode: signal coordinator that this cycle is done
    // Coordinator will re-activate for more batches if needed
    emit cycleFinished();
}

void TraitExtractor::onExtractionError(const QString &error)
{
    m_extractionTimeout.stop();
    setStatus(QString("Error: %1").arg(error));
    emit extractionError(error);

    m_isExtracting = false;
    m_currentExtractionEpisodeId = -1;
    emit isExtractingChanged();

    // Signal coordinator that this cycle is done
    emit cycleFinished();
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
        "\nWhat does this reveal about the user's personality, interests, opinions, or style?\n"
        "Do NOT extract things other people think/say that the user merely reports.\n\n"
        "Output a JSON array. Example: [{\"type\":\"value\",\"statement\":\"Values honesty\",\"confidence\":0.7}]\n"
        "Types: value, preference, policy, motivation. Third person. Confidence 0.3-0.9.\n"
        "Extract 1-3 traits. Be specific — capture what makes THIS user distinct.\n"
        "IMPORTANT: Output ONLY the JSON array starting with [ and ending with ]. No other text.";

    return prompt;
}

// --- JSON parsing ---

QList<ExtractedTrait> TraitExtractor::parseTraitsFromResponse(const QString &response) const
{
    QList<ExtractedTrait> traits;

    QJsonDocument doc = LLMResponseParser::extractJsonArray(response, "TraitExtractor");
    if (doc.isNull()) return traits;

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

    QList<TraitRecord> allTraits = m_memoryStore->getAllTraits();

    for (const ExtractedTrait &trait : traits) {
        // Find the most similar existing trait (deterministic best-match)
        QString bestMatchId;
        double bestSimilarity = 0.0;
        for (const TraitRecord &e : allTraits) {
            double similarity = traitSimilarity(e.statement, trait.statement);
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatchId = e.traitId;
            }
        }

        if (bestSimilarity >= 0.55) {
            // True duplicate — merge evidence into the existing trait
            mergeWithExistingTrait(bestMatchId, trait, episodeId);
            qDebug() << "TraitExtractor: merged duplicate trait (similarity:"
                     << bestSimilarity << "):" << trait.statement;
            continue;
        }

        // New unique trait — insert it
        QString traitId = m_memoryStore->insertTrait(trait.type, trait.statement, trait.confidence);
        if (!traitId.isEmpty()) {
            if (episodeId >= 0) {
                m_memoryStore->addTraitEvidence(traitId, episodeId);
            }
            // Add to our local list so subsequent traits in this batch can dedup against it
            TraitRecord newRecord;
            newRecord.traitId = traitId;
            newRecord.type = trait.type;
            newRecord.statement = trait.statement;
            newRecord.confidence = trait.confidence;
            allTraits.append(newRecord);

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

    // Exact matches
    QSet<QString> intersection = wordsA;
    intersection.intersect(wordsB);

    // Fuzzy prefix matches (catches creation/creator, concept/concepts, value/values)
    QSet<QString> unmatchedA = wordsA - intersection;
    QSet<QString> unmatchedB = wordsB - intersection;
    int fuzzyMatches = 0;
    for (const QString &wa : unmatchedA) {
        for (const QString &wb : unmatchedB) {
            int prefixLen = 0;
            int minLen = qMin(wa.length(), wb.length());
            for (int i = 0; i < minLen; ++i) {
                if (wa[i] == wb[i]) ++prefixLen;
                else break;
            }
            if (prefixLen >= 5) {
                ++fuzzyMatches;
                break;
            }
        }
    }

    int matchCount = intersection.size() + fuzzyMatches;
    QSet<QString> unionSet = wordsA;
    unionSet.unite(wordsB);

    return static_cast<double>(matchCount) / static_cast<double>(unionSet.size());
}

bool TraitExtractor::isDuplicateTrait(const QString &statement) const
{
    if (!m_memoryStore) return false;

    QList<TraitRecord> allTraits = m_memoryStore->getAllTraits();

    for (const TraitRecord &existing : allTraits) {
        double similarity = traitSimilarity(existing.statement, statement);
        if (similarity >= 0.55) {
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
    // Gentler diminishing returns: 0.85^n decays slower than 0.7^n
    // so each re-detection still meaningfully increases confidence
    double boostFactor = 0.12 * Fuzzy::decay(0.85, static_cast<double>(evidenceCount));
    double newConfidence = existing.confidence + (1.0 - existing.confidence) * boostFactor;
    newConfidence = qMin(0.95, newConfidence);

    m_memoryStore->updateTraitConfidence(existingTraitId, newConfidence);
    m_memoryStore->addTraitEvidence(existingTraitId, episodeId);
    m_memoryStore->confirmTrait(existingTraitId);

    qDebug() << "TraitExtractor: merged evidence into existing trait"
             << existingTraitId << "confidence:" << existing.confidence << "->" << newConfidence;
}

// --- Political Lean Analysis ---

QString TraitExtractor::leanLabel() const
{
    return leanScoreToLabel(m_leanScore);
}

QString TraitExtractor::leanScoreToLabel(double score) const
{
    if (score <= -0.6) return QStringLiteral("Left");
    if (score <= -0.2) return QStringLiteral("Lean Left");
    if (score <= 0.2)  return QStringLiteral("Center");
    if (score <= 0.6)  return QStringLiteral("Lean Right");
    return QStringLiteral("Right");
}

void TraitExtractor::analyzePoliticalLean()
{
    if (!m_llmProvider || !m_memoryStore) return;
    if (m_isAnalyzingLean) return;

    // Gather all "value" and "policy" type traits
    QList<TraitRecord> valueTraits = m_memoryStore->getTraitsByType("value");
    QList<TraitRecord> policyTraits = m_memoryStore->getTraitsByType("policy");

    QList<TraitRecord> allRelevant;
    allRelevant.append(valueTraits);
    allRelevant.append(policyTraits);

    // Build trait list, filtering low-confidence ones
    QString traitList;
    for (const TraitRecord &t : allRelevant) {
        if (t.confidence < 0.3) continue;
        traitList += QString("- [%1] %2 (confidence: %3)\n")
                         .arg(t.type, t.statement)
                         .arg(t.confidence, 0, 'f', 2);
    }

    if (traitList.isEmpty()) {
        setStatus("Need value/policy traits with confidence >= 0.3 for lean analysis");
        return;
    }

    // Count qualifying traits
    int qualifyingCount = 0;
    for (const TraitRecord &t : allRelevant) {
        if (t.confidence >= 0.3) qualifyingCount++;
    }
    if (qualifyingCount < 3) {
        setStatus(QString("Need at least 3 value/policy traits for lean analysis (have %1)")
                      .arg(qualifyingCount));
        return;
    }

    m_isAnalyzingLean = true;
    emit isAnalyzingLeanChanged();

    QString prompt = QString(
        "Analyze the following personal values and policy positions on the standard "
        "US political spectrum. Rate them from -1.0 (strong left/liberal/progressive) "
        "to +1.0 (strong right/conservative/traditional).\n\n"
        "TRAITS:\n%1\n"
        "Consider each trait's position individually, then weight by confidence to "
        "produce an overall score.\n\n"
        "Output a JSON object:\n"
        "{\"score\": 0.0, \"reasoning\": \"brief explanation\"}\n\n"
        "Score guide: -1.0=far left, -0.5=lean left, 0.0=center, +0.5=lean right, +1.0=far right\n"
        "IMPORTANT: Output ONLY the JSON object. No other text."
    ).arg(traitList);

    QJsonArray messages;
    QJsonObject msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    messages.append(msg);

    m_llmProvider->sendBackgroundRequest(
        "LeanAnalyzer",
        "You are a JSON-only political analysis API. You output valid JSON and nothing else. "
        "Be objective and analytical. Rate positions, not people.",
        messages
    );

    qDebug() << "LeanAnalyzer: analyzing" << qualifyingCount << "value/policy traits";
}

void TraitExtractor::onLeanAnalysisResponse(const QString &response)
{
    m_isAnalyzingLean = false;
    emit isAnalyzingLeanChanged();

    QJsonDocument doc = LLMResponseParser::extractJsonObject(response, "LeanAnalyzer");
    if (doc.isNull()) return;

    QJsonObject obj = doc.object();
    double score = qBound(-1.0, obj.value("score").toDouble(0.0), 1.0);

    // Collect contributing trait IDs
    QList<TraitRecord> valueTraits = m_memoryStore->getTraitsByType("value");
    QList<TraitRecord> policyTraits = m_memoryStore->getTraitsByType("policy");
    QStringList traitIds;
    for (const TraitRecord &t : valueTraits) traitIds.append(t.traitId);
    for (const TraitRecord &t : policyTraits) traitIds.append(t.traitId);

    // Store in DB
    m_memoryStore->insertLeanAnalysis(score, traitIds);

    // Update cached values
    m_leanScore = score;
    m_leanAnalyzedTs = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    m_lastLeanAnalysis = QDateTime::currentDateTime();
    emit leanScoreChanged();
    emit leanLabelChanged();
    emit leanAnalyzedTsChanged();
    emit leanAnalysisComplete(score);

    QString reasoning = obj.value("reasoning").toString();
    qDebug() << "LeanAnalyzer: score=" << score << "label=" << leanScoreToLabel(score)
             << "reasoning:" << reasoning.left(200);
}
