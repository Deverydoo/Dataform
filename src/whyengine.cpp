#include "whyengine.h"
#include "fuzzylogic.h"
#include "memorystore.h"
#include <QDebug>
#include <QRandomGenerator>
#include <QRegularExpression>

// Static template initialization
QList<WhyEngine::InquiryTemplate> WhyEngine::s_templates = WhyEngine::initTemplates();

QList<WhyEngine::InquiryTemplate> WhyEngine::initTemplates()
{
    return {
        // Topic shift - user changed subjects
        {"topic_shift", "Interesting. What prompted the change in topic?"},
        {"topic_shift", "That is a different subject. Is it connected to what we were discussing, or something new?"},
        {"topic_shift", "What made you think of this now?"},

        // Emotional content - user expressing feelings or stress
        {"emotional", "This seems significant to you. What is the reasoning behind that?"},
        {"emotional", "I notice this carries weight for you. Can you explain why?"},
        {"emotional", "What would a good outcome look like here, specifically?"},

        // Decision-related - user weighing options
        {"decision", "What is the primary factor in this decision for you?"},
        {"decision", "Are you looking for analysis, or thinking out loud?"},
        {"decision", "What evidence would resolve this for you?"},

        // Novel topic - something DATAFORM hasn't seen before
        {"novel", "I have no prior data on this topic. What is your experience with it?"},
        {"novel", "This is unfamiliar to me. What draws you to this subject?"},
        {"novel", "What specifically are you trying to accomplish here?"},

        // Spot check - periodic low-frequency check-in
        {"spot_check", "Is my current approach useful, or would you prefer I adjust?"},
        {"spot_check", "Am I understanding your requirements correctly?"},
        {"spot_check", "Is there context I am missing?"},
    };
}

WhyEngine::WhyEngine(QObject *parent)
    : QObject(parent)
{
}

WhyEngine::~WhyEngine()
{
}

void WhyEngine::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

bool WhyEngine::shouldInquire(const QString &userMessage, const QString &currentTopic) const
{
    // Never ask if we're awaiting a response to a previous inquiry
    if (m_awaitingInquiryResponse) return false;

    // Budget check: respect minimum turns between inquiries
    if (!isWithinBudget()) return false;

    // Evaluate trigger conditions
    bool topicShift = detectTopicShift(userMessage, currentTopic);
    bool emotional = isEmotionalContent(userMessage);
    bool decision = isDecisionRelated(userMessage);
    bool novel = isNovelTopic(userMessage);

    // Short messages are unlikely targets for inquiry (soft gate)
    double lengthFactor = Fuzzy::sigmoid(
        static_cast<double>(userMessage.trimmed().length()), 10.0, 4.0
    );
    if (QRandomGenerator::global()->generateDouble() > lengthFactor) return false;

    // Strong triggers always fire (if within budget)
    if (emotional || decision) return true;

    // Topic shift triggers with high probability
    if (topicShift) {
        double roll = QRandomGenerator::global()->generateDouble();
        return roll < 0.7;
    }

    // Novel topics trigger based on curiosity level (high curiosity = more likely)
    if (novel) {
        double roll = QRandomGenerator::global()->generateDouble();
        return roll < (0.4 + 0.4 * m_curiosityLevel);  // 0.4 to 0.8 based on curiosity
    }

    // Spot check: low-frequency random inquiry
    if (m_turnsSinceLastInquiry >= 8) {
        double roll = QRandomGenerator::global()->generateDouble();
        return roll < BASE_INQUIRY_PROBABILITY * m_curiosityLevel;
    }

    return false;
}

QString WhyEngine::buildCuriosityDirective(const QString &userMessage,
                                             const QString &currentTopic) const
{
    if (!shouldInquire(userMessage, currentTopic)) {
        return QString();
    }

    // Determine the trigger reason
    QString triggerReason;
    if (isEmotionalContent(userMessage)) {
        triggerReason = "emotional";
    } else if (isDecisionRelated(userMessage)) {
        triggerReason = "decision";
    } else if (detectTopicShift(userMessage, currentTopic)) {
        triggerReason = "topic_shift";
    } else if (isNovelTopic(userMessage)) {
        triggerReason = "novel";
        // Queue novel topics for idle-time research
        QStringList words = userMessage.split(QRegularExpression("\\W+"), Qt::SkipEmptyParts);
        QStringList keywords;
        for (const QString &w : words) {
            if (w.length() > 4) keywords.append(w.toLower());
        }
        if (!keywords.isEmpty()) {
            const_cast<WhyEngine*>(this)->queueNovelTopic(keywords.join(' '));
        }
    } else {
        triggerReason = "spot_check";
    }

    QString inquiry = selectInquiryTemplate(userMessage, triggerReason);

    return QString(
        "\n\nCURIOSITY DIRECTIVE: After your response, you may ask ONE brief question "
        "about the reasoning behind what the user said. "
        "Suggested angle: \"%1\" "
        "Keep it short and direct. Do not therapize. If it would feel forced, skip it."
    ).arg(inquiry);
}

void WhyEngine::onResponseDelivered(const QString &response, bool containedInquiry)
{
    m_totalTurns++;
    m_turnsSinceLastInquiry++;

    if (containedInquiry) {
        m_awaitingInquiryResponse = true;
        m_inquiryTriggered = true;
        emit inquiryTriggeredChanged();
    }

    updateCuriosityLevel();
}

void WhyEngine::updateTopicKeywords(const QString &userMessage)
{
    // Extract keywords (words > 4 chars) and maintain a sliding window
    QStringList words = userMessage.toLower().split(
        QRegularExpression("\\W+"), Qt::SkipEmptyParts);

    for (const QString &w : words) {
        if (w.length() > 4 && !m_recentTopicKeywords.contains(w)) {
            m_recentTopicKeywords.append(w);
        }
    }

    // Keep only the last 20 keywords (sliding window of recent context)
    while (m_recentTopicKeywords.size() > 20) {
        m_recentTopicKeywords.removeFirst();
    }
}

bool WhyEngine::isAnsweringInquiry(const QString &userMessage) const
{
    if (!m_awaitingInquiryResponse) return false;

    // Heuristics: if the user's message is longer than typical and seems explanatory
    if (userMessage.length() > 50) return true;

    // Check for common "answering" patterns
    static const QStringList answerPatterns = {
        "because", "the reason", "i think", "i feel", "i want",
        "i need", "it's about", "well,", "honestly", "actually",
        "for me", "my goal", "what i", "i'm trying"
    };

    QString lower = userMessage.toLower();
    for (const QString &pattern : answerPatterns) {
        if (lower.contains(pattern)) return true;
    }

    return false;
}

void WhyEngine::recordInquiry(const QString &inquiry)
{
    m_lastInquiry = inquiry;
    m_lastInquiryTime = QDateTime::currentDateTime();
    m_inquiryCount++;
    m_consecutiveInquiries++;
    m_turnsSinceLastInquiry = 0;
    m_awaitingInquiryResponse = true;

    emit lastInquiryChanged();
    emit inquiryCountChanged();

    qDebug() << "WhyEngine: recorded inquiry #" << m_inquiryCount << ":" << inquiry.left(60);
}

void WhyEngine::queueNovelTopic(const QString &topic)
{
    if (topic.isEmpty()) return;

    // Avoid duplicates
    for (const QString &existing : m_novelTopicsQueue) {
        if (existing.toLower() == topic.toLower()) return;
    }

    m_novelTopicsQueue.append(topic);

    // Cap the queue
    while (m_novelTopicsQueue.size() > MAX_NOVEL_TOPICS) {
        m_novelTopicsQueue.removeFirst();
    }

    qDebug() << "WhyEngine: queued novel topic for research:" << topic
             << "(queue size:" << m_novelTopicsQueue.size() << ")";
}

QStringList WhyEngine::getNovelTopicsForResearch()
{
    QStringList topics = m_novelTopicsQueue;
    m_novelTopicsQueue.clear();
    return topics;
}

void WhyEngine::reset()
{
    m_inquiryTriggered = false;
    m_lastInquiry.clear();
    m_awaitingInquiryResponse = false;
    m_consecutiveInquiries = 0;
    m_turnsSinceLastInquiry = 0;
    m_totalTurns = 0;
    m_recentTopicKeywords.clear();

    emit inquiryTriggeredChanged();
    emit lastInquiryChanged();
}

// --- Trigger detection ---

bool WhyEngine::detectTopicShift(const QString &userMessage, const QString &currentTopic) const
{
    if (m_recentTopicKeywords.isEmpty()) return false;

    // Extract keywords from user message (simple: words > 4 chars, lowered)
    QStringList messageWords = userMessage.toLower().split(
        QRegularExpression("\\W+"), Qt::SkipEmptyParts);

    QStringList keywords;
    for (const QString &w : messageWords) {
        if (w.length() > 4) keywords.append(w);
    }

    if (keywords.isEmpty()) return false;

    // Compare with recent topic keywords - low overlap = topic shift
    int overlap = 0;
    for (const QString &kw : keywords) {
        if (m_recentTopicKeywords.contains(kw)) overlap++;
    }

    double overlapRatio = keywords.isEmpty() ? 0.0
        : static_cast<double>(overlap) / keywords.size();

    return overlapRatio < (1.0 - TOPIC_SHIFT_THRESHOLD);
}

bool WhyEngine::isEmotionalContent(const QString &message) const
{
    static const QStringList emotionalMarkers = {
        "frustrated", "worried", "anxious", "excited", "angry", "sad",
        "stressed", "overwhelmed", "confused", "scared", "happy",
        "disappointed", "hurt", "tired", "lost", "stuck",
        "i feel", "i'm feeling", "it bothers me", "i can't stop thinking",
        "i'm afraid", "i'm worried", "breaking point", "i don't know what to do"
    };

    QString lower = message.toLower();
    for (const QString &marker : emotionalMarkers) {
        if (lower.contains(marker)) return true;
    }
    return false;
}

bool WhyEngine::isDecisionRelated(const QString &message) const
{
    static const QStringList decisionMarkers = {
        "should i", "which one", "what would you", "or should",
        "trying to decide", "can't choose", "weighing", "pros and cons",
        "better option", "trade-off", "trade off", "dilemma",
        "on one hand", "on the other", "what do you think",
        "help me decide", "not sure whether", "debating"
    };

    QString lower = message.toLower();
    for (const QString &marker : decisionMarkers) {
        if (lower.contains(marker)) return true;
    }
    return false;
}

bool WhyEngine::isNovelTopic(const QString &userMessage) const
{
    if (!m_memoryStore) return true; // No memory = everything is novel

    // Check if we have any episodes touching similar keywords
    QStringList words = userMessage.toLower().split(
        QRegularExpression("\\W+"), Qt::SkipEmptyParts);

    // Search for the longest meaningful word in episodic memory
    for (const QString &w : words) {
        if (w.length() <= 4) continue;
        QList<EpisodicRecord> matches = m_memoryStore->searchEpisodes(w, 3);
        if (!matches.isEmpty()) return false; // Found prior context
    }

    return true; // No prior episodes match
}

double WhyEngine::computeTraitCoverage(const QString &topic) const
{
    if (!m_memoryStore) return 0.0;

    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    if (traits.isEmpty()) return 0.0;

    // Check how many traits relate to this topic
    int related = 0;
    QString lower = topic.toLower();
    for (const TraitRecord &t : traits) {
        if (t.statement.toLower().contains(lower)) related++;
    }

    return traits.isEmpty() ? 0.0 : static_cast<double>(related) / traits.size();
}

// --- Inquiry template selection ---

QString WhyEngine::selectInquiryTemplate(const QString &userMessage,
                                           const QString &triggerReason) const
{
    // Filter templates by category
    QList<InquiryTemplate> candidates;
    for (const InquiryTemplate &t : s_templates) {
        if (t.category == triggerReason) {
            candidates.append(t);
        }
    }

    if (candidates.isEmpty()) {
        // Fallback to spot_check
        for (const InquiryTemplate &t : s_templates) {
            if (t.category == "spot_check") candidates.append(t);
        }
    }

    if (candidates.isEmpty()) {
        return "What is your reasoning here?";
    }

    // Random selection from matching templates
    int idx = QRandomGenerator::global()->bounded(candidates.size());
    return candidates[idx].templateText;
}

// --- Budget / cooldown ---

bool WhyEngine::isWithinBudget() const
{
    // Respect minimum turns between inquiries
    if (m_turnsSinceLastInquiry < MIN_TURNS_BETWEEN_INQUIRIES) return false;

    // Don't ask too many in a row
    if (m_consecutiveInquiries >= MAX_CONSECUTIVE_INQUIRIES) return false;

    // Soft time-based cooldown centered at 30 seconds
    if (m_lastInquiryTime.isValid()) {
        qint64 secsSinceLast = m_lastInquiryTime.secsTo(QDateTime::currentDateTime());
        double cooldownReady = Fuzzy::sigmoid(static_cast<double>(secsSinceLast), 30.0, 8.0);
        if (QRandomGenerator::global()->generateDouble() > cooldownReady) return false;
    }

    return true;
}

void WhyEngine::updateCuriosityLevel()
{
    // Curiosity level adjusts based on how much we know
    if (!m_memoryStore) return;

    int traitCount = m_memoryStore->traitCount();
    int episodeCount = m_memoryStore->episodeCount();

    // Trait saturation: S-curve — first few traits barely reduce curiosity,
    // 15+ traits reduce meaningfully. Floor at 0.15.
    double traitFactor = 1.0 - Fuzzy::sigmoid(
        static_cast<double>(traitCount), 12.0, 5.0
    ) * 0.85;

    // Novice factor: smooth transition instead of step at 20 episodes
    // At 5 episodes: ~0.95, at 20: ~0.85, at 50: ~0.72
    double noviceFactor = 0.7 + 0.3 * (1.0 - Fuzzy::sigmoid(
        static_cast<double>(episodeCount), 20.0, 10.0
    ));

    m_curiosityLevel = Fuzzy::clamp01(
        std::max(0.15, traitFactor * noviceFactor)
    );
    emit curiosityLevelChanged();
}
