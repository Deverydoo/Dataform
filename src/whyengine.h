#ifndef WHYENGINE_H
#define WHYENGINE_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QList>

class MemoryStore;

class WhyEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool inquiryTriggered READ inquiryTriggered NOTIFY inquiryTriggeredChanged)
    Q_PROPERTY(QString lastInquiry READ lastInquiry NOTIFY lastInquiryChanged)
    Q_PROPERTY(int inquiryCount READ inquiryCount NOTIFY inquiryCountChanged)
    Q_PROPERTY(double curiosityLevel READ curiosityLevel NOTIFY curiosityLevelChanged)

public:
    explicit WhyEngine(QObject *parent = nullptr);
    ~WhyEngine();

    void setMemoryStore(MemoryStore *store);

    bool inquiryTriggered() const { return m_inquiryTriggered; }
    QString lastInquiry() const { return m_lastInquiry; }
    int inquiryCount() const { return m_inquiryCount; }
    double curiosityLevel() const { return m_curiosityLevel; }

    // Core decision: should DATAFORM ask a "why" question this turn?
    bool shouldInquire(const QString &userMessage, const QString &currentTopic) const;

    // Build the curiosity directive for the system prompt
    // Returns empty string if no inquiry should be made
    QString buildCuriosityDirective(const QString &userMessage, const QString &currentTopic) const;

    // Called after a response is delivered - tracks inquiry state
    void onResponseDelivered(const QString &response, bool containedInquiry);

    // Called when user's message appears to answer a previous inquiry
    bool isAnsweringInquiry(const QString &userMessage) const;

    // Record that we asked an inquiry (updates cooldown/budget)
    void recordInquiry(const QString &inquiry);

    // Track topic keywords from user messages (for shift detection)
    void updateTopicKeywords(const QString &userMessage);

    // Novel topic queue for idle-time research
    void queueNovelTopic(const QString &topic);
    QStringList getNovelTopicsForResearch();
    QStringList recentTopicKeywords() const { return m_recentTopicKeywords; }

    // Reset inquiry tracking (e.g., on conversation clear)
    Q_INVOKABLE void reset();

signals:
    void inquiryTriggeredChanged();
    void lastInquiryChanged();
    void inquiryCountChanged();
    void curiosityLevelChanged();

private:
    // --- Trigger evaluation ---
    bool detectTopicShift(const QString &userMessage, const QString &currentTopic) const;
    bool isEmotionalContent(const QString &message) const;
    bool isDecisionRelated(const QString &message) const;
    bool isNovelTopic(const QString &userMessage) const;
    double computeTraitCoverage(const QString &topic) const;

    // --- Inquiry generation ---
    QString selectInquiryTemplate(const QString &userMessage, const QString &triggerReason) const;

    // --- Budget / cooldown ---
    bool isWithinBudget() const;
    void updateCuriosityLevel();

    MemoryStore *m_memoryStore = nullptr;

    bool m_inquiryTriggered = false;
    QString m_lastInquiry;
    int m_inquiryCount = 0;
    double m_curiosityLevel = 0.5;  // 0.0 = never ask, 1.0 = always ask

    // Inquiry tracking
    QDateTime m_lastInquiryTime;
    bool m_awaitingInquiryResponse = false;
    int m_consecutiveInquiries = 0;
    int m_turnsSinceLastInquiry = 0;
    int m_totalTurns = 0;

    // Recent topics for shift detection
    QStringList m_recentTopicKeywords;

    // Novel topics discovered during conversation (for research)
    QStringList m_novelTopicsQueue;
    static constexpr int MAX_NOVEL_TOPICS = 10;

    // Configurable thresholds
    static constexpr int MIN_TURNS_BETWEEN_INQUIRIES = 2;
    static constexpr int MAX_CONSECUTIVE_INQUIRIES = 2;
    static constexpr double TOPIC_SHIFT_THRESHOLD = 0.6;
    static constexpr double BASE_INQUIRY_PROBABILITY = 0.35;

    // Inquiry templates by trigger category
    struct InquiryTemplate {
        QString category;   // "topic_shift", "emotional", "decision", "novel", "spot_check"
        QString templateText;
    };
    static QList<InquiryTemplate> s_templates;
    static QList<InquiryTemplate> initTemplates();
};

#endif // WHYENGINE_H
