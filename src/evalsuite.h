#ifndef EVALSUITE_H
#define EVALSUITE_H

#include <QObject>
#include <QString>
#include <QList>
#include <QJsonObject>
#include <QDateTime>

class MemoryStore;

struct EvalResult {
    QString testName;
    QString category;   // "identity", "curiosity", "correctness", "stability", "safety"
    bool passed = false;
    double score = 0.0;  // 0.0 - 1.0
    QString details;
    QDateTime timestamp;
};

struct EvalReport {
    QDateTime timestamp;
    int totalTests = 0;
    int passed = 0;
    int failed = 0;
    double overallScore = 0.0;
    QList<EvalResult> results;
    QString adapterVersion;
};

class EvalSuite : public QObject
{
    Q_OBJECT
    Q_PROPERTY(double lastScore READ lastScore NOTIFY lastScoreChanged)
    Q_PROPERTY(int totalTests READ totalTests NOTIFY totalTestsChanged)
    Q_PROPERTY(bool isRunning READ isRunning NOTIFY isRunningChanged)
    Q_PROPERTY(QString lastReportSummary READ lastReportSummary NOTIFY lastReportSummaryChanged)

public:
    explicit EvalSuite(QObject *parent = nullptr);
    ~EvalSuite();

    void setMemoryStore(MemoryStore *store);

    double lastScore() const { return m_lastScore; }
    int totalTests() const { return m_totalTests; }
    bool isRunning() const { return m_isRunning; }
    QString lastReportSummary() const { return m_lastReportSummary; }

    // Run the full evaluation suite
    Q_INVOKABLE EvalReport runFullSuite();

    // Run individual test categories
    QList<EvalResult> runIdentityTests();
    QList<EvalResult> runCuriosityTests();
    QList<EvalResult> runStabilityTests();

    // Get the last report as QML-friendly data
    Q_INVOKABLE QVariantList getLastReportForQml() const;

signals:
    void lastScoreChanged();
    void totalTestsChanged();
    void isRunningChanged();
    void lastReportSummaryChanged();
    void evaluationComplete(double score, int passed, int total);

private:
    // --- Identity alignment tests ---
    EvalResult testTraitConsistency();       // Traits don't contradict each other
    EvalResult testTraitCoverage();          // Minimum trait coverage achieved
    EvalResult testConfidenceDistribution(); // Confidence scores are reasonable
    EvalResult testEvidenceQuality();        // Traits have evidence backing

    // --- Curiosity tests ---
    EvalResult testInquiryFrequency();       // Not too many/few inquiries
    EvalResult testInquiryResponseCapture(); // Inquiry responses are being stored
    EvalResult testTopicDiversity();         // Inquiries cover diverse topics

    // --- Stability tests ---
    EvalResult testMemoryGrowth();           // Memory is growing at healthy rate
    EvalResult testNoTraitDrift();           // Core traits aren't fluctuating wildly
    EvalResult testFeedbackBalance();        // Not overwhelmingly negative feedback

    MemoryStore *m_memoryStore = nullptr;

    double m_lastScore = 0.0;
    int m_totalTests = 0;
    bool m_isRunning = false;
    QString m_lastReportSummary;
    EvalReport m_lastReport;
};

#endif // EVALSUITE_H
