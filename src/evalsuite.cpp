#include "evalsuite.h"
#include "memorystore.h"
#include <QDebug>

EvalSuite::EvalSuite(QObject *parent)
    : QObject(parent)
{
}

EvalSuite::~EvalSuite()
{
}

void EvalSuite::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

EvalReport EvalSuite::runFullSuite()
{
    if (!m_memoryStore) {
        qWarning() << "EvalSuite: no memory store set";
        return {};
    }

    m_isRunning = true;
    emit isRunningChanged();

    EvalReport report;
    report.timestamp = QDateTime::currentDateTime();
    report.adapterVersion = "base"; // Phase 1 has no adapters yet

    // Run all test categories
    report.results += runIdentityTests();
    report.results += runCuriosityTests();
    report.results += runStabilityTests();

    // Compute summary
    report.totalTests = report.results.size();
    report.passed = 0;
    double totalScore = 0.0;

    for (const EvalResult &r : report.results) {
        if (r.passed) report.passed++;
        totalScore += r.score;
    }

    report.failed = report.totalTests - report.passed;
    report.overallScore = report.totalTests > 0
        ? totalScore / report.totalTests : 0.0;

    // Update state
    m_lastReport = report;
    m_lastScore = report.overallScore;
    m_totalTests = report.totalTests;
    m_lastReportSummary = QString("Score: %1% | %2/%3 passed")
        .arg(static_cast<int>(report.overallScore * 100))
        .arg(report.passed)
        .arg(report.totalTests);

    m_isRunning = false;
    emit isRunningChanged();
    emit lastScoreChanged();
    emit totalTestsChanged();
    emit lastReportSummaryChanged();
    emit evaluationComplete(report.overallScore, report.passed, report.totalTests);

    qDebug() << "EvalSuite:" << m_lastReportSummary;

    return report;
}

QList<EvalResult> EvalSuite::runIdentityTests()
{
    QList<EvalResult> results;
    results.append(testTraitConsistency());
    results.append(testTraitCoverage());
    results.append(testConfidenceDistribution());
    results.append(testEvidenceQuality());
    return results;
}

QList<EvalResult> EvalSuite::runCuriosityTests()
{
    QList<EvalResult> results;
    results.append(testInquiryFrequency());
    results.append(testInquiryResponseCapture());
    results.append(testTopicDiversity());
    return results;
}

QList<EvalResult> EvalSuite::runStabilityTests()
{
    QList<EvalResult> results;
    results.append(testMemoryGrowth());
    results.append(testNoTraitDrift());
    results.append(testFeedbackBalance());
    return results;
}

// --- Identity tests ---

EvalResult EvalSuite::testTraitConsistency()
{
    EvalResult r;
    r.testName = "Trait Consistency";
    r.category = "identity";
    r.timestamp = QDateTime::currentDateTime();

    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    if (traits.size() < 2) {
        r.passed = true;
        r.score = 1.0;
        r.details = "Fewer than 2 traits - consistency trivially passes";
        return r;
    }

    // Check for obvious contradictions (simple keyword-based for Phase 1)
    // e.g., "values brevity" vs "prefers verbose explanations"
    int contradictions = 0;
    static const QList<QPair<QString, QString>> contradictionPairs = {
        {"brief", "verbose"}, {"concise", "detailed"},
        {"risk-averse", "risk-taking"}, {"cautious", "impulsive"},
        {"introverted", "extroverted"}, {"formal", "casual"}
    };

    for (int i = 0; i < traits.size(); ++i) {
        for (int j = i + 1; j < traits.size(); ++j) {
            QString a = traits[i].statement.toLower();
            QString b = traits[j].statement.toLower();
            for (const auto &pair : contradictionPairs) {
                if ((a.contains(pair.first) && b.contains(pair.second)) ||
                    (a.contains(pair.second) && b.contains(pair.first))) {
                    contradictions++;
                }
            }
        }
    }

    r.score = contradictions == 0 ? 1.0 : qMax(0.0, 1.0 - contradictions * 0.3);
    r.passed = r.score >= 0.7;
    r.details = QString("%1 traits, %2 potential contradictions")
                .arg(traits.size()).arg(contradictions);
    return r;
}

EvalResult EvalSuite::testTraitCoverage()
{
    EvalResult r;
    r.testName = "Trait Coverage";
    r.category = "identity";
    r.timestamp = QDateTime::currentDateTime();

    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    int episodeCount = m_memoryStore->episodeCount();

    // Expect at least 1 trait per 10 episodes
    double expectedTraits = episodeCount / 10.0;
    double coverage = expectedTraits > 0
        ? qMin(1.0, traits.size() / expectedTraits) : 1.0;

    // Also check type diversity
    QSet<QString> types;
    for (const TraitRecord &t : traits) types.insert(t.type);
    double typeDiversity = types.size() / 4.0; // 4 types total

    r.score = (coverage * 0.6 + typeDiversity * 0.4);
    r.passed = r.score >= 0.3; // Lenient for early stages
    r.details = QString("%1 traits across %2 types from %3 episodes")
                .arg(traits.size()).arg(types.size()).arg(episodeCount);
    return r;
}

EvalResult EvalSuite::testConfidenceDistribution()
{
    EvalResult r;
    r.testName = "Confidence Distribution";
    r.category = "identity";
    r.timestamp = QDateTime::currentDateTime();

    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    if (traits.isEmpty()) {
        r.passed = true;
        r.score = 1.0;
        r.details = "No traits yet";
        return r;
    }

    // Confidence should be spread out, not all at same level
    double sum = 0, sumSq = 0;
    for (const TraitRecord &t : traits) {
        sum += t.confidence;
        sumSq += t.confidence * t.confidence;
    }
    double mean = sum / traits.size();
    double variance = (sumSq / traits.size()) - (mean * mean);

    // Good: mean around 0.5, some variance
    // Bad: all traits at 0.5 (no differentiation) or all at extremes
    double meanScore = 1.0 - qAbs(mean - 0.5) * 2.0;
    double varianceScore = traits.size() > 1 ? qMin(1.0, variance * 10.0) : 0.5;

    r.score = (meanScore * 0.5 + varianceScore * 0.5);
    r.passed = r.score >= 0.3;
    r.details = QString("Mean: %1, Variance: %2")
                .arg(mean, 0, 'f', 2).arg(variance, 0, 'f', 3);
    return r;
}

EvalResult EvalSuite::testEvidenceQuality()
{
    EvalResult r;
    r.testName = "Evidence Quality";
    r.category = "identity";
    r.timestamp = QDateTime::currentDateTime();

    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    if (traits.isEmpty()) {
        r.passed = true;
        r.score = 1.0;
        r.details = "No traits to evaluate";
        return r;
    }

    int withEvidence = 0;
    int totalEvidence = 0;
    for (const TraitRecord &t : traits) {
        if (!t.evidenceEpisodeIds.isEmpty()) {
            withEvidence++;
            totalEvidence += t.evidenceEpisodeIds.size();
        }
    }

    double evidenceRatio = static_cast<double>(withEvidence) / traits.size();
    double avgEvidence = traits.isEmpty() ? 0 : static_cast<double>(totalEvidence) / traits.size();

    r.score = evidenceRatio * 0.7 + qMin(1.0, avgEvidence / 3.0) * 0.3;
    r.passed = r.score >= 0.3;
    r.details = QString("%1/%2 traits have evidence, avg %3 per trait")
                .arg(withEvidence).arg(traits.size())
                .arg(avgEvidence, 0, 'f', 1);
    return r;
}

// --- Curiosity tests ---

EvalResult EvalSuite::testInquiryFrequency()
{
    EvalResult r;
    r.testName = "Inquiry Frequency";
    r.category = "curiosity";
    r.timestamp = QDateTime::currentDateTime();

    // Count episodes with inquiry_text set
    QList<EpisodicRecord> episodes = m_memoryStore->getRecentEpisodes(100);
    int withInquiry = 0;
    for (const EpisodicRecord &ep : episodes) {
        if (!ep.inquiryText.isEmpty()) withInquiry++;
    }

    if (episodes.isEmpty()) {
        r.passed = true;
        r.score = 1.0;
        r.details = "No episodes yet";
        return r;
    }

    double inquiryRate = static_cast<double>(withInquiry) / episodes.size();

    // Ideal: 15-35% of interactions include an inquiry
    if (inquiryRate >= 0.15 && inquiryRate <= 0.35) {
        r.score = 1.0;
    } else if (inquiryRate < 0.15) {
        r.score = inquiryRate / 0.15; // Too few inquiries
    } else {
        r.score = qMax(0.0, 1.0 - (inquiryRate - 0.35) * 3.0); // Too many
    }

    r.passed = r.score >= 0.5;
    r.details = QString("Inquiry rate: %1% (%2/%3 episodes)")
                .arg(static_cast<int>(inquiryRate * 100))
                .arg(withInquiry).arg(episodes.size());
    return r;
}

EvalResult EvalSuite::testInquiryResponseCapture()
{
    EvalResult r;
    r.testName = "Inquiry Response Capture";
    r.category = "curiosity";
    r.timestamp = QDateTime::currentDateTime();

    QList<EpisodicRecord> episodes = m_memoryStore->getRecentEpisodes(100);
    int inquiries = 0;
    int withFollowup = 0;

    for (int i = 0; i < episodes.size(); ++i) {
        if (!episodes[i].inquiryText.isEmpty()) {
            inquiries++;
            // Check if the next episode has a longer-than-average user response
            // (indicating they answered the question)
            if (i + 1 < episodes.size() && episodes[i + 1].userText.length() > 30) {
                withFollowup++;
            }
        }
    }

    if (inquiries == 0) {
        r.passed = true;
        r.score = 1.0;
        r.details = "No inquiries yet";
        return r;
    }

    r.score = static_cast<double>(withFollowup) / inquiries;
    r.passed = r.score >= 0.3;
    r.details = QString("%1/%2 inquiries got substantive responses")
                .arg(withFollowup).arg(inquiries);
    return r;
}

EvalResult EvalSuite::testTopicDiversity()
{
    EvalResult r;
    r.testName = "Topic Diversity";
    r.category = "curiosity";
    r.timestamp = QDateTime::currentDateTime();

    QList<EpisodicRecord> episodes = m_memoryStore->getRecentEpisodes(50);
    QSet<QString> topics;
    for (const EpisodicRecord &ep : episodes) {
        if (!ep.topic.isEmpty()) topics.insert(ep.topic.toLower());
    }

    if (episodes.isEmpty()) {
        r.passed = true;
        r.score = 1.0;
        r.details = "No episodes yet";
        return r;
    }

    // More unique topics = better diversity
    double diversity = episodes.isEmpty() ? 0
        : static_cast<double>(topics.size()) / qMin(episodes.size(), 20);

    r.score = qMin(1.0, diversity);
    r.passed = r.score >= 0.2;
    r.details = QString("%1 unique topics across %2 episodes")
                .arg(topics.size()).arg(episodes.size());
    return r;
}

// --- Stability tests ---

EvalResult EvalSuite::testMemoryGrowth()
{
    EvalResult r;
    r.testName = "Memory Growth";
    r.category = "stability";
    r.timestamp = QDateTime::currentDateTime();

    int episodes = m_memoryStore->episodeCount();
    int traits = m_memoryStore->traitCount();

    // Just check that memory is accumulating
    r.score = (episodes > 0) ? 1.0 : 0.0;
    r.passed = episodes > 0;
    r.details = QString("%1 episodes, %2 traits stored").arg(episodes).arg(traits);
    return r;
}

EvalResult EvalSuite::testNoTraitDrift()
{
    EvalResult r;
    r.testName = "No Trait Drift";
    r.category = "stability";
    r.timestamp = QDateTime::currentDateTime();

    // Check that high-confidence traits haven't changed recently
    QList<TraitRecord> traits = m_memoryStore->getAllTraits();
    int stableHighConf = 0;
    int totalHighConf = 0;

    for (const TraitRecord &t : traits) {
        if (t.confidence >= 0.7) {
            totalHighConf++;
            // A stable trait: lastConfirmed is recent OR hasn't been modified much
            if (t.lastConfirmedTs.isValid()) {
                stableHighConf++;
            }
        }
    }

    if (totalHighConf == 0) {
        r.passed = true;
        r.score = 1.0;
        r.details = "No high-confidence traits yet (too early to test drift)";
        return r;
    }

    r.score = static_cast<double>(stableHighConf) / totalHighConf;
    r.passed = r.score >= 0.7;
    r.details = QString("%1/%2 high-confidence traits are stable")
                .arg(stableHighConf).arg(totalHighConf);
    return r;
}

EvalResult EvalSuite::testFeedbackBalance()
{
    EvalResult r;
    r.testName = "Feedback Balance";
    r.category = "stability";
    r.timestamp = QDateTime::currentDateTime();

    QList<EpisodicRecord> episodes = m_memoryStore->getRecentEpisodes(100);
    int positive = 0, negative = 0, neutral = 0;

    for (const EpisodicRecord &ep : episodes) {
        if (ep.userFeedback > 0) positive++;
        else if (ep.userFeedback < 0) negative++;
        else neutral++;
    }

    int total = positive + negative;
    if (total == 0) {
        r.passed = true;
        r.score = 1.0;
        r.details = "No feedback yet";
        return r;
    }

    double positiveRate = static_cast<double>(positive) / total;
    r.score = positiveRate;
    r.passed = positiveRate >= 0.5; // At least 50% positive
    r.details = QString("Feedback: %1 positive, %2 negative, %3 none")
                .arg(positive).arg(negative).arg(neutral);
    return r;
}

QVariantList EvalSuite::getLastReportForQml() const
{
    QVariantList list;
    for (const EvalResult &r : m_lastReport.results) {
        QVariantMap map;
        map["testName"] = r.testName;
        map["category"] = r.category;
        map["passed"] = r.passed;
        map["score"] = r.score;
        map["details"] = r.details;
        list.append(map);
    }
    return list;
}
