#include "researchstore.h"
#include "memorystore.h"
#include <QSqlQuery>
#include <QSqlError>
#include <QDebug>

ResearchStore::ResearchStore(QObject *parent)
    : QObject(parent)
{
}

void ResearchStore::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

bool ResearchStore::initialize()
{
    if (!m_memoryStore) {
        qWarning() << "ResearchStore: no MemoryStore set";
        return false;
    }

    // The research_findings table is created by the episodic DB migration v1→v2
    // in MemoryStore::migrateEpisodicSchema(). We just verify it exists.
    QSqlDatabase db = m_memoryStore->episodicDatabase();
    if (!db.isOpen()) {
        qWarning() << "ResearchStore: episodic database not open";
        return false;
    }

    QStringList tables = db.tables();
    if (!tables.contains("research_findings")) {
        qWarning() << "ResearchStore: research_findings table not found, "
                       "will be created by next migration";
        return false;
    }

    m_initialized = true;
    qDebug() << "ResearchStore initialized. Pending:" << pendingCount()
             << "Approved:" << approvedCount();
    return true;
}

qint64 ResearchStore::insertFinding(const ResearchFinding &finding)
{
    if (!m_initialized) return -1;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.prepare("INSERT INTO research_findings "
              "(topic, search_query, source_url, source_title, raw_snippet, "
              "llm_summary, relevance_reason, status, relevance_score, model_id) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
    q.addBindValue(finding.topic);
    q.addBindValue(finding.searchQuery);
    q.addBindValue(finding.sourceUrl);
    q.addBindValue(finding.sourceTitle);
    q.addBindValue(finding.rawSnippet);
    q.addBindValue(finding.llmSummary);
    q.addBindValue(finding.relevanceReason);
    q.addBindValue(finding.status);
    q.addBindValue(finding.relevanceScore);
    q.addBindValue(finding.modelId);

    if (!q.exec()) {
        qWarning() << "ResearchStore: insert failed:" << q.lastError().text();
        return -1;
    }

    qint64 id = q.lastInsertId().toLongLong();
    emit findingInserted(id);
    emit countsChanged();
    return id;
}

bool ResearchStore::updateFindingStatus(qint64 findingId, int status)
{
    if (!m_initialized) return false;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.prepare("UPDATE research_findings SET status = ? WHERE id = ?");
    q.addBindValue(status);
    q.addBindValue(findingId);

    if (!q.exec()) {
        qWarning() << "ResearchStore: update status failed:" << q.lastError().text();
        return false;
    }

    emit findingStatusChanged(findingId, status);
    emit countsChanged();
    return true;
}

ResearchFinding ResearchStore::getFinding(qint64 id) const
{
    ResearchFinding finding;
    if (!m_initialized) return finding;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.prepare("SELECT * FROM research_findings WHERE id = ?");
    q.addBindValue(id);

    if (q.exec() && q.next()) {
        finding.findingId = q.value("id").toLongLong();
        finding.timestamp = QDateTime::fromString(q.value("timestamp").toString(), Qt::ISODate);
        finding.topic = q.value("topic").toString();
        finding.searchQuery = q.value("search_query").toString();
        finding.sourceUrl = q.value("source_url").toString();
        finding.sourceTitle = q.value("source_title").toString();
        finding.rawSnippet = q.value("raw_snippet").toString();
        finding.llmSummary = q.value("llm_summary").toString();
        finding.relevanceReason = q.value("relevance_reason").toString();
        finding.status = q.value("status").toInt();
        finding.relevanceScore = q.value("relevance_score").toDouble();
        finding.modelId = q.value("model_id").toString();
    }

    return finding;
}

QList<ResearchFinding> ResearchStore::getPendingFindings() const
{
    QList<ResearchFinding> findings;
    if (!m_initialized) return findings;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    if (!q.exec("SELECT * FROM research_findings WHERE status = 0 ORDER BY timestamp DESC")) {
        qWarning() << "ResearchStore: getPendingFindings query failed:" << q.lastError().text();
        return findings;
    }

    while (q.next()) {
        ResearchFinding f;
        f.findingId = q.value("id").toLongLong();
        f.timestamp = QDateTime::fromString(q.value("timestamp").toString(), Qt::ISODate);
        f.topic = q.value("topic").toString();
        f.searchQuery = q.value("search_query").toString();
        f.sourceUrl = q.value("source_url").toString();
        f.sourceTitle = q.value("source_title").toString();
        f.rawSnippet = q.value("raw_snippet").toString();
        f.llmSummary = q.value("llm_summary").toString();
        f.relevanceReason = q.value("relevance_reason").toString();
        f.status = q.value("status").toInt();
        f.relevanceScore = q.value("relevance_score").toDouble();
        f.modelId = q.value("model_id").toString();
        findings.append(f);
    }

    return findings;
}

QList<ResearchFinding> ResearchStore::getApprovedFindings() const
{
    QList<ResearchFinding> findings;
    if (!m_initialized) return findings;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    if (!q.exec("SELECT * FROM research_findings WHERE status = 1 "
           "ORDER BY relevance_score DESC, timestamp DESC")) {
        qWarning() << "ResearchStore: getApprovedFindings query failed:" << q.lastError().text();
        return findings;
    }

    while (q.next()) {
        ResearchFinding f;
        f.findingId = q.value("id").toLongLong();
        f.timestamp = QDateTime::fromString(q.value("timestamp").toString(), Qt::ISODate);
        f.topic = q.value("topic").toString();
        f.sourceUrl = q.value("source_url").toString();
        f.sourceTitle = q.value("source_title").toString();
        f.llmSummary = q.value("llm_summary").toString();
        f.relevanceReason = q.value("relevance_reason").toString();
        f.status = 1;
        f.relevanceScore = q.value("relevance_score").toDouble();
        findings.append(f);
    }

    return findings;
}

QList<ResearchFinding> ResearchStore::getRecentFindings(int limit) const
{
    QList<ResearchFinding> findings;
    if (!m_initialized) return findings;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.prepare("SELECT * FROM research_findings ORDER BY timestamp DESC LIMIT ?");
    q.addBindValue(limit);

    if (q.exec()) {
        while (q.next()) {
            ResearchFinding f;
            f.findingId = q.value("id").toLongLong();
            f.timestamp = QDateTime::fromString(q.value("timestamp").toString(), Qt::ISODate);
            f.topic = q.value("topic").toString();
            f.searchQuery = q.value("search_query").toString();
            f.sourceUrl = q.value("source_url").toString();
            f.sourceTitle = q.value("source_title").toString();
            f.llmSummary = q.value("llm_summary").toString();
            f.relevanceReason = q.value("relevance_reason").toString();
            f.status = q.value("status").toInt();
            f.relevanceScore = q.value("relevance_score").toDouble();
            findings.append(f);
        }
    }

    return findings;
}

QList<ResearchFinding> ResearchStore::getApprovedFindingsForTopic(const QString &topic,
                                                                    int limit) const
{
    QList<ResearchFinding> findings;
    if (!m_initialized || topic.isEmpty()) return findings;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.prepare("SELECT * FROM research_findings WHERE status = 1 "
              "AND (topic LIKE ? OR llm_summary LIKE ?) "
              "ORDER BY relevance_score DESC LIMIT ?");
    QString pattern = "%" + topic + "%";
    q.addBindValue(pattern);
    q.addBindValue(pattern);
    q.addBindValue(limit);

    if (q.exec()) {
        while (q.next()) {
            ResearchFinding f;
            f.findingId = q.value("id").toLongLong();
            f.topic = q.value("topic").toString();
            f.sourceUrl = q.value("source_url").toString();
            f.llmSummary = q.value("llm_summary").toString();
            f.relevanceReason = q.value("relevance_reason").toString();
            f.relevanceScore = q.value("relevance_score").toDouble();
            findings.append(f);
        }
    }

    return findings;
}

QList<ResearchFinding> ResearchStore::searchFindings(const QString &query, int limit) const
{
    QList<ResearchFinding> findings;
    if (!m_initialized || query.isEmpty()) return findings;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.prepare("SELECT * FROM research_findings "
              "WHERE llm_summary LIKE ? OR topic LIKE ? OR source_title LIKE ? "
              "ORDER BY timestamp DESC LIMIT ?");
    QString pattern = "%" + query + "%";
    q.addBindValue(pattern);
    q.addBindValue(pattern);
    q.addBindValue(pattern);
    q.addBindValue(limit);

    if (q.exec()) {
        while (q.next()) {
            ResearchFinding f;
            f.findingId = q.value("id").toLongLong();
            f.timestamp = QDateTime::fromString(q.value("timestamp").toString(), Qt::ISODate);
            f.topic = q.value("topic").toString();
            f.sourceUrl = q.value("source_url").toString();
            f.sourceTitle = q.value("source_title").toString();
            f.llmSummary = q.value("llm_summary").toString();
            f.status = q.value("status").toInt();
            f.relevanceScore = q.value("relevance_score").toDouble();
            findings.append(f);
        }
    }

    return findings;
}

bool ResearchStore::hasRecentResearch(const QString &topic, int withinHours) const
{
    if (!m_initialized) return false;

    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.prepare("SELECT COUNT(*) FROM research_findings "
              "WHERE topic LIKE ? AND timestamp > datetime('now', ? || ' hours')");
    q.addBindValue("%" + topic + "%");
    q.addBindValue(QString("-%1").arg(withinHours));

    if (q.exec() && q.next()) {
        return q.value(0).toInt() > 0;
    }

    return false;
}

int ResearchStore::pendingCount() const
{
    if (!m_initialized) return 0;
    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.exec("SELECT COUNT(*) FROM research_findings WHERE status = 0");
    return (q.next()) ? q.value(0).toInt() : 0;
}

int ResearchStore::approvedCount() const
{
    if (!m_initialized) return 0;
    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.exec("SELECT COUNT(*) FROM research_findings WHERE status = 1");
    return (q.next()) ? q.value(0).toInt() : 0;
}

int ResearchStore::totalCount() const
{
    if (!m_initialized) return 0;
    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.exec("SELECT COUNT(*) FROM research_findings");
    return (q.next()) ? q.value(0).toInt() : 0;
}

QVariantList ResearchStore::getPendingFindingsForQml() const
{
    QVariantList list;
    for (const ResearchFinding &f : getPendingFindings()) {
        list.append(findingToVariantMap(f));
    }
    return list;
}

QVariantList ResearchStore::getApprovedFindingsForQml() const
{
    QVariantList list;
    for (const ResearchFinding &f : getApprovedFindings()) {
        list.append(findingToVariantMap(f));
    }
    return list;
}

QVariantList ResearchStore::getRecentFindingsForQml(int limit) const
{
    QVariantList list;
    for (const ResearchFinding &f : getRecentFindings(limit)) {
        list.append(findingToVariantMap(f));
    }
    return list;
}

void ResearchStore::approveFinding(qint64 findingId)
{
    updateFindingStatus(findingId, 1);
}

void ResearchStore::rejectFinding(qint64 findingId)
{
    updateFindingStatus(findingId, -1);
}

void ResearchStore::approveAllPending()
{
    if (!m_initialized) return;
    QSqlDatabase db = m_memoryStore->episodicDatabase();
    QSqlQuery q(db);
    q.exec("UPDATE research_findings SET status = 1 WHERE status = 0");
    emit countsChanged();
}

QVariantMap ResearchStore::findingToVariantMap(const ResearchFinding &finding) const
{
    QVariantMap map;
    map["findingId"] = finding.findingId;
    map["timestamp"] = finding.timestamp.toString("yyyy-MM-dd HH:mm");
    map["topic"] = finding.topic;
    map["searchQuery"] = finding.searchQuery;
    map["sourceUrl"] = finding.sourceUrl;
    map["sourceTitle"] = finding.sourceTitle;
    map["rawSnippet"] = finding.rawSnippet;
    map["llmSummary"] = finding.llmSummary;
    map["relevanceReason"] = finding.relevanceReason;
    map["status"] = finding.status;
    map["relevanceScore"] = finding.relevanceScore;
    return map;
}
