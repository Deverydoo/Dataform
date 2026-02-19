#ifndef RESEARCHSTORE_H
#define RESEARCHSTORE_H

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QList>
#include <QVariantList>
#include <QVariantMap>

class MemoryStore;

struct ResearchFinding {
    qint64 findingId = -1;
    QDateTime timestamp;
    QString topic;
    QString searchQuery;
    QString sourceUrl;
    QString sourceTitle;
    QString rawSnippet;
    QString llmSummary;
    QString relevanceReason;
    int status = 0;                 // 0=pending, 1=approved, -1=rejected
    double relevanceScore = 0.0;
    QString modelId;
};

class ResearchStore : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int pendingCount READ pendingCount NOTIFY countsChanged)
    Q_PROPERTY(int approvedCount READ approvedCount NOTIFY countsChanged)
    Q_PROPERTY(int totalCount READ totalCount NOTIFY countsChanged)

public:
    explicit ResearchStore(QObject *parent = nullptr);

    void setMemoryStore(MemoryStore *store);
    bool initialize();

    // CRUD
    qint64 insertFinding(const ResearchFinding &finding);
    bool updateFindingStatus(qint64 findingId, int status);
    ResearchFinding getFinding(qint64 id) const;

    // Queries
    QList<ResearchFinding> getPendingFindings() const;
    QList<ResearchFinding> getApprovedFindings() const;
    QList<ResearchFinding> getRecentFindings(int limit = 20) const;
    QList<ResearchFinding> getApprovedFindingsForTopic(const QString &topic,
                                                        int limit = 3) const;
    QList<ResearchFinding> searchFindings(const QString &query, int limit = 10) const;
    bool hasRecentResearch(const QString &topic, int withinHours) const;

    // Counts
    int pendingCount() const;
    int approvedCount() const;
    int totalCount() const;

    // QML interface
    Q_INVOKABLE QVariantList getPendingFindingsForQml() const;
    Q_INVOKABLE QVariantList getApprovedFindingsForQml() const;
    Q_INVOKABLE QVariantList getRecentFindingsForQml(int limit = 20) const;
    Q_INVOKABLE void approveFinding(qint64 findingId);
    Q_INVOKABLE void rejectFinding(qint64 findingId);
    Q_INVOKABLE void approveAllPending();

signals:
    void countsChanged();
    void findingInserted(qint64 findingId);
    void findingStatusChanged(qint64 findingId, int status);

private:
    QVariantMap findingToVariantMap(const ResearchFinding &finding) const;

    MemoryStore *m_memoryStore = nullptr;
    bool m_initialized = false;
};

#endif // RESEARCHSTORE_H
