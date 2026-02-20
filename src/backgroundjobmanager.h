#ifndef BACKGROUNDJOBMANAGER_H
#define BACKGROUNDJOBMANAGER_H

#include <QObject>
#include <QThreadPool>
#include <QRunnable>
#include <QAtomicInt>
#include <QMutex>
#include <QMutexLocker>
#include <QSet>
#include <QMetaObject>
#include <functional>

// A lightweight job queue that runs CPU-intensive work off the main
// thread and delivers results back via Qt signals.
//
// Usage:
//   auto jobId = BackgroundJobManager::instance()->submit<QList<SearchResult>>(
//       "semantic_search",
//       [cache, queryVec]() -> QList<SearchResult> {
//           // runs on thread pool
//           return computeSearch(cache, queryVec);
//       },
//       [this](const QList<SearchResult> &results) {
//           // runs on main thread
//           handleResults(results);
//       }
//   );
//
// Cancel:
//   BackgroundJobManager::instance()->cancel(jobId);
//
// Shutdown:
//   BackgroundJobManager::instance()->cancelAll();
//   BackgroundJobManager::instance()->waitForDone(5000);

class BackgroundJobManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int activeJobCount READ activeJobCount NOTIFY activeJobCountChanged)

public:
    static BackgroundJobManager *instance();

    // Submit a typed job. Returns a job ID for cancellation.
    // WorkFn runs on a thread pool thread.
    // ResultFn runs on the main thread (via QueuedConnection).
    template<typename ResultType>
    int submit(const QString &tag,
               std::function<ResultType()> workFn,
               std::function<void(const ResultType &)> resultFn);

    // Submit a void job (no result to deliver).
    int submitVoid(const QString &tag,
                   std::function<void()> workFn,
                   std::function<void()> completionFn = nullptr);

    // Cancel a job by ID. If running, the work function must check isCancelled().
    void cancel(int jobId);

    // Cancel all pending and running jobs.
    void cancelAll();

    // Wait for all jobs to complete (for shutdown).
    bool waitForDone(int msTimeout = 5000);

    // Check if a specific job was cancelled (callable from work function).
    bool isCancelled(int jobId) const;

    int activeJobCount() const { return m_activeCount.loadRelaxed(); }

    // Configure thread pool size (default: 2 threads for background work)
    void setMaxThreadCount(int count);

signals:
    void activeJobCountChanged();
    void jobCompleted(int jobId, const QString &tag);
    void jobFailed(int jobId, const QString &tag, const QString &error);

private:
    explicit BackgroundJobManager(QObject *parent = nullptr);
    ~BackgroundJobManager();

    // Type-erased runnable for the thread pool
    class GenericRunnable : public QRunnable {
    public:
        std::function<void()> m_fn;
        void run() override { if (m_fn) m_fn(); }
    };

    void decrementActive();

    QThreadPool m_pool;
    QAtomicInt m_nextJobId{1};
    QAtomicInt m_activeCount{0};

    mutable QMutex m_cancelMutex;
    QSet<int> m_cancelledJobs;

    Q_DISABLE_COPY(BackgroundJobManager)
};

// ---- Template implementation ----

template<typename ResultType>
int BackgroundJobManager::submit(const QString &tag,
                                  std::function<ResultType()> workFn,
                                  std::function<void(const ResultType &)> resultFn)
{
    int jobId = m_nextJobId.fetchAndAddRelaxed(1);
    m_activeCount.fetchAndAddRelaxed(1);
    emit activeJobCountChanged();

    auto *runnable = new GenericRunnable;
    runnable->setAutoDelete(true);
    runnable->m_fn = [this, jobId, tag, workFn = std::move(workFn),
                       resultFn = std::move(resultFn)]() {
        if (isCancelled(jobId)) {
            decrementActive();
            return;
        }
        try {
            ResultType result = workFn();
            QMetaObject::invokeMethod(this, [this, jobId, tag,
                                              result = std::move(result),
                                              resultFn]() {
                if (!isCancelled(jobId) && resultFn) {
                    resultFn(result);
                }
                decrementActive();
                emit jobCompleted(jobId, tag);
            }, Qt::QueuedConnection);
        } catch (const std::exception &e) {
            QString error = QString::fromStdString(e.what());
            QMetaObject::invokeMethod(this, [this, jobId, tag, error]() {
                decrementActive();
                emit jobFailed(jobId, tag, error);
            }, Qt::QueuedConnection);
        }
    };

    m_pool.start(runnable);
    return jobId;
}

#endif // BACKGROUNDJOBMANAGER_H
