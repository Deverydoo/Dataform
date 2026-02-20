#include "backgroundjobmanager.h"
#include <QDebug>

BackgroundJobManager *BackgroundJobManager::instance()
{
    static BackgroundJobManager s_instance;
    return &s_instance;
}

BackgroundJobManager::BackgroundJobManager(QObject *parent)
    : QObject(parent)
{
    // Default: 2 background threads. ORT training and llama.cpp
    // have their own threads, so this pool is only for short
    // CPU-intensive tasks (search, data generation, file I/O).
    m_pool.setMaxThreadCount(2);
}

BackgroundJobManager::~BackgroundJobManager()
{
    cancelAll();
    m_pool.waitForDone(5000);
}

int BackgroundJobManager::submitVoid(const QString &tag,
                                      std::function<void()> workFn,
                                      std::function<void()> completionFn)
{
    int jobId = m_nextJobId.fetchAndAddRelaxed(1);
    m_activeCount.fetchAndAddRelaxed(1);
    emit activeJobCountChanged();

    auto *runnable = new GenericRunnable;
    runnable->setAutoDelete(true);
    runnable->m_fn = [this, jobId, tag, workFn = std::move(workFn),
                       completionFn = std::move(completionFn)]() {
        if (isCancelled(jobId)) {
            decrementActive();
            return;
        }
        try {
            workFn();
            QMetaObject::invokeMethod(this, [this, jobId, tag, completionFn]() {
                if (!isCancelled(jobId) && completionFn) {
                    completionFn();
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

void BackgroundJobManager::cancel(int jobId)
{
    QMutexLocker lock(&m_cancelMutex);
    m_cancelledJobs.insert(jobId);
}

void BackgroundJobManager::cancelAll()
{
    m_pool.clear();
    QMutexLocker lock(&m_cancelMutex);
    for (int i = 1; i < m_nextJobId.loadRelaxed(); ++i) {
        m_cancelledJobs.insert(i);
    }
}

bool BackgroundJobManager::waitForDone(int msTimeout)
{
    return m_pool.waitForDone(msTimeout);
}

bool BackgroundJobManager::isCancelled(int jobId) const
{
    QMutexLocker lock(&m_cancelMutex);
    return m_cancelledJobs.contains(jobId);
}

void BackgroundJobManager::setMaxThreadCount(int count)
{
    m_pool.setMaxThreadCount(count);
}

void BackgroundJobManager::decrementActive()
{
    m_activeCount.fetchAndSubRelaxed(1);
    emit activeJobCountChanged();
}
