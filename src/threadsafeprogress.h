#ifndef THREADSAFEPROGRESS_H
#define THREADSAFEPROGRESS_H

#include <QMutex>
#include <QMutexLocker>
#include <QString>

// Thread-safe progress state container for components that run
// background threads and expose progress via Q_PROPERTY.
// Write from background thread, read from main thread.
class ThreadSafeProgress
{
public:
    ThreadSafeProgress() = default;

    // --- Writers (call from background thread) ---

    void setCurrentStep(int step) {
        QMutexLocker lock(&m_mutex);
        m_currentStep = step;
    }

    void setTotalSteps(int total) {
        QMutexLocker lock(&m_mutex);
        m_totalSteps = total;
    }

    void setCurrentLoss(float loss) {
        QMutexLocker lock(&m_mutex);
        m_currentLoss = loss;
    }

    void setStatus(const QString &status) {
        QMutexLocker lock(&m_mutex);
        m_status = status;
    }

    // --- Readers (call from main thread / QML) ---

    int currentStep() const {
        QMutexLocker lock(&m_mutex);
        return m_currentStep;
    }

    int totalSteps() const {
        QMutexLocker lock(&m_mutex);
        return m_totalSteps;
    }

    float currentLoss() const {
        QMutexLocker lock(&m_mutex);
        return m_currentLoss;
    }

    QString status() const {
        QMutexLocker lock(&m_mutex);
        return m_status;
    }

    // --- Bulk reset ---

    void reset() {
        QMutexLocker lock(&m_mutex);
        m_currentStep = 0;
        m_totalSteps = 0;
        m_currentLoss = 0.0f;
        m_status.clear();
    }

private:
    mutable QMutex m_mutex;
    int m_currentStep = 0;
    int m_totalSteps = 0;
    float m_currentLoss = 0.0f;
    QString m_status;
};

#endif // THREADSAFEPROGRESS_H
