#include "idlescheduler.h"
#include <QDebug>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

IdleScheduler::IdleScheduler(QObject *parent)
    : QObject(parent)
    , m_pollTimer(new QTimer(this))
{
    connect(m_pollTimer, &QTimer::timeout, this, &IdleScheduler::pollSystemState);
}

IdleScheduler::~IdleScheduler()
{
    stop();
}

void IdleScheduler::setMemoryStore(MemoryStore *store)
{
    m_memoryStore = store;
}

void IdleScheduler::setComputeBudgetPercent(int percent)
{
    percent = qBound(1, percent, 50);
    if (m_computeBudgetPercent != percent) {
        m_computeBudgetPercent = percent;
        emit computeBudgetPercentChanged();
    }
}

void IdleScheduler::setEnabled(bool enabled)
{
    if (m_enabled != enabled) {
        m_enabled = enabled;
        emit enabledChanged();

        if (!enabled && m_isSchedulerActive) {
            m_isSchedulerActive = false;
            emit isSchedulerActiveChanged();
            emit idleWindowClosed();
        }

        m_schedulerStatus = buildStatusString();
        emit schedulerStatusChanged();
    }
}

void IdleScheduler::start()
{
    if (m_pollTimer->isActive()) return;

    m_pollTimer->start(POLL_INTERVAL_MS);
    m_schedulerStatus = m_enabled ? "Monitoring" : "Disabled";
    emit schedulerStatusChanged();
    qDebug() << "IdleScheduler started, polling every" << POLL_INTERVAL_MS << "ms";
}

void IdleScheduler::stop()
{
    m_pollTimer->stop();

    if (m_isSchedulerActive) {
        m_isSchedulerActive = false;
        emit isSchedulerActiveChanged();
        emit idleWindowClosed();
    }

    m_schedulerStatus = "Stopped";
    emit schedulerStatusChanged();
    qDebug() << "IdleScheduler stopped";
}

void IdleScheduler::pauseImmediately()
{
    // Called on any user interaction for instant deactivation
    if (m_isSchedulerActive) {
        m_isSchedulerActive = false;
        m_schedulerStatus = "Paused - user active";
        emit isSchedulerActiveChanged();
        emit schedulerStatusChanged();
        emit idleWindowClosed();
    }

    // Reset idle time tracking
    m_idleTimeSeconds = 0;
    m_isIdle = false;
    emit idleTimeSecondsChanged();
    emit isIdleChanged();
}

void IdleScheduler::pollSystemState()
{
    // Query system state
    int idleMs = queryIdleTimeMs();
    int newIdleSeconds = idleMs / 1000;
    bool newPluggedIn = queryPluggedIn();
    double newTemp = queryCpuTemperature();

    // Update idle time
    if (m_idleTimeSeconds != newIdleSeconds) {
        m_idleTimeSeconds = newIdleSeconds;
        emit idleTimeSecondsChanged();
    }

    // Update idle state
    bool newIdle = (m_idleTimeSeconds >= IDLE_THRESHOLD_SECONDS);
    if (m_isIdle != newIdle) {
        m_isIdle = newIdle;
        emit isIdleChanged();
    }

    // Update plugged-in state
    if (m_isPluggedIn != newPluggedIn) {
        m_isPluggedIn = newPluggedIn;
        emit isPluggedInChanged();
    }

    // Update temperature
    if (qAbs(m_cpuTemperature - newTemp) > 0.5) {
        m_cpuTemperature = newTemp;
        emit cpuTemperatureChanged();
    }

    // Evaluate whether scheduler should be active
    evaluateConditions();
}

void IdleScheduler::evaluateConditions()
{
    bool shouldBeActive = m_enabled
                          && m_isIdle
                          && m_isPluggedIn
                          && (m_cpuTemperature < THERMAL_LIMIT_CELSIUS || m_cpuTemperature == 0.0);

    if (shouldBeActive && !m_isSchedulerActive) {
        m_isSchedulerActive = true;
        m_schedulerStatus = "Active - idle window open";
        emit isSchedulerActiveChanged();
        emit schedulerStatusChanged();
        emit idleWindowOpened();
        qDebug() << "Idle window OPENED - conditions met for background work";
    } else if (!shouldBeActive && m_isSchedulerActive) {
        m_isSchedulerActive = false;
        m_schedulerStatus = buildStatusString();
        emit isSchedulerActiveChanged();
        emit schedulerStatusChanged();
        emit idleWindowClosed();
        qDebug() << "Idle window CLOSED -" << m_schedulerStatus;
    } else if (!m_isSchedulerActive) {
        // Update status string even when not transitioning
        QString newStatus = buildStatusString();
        if (m_schedulerStatus != newStatus) {
            m_schedulerStatus = newStatus;
            emit schedulerStatusChanged();
        }
    }
}

QString IdleScheduler::buildStatusString() const
{
    if (!m_enabled) return "Disabled";
    if (m_isSchedulerActive) return "Active - idle window open";
    if (!m_isIdle) return QString("Waiting - user active (%1s idle)").arg(m_idleTimeSeconds);
    if (!m_isPluggedIn) return "Waiting - on battery";
    if (m_cpuTemperature >= THERMAL_LIMIT_CELSIUS && m_cpuTemperature > 0.0)
        return QString("Waiting - thermal limit (%1C)").arg(m_cpuTemperature, 0, 'f', 1);
    return "Monitoring";
}

int IdleScheduler::queryIdleTimeMs() const
{
#ifdef Q_OS_WIN
    LASTINPUTINFO lii;
    lii.cbSize = sizeof(LASTINPUTINFO);
    if (GetLastInputInfo(&lii)) {
        DWORD tickCount = GetTickCount();
        return static_cast<int>(tickCount - lii.dwTime);
    }
    return 0;
#else
    return 0;
#endif
}

bool IdleScheduler::queryPluggedIn() const
{
#ifdef Q_OS_WIN
    SYSTEM_POWER_STATUS sps;
    if (GetSystemPowerStatus(&sps)) {
        // ACLineStatus: 0 = battery, 1 = AC, 255 = unknown
        return sps.ACLineStatus == 1 || sps.ACLineStatus == 255;
    }
    return true; // Assume plugged in if query fails (desktop PC)
#else
    return true;
#endif
}

double IdleScheduler::queryCpuTemperature() const
{
    // CPU temperature via WMI requires admin privileges on many systems.
    // For Phase 0, return 0.0 which the thermal guard treats as "no data = OK".
    // Phase 1+ can integrate OpenHardwareMonitor or WMI queries.
    return 0.0;
}
