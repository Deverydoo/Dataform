#include "idlejobcoordinator.h"
#include <QDebug>

IdleJobCoordinator::IdleJobCoordinator(QObject *parent)
    : QObject(parent)
{
    m_scheduleTimer = new QTimer(this);
    m_scheduleTimer->setSingleShot(true);
    connect(m_scheduleTimer, &QTimer::timeout, this, &IdleJobCoordinator::scheduleNextEngine);
}

IdleJobCoordinator::~IdleJobCoordinator()
{
}

void IdleJobCoordinator::registerEngine(const QString &name, int priority,
                                         std::function<bool()> canStart,
                                         std::function<void()> start)
{
    EngineEntry entry{name, priority, canStart, start};

    // Insert sorted by priority (lower = higher priority)
    int insertAt = m_engines.size();
    for (int i = 0; i < m_engines.size(); ++i) {
        if (m_engines[i].priority > priority) {
            insertAt = i;
            break;
        }
    }
    m_engines.insert(insertAt, entry);
    emit pendingEngineCountChanged();
}

int IdleJobCoordinator::pendingEngineCount() const
{
    int count = 0;
    for (const auto &e : m_engines) {
        if (e.canStart && e.canStart()) count++;
    }
    return count;
}

void IdleJobCoordinator::onIdleWindowOpened()
{
    m_idleWindowOpen = true;
    m_currentIndex = 0;
    qDebug() << "IdleJobCoordinator: idle window opened, scheduling engines";

    if (!m_activeEngineName.isEmpty()) {
        // An engine is still running from before — let it finish
        return;
    }

    scheduleNextEngine();
}

void IdleJobCoordinator::onIdleWindowClosed()
{
    m_idleWindowOpen = false;
    m_scheduleTimer->stop();
    qDebug() << "IdleJobCoordinator: idle window closed";

    if (!m_activeEngineName.isEmpty()) {
        qDebug() << "IdleJobCoordinator: engine" << m_activeEngineName
                 << "still running, will finish its cycle";
    }
}

void IdleJobCoordinator::onEngineCycleFinished(const QString &name)
{
    if (name != m_activeEngineName) {
        // Stale signal from a previous activation — ignore
        return;
    }

    qDebug() << "IdleJobCoordinator: engine" << name << "cycle finished";
    m_activeEngineName.clear();
    emit activeEngineChanged();

    if (!m_idleWindowOpen || m_foregroundBusy) {
        return;
    }

    // Schedule next engine after a breathing delay
    m_scheduleTimer->start(INTER_ENGINE_DELAY_MS);
}

void IdleJobCoordinator::onForegroundBusy()
{
    m_foregroundBusy = true;
    m_scheduleTimer->stop();
    qDebug() << "IdleJobCoordinator: foreground busy, pausing scheduling";
}

void IdleJobCoordinator::onForegroundIdle()
{
    m_foregroundBusy = false;
    qDebug() << "IdleJobCoordinator: foreground idle, resuming";

    if (m_idleWindowOpen && m_activeEngineName.isEmpty()) {
        // Resume scheduling after a short delay
        m_scheduleTimer->start(INTER_ENGINE_DELAY_MS);
    }
}

void IdleJobCoordinator::scheduleNextEngine()
{
    if (!m_idleWindowOpen || m_foregroundBusy) return;
    if (!m_activeEngineName.isEmpty()) return; // already running

    // Try each engine starting from current index
    int tried = 0;
    while (tried < m_engines.size()) {
        int idx = m_currentIndex % m_engines.size();
        m_currentIndex++;
        tried++;

        const EngineEntry &entry = m_engines[idx];
        if (entry.canStart && entry.canStart()) {
            m_activeEngineName = entry.name;
            emit activeEngineChanged();
            emit pendingEngineCountChanged();
            qDebug() << "IdleJobCoordinator: activating engine" << entry.name
                     << "(priority:" << entry.priority << ")";
            entry.start();
            return;
        }
    }

    // All engines either can't start or on cooldown — retry after full round delay
    qDebug() << "IdleJobCoordinator: no engines ready, retrying in"
             << FULL_ROUND_DELAY_MS / 1000 << "seconds";
    m_currentIndex = 0;
    m_scheduleTimer->start(FULL_ROUND_DELAY_MS);
}
