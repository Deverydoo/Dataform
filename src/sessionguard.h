#ifndef SESSIONGUARD_H
#define SESSIONGUARD_H

#include <QReadWriteLock>
#include <QReadLocker>
#include <QWriteLocker>

// Protects a session/model pointer shared between main thread
// (model load/unload) and background thread (inference/generation).
//
// Background thread: QReadLocker lock(guard.rwLock());
//   -- Shared access; multiple background ops can overlap.
//   -- Blocks if a write lock is held (model being swapped).
//
// Main thread model swap: QWriteLocker lock(guard.rwLock());
//   -- Exclusive access; blocks until background thread releases.
class SessionGuard
{
public:
    SessionGuard() = default;

    QReadWriteLock *rwLock() { return &m_lock; }

private:
    QReadWriteLock m_lock;
};

#endif // SESSIONGUARD_H
