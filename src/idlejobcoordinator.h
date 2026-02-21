#ifndef IDLEJOBCOORDINATOR_H
#define IDLEJOBCOORDINATOR_H

#include <QObject>
#include <QString>
#include <QList>
#include <QTimer>
#include <functional>

class IdleJobCoordinator : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString activeEngine READ activeEngine NOTIFY activeEngineChanged)
    Q_PROPERTY(int pendingEngineCount READ pendingEngineCount NOTIFY pendingEngineCountChanged)

public:
    explicit IdleJobCoordinator(QObject *parent = nullptr);
    ~IdleJobCoordinator();

    struct EngineEntry {
        QString name;
        int priority;                   // lower = runs first
        std::function<bool()> canStart; // engine's canStartCycle() check
        std::function<void()> start;    // calls engine's requestStart()
    };

    void registerEngine(const QString &name, int priority,
                        std::function<bool()> canStart,
                        std::function<void()> start);

    QString activeEngine() const { return m_activeEngineName; }
    int pendingEngineCount() const;

public slots:
    void onIdleWindowOpened();
    void onIdleWindowClosed();
    void onEngineCycleFinished(const QString &name);
    void onForegroundBusy();
    void onForegroundIdle();

signals:
    void activeEngineChanged();
    void pendingEngineCountChanged();

private:
    void scheduleNextEngine();
    void onEngineTimeout();

    QList<EngineEntry> m_engines;       // priority-sorted
    QString m_activeEngineName;
    bool m_idleWindowOpen = false;
    bool m_foregroundBusy = false;
    int m_currentIndex = 0;             // round-robin position
    QTimer *m_scheduleTimer = nullptr;
    QTimer *m_watchdogTimer = nullptr;

    static constexpr int INTER_ENGINE_DELAY_MS = 2000;
    static constexpr int FULL_ROUND_DELAY_MS = 30000;
    static constexpr int ENGINE_TIMEOUT_MS = 120000;  // 2 min safety net
};

#endif // IDLEJOBCOORDINATOR_H
