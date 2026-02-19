#ifndef IDLESCHEDULER_H
#define IDLESCHEDULER_H

#include <QObject>
#include <QTimer>
#include <QString>

class MemoryStore;

class IdleScheduler : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isIdle READ isIdle NOTIFY isIdleChanged)
    Q_PROPERTY(bool isPluggedIn READ isPluggedIn NOTIFY isPluggedInChanged)
    Q_PROPERTY(bool isSchedulerActive READ isSchedulerActive NOTIFY isSchedulerActiveChanged)
    Q_PROPERTY(double cpuTemperature READ cpuTemperature NOTIFY cpuTemperatureChanged)
    Q_PROPERTY(int idleTimeSeconds READ idleTimeSeconds NOTIFY idleTimeSecondsChanged)
    Q_PROPERTY(int computeBudgetPercent READ computeBudgetPercent WRITE setComputeBudgetPercent NOTIFY computeBudgetPercentChanged)
    Q_PROPERTY(QString schedulerStatus READ schedulerStatus NOTIFY schedulerStatusChanged)
    Q_PROPERTY(bool enabled READ enabled WRITE setEnabled NOTIFY enabledChanged)

public:
    explicit IdleScheduler(QObject *parent = nullptr);
    ~IdleScheduler();

    void setMemoryStore(MemoryStore *store);

    bool isIdle() const { return m_isIdle; }
    bool isPluggedIn() const { return m_isPluggedIn; }
    bool isSchedulerActive() const { return m_isSchedulerActive; }
    double cpuTemperature() const { return m_cpuTemperature; }
    int idleTimeSeconds() const { return m_idleTimeSeconds; }
    int computeBudgetPercent() const { return m_computeBudgetPercent; }
    QString schedulerStatus() const { return m_schedulerStatus; }
    bool enabled() const { return m_enabled; }

    void setComputeBudgetPercent(int percent);
    void setEnabled(bool enabled);

    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
    Q_INVOKABLE void pauseImmediately();

signals:
    void isIdleChanged();
    void isPluggedInChanged();
    void isSchedulerActiveChanged();
    void cpuTemperatureChanged();
    void idleTimeSecondsChanged();
    void computeBudgetPercentChanged();
    void schedulerStatusChanged();
    void enabledChanged();

    // Emitted when conditions are right for background work
    void idleWindowOpened();
    void idleWindowClosed();

private slots:
    void pollSystemState();

private:
    void evaluateConditions();
    int queryIdleTimeMs() const;
    bool queryPluggedIn() const;
    double queryCpuTemperature() const;
    QString buildStatusString() const;

    MemoryStore *m_memoryStore = nullptr;
    QTimer *m_pollTimer = nullptr;

    bool m_isIdle = false;
    bool m_isPluggedIn = true;
    bool m_isSchedulerActive = false;
    double m_cpuTemperature = 0.0;
    int m_idleTimeSeconds = 0;
    int m_computeBudgetPercent = 10;
    bool m_enabled = true;
    QString m_schedulerStatus = "Stopped";

    static constexpr int IDLE_THRESHOLD_SECONDS = 120;
    static constexpr double THERMAL_LIMIT_CELSIUS = 80.0;
    static constexpr int POLL_INTERVAL_MS = 5000;
};

#endif // IDLESCHEDULER_H
