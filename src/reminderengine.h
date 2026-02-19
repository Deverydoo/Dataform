#ifndef REMINDERENGINE_H
#define REMINDERENGINE_H

#include <QObject>
#include <QString>
#include <QTimer>
#include <QDateTime>

class MemoryStore;
class ThoughtEngine;
class SettingsManager;

class ReminderEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int pendingCount READ pendingCount NOTIFY pendingCountChanged)

public:
    explicit ReminderEngine(QObject *parent = nullptr);
    ~ReminderEngine();

    void setMemoryStore(MemoryStore *store);
    void setThoughtEngine(ThoughtEngine *engine);
    void setSettingsManager(SettingsManager *settings);

    int pendingCount() const { return m_pendingCount; }

    Q_INVOKABLE void addReminder(const QString &content, const QString &dueTsStr = QString());
    void detectAndStore(const QString &text, qint64 episodeId);

public slots:
    void start();
    void stop();
    void checkReminders();

signals:
    void reminderTriggered(qint64 id, const QString &content);
    void pendingCountChanged();
    void reminderStored(const QString &content, const QString &dueTs);

private:
    QDateTime parseTimeExpression(const QString &text) const;
    void refreshPendingCount();

    MemoryStore *m_memoryStore = nullptr;
    ThoughtEngine *m_thoughtEngine = nullptr;
    SettingsManager *m_settingsManager = nullptr;
    QTimer *m_pollTimer = nullptr;
    int m_pendingCount = 0;
};

#endif // REMINDERENGINE_H
