#include "reminderengine.h"
#include "memorystore.h"
#include "thoughtengine.h"
#include "settingsmanager.h"
#include <QDebug>
#include <QRegularExpression>

ReminderEngine::ReminderEngine(QObject *parent)
    : QObject(parent)
    , m_pollTimer(new QTimer(this))
{
    m_pollTimer->setInterval(60000); // 60 seconds
    connect(m_pollTimer, &QTimer::timeout, this, &ReminderEngine::checkReminders);
}

ReminderEngine::~ReminderEngine()
{
    stop();
}

void ReminderEngine::setMemoryStore(MemoryStore *store) { m_memoryStore = store; }
void ReminderEngine::setThoughtEngine(ThoughtEngine *engine) { m_thoughtEngine = engine; }
void ReminderEngine::setSettingsManager(SettingsManager *settings) { m_settingsManager = settings; }

void ReminderEngine::start()
{
    if (!m_pollTimer->isActive()) {
        m_pollTimer->start();
        qDebug() << "ReminderEngine: started polling every 60s";
    }
    refreshPendingCount();
}

void ReminderEngine::stop()
{
    m_pollTimer->stop();
}

void ReminderEngine::checkReminders()
{
    if (!m_memoryStore || !m_thoughtEngine) return;

    QList<ReminderRecord> due = m_memoryStore->getDueReminders();
    for (const ReminderRecord &r : due) {
        qDebug() << "ReminderEngine: triggering reminder" << r.id << r.content.left(50);

        ThoughtRecord thought;
        thought.type = "reminder";
        thought.title = r.content.left(80);
        thought.content = QString("You asked me to remind you: \"%1\"").arg(r.content);
        thought.priority = 0.9;
        thought.sourceType = "reminder";
        thought.sourceId = r.id;
        thought.generatedBy = "reminder_engine";
        m_thoughtEngine->insertThought(thought);

        m_memoryStore->updateReminderStatus(r.id, 1); // triggered
        emit reminderTriggered(r.id, r.content);
    }

    // Also check reminders with no due date that are older than 24h
    QList<ReminderRecord> pending = m_memoryStore->getPendingReminders();
    for (const ReminderRecord &r : pending) {
        if (!r.dueTs.isValid() && r.createdTs.isValid()) {
            qint64 hoursSince = r.createdTs.secsTo(QDateTime::currentDateTime()) / 3600;
            if (hoursSince >= 24) {
                ThoughtRecord thought;
                thought.type = "reminder";
                thought.title = r.content.left(80);
                thought.content = QString("You mentioned this a while ago: \"%1\". Did you want me to remind you about it?").arg(r.content);
                thought.priority = 0.7;
                thought.sourceType = "reminder";
                thought.sourceId = r.id;
                thought.generatedBy = "reminder_engine";
                m_thoughtEngine->insertThought(thought);

                m_memoryStore->updateReminderStatus(r.id, 1);
            }
        }
    }

    refreshPendingCount();
}

void ReminderEngine::addReminder(const QString &content, const QString &dueTsStr)
{
    if (!m_memoryStore) return;
    m_memoryStore->insertReminder(content, dueTsStr);
    refreshPendingCount();
    emit reminderStored(content, dueTsStr);
}

void ReminderEngine::detectAndStore(const QString &text, qint64 episodeId)
{
    if (!m_memoryStore) return;

    // Try to parse a time expression from the text
    QDateTime dueTime = parseTimeExpression(text);
    QString dueTsStr = dueTime.isValid() ? dueTime.toString(Qt::ISODate) : QString();

    // Clean up the reminder content
    QString content = text;
    // Remove common time phrases from the stored content
    static const QStringList timePhrases = {
        "tomorrow", "tonight", "next week", "in an hour",
        "in two hours", "in 2 hours", "in a day", "in a week"
    };
    QString lower = content.toLower();
    for (const QString &phrase : timePhrases) {
        int idx = lower.indexOf(phrase);
        if (idx >= 0) {
            // Keep the content before the time phrase if there's meaningful text
            QString before = content.left(idx).trimmed();
            if (before.length() > 3) {
                content = before;
            }
            break;
        }
    }

    // Remove leading "to " or "that "
    if (content.startsWith("to ", Qt::CaseInsensitive))
        content = content.mid(3);
    else if (content.startsWith("that ", Qt::CaseInsensitive))
        content = content.mid(5);

    content = content.trimmed();
    if (content.isEmpty()) content = text.trimmed();

    m_memoryStore->insertReminder(content, dueTsStr, episodeId);
    refreshPendingCount();

    QString timeDesc = dueTime.isValid()
        ? QString("for %1").arg(dueTime.toString("yyyy-MM-dd HH:mm"))
        : "with no specific time";
    qDebug() << "ReminderEngine: stored reminder" << content.left(50) << timeDesc;
    emit reminderStored(content, dueTsStr);
}

QDateTime ReminderEngine::parseTimeExpression(const QString &text) const
{
    QString lower = text.toLower();
    QDateTime now = QDateTime::currentDateTime();

    if (lower.contains("tomorrow")) {
        return now.addDays(1).date().startOfDay().addSecs(9 * 3600); // 9 AM tomorrow
    }
    if (lower.contains("tonight")) {
        return now.date().startOfDay().addSecs(20 * 3600); // 8 PM today
    }
    if (lower.contains("next week")) {
        return now.addDays(7).date().startOfDay().addSecs(9 * 3600);
    }

    // "in N hours/minutes/days"
    QRegularExpression inPattern("in\\s+(\\d+)\\s+(hour|minute|min|day|week)s?");
    auto match = inPattern.match(lower);
    if (match.hasMatch()) {
        int amount = match.captured(1).toInt();
        QString unit = match.captured(2);
        if (unit.startsWith("hour")) return now.addSecs(amount * 3600);
        if (unit.startsWith("min")) return now.addSecs(amount * 60);
        if (unit.startsWith("day")) return now.addDays(amount);
        if (unit.startsWith("week")) return now.addDays(amount * 7);
    }

    // "in an hour" / "in a day"
    if (lower.contains("in an hour") || lower.contains("in one hour"))
        return now.addSecs(3600);
    if (lower.contains("in a day") || lower.contains("in one day"))
        return now.addDays(1);
    if (lower.contains("in a week") || lower.contains("in one week"))
        return now.addDays(7);

    return QDateTime(); // No time expression found
}

void ReminderEngine::refreshPendingCount()
{
    if (!m_memoryStore) return;
    int count = m_memoryStore->getPendingReminders().size();
    if (m_pendingCount != count) {
        m_pendingCount = count;
        emit pendingCountChanged();
    }
}
