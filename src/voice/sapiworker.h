#ifndef SAPIWORKER_H
#define SAPIWORKER_H

// QtGlobal defines Q_OS_WIN, so it has to land before the platform check.
#include <QtGlobal>

#if defined(DATAFORM_VOICE_ENABLED) && defined(Q_OS_WIN)

#include <QObject>
#include <QString>
#include <QStringList>
#include <QAtomicInt>

struct ISpVoice;

/**
 * Owns the SAPI COM voice on a dedicated thread.
 *
 * Only sapi.h is used: sphelper.h and ATL are avoided because neither is
 * guaranteed to be installed with the MSVC toolchain, and the token enumeration
 * they provide is short enough to do directly.
 *
 * Speaking starts asynchronously and is then waited on in short slices, which is
 * what lets a cancel from the UI thread stop playback inside the response budget
 * rather than after the current sentence.
 */
class SapiWorker : public QObject
{
    Q_OBJECT

public:
    SapiWorker(QString voiceName, int rate);
    ~SapiWorker() override;

    /** Set from the owning thread, polled by the speak loop. */
    QAtomicInt cancelRequested{0};

    /** Enumerates installed voice names. Safe to call before initialise(). */
    static QStringList availableVoices();

    /**
     * Outcome of initialise(), read directly rather than signalled.
     *
     * The owner invokes initialise() with a blocking queued call so that
     * isAvailable() is meaningful the moment the backend is constructed. A signal
     * could not serve that: it would be queued to the owner's event loop and so
     * arrive only after construction had already reported "unavailable".
     * Reading these after the blocking call returns is safe — that call
     * establishes the ordering, and the worker is idle by then.
     */
    bool initOk() const { return m_initOk; }
    QString initMessage() const { return m_initMessage; }

public slots:
    void initialise();
    void speak(quint64 requestId, const QString &text);
    void shutdown();

signals:
    void speechFinished(quint64 requestId);
    void speechError(quint64 requestId, const QString &message);

private:
    bool selectVoice(const QString &wanted);

    ISpVoice *m_voice = nullptr;
    QString m_voiceName;
    int m_rate = 0;
    bool m_comReady = false;
    bool m_initOk = false;
    QString m_initMessage;
};

#endif // DATAFORM_VOICE_ENABLED && Q_OS_WIN
#endif // SAPIWORKER_H
