#ifndef TTSBACKENDS_H
#define TTSBACKENDS_H

#ifdef DATAFORM_VOICE_ENABLED

#include "speechsynthesizer.h"

#include <QString>
#include <QThread>
#include <QAtomicInt>
#include <QByteArray>

class QProcess;
class SapiWorker;

/**
 * Piper text-to-speech, driven as a child process.
 *
 * Piper is invoked once per utterance with --output_raw and its PCM is forwarded
 * as it arrives, so playback starts well before synthesis finishes. Running a
 * process per utterance costs one model load (roughly 100-200 ms for a medium
 * voice) but buys exact cancellation: killing the process stops audio at once,
 * with no partially-consumed model state to reset.
 */
class PiperBackend : public TtsBackend
{
    Q_OBJECT

public:
    /**
     * @param exePath   Path to piper.exe
     * @param modelPath Path to the voice .onnx; its .onnx.json sibling supplies the sample rate
     * @param speed     Length scale multiplier; >1 is slower speech
     */
    PiperBackend(const QString &exePath, const QString &modelPath,
                 double speed = 1.0, QObject *parent = nullptr);
    ~PiperBackend() override;

    QString name() const override { return QStringLiteral("Piper"); }
    bool isAvailable() const override;
    QString unavailableReason() const override { return m_unavailableReason; }
    int sampleRate() const override { return m_sampleRate; }

    void speak(quint64 requestId, const QString &text) override;
    void cancel() override;

private:
    void readAvailableAudio();
    void finishProcess();

    QString m_exePath;
    QString m_modelPath;
    double m_speed = 1.0;
    int m_sampleRate = 22050;
    QString m_unavailableReason;

    QProcess *m_process = nullptr;
    quint64 m_currentRequest = 0;
    QByteArray m_carry;      // odd trailing byte between reads of 16-bit samples
    bool m_cancelled = false;
};

#ifdef Q_OS_WIN

/**
 * Windows SAPI voice.
 *
 * Present so the voice loop is usable before Piper is installed. SAPI drives the
 * output device itself, so it reports sampleRate() == 0 and emits no samples;
 * barge-in goes through its own purge rather than through AudioPlayback.
 */
class SapiBackend : public TtsBackend
{
    Q_OBJECT

public:
    explicit SapiBackend(const QString &voiceName = QString(), int rate = 0,
                         QObject *parent = nullptr);
    ~SapiBackend() override;

    QString name() const override { return QStringLiteral("Windows SAPI"); }
    bool isAvailable() const override { return m_available; }
    QString unavailableReason() const override { return m_unavailableReason; }
    int sampleRate() const override { return 0; }   // self-playing

    void speak(quint64 requestId, const QString &text) override;
    void cancel() override;

    static QStringList availableVoices();

signals:
    void requestSpeak(quint64 requestId, const QString &text);

private:
    QThread m_thread;
    SapiWorker *m_worker = nullptr;
    bool m_available = false;
    QString m_unavailableReason;
};

#endif // Q_OS_WIN

#endif // DATAFORM_VOICE_ENABLED
#endif // TTSBACKENDS_H
