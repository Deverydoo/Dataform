#ifndef SPEECHSYNTHESIZER_H
#define SPEECHSYNTHESIZER_H

#ifdef DATAFORM_VOICE_ENABLED

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QQueue>
#include <QJsonObject>
#include <memory>

/**
 * Backend contract for speech synthesis.
 *
 * Backends stream: they emit audio as it is produced so playback can start on the
 * first syllable rather than the last. A backend that owns its own output device
 * (SAPI) reports sampleRate() == 0 and emits no samples.
 */
class TtsBackend : public QObject
{
    Q_OBJECT

public:
    explicit TtsBackend(QObject *parent = nullptr) : QObject(parent) {}
    ~TtsBackend() override = default;

    virtual QString name() const = 0;
    virtual bool isAvailable() const = 0;
    virtual QString unavailableReason() const { return QString(); }

    /** 0 when the backend plays audio itself instead of returning samples. */
    virtual int sampleRate() const = 0;

    virtual void speak(quint64 requestId, const QString &text) = 0;
    virtual void cancel() = 0;

signals:
    void samplesReady(quint64 requestId, const QVector<float> &samples);
    void speechFinished(quint64 requestId);
    void speechError(quint64 requestId, const QString &message);
};

/**
 * Text to speech for assistant replies.
 *
 * Sentences are queued and synthesised one at a time so a long reply starts
 * speaking early, and so a barge-in drops only what has not been spoken yet.
 */
class SpeechSynthesizer : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isSpeaking READ isSpeaking NOTIFY isSpeakingChanged)
    Q_PROPERTY(bool isAvailable READ isAvailable NOTIFY backendChanged)
    Q_PROPERTY(QString backendName READ backendName NOTIFY backendChanged)
    Q_PROPERTY(QString unavailableReason READ unavailableReason NOTIFY backendChanged)

public:
    explicit SpeechSynthesizer(QObject *parent = nullptr);
    ~SpeechSynthesizer();

    bool isSpeaking() const { return m_isSpeaking; }
    bool isAvailable() const;
    QString backendName() const;
    QString unavailableReason() const;

    /** Backend ids: "piper", "sapi", "none". Unknown ids fall back to "none". */
    void configureBackend(const QString &backendId, const QJsonObject &config);

    /** Whether the current backend hands back samples for AudioPlayback to play. */
    bool producesSamples() const;
    int sampleRate() const;

    /**
     * Queues text for the given turn. Splitting happens here so callers can push
     * a whole reply, or push it sentence by sentence as it streams in.
     */
    void speak(quint64 turnId, const QString &text);

    /** Marks the end of a streamed reply; the turn finishes once the queue drains. */
    void finishTurn(quint64 turnId);

    void cancel();

    /** Splits text at sentence boundaries, keeping abbreviations and decimals intact. */
    static QStringList splitIntoSentences(const QString &text);

    /** Installed Windows SAPI voice names; empty on other platforms. */
    Q_INVOKABLE QStringList availableSapiVoices() const;

signals:
    void isSpeakingChanged();
    void backendChanged();

    void samplesReady(quint64 turnId, const QVector<float> &samples);
    void turnSpeechStarted(quint64 turnId);
    void turnSpeechFinished(quint64 turnId);
    void speechError(quint64 turnId, const QString &message);

private:
    struct QueuedUtterance {
        quint64 turnId;
        QString text;
    };

    void pumpQueue();
    void onBackendFinished(quint64 requestId);
    void setSpeaking(bool speaking);

    std::unique_ptr<TtsBackend> m_backend;
    QString m_backendId;
    QQueue<QueuedUtterance> m_queue;
    quint64 m_activeTurn = 0;
    quint64 m_nextRequestId = 1;
    bool m_backendBusy = false;
    bool m_isSpeaking = false;
    bool m_turnComplete = true;
};

#endif // DATAFORM_VOICE_ENABLED
#endif // SPEECHSYNTHESIZER_H
