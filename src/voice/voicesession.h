#ifndef VOICESESSION_H
#define VOICESESSION_H

#ifdef DATAFORM_VOICE_ENABLED

#include <QObject>
#include <QString>
#include <QVector>
#include <QElapsedTimer>
#include <QTimer>

class AudioCapture;
class AudioPlayback;
class SpeechRecognizer;
class SpeechSynthesizer;
class Orchestrator;
class SettingsManager;

/**
 * The voice turn loop.
 *
 * Owns the states a spoken request moves through and the turn identity that makes
 * cancellation safe: every result carries the turn that asked for it, and results
 * belonging to a superseded turn are dropped rather than acted on. Typed requests
 * run through the same Orchestrator, so this class stays out of their way except
 * when a reply needs speaking.
 *
 * Capture is half-duplex by design. Nothing here cancels the echo of the
 * assistant's own voice, so the microphone is closed while it is speaking; going
 * full-duplex needs echo cancellation first.
 */
class VoiceSession : public QObject
{
    Q_OBJECT
    Q_PROPERTY(State state READ state NOTIFY stateChanged)
    Q_PROPERTY(QString stateLabel READ stateLabel NOTIFY stateChanged)
    Q_PROPERTY(QString partialTranscript READ partialTranscript NOTIFY partialTranscriptChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)
    Q_PROPERTY(bool voiceInputReady READ voiceInputReady NOTIFY readinessChanged)
    Q_PROPERTY(bool voiceOutputReady READ voiceOutputReady NOTIFY readinessChanged)
    Q_PROPERTY(bool speakReplies READ speakReplies WRITE setSpeakReplies NOTIFY speakRepliesChanged)
    Q_PROPERTY(qreal inputLevel READ inputLevel NOTIFY inputLevelChanged)
    Q_PROPERTY(bool canCancel READ canCancel NOTIFY stateChanged)

public:
    enum State {
        Idle,
        Capturing,
        Transcribing,
        Thinking,
        Executing,
        Speaking,
        Cancelled,
        Error,
    };
    Q_ENUM(State)

    explicit VoiceSession(QObject *parent = nullptr);
    ~VoiceSession();

    void setAudioCapture(AudioCapture *capture);
    void setAudioPlayback(AudioPlayback *playback);
    void setSpeechRecognizer(SpeechRecognizer *recognizer);
    void setSpeechSynthesizer(SpeechSynthesizer *synthesizer);
    void setOrchestrator(Orchestrator *orchestrator);
    void setSettingsManager(SettingsManager *settings);

    State state() const { return m_state; }
    QString stateLabel() const;
    QString partialTranscript() const { return m_partialTranscript; }
    QString lastError() const { return m_lastError; }
    bool voiceInputReady() const;
    bool voiceOutputReady() const;
    bool speakReplies() const { return m_speakReplies; }
    void setSpeakReplies(bool enabled);
    qreal inputLevel() const;
    bool canCancel() const;

    /** Applies the voice section of settings and loads/unloads engines to match. */
    Q_INVOKABLE void applySettings();

    /** Push-to-talk pressed. Interrupts any reply in progress. */
    Q_INVOKABLE void startListening();

    /** Push-to-talk released. Transcription finishes, then the turn continues. */
    Q_INVOKABLE void stopListening();

    /**
     * Abandons the current turn: stops capture, drops the transcript, silences
     * playback and cancels generation. Work a tool already committed is not
     * undone, and is reported rather than hidden.
     */
    Q_INVOKABLE void cancel();

    /** Speaks arbitrary text outside a turn (reminders, proactive thoughts). */
    Q_INVOKABLE void announce(const QString &text);

signals:
    void stateChanged();
    void partialTranscriptChanged();
    void lastErrorChanged();
    void readinessChanged();
    void speakRepliesChanged();
    void inputLevelChanged();

    /** A usable transcript was produced and handed to the assistant. */
    void transcriptCommitted(const QString &text);

    /** The turn ended without reaching the assistant, with a reason for the UI. */
    void turnAbandoned(const QString &reason);

    /** Per-turn timings in milliseconds, for the latency targets in the plan. */
    void turnTimings(qint64 transcribeMs, qint64 thinkMs, qint64 firstAudioMs, qint64 totalMs);

private slots:
    void onAudioChunk(const QVector<float> &samples);
    void onPartialTranscript(quint64 turnId, const QString &text);
    void onFinalTranscript(quint64 turnId, const QString &text, const QString &language);
    void onRecognitionError(quint64 turnId, const QString &error);
    void onAssistantResponse(const QString &response);
    void onOrchestratorError(const QString &error);
    void onSynthSamples(quint64 turnId, const QVector<float> &samples);
    void onSpeechFinished(quint64 turnId);
    void onSpeechError(quint64 turnId, const QString &message);
    void onPlaybackFinished();
    void onDeviceLost();
    void onCaptureError(const QString &message);

private:
    void setState(State state);
    void setError(const QString &error);
    void setPartial(const QString &text);
    void finishTurn();
    bool isCurrentTurn(quint64 turnId) const { return turnId == m_turnId && m_turnActive; }
    void configureSynthesizer();

    AudioCapture *m_capture = nullptr;
    AudioPlayback *m_playback = nullptr;
    SpeechRecognizer *m_recognizer = nullptr;
    SpeechSynthesizer *m_synthesizer = nullptr;
    Orchestrator *m_orchestrator = nullptr;
    SettingsManager *m_settings = nullptr;

    State m_state = Idle;
    QString m_partialTranscript;
    QString m_lastError;
    bool m_speakReplies = true;

    // Turn identity. Monotonic so a stale result can never match a live turn.
    quint64 m_turnId = 0;
    bool m_turnActive = false;

    // The assistant is shared with the typed UI; only replies to a turn this
    // session started are spoken.
    bool m_awaitingResponse = false;

    QElapsedTimer m_turnTimer;
    qint64 m_captureEndedMs = 0;
    qint64 m_transcriptReadyMs = 0;
    qint64 m_firstAudioMs = 0;

    /** Announcements reuse the speech path but must not be mistaken for a turn. */
    bool m_announcing = false;

    /**
     * Set once synthesis is done but queued audio is still draining.
     *
     * Capture is half-duplex, so a turn is not over until the speaker is actually
     * quiet — ending it at the last synthesised sample would let the microphone
     * reopen over the tail of the reply and transcribe it.
     */
    bool m_awaitingPlaybackDrain = false;
};

#endif // DATAFORM_VOICE_ENABLED
#endif // VOICESESSION_H
