#include "voicesession.h"

#ifdef DATAFORM_VOICE_ENABLED

#include "audiocapture.h"
#include "audioplayback.h"
#include "speechrecognizer.h"
#include "speechsynthesizer.h"
#include "../orchestrator.h"
#include "../settingsmanager.h"

#include <QJsonObject>
#include <QDebug>

VoiceSession::VoiceSession(QObject *parent)
    : QObject(parent)
{
}

VoiceSession::~VoiceSession() = default;

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

void VoiceSession::setAudioCapture(AudioCapture *capture)
{
    m_capture = capture;
    if (!m_capture)
        return;

    connect(m_capture, &AudioCapture::audioChunk, this, &VoiceSession::onAudioChunk);
    connect(m_capture, &AudioCapture::deviceLost, this, &VoiceSession::onDeviceLost);
    connect(m_capture, &AudioCapture::captureError, this, &VoiceSession::onCaptureError);
    connect(m_capture, &AudioCapture::inputLevelChanged, this, &VoiceSession::inputLevelChanged);
    connect(m_capture, &AudioCapture::availabilityChanged, this, &VoiceSession::readinessChanged);
}

void VoiceSession::setAudioPlayback(AudioPlayback *playback)
{
    m_playback = playback;
    if (!m_playback)
        return;

    connect(m_playback, &AudioPlayback::playbackFinished,
            this, &VoiceSession::onPlaybackFinished);
    connect(m_playback, &AudioPlayback::playbackError, this, [this](const QString &message) {
        setError(message);
        onPlaybackFinished();   // do not strand a turn on a dead output device
    });
}

void VoiceSession::setSpeechRecognizer(SpeechRecognizer *recognizer)
{
    m_recognizer = recognizer;
    if (!m_recognizer)
        return;

    connect(m_recognizer, &SpeechRecognizer::partialTranscript,
            this, &VoiceSession::onPartialTranscript);
    connect(m_recognizer, &SpeechRecognizer::finalTranscript,
            this, &VoiceSession::onFinalTranscript);
    connect(m_recognizer, &SpeechRecognizer::recognitionError,
            this, &VoiceSession::onRecognitionError);
    connect(m_recognizer, &SpeechRecognizer::isModelLoadedChanged,
            this, &VoiceSession::readinessChanged);
    connect(m_recognizer, &SpeechRecognizer::modelLoadFailed, this,
            [this](const QString &error) { setError(error); });
}

void VoiceSession::setSpeechSynthesizer(SpeechSynthesizer *synthesizer)
{
    m_synthesizer = synthesizer;
    if (!m_synthesizer)
        return;

    connect(m_synthesizer, &SpeechSynthesizer::samplesReady,
            this, &VoiceSession::onSynthSamples);
    connect(m_synthesizer, &SpeechSynthesizer::turnSpeechFinished,
            this, &VoiceSession::onSpeechFinished);
    connect(m_synthesizer, &SpeechSynthesizer::speechError,
            this, &VoiceSession::onSpeechError);
    connect(m_synthesizer, &SpeechSynthesizer::backendChanged,
            this, &VoiceSession::readinessChanged);
}

void VoiceSession::setOrchestrator(Orchestrator *orchestrator)
{
    m_orchestrator = orchestrator;
    if (!m_orchestrator)
        return;

    connect(m_orchestrator, &Orchestrator::assistantResponseReady,
            this, &VoiceSession::onAssistantResponse);
    connect(m_orchestrator, &Orchestrator::errorOccurred,
            this, &VoiceSession::onOrchestratorError);

    // A tool running is materially different from the model thinking: the plan
    // asks for both to be visible, and only one of them is safe to cancel
    // without leaving committed work unreported.
    connect(m_orchestrator, &Orchestrator::toolInvocationStarted, this, [this](const QString &) {
        if (m_awaitingResponse && m_turnActive)
            setState(Executing);
    });
}

void VoiceSession::setSettingsManager(SettingsManager *settings)
{
    m_settings = settings;
    if (!m_settings)
        return;

    m_speakReplies = m_settings->speakReplies();

    // Any change that affects which engine or device is in use warrants a reload;
    // applySettings() is idempotent so coalescing is unnecessary.
    connect(m_settings, &SettingsManager::voiceEnabledChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::sttModelPathChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::sttDeviceChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::sttComputeTypeChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::sttLanguageChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::micDeviceNameChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::ttsBackendChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::piperExePathChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::piperVoicePathChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::sapiVoiceNameChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::ttsSpeedChanged, this, &VoiceSession::applySettings);
    connect(m_settings, &SettingsManager::speakRepliesChanged, this, [this] {
        setSpeakReplies(m_settings->speakReplies());
    });
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

QString VoiceSession::stateLabel() const
{
    switch (m_state) {
    case Idle:         return QStringLiteral("Idle");
    case Capturing:    return QStringLiteral("Listening");
    case Transcribing: return QStringLiteral("Transcribing");
    case Thinking:     return QStringLiteral("Thinking");
    case Executing:    return QStringLiteral("Working");
    case Speaking:     return QStringLiteral("Speaking");
    case Cancelled:    return QStringLiteral("Cancelled");
    case Error:        return QStringLiteral("Error");
    }
    return QStringLiteral("Idle");
}

bool VoiceSession::voiceInputReady() const
{
    return m_recognizer && m_recognizer->isModelLoaded()
           && m_capture && m_capture->hasInputDevice();
}

bool VoiceSession::voiceOutputReady() const
{
    return m_synthesizer && m_synthesizer->isAvailable();
}

qreal VoiceSession::inputLevel() const
{
    return m_capture ? m_capture->inputLevel() : 0.0;
}

bool VoiceSession::canCancel() const
{
    return m_state == Capturing || m_state == Transcribing || m_state == Thinking
           || m_state == Executing || m_state == Speaking;
}

void VoiceSession::setState(State state)
{
    if (m_state == state)
        return;
    m_state = state;
    emit stateChanged();
}

void VoiceSession::setError(const QString &error)
{
    if (m_lastError == error)
        return;
    m_lastError = error;
    emit lastErrorChanged();
}

void VoiceSession::setPartial(const QString &text)
{
    if (m_partialTranscript == text)
        return;
    m_partialTranscript = text;
    emit partialTranscriptChanged();
}

void VoiceSession::setSpeakReplies(bool enabled)
{
    if (m_speakReplies == enabled)
        return;
    m_speakReplies = enabled;
    if (!enabled && m_synthesizer)
        m_synthesizer->cancel();
    emit speakRepliesChanged();
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void VoiceSession::applySettings()
{
    if (!m_settings)
        return;

    qInfo() << "VoiceSession: applying settings, voiceEnabled="
            << m_settings->voiceEnabled() << "tts=" << m_settings->ttsBackend();

    if (!m_settings->voiceEnabled()) {
        cancel();
        if (m_recognizer)
            m_recognizer->unloadModel();
        if (m_synthesizer)
            m_synthesizer->configureBackend(QStringLiteral("none"), {});
        emit readinessChanged();
        return;
    }

    if (m_capture)
        m_capture->setPreferredDevice(m_settings->micDeviceName());

    if (m_recognizer) {
        const QString modelPath = m_settings->sttModelPath();
        if (modelPath.isEmpty()) {
            setError(QStringLiteral("No speech model is configured. Set the Whisper model "
                                    "directory in Settings > Voice."));
        } else if (m_recognizer->modelPath() != modelPath || !m_recognizer->isModelLoaded()) {
            // Empty VAD path means "look beside the Whisper model"; Muninn drops
            // to energy VAD if nothing turns up.
            m_recognizer->loadModel(modelPath,
                                    m_settings->sttDevice(),
                                    m_settings->sttComputeType(),
                                    m_settings->sttLanguage(),
                                    m_settings->sttVadModelPath());
        }
    }

    configureSynthesizer();
    emit readinessChanged();
}

void VoiceSession::configureSynthesizer()
{
    if (!m_synthesizer || !m_settings)
        return;

    const QString backend = m_settings->ttsBackend();

    QJsonObject config;
    if (backend == QLatin1String("piper")) {
        config[QStringLiteral("exePath")] = m_settings->piperExePath();
        config[QStringLiteral("modelPath")] = m_settings->piperVoicePath();
        config[QStringLiteral("speed")] = m_settings->ttsSpeed();
    } else if (backend == QLatin1String("sapi")) {
        config[QStringLiteral("voiceName")] = m_settings->sapiVoiceName();
        // SAPI rate is -10..10 rather than a multiplier; map the shared speed
        // setting onto it so one control drives either backend.
        config[QStringLiteral("rate")] = int(qRound((m_settings->ttsSpeed() - 1.0) * 10.0));
    }

    m_synthesizer->configureBackend(backend, config);

    if (m_synthesizer->isAvailable()) {
        qInfo() << "VoiceSession: speech backend" << m_synthesizer->backendName()
                << "ready, sampleRate=" << m_synthesizer->sampleRate();
        // A usable backend can still have something worth saying, such as a named
        // voice that was not found and silently fell back to the default.
        const QString note = m_synthesizer->unavailableReason();
        if (!note.isEmpty())
            qWarning() << "VoiceSession:" << note;
    } else if (backend != QLatin1String("none")) {
        qWarning() << "VoiceSession: speech backend unavailable:"
                   << m_synthesizer->unavailableReason();
        setError(m_synthesizer->unavailableReason());
    }
}

// ---------------------------------------------------------------------------
// Turn lifecycle
// ---------------------------------------------------------------------------

void VoiceSession::startListening()
{
    if (!m_settings || !m_settings->voiceEnabled()) {
        setError(QStringLiteral("Voice is turned off. Enable it in Settings > Voice."));
        return;
    }
    if (!m_recognizer || !m_recognizer->isModelLoaded()) {
        setError(QStringLiteral("The speech model is not loaded yet."));
        return;
    }
    if (!m_capture || !m_capture->hasInputDevice()) {
        setError(QStringLiteral("No microphone is available. Typed requests still work."));
        return;
    }
    if (m_state == Capturing)
        return;

    // Pressing to talk while the assistant is replying is a barge-in, not an
    // error: the old turn goes away and this one takes over.
    if (canCancel())
        cancel();

    ++m_turnId;
    m_turnActive = true;
    m_awaitingResponse = false;
    m_announcing = false;
    m_awaitingPlaybackDrain = false;
    m_captureEndedMs = 0;
    m_transcriptReadyMs = 0;
    m_firstAudioMs = 0;
    m_turnTimer.restart();

    setError(QString());
    setPartial(QString());

    m_recognizer->beginUtterance(m_turnId);
    m_capture->start();

    if (!m_capture->isCapturing()) {
        // start() already reported why; do not leave a half-open turn behind.
        m_recognizer->cancel();
        m_turnActive = false;
        setState(Error);
        return;
    }

    setState(Capturing);
}

void VoiceSession::onAudioChunk(const QVector<float> &samples)
{
    if (m_state != Capturing || !m_turnActive || !m_recognizer)
        return;
    m_recognizer->pushAudio(samples, kCaptureSampleRate);
}

void VoiceSession::stopListening()
{
    if (m_state != Capturing || !m_turnActive)
        return;

    m_captureEndedMs = m_turnTimer.elapsed();

    if (m_capture)
        m_capture->stop();

    setState(Transcribing);
    if (m_recognizer)
        m_recognizer->endUtterance(m_turnId);
}

void VoiceSession::onPartialTranscript(quint64 turnId, const QString &text)
{
    if (!isCurrentTurn(turnId))
        return;
    setPartial(text);
}

void VoiceSession::onFinalTranscript(quint64 turnId, const QString &text, const QString &language)
{
    Q_UNUSED(language)

    if (!isCurrentTurn(turnId))
        return;   // a cancelled or superseded turn

    m_transcriptReadyMs = m_turnTimer.elapsed();

    const QString transcript = text.trimmed();

    // Nothing usable came back. Stop here rather than sending an empty prompt to
    // the model or, worse, acting on a stray hallucinated word.
    if (transcript.isEmpty()) {
        m_turnActive = false;
        setPartial(QString());
        setState(Idle);
        emit turnAbandoned(QStringLiteral("Nothing was heard."));
        return;
    }

    setPartial(transcript);

    if (!m_orchestrator) {
        m_turnActive = false;
        setState(Error);
        setError(QStringLiteral("The assistant is not available."));
        return;
    }

    setState(Thinking);
    m_awaitingResponse = true;
    emit transcriptCommitted(transcript);
    m_orchestrator->handleUserMessage(transcript);
}

void VoiceSession::onRecognitionError(quint64 turnId, const QString &error)
{
    if (!isCurrentTurn(turnId))
        return;

    m_turnActive = false;
    m_awaitingResponse = false;
    if (m_capture)
        m_capture->stop();

    setError(error);
    setState(Error);
}

void VoiceSession::onAssistantResponse(const QString &response)
{
    // The Orchestrator also serves the typed UI. Only a reply to a turn this
    // session started is spoken.
    if (!m_awaitingResponse || !m_turnActive)
        return;

    m_awaitingResponse = false;

    const bool canSpeak = m_speakReplies && m_synthesizer && m_synthesizer->isAvailable()
                          && !response.trimmed().isEmpty();
    if (!canSpeak) {
        finishTurn();
        return;
    }

    setState(Speaking);

    if (m_synthesizer->producesSamples() && m_playback) {
        m_playback->beginStream(m_synthesizer->sampleRate());
        // beginStream reports a dead output device synchronously, which already
        // closed the turn. Synthesising into a sink that is not there would burn
        // a Piper process for audio nobody can hear.
        if (!m_playback->isPlaying())
            return;
    }

    m_synthesizer->speak(m_turnId, response);
    m_synthesizer->finishTurn(m_turnId);
}

void VoiceSession::onOrchestratorError(const QString &error)
{
    if (!m_awaitingResponse || !m_turnActive)
        return;

    m_awaitingResponse = false;
    m_turnActive = false;
    setError(error);
    setState(Error);
}

void VoiceSession::onSynthSamples(quint64 turnId, const QVector<float> &samples)
{
    if (!m_announcing && !isCurrentTurn(turnId))
        return;
    if (!m_playback)
        return;

    if (m_firstAudioMs == 0 && !m_announcing)
        m_firstAudioMs = m_turnTimer.elapsed();

    m_playback->appendSamples(samples);
}

void VoiceSession::onSpeechFinished(quint64 turnId)
{
    const bool ours = m_announcing || isCurrentTurn(turnId);
    if (!ours)
        return;

    // A backend that hands back samples has only finished producing them; the
    // speaker is still working through the queue. Wait for the drain so the
    // microphone does not reopen over the end of the reply.
    if (m_synthesizer && m_synthesizer->producesSamples() && m_playback) {
        m_awaitingPlaybackDrain = true;
        m_playback->endStream();
        return;
    }

    // A self-playing backend (SAPI) is silent by the time it reports finished.
    onPlaybackFinished();
}

void VoiceSession::onPlaybackFinished()
{
    m_awaitingPlaybackDrain = false;

    if (m_announcing) {
        m_announcing = false;
        if (!m_turnActive)
            setState(Idle);
        return;
    }

    if (!m_turnActive)
        return;

    finishTurn();
}

void VoiceSession::onSpeechError(quint64 turnId, const QString &message)
{
    setError(message);

    if (!m_announcing && !isCurrentTurn(turnId))
        return;

    // Synthesis failing does not undo the turn; the reply is already on screen.
    // Close it out through the same path so no drain flag is left standing.
    onPlaybackFinished();
}

void VoiceSession::finishTurn()
{
    const qint64 totalMs = m_turnTimer.isValid() ? m_turnTimer.elapsed() : 0;
    const qint64 transcribeMs = m_transcriptReadyMs - m_captureEndedMs;
    const qint64 thinkMs = m_transcriptReadyMs > 0
                               ? (m_firstAudioMs > 0 ? m_firstAudioMs - m_transcriptReadyMs
                                                     : totalMs - m_transcriptReadyMs)
                               : 0;
    const qint64 firstAudioMs = m_firstAudioMs > 0 ? m_firstAudioMs - m_captureEndedMs : 0;

    m_turnActive = false;
    m_awaitingResponse = false;
    m_awaitingPlaybackDrain = false;
    setState(Idle);

    emit turnTimings(qMax<qint64>(0, transcribeMs), qMax<qint64>(0, thinkMs),
                     qMax<qint64>(0, firstAudioMs), totalMs);
}

void VoiceSession::cancel()
{
    const bool hadTurn = m_turnActive || m_announcing;

    // Bump the turn id first: anything still in flight now fails isCurrentTurn()
    // and is dropped when it arrives.
    ++m_turnId;
    m_turnActive = false;
    m_awaitingResponse = false;
    m_announcing = false;
    m_awaitingPlaybackDrain = false;

    // Order matters below: the flags above are already cleared, so the
    // playbackFinished that stopAll() emits falls through onPlaybackFinished
    // without resurrecting the turn it just cancelled.
    if (m_capture)
        m_capture->stop();
    if (m_recognizer)
        m_recognizer->cancel();
    if (m_synthesizer)
        m_synthesizer->cancel();
    if (m_playback)
        m_playback->stopAll();

    setPartial(QString());

    if (hadTurn) {
        setState(Cancelled);
        // Cancelled is a report, not a resting state; the UI shows it briefly.
        QTimer::singleShot(600, this, [this] {
            if (m_state == Cancelled)
                setState(Idle);
        });
    } else {
        setState(Idle);
    }
}

void VoiceSession::announce(const QString &text)
{
    if (!m_speakReplies || !m_synthesizer || !m_synthesizer->isAvailable())
        return;
    if (text.trimmed().isEmpty())
        return;

    // Never talk over a live turn; the user is mid-conversation.
    if (m_turnActive)
        return;

    m_announcing = true;
    setState(Speaking);

    // Announcements reuse the current turn id so the synthesizer treats them as
    // one unit, while m_announcing keeps them out of the turn accounting.
    if (m_synthesizer->producesSamples() && m_playback) {
        m_playback->beginStream(m_synthesizer->sampleRate());
        if (!m_playback->isPlaying()) {
            m_announcing = false;
            setState(Idle);
            return;
        }
    }

    m_synthesizer->speak(m_turnId, text);
    m_synthesizer->finishTurn(m_turnId);
}

void VoiceSession::onDeviceLost()
{
    if (m_state == Capturing || m_state == Transcribing) {
        cancel();
        setError(QStringLiteral("The microphone was disconnected. Typed requests still work."));
        setState(Error);
    }
    emit readinessChanged();
}

void VoiceSession::onCaptureError(const QString &message)
{
    setError(message);
    emit readinessChanged();
}

#endif // DATAFORM_VOICE_ENABLED
