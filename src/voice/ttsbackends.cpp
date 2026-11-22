#include "ttsbackends.h"

#ifdef DATAFORM_VOICE_ENABLED

#include <QProcess>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QDebug>

// ---------------------------------------------------------------------------
// PiperBackend
// ---------------------------------------------------------------------------

PiperBackend::PiperBackend(const QString &exePath, const QString &modelPath,
                           double speed, QObject *parent)
    : TtsBackend(parent)
    , m_exePath(exePath)
    , m_modelPath(modelPath)
    , m_speed(speed)
{
    if (m_exePath.isEmpty() || !QFileInfo::exists(m_exePath)) {
        m_unavailableReason = QStringLiteral(
            "piper.exe was not found at '%1'. Download a Piper release and set the "
            "path in Settings > Voice.").arg(m_exePath);
        return;
    }
    if (m_modelPath.isEmpty() || !QFileInfo::exists(m_modelPath)) {
        m_unavailableReason = QStringLiteral(
            "Piper voice model was not found at '%1'.").arg(m_modelPath);
        return;
    }

    // The voice config carries the sample rate; guessing it would detune playback.
    const QString configPath = m_modelPath + QStringLiteral(".json");
    QFile config(configPath);
    if (config.open(QIODevice::ReadOnly)) {
        const QJsonObject root = QJsonDocument::fromJson(config.readAll()).object();
        const int rate = root.value(QStringLiteral("audio")).toObject()
                             .value(QStringLiteral("sample_rate")).toInt(0);
        if (rate > 0)
            m_sampleRate = rate;
        else
            qWarning() << "PiperBackend: no audio.sample_rate in" << configPath
                       << "- assuming" << m_sampleRate;
    } else {
        qWarning() << "PiperBackend: could not read" << configPath
                   << "- assuming" << m_sampleRate << "Hz";
    }
}

PiperBackend::~PiperBackend()
{
    cancel();
}

bool PiperBackend::isAvailable() const
{
    return m_unavailableReason.isEmpty();
}

void PiperBackend::speak(quint64 requestId, const QString &text)
{
    if (!isAvailable()) {
        emit speechError(requestId, m_unavailableReason);
        emit speechFinished(requestId);
        return;
    }

    // One utterance at a time; the synthesizer queue guarantees this, but a stray
    // call must not leave an orphaned process behind.
    if (m_process)
        cancel();

    m_currentRequest = requestId;
    m_carry.clear();
    m_cancelled = false;

    m_process = new QProcess(this);
    m_process->setProcessChannelMode(QProcess::SeparateChannels);

    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &PiperBackend::readAvailableAudio);
    connect(m_process, &QProcess::finished,
            this, [this](int, QProcess::ExitStatus) { finishProcess(); });
    connect(m_process, &QProcess::errorOccurred, this,
            [this, requestId](QProcess::ProcessError error) {
                if (m_cancelled || error == QProcess::Crashed)
                    return;   // cancel() kills the process; that is not a failure
                emit speechError(requestId, QStringLiteral("Piper failed to run: %1")
                                                .arg(m_process ? m_process->errorString()
                                                               : QStringLiteral("unknown")));
            });

    QStringList args;
    args << QStringLiteral("--model") << m_modelPath
         << QStringLiteral("--output_raw");
    if (!qFuzzyCompare(m_speed, 1.0))
        args << QStringLiteral("--length_scale") << QString::number(m_speed, 'f', 3);

    m_process->start(m_exePath, args);
    if (!m_process->waitForStarted(3000)) {
        const QString err = m_process->errorString();
        m_process->deleteLater();
        m_process = nullptr;
        emit speechError(requestId, QStringLiteral("Piper did not start: %1").arg(err));
        emit speechFinished(requestId);
        return;
    }

    // Piper reads one utterance per line.
    QByteArray payload = text.simplified().toUtf8();
    payload.append('\n');
    m_process->write(payload);
    m_process->closeWriteChannel();
}

void PiperBackend::readAvailableAudio()
{
    if (!m_process || m_cancelled)
        return;

    QByteArray chunk = m_carry + m_process->readAllStandardOutput();
    m_carry.clear();

    // --output_raw is 16-bit signed mono; a read can land mid-sample.
    if (chunk.size() % 2 != 0) {
        m_carry = chunk.right(1);
        chunk.chop(1);
    }
    if (chunk.isEmpty())
        return;

    const int count = chunk.size() / 2;
    const auto *pcm = reinterpret_cast<const qint16 *>(chunk.constData());

    QVector<float> samples;
    samples.resize(count);
    for (int i = 0; i < count; ++i)
        samples[i] = float(pcm[i]) / 32768.0f;

    emit samplesReady(m_currentRequest, samples);
}

void PiperBackend::finishProcess()
{
    if (!m_process)
        return;

    readAvailableAudio();

    const quint64 requestId = m_currentRequest;
    const bool cancelled = m_cancelled;
    const int exitCode = m_process->exitCode();
    const QByteArray stderrOutput = m_process->readAllStandardError();

    m_process->deleteLater();
    m_process = nullptr;
    m_currentRequest = 0;
    m_carry.clear();

    if (cancelled)
        return;

    if (exitCode != 0) {
        emit speechError(requestId, QStringLiteral("Piper exited with code %1: %2")
                                        .arg(exitCode)
                                        .arg(QString::fromUtf8(stderrOutput).trimmed()));
    }
    emit speechFinished(requestId);
}

void PiperBackend::cancel()
{
    if (!m_process)
        return;

    m_cancelled = true;
    m_process->kill();
    m_process->waitForFinished(500);
    m_process->deleteLater();
    m_process = nullptr;
    m_currentRequest = 0;
    m_carry.clear();
}

// ---------------------------------------------------------------------------
// SapiBackend
// ---------------------------------------------------------------------------

#ifdef Q_OS_WIN

#include "sapiworker.h"

SapiBackend::SapiBackend(const QString &voiceName, int rate, QObject *parent)
    : TtsBackend(parent)
    , m_worker(new SapiWorker(voiceName, rate))
{
    m_worker->moveToThread(&m_thread);
    connect(&m_thread, &QThread::finished, m_worker, &QObject::deleteLater);

    connect(m_worker, &SapiWorker::speechFinished, this, &TtsBackend::speechFinished);
    connect(m_worker, &SapiWorker::speechError, this, &TtsBackend::speechError);
    connect(this, &SapiBackend::requestSpeak, m_worker, &SapiWorker::speak,
            Qt::QueuedConnection);

    m_thread.setObjectName(QStringLiteral("dataform-sapi"));
    m_thread.start();

    // Block so isAvailable() is meaningful to the caller that just built us, and
    // read the outcome directly: a signal from the worker would be queued to this
    // thread and could not arrive before this constructor returns.
    QMetaObject::invokeMethod(m_worker, "initialise", Qt::BlockingQueuedConnection);

    m_available = m_worker->initOk();
    m_unavailableReason = m_worker->initMessage();
}

SapiBackend::~SapiBackend()
{
    m_worker->cancelRequested.storeRelaxed(1);
    QMetaObject::invokeMethod(m_worker, "shutdown", Qt::BlockingQueuedConnection);
    m_thread.quit();
    m_thread.wait(5000);
}

void SapiBackend::speak(quint64 requestId, const QString &text)
{
    if (!m_available) {
        emit speechError(requestId, m_unavailableReason.isEmpty()
                                        ? QStringLiteral("SAPI voice is not available.")
                                        : m_unavailableReason);
        emit speechFinished(requestId);
        return;
    }
    emit requestSpeak(requestId, text);
}

void SapiBackend::cancel()
{
    m_worker->cancelRequested.storeRelaxed(1);
}

QStringList SapiBackend::availableVoices()
{
    return SapiWorker::availableVoices();
}

#endif // Q_OS_WIN

#endif // DATAFORM_VOICE_ENABLED
