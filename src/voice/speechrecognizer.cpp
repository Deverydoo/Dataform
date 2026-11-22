#include "speechrecognizer.h"

#ifdef DATAFORM_VOICE_ENABLED

#include <QTimer>
#include <QElapsedTimer>
#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <QMetaObject>

#include "muninn/streaming_transcriber.h"
#include "muninn/types.h"

namespace {
// Muninn reports partials from its own worker; 120 ms is often enough to look
// live in the UI without repainting the transcript on every frame.
constexpr int kPartialPollMs = 120;

muninn::DeviceType deviceFromString(const QString &device)
{
    if (device.compare(QLatin1String("cuda"), Qt::CaseInsensitive) == 0)
        return muninn::DeviceType::CUDA;
    if (device.compare(QLatin1String("cpu"), Qt::CaseInsensitive) == 0)
        return muninn::DeviceType::CPU;
    return muninn::DeviceType::Auto;
}

muninn::ComputeType computeFromString(const QString &compute)
{
    if (compute.compare(QLatin1String("int8"), Qt::CaseInsensitive) == 0)
        return muninn::ComputeType::Int8;
    if (compute.compare(QLatin1String("float32"), Qt::CaseInsensitive) == 0)
        return muninn::ComputeType::Float32;
    return muninn::ComputeType::Float16;
}
}

// ---------------------------------------------------------------------------
// SttWorker
// ---------------------------------------------------------------------------

SttWorker::SttWorker(QObject *parent)
    : QObject(parent)
{
}

SttWorker::~SttWorker()
{
    unloadModel();
}

void SttWorker::loadModel(const QString &modelPath, const QString &device,
                          const QString &computeType, const QString &language,
                          const QString &vadModelPath)
{
    if (!QDir(modelPath).exists()) {
        emit modelLoadFailed(QStringLiteral("Whisper model directory not found: %1").arg(modelPath));
        return;
    }

    m_language = language;
    m_vadModelPath = vadModelPath;

    if (m_vadModelPath.isEmpty())
        qInfo() << "Muninn: no Silero model configured — endpointing on energy VAD";
    else
        qInfo() << "Muninn: Silero VAD model" << m_vadModelPath;

    qInfo() << "Muninn: loading" << modelPath << "device=" << device
            << "compute=" << computeType << "language=" << language;

    QElapsedTimer loadTimer;
    loadTimer.start();

    try {
        muninn::ModelOptions options;
        options.model_path = modelPath.toStdString();
        options.device = deviceFromString(device);
        options.compute_type = computeFromString(computeType);

        m_transcriber = std::make_unique<muninn::StreamingTranscriber>(options);
        m_modelPath = modelPath;
    } catch (const std::exception &e) {
        m_transcriber.reset();
        emit modelLoadFailed(QStringLiteral("Muninn failed to load '%1': %2")
                                 .arg(modelPath, QString::fromUtf8(e.what())));
        return;
    } catch (...) {
        m_transcriber.reset();
        emit modelLoadFailed(QStringLiteral("Muninn failed to load '%1': unknown error").arg(modelPath));
        return;
    }

    if (!m_pollTimer) {
        m_pollTimer = new QTimer(this);
        m_pollTimer->setInterval(kPartialPollMs);
        connect(m_pollTimer, &QTimer::timeout, this, &SttWorker::pollPartial);
    }

    qInfo() << "Muninn: model ready in" << loadTimer.elapsed() << "ms";
    emit modelLoaded(modelPath);
}

void SttWorker::unloadModel()
{
    if (m_pollTimer)
        m_pollTimer->stop();

    if (m_transcriber && m_streaming) {
        try {
            m_transcriber->stop();
        } catch (...) {
            // Teardown path; a failed flush must not prevent releasing the model.
        }
    }
    m_streaming = false;
    m_transcriber.reset();
    m_modelPath.clear();
}

void SttWorker::beginUtterance(quint64 turnId)
{
    if (!m_transcriber) {
        emit recognitionError(turnId, QStringLiteral("No speech model is loaded."));
        return;
    }

    // A previous utterance that was never ended would otherwise leak its audio
    // into this one.
    if (m_streaming) {
        try {
            m_transcriber->stop();
        } catch (...) {
        }
        m_streaming = false;
    }

    m_activeTurn = turnId;
    m_lastPartial.clear();

    muninn::StreamingOptions options;
    options.language = m_language.isEmpty() ? std::string("auto") : m_language.toStdString();
    options.enable_vad = true;

    // Silero is the better endpointer on clean speech, but it is an ONNX model,
    // so it only works when the single onnxruntime.dll in this process is new
    // enough for Muninn. CMake compares ORT_API_VERSION against what Muninn was
    // built for and defines DATAFORM_MUNINN_NEURAL_VAD when they line up; energy
    // VAD touches no ONNX at all and is the fallback.
    // See Docs/decisions/0002-onnxruntime-unification.md.
    //
    // Muninn falls back to energy on its own if the model is missing or fails to
    // load, so an unset path degrades rather than breaking the turn.
#ifdef DATAFORM_MUNINN_NEURAL_VAD
    options.vad_type = m_vadModelPath.isEmpty() ? muninn::VADType::Energy
                                                : muninn::VADType::Silero;
    options.vad_model_path = m_vadModelPath.toStdString();
#else
    options.vad_type = muninn::VADType::Energy;
#endif
    options.chunk_length_s = 1.5f;
    options.overlap_s = 0.3f;
    options.word_timestamps = false;      // not needed for commands; saves work
    options.filter_hallucinations = true;

    try {
        m_transcriber->reset();
        m_transcriber->start(options);
        m_streaming = true;
    } catch (const std::exception &e) {
        emit recognitionError(turnId, QStringLiteral("Could not start transcription: %1")
                                          .arg(QString::fromUtf8(e.what())));
        return;
    }

    if (m_pollTimer)
        m_pollTimer->start();
}

void SttWorker::pushAudio(const QVector<float> &samples, int sampleRate)
{
    if (!m_transcriber || !m_streaming || samples.isEmpty())
        return;
    if (cancelRequested.loadRelaxed())
        return;

    try {
        m_transcriber->push_audio(samples.constData(), size_t(samples.size()), sampleRate, 1);
    } catch (const std::exception &e) {
        qWarning() << "SttWorker: push_audio failed:" << e.what();
    }
}

void SttWorker::pollPartial()
{
    if (!m_transcriber || !m_streaming)
        return;
    if (cancelRequested.loadRelaxed())
        return;

    std::string current;
    try {
        current = m_transcriber->get_current_text(true);
    } catch (...) {
        return;
    }

    const QString text = QString::fromStdString(current).trimmed();
    if (text != m_lastPartial) {
        m_lastPartial = text;
        emit partialTranscript(m_activeTurn, text);
    }
}

void SttWorker::endUtterance(quint64 turnId)
{
    if (m_pollTimer)
        m_pollTimer->stop();

    if (!m_transcriber || !m_streaming) {
        emit finalTranscript(turnId, QString(), QString());
        return;
    }

    std::vector<muninn::Segment> tail;
    try {
        tail = m_transcriber->stop();
    } catch (const std::exception &e) {
        m_streaming = false;
        emit recognitionError(turnId, QStringLiteral("Transcription failed: %1")
                                          .arg(QString::fromUtf8(e.what())));
        return;
    }
    m_streaming = false;

    // A cancelled turn still had to be flushed to leave Muninn in a clean state,
    // but its text is discarded rather than reported.
    if (cancelRequested.loadRelaxed())
        return;

    QString text;
    QString language;
    try {
        for (const muninn::Segment &seg : m_transcriber->get_all_segments()) {
            const QString piece = QString::fromStdString(seg.text).trimmed();
            if (piece.isEmpty())
                continue;
            if (!text.isEmpty())
                text += QLatin1Char(' ');
            text += piece;
            if (language.isEmpty())
                language = QString::fromStdString(seg.language);
        }
    } catch (...) {
    }

    for (const muninn::Segment &seg : tail) {
        const QString piece = QString::fromStdString(seg.text).trimmed();
        if (piece.isEmpty() || text.contains(piece))
            continue;
        if (!text.isEmpty())
            text += QLatin1Char(' ');
        text += piece;
    }

    emit finalTranscript(turnId, text.trimmed(), language);
}

// ---------------------------------------------------------------------------
// SpeechRecognizer
// ---------------------------------------------------------------------------

SpeechRecognizer::SpeechRecognizer(QObject *parent)
    : QObject(parent)
    , m_worker(new SttWorker)
{
    m_worker->moveToThread(&m_thread);
    connect(&m_thread, &QThread::finished, m_worker, &QObject::deleteLater);

    connect(m_worker, &SttWorker::modelLoaded, this, [this](const QString &path) {
        m_isModelLoaded = true;
        m_modelPath = path;
        emit isModelLoadedChanged();
        emit modelLoaded(path);
    });

    connect(m_worker, &SttWorker::modelLoadFailed, this, [this](const QString &error) {
        m_isModelLoaded = false;
        m_modelPath.clear();
        emit isModelLoadedChanged();
        setError(error);
        emit modelLoadFailed(error);
    });

    connect(m_worker, &SttWorker::partialTranscript, this, &SpeechRecognizer::partialTranscript);
    connect(m_worker, &SttWorker::finalTranscript, this, &SpeechRecognizer::finalTranscript);
    connect(m_worker, &SttWorker::recognitionError, this,
            [this](quint64 turnId, const QString &error) {
                setError(error);
                emit recognitionError(turnId, error);
            });

    m_thread.setObjectName(QStringLiteral("dataform-stt"));
    m_thread.start();
}

SpeechRecognizer::~SpeechRecognizer()
{
    m_worker->cancelRequested.storeRelaxed(1);
    QMetaObject::invokeMethod(m_worker, "unloadModel", Qt::BlockingQueuedConnection);
    m_thread.quit();
    m_thread.wait(10000);
}

QString SpeechRecognizer::defaultVadModelPath(const QString &sttModelPath)
{
    if (sttModelPath.isEmpty())
        return QString();

    // Muninn keeps silero_vad.onnx alongside the model directories, so it sits
    // one level up from the Whisper model folder.
    const QDir modelDir(sttModelPath);
    QDir parent(modelDir);
    if (!parent.cdUp())
        return QString();

    const QString candidate = parent.filePath(QStringLiteral("silero_vad.onnx"));
    return QFileInfo::exists(candidate) ? candidate : QString();
}

void SpeechRecognizer::loadModel(const QString &modelPath, const QString &device,
                                 const QString &computeType, const QString &language,
                                 const QString &vadModelPath)
{
    const QString vadPath = vadModelPath.isEmpty() ? defaultVadModelPath(modelPath)
                                                   : vadModelPath;

    QMetaObject::invokeMethod(m_worker, "loadModel", Qt::QueuedConnection,
                              Q_ARG(QString, modelPath),
                              Q_ARG(QString, device),
                              Q_ARG(QString, computeType),
                              Q_ARG(QString, language),
                              Q_ARG(QString, vadPath));
}

void SpeechRecognizer::unloadModel()
{
    QMetaObject::invokeMethod(m_worker, "unloadModel", Qt::QueuedConnection);
    m_isModelLoaded = false;
    m_modelPath.clear();
    emit isModelLoadedChanged();
}

void SpeechRecognizer::beginUtterance(quint64 turnId)
{
    m_worker->cancelRequested.storeRelaxed(0);
    QMetaObject::invokeMethod(m_worker, "beginUtterance", Qt::QueuedConnection,
                              Q_ARG(quint64, turnId));
}

void SpeechRecognizer::pushAudio(const QVector<float> &samples, int sampleRate)
{
    QMetaObject::invokeMethod(m_worker, "pushAudio", Qt::QueuedConnection,
                              Q_ARG(QVector<float>, samples),
                              Q_ARG(int, sampleRate));
}

void SpeechRecognizer::endUtterance(quint64 turnId)
{
    QMetaObject::invokeMethod(m_worker, "endUtterance", Qt::QueuedConnection,
                              Q_ARG(quint64, turnId));
}

void SpeechRecognizer::cancel()
{
    // The worker may be mid-flush; the flag makes it drop whatever it produces.
    m_worker->cancelRequested.storeRelaxed(1);
}

void SpeechRecognizer::setError(const QString &error)
{
    if (m_lastError == error)
        return;
    m_lastError = error;
    emit lastErrorChanged();
}

#endif // DATAFORM_VOICE_ENABLED
