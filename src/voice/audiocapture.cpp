#include "audiocapture.h"

#ifdef DATAFORM_VOICE_ENABLED

#include <QAudioSource>
#include <QMediaDevices>
#include <QDebug>
#include <cmath>

namespace {
// 20 ms of audio. Small enough that partial transcripts feel live, large enough
// that the queued-connection traffic to the recogniser stays negligible.
constexpr int kChunkSamples = kCaptureSampleRate / 50;
}

/**
 * Push-mode sink. QAudioSource writes into this device from its own thread; we
 * accumulate into whole chunks and hand them up as complete units.
 */
class AudioCapture::SampleSink : public QIODevice
{
public:
    explicit SampleSink(AudioCapture *owner) : QIODevice(owner), m_owner(owner)
    {
        m_pending.reserve(kChunkSamples * 2);
    }

    void clearPending()
    {
        m_pending.clear();
    }

protected:
    qint64 readData(char *, qint64) override { return -1; }

    qint64 writeData(const char *data, qint64 len) override
    {
        const auto *samples = reinterpret_cast<const float *>(data);
        const qint64 count = len / qint64(sizeof(float));

        for (qint64 i = 0; i < count; ++i) {
            m_pending.append(samples[i]);
            if (m_pending.size() >= kChunkSamples) {
                m_owner->handleSamples(m_pending);
                m_pending.clear();
            }
        }
        return len;
    }

private:
    AudioCapture *m_owner;
    QVector<float> m_pending;
};

AudioCapture::AudioCapture(QObject *parent)
    : QObject(parent)
    , m_mediaDevices(new QMediaDevices(this))
{
    m_format.setSampleRate(kCaptureSampleRate);
    m_format.setChannelCount(1);
    m_format.setSampleFormat(QAudioFormat::Float);

    connect(m_mediaDevices, &QMediaDevices::audioInputsChanged,
            this, &AudioCapture::onDevicesChanged);

    m_device = QMediaDevices::defaultAudioInput();
}

AudioCapture::~AudioCapture()
{
    stop();
}

bool AudioCapture::hasInputDevice() const
{
    return !m_device.isNull();
}

QString AudioCapture::deviceName() const
{
    return m_device.isNull() ? QString() : m_device.description();
}

QStringList AudioCapture::availableDevices() const
{
    QStringList names;
    const auto inputs = QMediaDevices::audioInputs();
    names.reserve(inputs.size());
    for (const QAudioDevice &dev : inputs)
        names << dev.description();
    return names;
}

void AudioCapture::setPreferredDevice(const QString &description)
{
    if (m_preferredDevice == description)
        return;
    m_preferredDevice = description;
    rebuildSource();
}

void AudioCapture::rebuildSource()
{
    const bool wasCapturing = m_isCapturing;
    stop();

    QAudioDevice chosen = QMediaDevices::defaultAudioInput();
    if (!m_preferredDevice.isEmpty()) {
        for (const QAudioDevice &dev : QMediaDevices::audioInputs()) {
            if (dev.description() == m_preferredDevice) {
                chosen = dev;
                break;
            }
        }
    }

    m_device = chosen;
    m_source.reset();
    emit availabilityChanged();

    if (wasCapturing)
        start();
}

void AudioCapture::start()
{
    if (m_isCapturing)
        return;

    if (m_device.isNull()) {
        emit captureError(QStringLiteral("No microphone is available."));
        return;
    }

    QAudioFormat format = m_format;
    if (!m_device.isFormatSupported(format)) {
        // Fall back to whatever the device prefers; Muninn handles the resample.
        format = m_device.preferredFormat();
        if (format.sampleFormat() != QAudioFormat::Float) {
            emit captureError(QStringLiteral("Microphone '%1' does not support float capture.")
                                  .arg(m_device.description()));
            return;
        }
        qWarning() << "AudioCapture: falling back to device format"
                   << format.sampleRate() << "Hz," << format.channelCount() << "ch";
    }

    if (!m_source || m_source->format() != format) {
        m_source = std::make_unique<QAudioSource>(m_device, format, this);
        connect(m_source.get(), &QAudioSource::stateChanged, this,
                [this](QAudio::State state) {
                    if (state == QAudio::StoppedState && m_source
                        && m_source->error() != QAudio::NoError && m_isCapturing) {
                        m_isCapturing = false;
                        emit isCapturingChanged();
                        emit deviceLost();
                    }
                });
    }

    if (!m_sink)
        m_sink = std::make_unique<SampleSink>(this);
    m_sink->clearPending();
    m_sink->open(QIODevice::WriteOnly);

    m_source->start(m_sink.get());

    if (m_source->error() != QAudio::NoError) {
        emit captureError(QStringLiteral("Could not open microphone '%1'.")
                              .arg(m_device.description()));
        return;
    }

    m_isCapturing = true;
    emit isCapturingChanged();
}

void AudioCapture::stop()
{
    if (m_source)
        m_source->stop();
    if (m_sink && m_sink->isOpen())
        m_sink->close();

    if (m_isCapturing) {
        m_isCapturing = false;
        emit isCapturingChanged();
    }

    if (!qFuzzyIsNull(m_inputLevel)) {
        m_inputLevel = 0.0;
        emit inputLevelChanged();
    }
}

void AudioCapture::handleSamples(const QVector<float> &samples)
{
    double sumSquares = 0.0;
    for (float s : samples)
        sumSquares += double(s) * double(s);

    const qreal rms = samples.isEmpty() ? 0.0 : std::sqrt(sumSquares / samples.size());
    // The meter only drives a UI bar, so a coarse threshold avoids flooding QML
    // with a property change every 20 ms.
    if (std::abs(rms - m_inputLevel) > 0.01) {
        m_inputLevel = rms;
        emit inputLevelChanged();
    }

    emit audioChunk(samples);
}

void AudioCapture::onDevicesChanged()
{
    const QAudioDevice current = m_device;
    const auto inputs = QMediaDevices::audioInputs();

    bool stillPresent = false;
    for (const QAudioDevice &dev : inputs) {
        if (dev.id() == current.id()) {
            stillPresent = true;
            break;
        }
    }

    if (!stillPresent) {
        const bool wasCapturing = m_isCapturing;
        rebuildSource();
        if (wasCapturing && m_device.isNull())
            emit deviceLost();
    }

    emit availabilityChanged();
}

#endif // DATAFORM_VOICE_ENABLED
