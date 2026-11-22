#include "audioplayback.h"

#ifdef DATAFORM_VOICE_ENABLED

#include <QAudioSink>
#include <QMediaDevices>
#include <QAudioDevice>
#include <QDebug>
#include <algorithm>
#include <cstring>

/**
 * Ring-free FIFO the sink pulls from.
 *
 * readData returns silence rather than 0 while the stream is still open: a
 * short-read makes QAudioSink go idle and emit stateChanged, which would end the
 * turn every time synthesis fell momentarily behind playback.
 */
class AudioPlayback::StreamBuffer : public QIODevice
{
public:
    explicit StreamBuffer(QObject *parent) : QIODevice(parent) {}

    void append(const QVector<float> &samples)
    {
        QMutexLocker lock(&m_mutex);
        m_data.append(samples);
    }

    void markComplete()
    {
        QMutexLocker lock(&m_mutex);
        m_complete = true;
    }

    void clear()
    {
        QMutexLocker lock(&m_mutex);
        m_data.clear();
        m_readPos = 0;
        m_complete = true;
    }

    void restart()
    {
        QMutexLocker lock(&m_mutex);
        m_data.clear();
        m_readPos = 0;
        m_complete = false;
    }

    bool drained() const
    {
        QMutexLocker lock(&m_mutex);
        return m_complete && m_readPos >= m_data.size();
    }

    bool isSequential() const override { return true; }
    qint64 bytesAvailable() const override { return QIODevice::bytesAvailable() + 1; }

protected:
    qint64 writeData(const char *, qint64) override { return -1; }

    qint64 readData(char *data, qint64 maxLen) override
    {
        QMutexLocker lock(&m_mutex);

        const qint64 wanted = maxLen / qint64(sizeof(float));
        const qint64 have = qint64(m_data.size()) - m_readPos;

        if (have <= 0) {
            if (m_complete)
                return 0;
            // Underrun mid-stream: emit silence so the sink stays active.
            const qint64 pad = std::min<qint64>(wanted, 256);
            std::memset(data, 0, size_t(pad) * sizeof(float));
            return pad * qint64(sizeof(float));
        }

        const qint64 count = std::min(wanted, have);
        std::memcpy(data, m_data.constData() + m_readPos, size_t(count) * sizeof(float));
        m_readPos += count;

        // Reclaim consumed samples once they dominate the buffer.
        if (m_readPos > 48000) {
            m_data.remove(0, int(m_readPos));
            m_readPos = 0;
        }
        return count * qint64(sizeof(float));
    }

private:
    mutable QMutex m_mutex;
    QVector<float> m_data;
    qint64 m_readPos = 0;
    bool m_complete = false;
};

AudioPlayback::AudioPlayback(QObject *parent)
    : QObject(parent)
{
    m_format.setChannelCount(1);
    m_format.setSampleFormat(QAudioFormat::Float);
}

AudioPlayback::~AudioPlayback()
{
    teardown();
}

void AudioPlayback::beginStream(int sampleRate)
{
    if (sampleRate <= 0) {
        emit playbackError(QStringLiteral("Invalid sample rate for playback."));
        return;
    }

    // A rate change means a new sink; otherwise reuse the open one so back-to-back
    // sentences do not click.
    if (m_sink && m_sampleRate == sampleRate && m_buffer) {
        m_buffer->restart();
        if (m_sink->state() != QAudio::ActiveState)
            m_sink->start(m_buffer.get());
        updatePlaying(true);
        emit playbackStarted();
        return;
    }

    teardown();

    m_sampleRate = sampleRate;
    m_format.setSampleRate(sampleRate);

    const QAudioDevice output = QMediaDevices::defaultAudioOutput();
    if (output.isNull()) {
        emit playbackError(QStringLiteral("No audio output device is available."));
        return;
    }

    QAudioFormat format = m_format;
    if (!output.isFormatSupported(format)) {
        emit playbackError(QStringLiteral("Audio output does not support %1 Hz float playback.")
                               .arg(sampleRate));
        return;
    }

    m_buffer = std::make_unique<StreamBuffer>(this);
    m_buffer->restart();
    m_buffer->open(QIODevice::ReadOnly);

    m_sink = std::make_unique<QAudioSink>(output, format, this);
    m_sink->setVolume(m_volume);

    connect(m_sink.get(), &QAudioSink::stateChanged, this, [this](QAudio::State state) {
        if (state == QAudio::IdleState && m_buffer && m_buffer->drained()) {
            updatePlaying(false);
            emit playbackFinished();
        } else if (state == QAudio::StoppedState && m_sink
                   && m_sink->error() != QAudio::NoError) {
            updatePlaying(false);
            emit playbackError(QStringLiteral("Audio output stopped unexpectedly."));
        }
    });

    m_sink->start(m_buffer.get());
    updatePlaying(true);
    emit playbackStarted();
}

void AudioPlayback::appendSamples(const QVector<float> &samples)
{
    if (!m_buffer || samples.isEmpty())
        return;
    m_buffer->append(samples);
}

void AudioPlayback::endStream()
{
    if (m_buffer)
        m_buffer->markComplete();
}

void AudioPlayback::setVolume(qreal volume)
{
    m_volume = qBound(0.0, volume, 1.0);
    if (m_sink)
        m_sink->setVolume(m_volume);
}

void AudioPlayback::stopAll()
{
    const bool was = m_isPlaying;

    if (m_buffer)
        m_buffer->clear();
    if (m_sink) {
        m_sink->reset();   // discards the device-side buffer, not just the queue
        m_sink->stop();
    }

    updatePlaying(false);
    if (was)
        emit playbackFinished();
}

void AudioPlayback::teardown()
{
    if (m_sink) {
        m_sink->stop();
        m_sink.reset();
    }
    if (m_buffer) {
        m_buffer->close();
        m_buffer.reset();
    }
    updatePlaying(false);
}

void AudioPlayback::updatePlaying(bool playing)
{
    if (m_isPlaying == playing)
        return;
    m_isPlaying = playing;
    emit isPlayingChanged();
}

#endif // DATAFORM_VOICE_ENABLED
