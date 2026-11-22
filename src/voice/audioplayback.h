#ifndef AUDIOPLAYBACK_H
#define AUDIOPLAYBACK_H

#ifdef DATAFORM_VOICE_ENABLED

#include <QObject>
#include <QIODevice>
#include <QAudioFormat>
#include <QVector>
#include <QQueue>
#include <QMutex>
#include <memory>

class QAudioSink;

/**
 * Playback for synthesised speech.
 *
 * Audio is appended as it is produced so the first sentence can start playing
 * while later ones are still being synthesised. stopAll() exists to satisfy the
 * barge-in requirement: it drops queued audio and silences the sink immediately
 * rather than waiting for the current buffer to drain.
 */
class AudioPlayback : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isPlaying READ isPlaying NOTIFY isPlayingChanged)

public:
    explicit AudioPlayback(QObject *parent = nullptr);
    ~AudioPlayback();

    bool isPlaying() const { return m_isPlaying; }

    /** Opens the sink for a stream of chunks at the given rate. Safe to call repeatedly. */
    void beginStream(int sampleRate);

    /** Appends mono float32 samples to the current stream. */
    void appendSamples(const QVector<float> &samples);

    /** Signals that no further audio is coming; playback ends once the buffer drains. */
    void endStream();

    void setVolume(qreal volume);

public slots:
    /** Barge-in: discard queued audio and stop output now. */
    void stopAll();

signals:
    void isPlayingChanged();
    void playbackStarted();
    void playbackFinished();
    void playbackError(const QString &message);

private:
    class StreamBuffer;

    void teardown();
    void updatePlaying(bool playing);

    std::unique_ptr<QAudioSink> m_sink;
    std::unique_ptr<StreamBuffer> m_buffer;
    QAudioFormat m_format;
    int m_sampleRate = 0;
    bool m_isPlaying = false;
    qreal m_volume = 1.0;
};

#endif // DATAFORM_VOICE_ENABLED
#endif // AUDIOPLAYBACK_H
