#ifndef AUDIOCAPTURE_H
#define AUDIOCAPTURE_H

#ifdef DATAFORM_VOICE_ENABLED

#include <QObject>
#include <QIODevice>
#include <QAudioFormat>
#include <QAudioDevice>
#include <QVector>
#include <memory>

class QAudioSource;
class QMediaDevices;

// Muninn resamples whatever it is handed, but capturing at its native rate keeps
// the conversion out of the hot path and halves the bytes crossing threads.
constexpr int kCaptureSampleRate = 16000;

/**
 * Microphone capture for the voice turn loop.
 *
 * Delivers fixed-size mono float32 chunks as they arrive rather than buffering a
 * whole utterance, so transcription can run while the user is still speaking.
 * Device loss is surfaced rather than swallowed: the plan requires typed input to
 * keep working when the microphone disappears.
 */
class AudioCapture : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isCapturing READ isCapturing NOTIFY isCapturingChanged)
    Q_PROPERTY(qreal inputLevel READ inputLevel NOTIFY inputLevelChanged)
    Q_PROPERTY(bool hasInputDevice READ hasInputDevice NOTIFY availabilityChanged)
    Q_PROPERTY(QString deviceName READ deviceName NOTIFY availabilityChanged)

public:
    explicit AudioCapture(QObject *parent = nullptr);
    ~AudioCapture();

    bool isCapturing() const { return m_isCapturing; }
    qreal inputLevel() const { return m_inputLevel; }
    bool hasInputDevice() const;
    QString deviceName() const;

    Q_INVOKABLE QStringList availableDevices() const;

    /** Selects an input by description; an empty or unknown name falls back to the system default. */
    void setPreferredDevice(const QString &description);

public slots:
    void start();
    void stop();

signals:
    void isCapturingChanged();
    void inputLevelChanged();
    void availabilityChanged();

    /** Mono float32 samples at kCaptureSampleRate. */
    void audioChunk(const QVector<float> &samples);

    void captureError(const QString &message);
    void deviceLost();

private slots:
    void onDevicesChanged();

private:
    class SampleSink;

    void rebuildSource();
    void handleSamples(const QVector<float> &samples);

    QMediaDevices *m_mediaDevices = nullptr;
    std::unique_ptr<QAudioSource> m_source;
    std::unique_ptr<SampleSink> m_sink;
    QAudioFormat m_format;
    QAudioDevice m_device;
    QString m_preferredDevice;
    bool m_isCapturing = false;
    qreal m_inputLevel = 0.0;
};

#endif // DATAFORM_VOICE_ENABLED
#endif // AUDIOCAPTURE_H
