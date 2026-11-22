#ifndef SPEECHRECOGNIZER_H
#define SPEECHRECOGNIZER_H

#ifdef DATAFORM_VOICE_ENABLED

#include <QObject>
#include <QString>
#include <QVector>
#include <QThread>
#include <QAtomicInt>
#include <memory>

namespace muninn {
class StreamingTranscriber;
}

/**
 * Worker that owns the Muninn streaming session.
 *
 * Muninn's model load and its flushing stop() both block for a noticeable time,
 * so the whole transcriber lives on a dedicated thread and the facade below talks
 * to it with queued calls only.
 */
class SttWorker : public QObject
{
    Q_OBJECT

public:
    explicit SttWorker(QObject *parent = nullptr);
    ~SttWorker();

    /** Set from the facade's thread; read from the worker to abandon stale utterances. */
    QAtomicInt cancelRequested{0};

public slots:
    void loadModel(const QString &modelPath, const QString &device,
                   const QString &computeType, const QString &language,
                   const QString &vadModelPath);
    void unloadModel();
    void beginUtterance(quint64 turnId);
    void pushAudio(const QVector<float> &samples, int sampleRate);
    void endUtterance(quint64 turnId);

signals:
    void modelLoaded(const QString &modelPath);
    void modelLoadFailed(const QString &error);
    void partialTranscript(quint64 turnId, const QString &text);
    void finalTranscript(quint64 turnId, const QString &text, const QString &language);
    void recognitionError(quint64 turnId, const QString &error);

private slots:
    void pollPartial();

private:
    std::unique_ptr<muninn::StreamingTranscriber> m_transcriber;
    QString m_modelPath;
    QString m_language;
    QString m_vadModelPath;
    quint64 m_activeTurn = 0;
    bool m_streaming = false;
    QString m_lastPartial;
    class QTimer *m_pollTimer = nullptr;
};

/**
 * Main-thread facade over Muninn streaming transcription.
 *
 * Transcripts are tagged with the turn that requested them so the session can
 * drop results that arrive after a cancellation.
 */
class SpeechRecognizer : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isModelLoaded READ isModelLoaded NOTIFY isModelLoadedChanged)
    Q_PROPERTY(QString modelPath READ modelPath NOTIFY isModelLoadedChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)

public:
    explicit SpeechRecognizer(QObject *parent = nullptr);
    ~SpeechRecognizer();

    bool isModelLoaded() const { return m_isModelLoaded; }
    QString modelPath() const { return m_modelPath; }
    QString lastError() const { return m_lastError; }

    /** Kicks off an asynchronous load; completion arrives via modelLoaded/modelLoadFailed. */
    /**
     * @param vadModelPath silero_vad.onnx for neural endpointing. Empty falls
     *        back to energy VAD, which needs no model.
     */
    Q_INVOKABLE void loadModel(const QString &modelPath,
                               const QString &device = QStringLiteral("cuda"),
                               const QString &computeType = QStringLiteral("float16"),
                               const QString &language = QStringLiteral("en"),
                               const QString &vadModelPath = QString());

    /** silero_vad.onnx beside the Whisper model directory, or empty if absent. */
    static QString defaultVadModelPath(const QString &sttModelPath);
    Q_INVOKABLE void unloadModel();

    void beginUtterance(quint64 turnId);
    void pushAudio(const QVector<float> &samples, int sampleRate);
    void endUtterance(quint64 turnId);

    /** Abandons the in-flight utterance; no final transcript will be emitted for it. */
    void cancel();

signals:
    void isModelLoadedChanged();
    void lastErrorChanged();

    void modelLoaded(const QString &modelPath);
    void modelLoadFailed(const QString &error);
    void partialTranscript(quint64 turnId, const QString &text);
    void finalTranscript(quint64 turnId, const QString &text, const QString &language);
    void recognitionError(quint64 turnId, const QString &error);

private:
    void setError(const QString &error);

    QThread m_thread;
    SttWorker *m_worker = nullptr;
    bool m_isModelLoaded = false;
    QString m_modelPath;
    QString m_lastError;
};

#endif // DATAFORM_VOICE_ENABLED
#endif // SPEECHRECOGNIZER_H
