#ifndef WEIGHTMERGER_H
#define WEIGHTMERGER_H

#ifdef DATAFORM_TRAINING_ENABLED

#include <QObject>
#include <QString>
#include <QList>

class WeightMerger : public QObject
{
    Q_OBJECT

public:
    explicit WeightMerger(QObject *parent = nullptr);
    ~WeightMerger();

    struct MergeInput {
        QString checkpointPath;
        double weight = 1.0;
    };

    // Average multiple checkpoints into a new checkpoint
    bool mergeCheckpoints(const QList<MergeInput> &inputs,
                          const QString &outputCheckpointPath,
                          const QString &artifactsDir);

    // Export merged checkpoint as inference model
    bool exportMerged(const QString &checkpointPath,
                      const QString &artifactsDir,
                      const QString &outputModelPath);

    QString lastError() const { return m_lastError; }

signals:
    void mergeComplete(const QString &outputPath);
    void mergeError(const QString &error);

private:
    QString m_lastError;
};

#endif // DATAFORM_TRAINING_ENABLED
#endif // WEIGHTMERGER_H
