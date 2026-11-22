#include "speechsynthesizer.h"

#ifdef DATAFORM_VOICE_ENABLED

#include "ttsbackends.h"

#include <QRegularExpression>
#include <QDebug>

namespace {

// Abbreviations whose trailing period does not end a sentence. Splitting after
// "Dr." would make the voice pause mid-name and hand Piper a one-word utterance.
const QStringList &abbreviations()
{
    static const QStringList list = {
        QStringLiteral("mr"),  QStringLiteral("mrs"), QStringLiteral("ms"),
        QStringLiteral("dr"),  QStringLiteral("prof"), QStringLiteral("st"),
        QStringLiteral("vs"),  QStringLiteral("etc"), QStringLiteral("e.g"),
        QStringLiteral("i.e"), QStringLiteral("approx"), QStringLiteral("no"),
        QStringLiteral("fig"), QStringLiteral("jr"),  QStringLiteral("sr"),
    };
    return list;
}

bool endsWithAbbreviation(const QString &text)
{
    // Take the last whitespace-delimited token and strip its trailing period.
    int start = text.size() - 1;
    while (start >= 0 && !text.at(start).isSpace())
        --start;
    QString token = text.mid(start + 1);
    if (token.endsWith(QLatin1Char('.')))
        token.chop(1);
    token = token.toLower();

    if (token.isEmpty())
        return false;

    // A single initial ("J.") is also not a sentence end.
    if (token.size() == 1 && token.at(0).isLetter())
        return true;

    return abbreviations().contains(token);
}

} // namespace

SpeechSynthesizer::SpeechSynthesizer(QObject *parent)
    : QObject(parent)
{
}

SpeechSynthesizer::~SpeechSynthesizer()
{
    if (m_backend)
        m_backend->cancel();
}

bool SpeechSynthesizer::isAvailable() const
{
    return m_backend && m_backend->isAvailable();
}

QString SpeechSynthesizer::backendName() const
{
    return m_backend ? m_backend->name() : QStringLiteral("None");
}

QString SpeechSynthesizer::unavailableReason() const
{
    if (!m_backend)
        return QStringLiteral("Spoken replies are turned off.");
    return m_backend->unavailableReason();
}

bool SpeechSynthesizer::producesSamples() const
{
    return m_backend && m_backend->sampleRate() > 0;
}

int SpeechSynthesizer::sampleRate() const
{
    return m_backend ? m_backend->sampleRate() : 0;
}

void SpeechSynthesizer::configureBackend(const QString &backendId, const QJsonObject &config)
{
    cancel();

    // Always rebuilt rather than diffed: the config carries voice, speed and
    // paths, and a stale backend that silently keeps the old voice is worse than
    // a load that is a few hundred milliseconds slower on a settings change.
    m_backend.reset();
    m_backendId = backendId;

    if (backendId.compare(QLatin1String("piper"), Qt::CaseInsensitive) == 0) {
        auto backend = std::make_unique<PiperBackend>(
            config.value(QStringLiteral("exePath")).toString(),
            config.value(QStringLiteral("modelPath")).toString(),
            config.value(QStringLiteral("speed")).toDouble(1.0));
        m_backend = std::move(backend);
#ifdef Q_OS_WIN
    } else if (backendId.compare(QLatin1String("sapi"), Qt::CaseInsensitive) == 0) {
        auto backend = std::make_unique<SapiBackend>(
            config.value(QStringLiteral("voiceName")).toString(),
            config.value(QStringLiteral("rate")).toInt(0));
        m_backend = std::move(backend);
#endif
    } else {
        if (backendId.compare(QLatin1String("none"), Qt::CaseInsensitive) != 0)
            qWarning() << "SpeechSynthesizer: unknown TTS backend" << backendId;
        emit backendChanged();
        return;
    }

    connect(m_backend.get(), &TtsBackend::samplesReady, this,
            [this](quint64, const QVector<float> &samples) {
                emit samplesReady(m_activeTurn, samples);
            });
    connect(m_backend.get(), &TtsBackend::speechFinished,
            this, &SpeechSynthesizer::onBackendFinished);
    connect(m_backend.get(), &TtsBackend::speechError, this,
            [this](quint64, const QString &message) {
                emit speechError(m_activeTurn, message);
            });
    emit backendChanged();
}

QStringList SpeechSynthesizer::availableSapiVoices() const
{
#ifdef Q_OS_WIN
    return SapiBackend::availableVoices();
#else
    return {};
#endif
}

QStringList SpeechSynthesizer::splitIntoSentences(const QString &text)
{
    QStringList sentences;
    const QString source = text.simplified();
    if (source.isEmpty())
        return sentences;

    int start = 0;
    for (int i = 0; i < source.size(); ++i) {
        const QChar c = source.at(i);
        const bool terminator = (c == QLatin1Char('.') || c == QLatin1Char('!')
                                 || c == QLatin1Char('?'));
        if (!terminator)
            continue;

        // Consume runs like "?!" or an ellipsis as a single terminator.
        int end = i;
        while (end + 1 < source.size()
               && (source.at(end + 1) == QLatin1Char('.')
                   || source.at(end + 1) == QLatin1Char('!')
                   || source.at(end + 1) == QLatin1Char('?'))) {
            ++end;
        }

        // A period between digits is a decimal, not a sentence break.
        if (c == QLatin1Char('.') && end == i && i > 0 && i + 1 < source.size()
            && source.at(i - 1).isDigit() && source.at(i + 1).isDigit()) {
            continue;
        }

        const QString candidate = source.mid(start, end - start + 1).trimmed();
        if (c == QLatin1Char('.') && end == i && endsWithAbbreviation(candidate)) {
            i = end;
            continue;
        }

        // Require the terminator to be followed by whitespace or end of text.
        if (end + 1 < source.size() && !source.at(end + 1).isSpace()) {
            i = end;
            continue;
        }

        if (!candidate.isEmpty())
            sentences << candidate;
        start = end + 1;
        i = end;
    }

    const QString tail = source.mid(start).trimmed();
    if (!tail.isEmpty())
        sentences << tail;

    return sentences;
}

void SpeechSynthesizer::speak(quint64 turnId, const QString &text)
{
    if (!m_backend || !m_backend->isAvailable())
        return;

    // A new turn supersedes anything still queued from an older one.
    if (turnId != m_activeTurn) {
        m_queue.clear();
        m_activeTurn = turnId;
    }
    m_turnComplete = false;

    const QStringList sentences = splitIntoSentences(text);
    for (const QString &sentence : sentences) {
        // Punctuation-only fragments would make the backend emit nothing and
        // waste a process launch.
        bool hasLetter = false;
        for (const QChar &c : sentence) {
            if (c.isLetterOrNumber()) {
                hasLetter = true;
                break;
            }
        }
        if (hasLetter)
            m_queue.enqueue({turnId, sentence});
    }

    pumpQueue();
}

void SpeechSynthesizer::finishTurn(quint64 turnId)
{
    if (turnId != m_activeTurn)
        return;

    m_turnComplete = true;

    // Nothing left to say — either the reply was silent or the last sentence has
    // already finished. Close the turn now rather than leaving the session
    // waiting on speech that will never arrive.
    if (m_queue.isEmpty() && !m_backendBusy) {
        setSpeaking(false);
        emit turnSpeechFinished(turnId);
    }
}

void SpeechSynthesizer::pumpQueue()
{
    if (m_backendBusy || m_queue.isEmpty() || !m_backend)
        return;

    const QueuedUtterance next = m_queue.dequeue();
    if (next.turnId != m_activeTurn) {
        pumpQueue();   // stale entry from a superseded turn
        return;
    }

    if (!m_isSpeaking) {
        setSpeaking(true);
        emit turnSpeechStarted(next.turnId);
    }

    m_backendBusy = true;
    m_backend->speak(m_nextRequestId++, next.text);
}

void SpeechSynthesizer::onBackendFinished(quint64)
{
    m_backendBusy = false;

    if (!m_queue.isEmpty()) {
        pumpQueue();
        return;
    }

    // Queue is empty, but the reply may still be streaming in; only a completed
    // turn ends the speech.
    if (m_turnComplete) {
        const quint64 finished = m_activeTurn;
        setSpeaking(false);
        emit turnSpeechFinished(finished);
    }
}

void SpeechSynthesizer::cancel()
{
    m_queue.clear();
    m_turnComplete = true;
    m_backendBusy = false;

    if (m_backend)
        m_backend->cancel();

    if (m_isSpeaking) {
        const quint64 cancelled = m_activeTurn;
        setSpeaking(false);
        emit turnSpeechFinished(cancelled);
    }
}

void SpeechSynthesizer::setSpeaking(bool speaking)
{
    if (m_isSpeaking == speaking)
        return;
    m_isSpeaking = speaking;
    emit isSpeakingChanged();
}

#endif // DATAFORM_VOICE_ENABLED
