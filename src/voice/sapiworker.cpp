#include "sapiworker.h"

#if defined(DATAFORM_VOICE_ENABLED) && defined(Q_OS_WIN)

#include <windows.h>
#include <objbase.h>
#include <sapi.h>

namespace {

/**
 * Reads a voice token's display name.
 *
 * Tokens expose their friendly name under the Attributes key; the token's own
 * default string value is the registry path, which is not what a user picks from.
 */
QString tokenDisplayName(ISpObjectToken *token)
{
    if (!token)
        return QString();

    ISpDataKey *attributes = nullptr;
    if (FAILED(token->OpenKey(L"Attributes", &attributes)) || !attributes)
        return QString();

    LPWSTR value = nullptr;
    QString name;
    if (SUCCEEDED(attributes->GetStringValue(L"Name", &value)) && value) {
        name = QString::fromWCharArray(value);
        ::CoTaskMemFree(value);
    }
    attributes->Release();
    return name;
}

/** Opens the installed-voices category without pulling in sphelper.h. */
ISpObjectTokenCategory *openVoiceCategory()
{
    ISpObjectTokenCategory *category = nullptr;
    if (FAILED(::CoCreateInstance(CLSID_SpObjectTokenCategory, nullptr, CLSCTX_ALL,
                                  IID_ISpObjectTokenCategory,
                                  reinterpret_cast<void **>(&category)))) {
        return nullptr;
    }
    if (FAILED(category->SetId(SPCAT_VOICES, FALSE))) {
        category->Release();
        return nullptr;
    }
    return category;
}

} // namespace

SapiWorker::SapiWorker(QString voiceName, int rate)
    : m_voiceName(std::move(voiceName))
    , m_rate(rate)
{
}

SapiWorker::~SapiWorker()
{
    shutdown();
}

void SapiWorker::initialise()
{
    m_initOk = false;
    m_initMessage.clear();

    // Apartment-threaded: SAPI is happiest there, and this thread owns the voice
    // for its whole lifetime so no marshalling is needed.
    const HRESULT coHr = ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(coHr)) {
        m_initMessage = QStringLiteral("Could not initialise COM for SAPI.");
        return;
    }
    m_comReady = (coHr != RPC_E_CHANGED_MODE);

    if (FAILED(::CoCreateInstance(CLSID_SpVoice, nullptr, CLSCTX_ALL, IID_ISpVoice,
                                  reinterpret_cast<void **>(&m_voice)))
        || !m_voice) {
        m_initMessage = QStringLiteral("No SAPI voice is installed on this system.");
        return;
    }

    // A missing named voice is not fatal — the default still speaks — but the
    // reason is kept so the UI can say why it is not the voice that was chosen.
    if (!m_voiceName.isEmpty() && !selectVoice(m_voiceName)) {
        m_initMessage = QStringLiteral("SAPI voice '%1' was not found; using the system default.")
                            .arg(m_voiceName);
    }

    if (m_rate != 0)
        m_voice->SetRate(LONG(qBound(-10, m_rate, 10)));

    m_initOk = true;
}

bool SapiWorker::selectVoice(const QString &wanted)
{
    ISpObjectTokenCategory *category = openVoiceCategory();
    if (!category)
        return false;

    IEnumSpObjectTokens *tokens = nullptr;
    if (FAILED(category->EnumTokens(nullptr, nullptr, &tokens)) || !tokens) {
        category->Release();
        return false;
    }

    bool selected = false;
    ISpObjectToken *token = nullptr;
    while (!selected && tokens->Next(1, &token, nullptr) == S_OK) {
        const QString name = tokenDisplayName(token);
        if (!name.isEmpty() && name.contains(wanted, Qt::CaseInsensitive)) {
            selected = SUCCEEDED(m_voice->SetVoice(token));
        }
        token->Release();
        token = nullptr;
    }

    tokens->Release();
    category->Release();
    return selected;
}

void SapiWorker::speak(quint64 requestId, const QString &text)
{
    if (!m_voice) {
        emit speechError(requestId, QStringLiteral("SAPI voice is not available."));
        emit speechFinished(requestId);
        return;
    }

    cancelRequested.storeRelaxed(0);

    const std::wstring wide = text.toStdWString();
    const HRESULT hr = m_voice->Speak(wide.c_str(), SPF_ASYNC | SPF_PURGEBEFORESPEAK, nullptr);
    if (FAILED(hr)) {
        emit speechError(requestId, QStringLiteral("SAPI refused the text (0x%1).")
                                        .arg(quint32(hr), 8, 16, QLatin1Char('0')));
        emit speechFinished(requestId);
        return;
    }

    // Slicing the wait keeps a cancel inside the 300 ms stop budget.
    while (m_voice->WaitUntilDone(50) == S_FALSE) {
        if (cancelRequested.loadRelaxed()) {
            m_voice->Speak(nullptr, SPF_PURGEBEFORESPEAK, nullptr);
            return;
        }
    }

    if (!cancelRequested.loadRelaxed())
        emit speechFinished(requestId);
}

void SapiWorker::shutdown()
{
    if (m_voice) {
        m_voice->Speak(nullptr, SPF_PURGEBEFORESPEAK, nullptr);
        m_voice->Release();
        m_voice = nullptr;
    }
    if (m_comReady) {
        ::CoUninitialize();
        m_comReady = false;
    }
}

QStringList SapiWorker::availableVoices()
{
    QStringList names;

    const HRESULT coHr = ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const bool ownsCom = SUCCEEDED(coHr) && coHr != RPC_E_CHANGED_MODE;
    if (FAILED(coHr) && coHr != RPC_E_CHANGED_MODE)
        return names;

    if (ISpObjectTokenCategory *category = openVoiceCategory()) {
        IEnumSpObjectTokens *tokens = nullptr;
        if (SUCCEEDED(category->EnumTokens(nullptr, nullptr, &tokens)) && tokens) {
            ISpObjectToken *token = nullptr;
            while (tokens->Next(1, &token, nullptr) == S_OK) {
                const QString name = tokenDisplayName(token);
                if (!name.isEmpty())
                    names << name;
                token->Release();
                token = nullptr;
            }
            tokens->Release();
        }
        category->Release();
    }

    if (ownsCom)
        ::CoUninitialize();

    return names;
}

#endif // DATAFORM_VOICE_ENABLED && Q_OS_WIN
