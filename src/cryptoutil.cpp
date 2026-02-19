#include "cryptoutil.h"
#include <QCryptographicHash>
#include <QRandomGenerator>
#include <QSysInfo>
#include <QFile>
#include <QDebug>

#ifdef Q_OS_WIN
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

QByteArray CryptoUtil::deriveKey(const QString &passphrase, const QByteArray &salt)
{
    // Iterated SHA-256 key derivation (10000 rounds)
    QByteArray data = passphrase.toUtf8() + salt;
    for (int i = 0; i < 10000; ++i) {
        data = QCryptographicHash::hash(data, QCryptographicHash::Sha256);
    }
    return data; // 32 bytes = 256 bits
}

QString CryptoUtil::machinePassphrase()
{
    // Profile-based passphrase: portable across machines while still
    // encrypting data at rest against casual disk access.
    // Previously used QSysInfo::machineUniqueId() which locked data to one machine.
    return "DATAFORM_PROFILE_KEY_v1";
}

QString CryptoUtil::legacyMachinePassphrase()
{
    // The old passphrase that was machine-locked. Used for one-time migration
    // of databases encrypted before the switch to a portable key.
    return QString::fromUtf8(QSysInfo::machineUniqueId());
}

QByteArray CryptoUtil::generateRandomBytes(int length)
{
#ifdef Q_OS_WIN
    QByteArray bytes(length, 0);
    NTSTATUS status = BCryptGenRandom(
        nullptr,
        reinterpret_cast<PUCHAR>(bytes.data()),
        static_cast<ULONG>(length),
        BCRYPT_USE_SYSTEM_PREFERRED_RNG
    );
    if (status != 0) {
        qWarning() << "BCryptGenRandom failed with status:" << status;
        // Fallback: use Qt's random
        for (int i = 0; i < length; ++i) {
            bytes[i] = static_cast<char>(QRandomGenerator::securelySeeded().generate() & 0xFF);
        }
    }
    return bytes;
#else
    QByteArray bytes(length, 0);
    for (int i = 0; i < length; ++i) {
        bytes[i] = static_cast<char>(QRandomGenerator::securelySeeded().generate() & 0xFF);
    }
    return bytes;
#endif
}

QByteArray CryptoUtil::pkcs7Pad(const QByteArray &data, int blockSize)
{
    int padding = blockSize - (data.size() % blockSize);
    QByteArray padded = data;
    padded.append(QByteArray(padding, static_cast<char>(padding)));
    return padded;
}

QByteArray CryptoUtil::pkcs7Unpad(const QByteArray &data)
{
    if (data.isEmpty()) return data;
    int padding = static_cast<unsigned char>(data.back());
    if (padding <= 0 || padding > 16 || padding > data.size()) {
        qWarning() << "Invalid PKCS7 padding";
        return QByteArray();
    }
    // Verify all padding bytes are correct
    for (int i = data.size() - padding; i < data.size(); ++i) {
        if (static_cast<unsigned char>(data[i]) != padding) {
            qWarning() << "Invalid PKCS7 padding bytes";
            return QByteArray();
        }
    }
    return data.left(data.size() - padding);
}

QByteArray CryptoUtil::encrypt(const QByteArray &plaintext, const QByteArray &key)
{
#ifdef Q_OS_WIN
    if (key.size() != 32) {
        qWarning() << "Encryption key must be 32 bytes (AES-256)";
        return QByteArray();
    }

    QByteArray iv = generateRandomBytes(16);
    QByteArray padded = pkcs7Pad(plaintext);

    BCRYPT_ALG_HANDLE hAlg = nullptr;
    BCRYPT_KEY_HANDLE hKey = nullptr;
    QByteArray result;

    NTSTATUS status = BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_AES_ALGORITHM, nullptr, 0);
    if (status != 0) {
        qWarning() << "BCryptOpenAlgorithmProvider failed:" << status;
        return QByteArray();
    }

    status = BCryptSetProperty(hAlg, BCRYPT_CHAINING_MODE,
                               (PUCHAR)BCRYPT_CHAIN_MODE_CBC,
                               sizeof(BCRYPT_CHAIN_MODE_CBC), 0);
    if (status != 0) {
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return QByteArray();
    }

    status = BCryptGenerateSymmetricKey(hAlg, &hKey, nullptr, 0,
                                         (PUCHAR)key.data(), key.size(), 0);
    if (status != 0) {
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return QByteArray();
    }

    // Determine output size
    ULONG cbCipherText = 0;
    QByteArray ivCopy = iv; // BCryptEncrypt modifies the IV buffer
    status = BCryptEncrypt(hKey,
                           (PUCHAR)padded.data(), padded.size(),
                           nullptr,
                           (PUCHAR)ivCopy.data(), ivCopy.size(),
                           nullptr, 0, &cbCipherText, 0);
    if (status != 0) {
        BCryptDestroyKey(hKey);
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return QByteArray();
    }

    QByteArray cipherText(static_cast<int>(cbCipherText), 0);
    ivCopy = iv; // Reset IV
    status = BCryptEncrypt(hKey,
                           (PUCHAR)padded.data(), padded.size(),
                           nullptr,
                           (PUCHAR)ivCopy.data(), ivCopy.size(),
                           (PUCHAR)cipherText.data(), cbCipherText,
                           &cbCipherText, 0);

    BCryptDestroyKey(hKey);
    BCryptCloseAlgorithmProvider(hAlg, 0);

    if (status != 0) {
        qWarning() << "BCryptEncrypt failed:" << status;
        return QByteArray();
    }

    // Return: IV(16) + ciphertext
    result = iv + cipherText;
    return result;
#else
    Q_UNUSED(plaintext);
    Q_UNUSED(key);
    qWarning() << "Encryption not implemented for this platform";
    return QByteArray();
#endif
}

QByteArray CryptoUtil::decrypt(const QByteArray &ciphertext, const QByteArray &key)
{
#ifdef Q_OS_WIN
    if (key.size() != 32) {
        qWarning() << "Decryption key must be 32 bytes (AES-256)";
        return QByteArray();
    }
    if (ciphertext.size() < 32) { // 16 IV + at least 16 ciphertext
        qWarning() << "Ciphertext too short";
        return QByteArray();
    }

    QByteArray iv = ciphertext.left(16);
    QByteArray encrypted = ciphertext.mid(16);

    BCRYPT_ALG_HANDLE hAlg = nullptr;
    BCRYPT_KEY_HANDLE hKey = nullptr;

    NTSTATUS status = BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_AES_ALGORITHM, nullptr, 0);
    if (status != 0) return QByteArray();

    status = BCryptSetProperty(hAlg, BCRYPT_CHAINING_MODE,
                               (PUCHAR)BCRYPT_CHAIN_MODE_CBC,
                               sizeof(BCRYPT_CHAIN_MODE_CBC), 0);
    if (status != 0) {
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return QByteArray();
    }

    status = BCryptGenerateSymmetricKey(hAlg, &hKey, nullptr, 0,
                                         (PUCHAR)key.data(), key.size(), 0);
    if (status != 0) {
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return QByteArray();
    }

    ULONG cbPlainText = 0;
    QByteArray ivCopy = iv;
    status = BCryptDecrypt(hKey,
                           (PUCHAR)encrypted.data(), encrypted.size(),
                           nullptr,
                           (PUCHAR)ivCopy.data(), ivCopy.size(),
                           nullptr, 0, &cbPlainText, 0);
    if (status != 0) {
        BCryptDestroyKey(hKey);
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return QByteArray();
    }

    QByteArray plainText(static_cast<int>(cbPlainText), 0);
    ivCopy = iv;
    status = BCryptDecrypt(hKey,
                           (PUCHAR)encrypted.data(), encrypted.size(),
                           nullptr,
                           (PUCHAR)ivCopy.data(), ivCopy.size(),
                           (PUCHAR)plainText.data(), cbPlainText,
                           &cbPlainText, 0);

    BCryptDestroyKey(hKey);
    BCryptCloseAlgorithmProvider(hAlg, 0);

    if (status != 0) {
        qWarning() << "BCryptDecrypt failed:" << status;
        return QByteArray();
    }

    return pkcs7Unpad(plainText);
#else
    Q_UNUSED(ciphertext);
    Q_UNUSED(key);
    return QByteArray();
#endif
}

bool CryptoUtil::encryptFile(const QString &plaintextPath, const QString &encryptedPath,
                              const QByteArray &key)
{
    QFile inFile(plaintextPath);
    if (!inFile.open(QIODevice::ReadOnly)) {
        qWarning() << "Cannot open file for encryption:" << plaintextPath;
        return false;
    }
    QByteArray plaintext = inFile.readAll();
    inFile.close();

    QByteArray encrypted = encrypt(plaintext, key);
    if (encrypted.isEmpty() && !plaintext.isEmpty()) {
        qWarning() << "Encryption produced empty output";
        return false;
    }

    QFile outFile(encryptedPath);
    if (!outFile.open(QIODevice::WriteOnly)) {
        qWarning() << "Cannot open file for writing:" << encryptedPath;
        return false;
    }
    outFile.write(encrypted);
    outFile.close();

    qDebug() << "Encrypted" << plaintextPath << "->" << encryptedPath
             << "(" << plaintext.size() << "->" << encrypted.size() << "bytes)";
    return true;
}

bool CryptoUtil::decryptFile(const QString &encryptedPath, const QString &plaintextPath,
                              const QByteArray &key)
{
    QFile inFile(encryptedPath);
    if (!inFile.open(QIODevice::ReadOnly)) {
        qDebug() << "No encrypted file found:" << encryptedPath << "(first run?)";
        return false;
    }
    QByteArray encrypted = inFile.readAll();
    inFile.close();

    QByteArray plaintext = decrypt(encrypted, key);
    if (plaintext.isEmpty() && !encrypted.isEmpty()) {
        qWarning() << "Decryption failed for:" << encryptedPath;
        return false;
    }

    QFile outFile(plaintextPath);
    if (!outFile.open(QIODevice::WriteOnly)) {
        qWarning() << "Cannot open file for writing:" << plaintextPath;
        return false;
    }
    outFile.write(plaintext);
    outFile.close();

    qDebug() << "Decrypted" << encryptedPath << "->" << plaintextPath
             << "(" << encrypted.size() << "->" << plaintext.size() << "bytes)";
    return true;
}
