#ifndef CRYPTOUTIL_H
#define CRYPTOUTIL_H

#include <QByteArray>
#include <QString>

class CryptoUtil
{
public:
    // Derive a 256-bit key from a passphrase + salt using iterated SHA-256
    static QByteArray deriveKey(const QString &passphrase, const QByteArray &salt);

    // Generate the current portable passphrase
    static QString machinePassphrase();

    // Legacy machine-specific passphrase (for migrating old encrypted data)
    static QString legacyMachinePassphrase();

    // Encrypt raw bytes with AES-256-CBC via Windows BCrypt
    // Returns: salt(16) + IV(16) + ciphertext
    static QByteArray encrypt(const QByteArray &plaintext, const QByteArray &key);

    // Decrypt: expects salt(16) + IV(16) + ciphertext
    static QByteArray decrypt(const QByteArray &ciphertext, const QByteArray &key);

    // File-level encrypt/decrypt
    static bool encryptFile(const QString &plaintextPath, const QString &encryptedPath,
                            const QByteArray &key);
    static bool decryptFile(const QString &encryptedPath, const QString &plaintextPath,
                            const QByteArray &key);

    // Generate cryptographically random bytes
    static QByteArray generateRandomBytes(int length);

private:
    static QByteArray pkcs7Pad(const QByteArray &data, int blockSize = 16);
    static QByteArray pkcs7Unpad(const QByteArray &data);
};

#endif // CRYPTOUTIL_H
