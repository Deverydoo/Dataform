#ifndef CLIPBOARDHELPER_H
#define CLIPBOARDHELPER_H

#include <QObject>
#include <QString>

class ClipboardHelper : public QObject
{
    Q_OBJECT

public:
    explicit ClipboardHelper(QObject *parent = nullptr);

    // Check if clipboard contains an image
    Q_INVOKABLE bool hasImage() const;

    // Get clipboard image as PNG base64 string
    Q_INVOKABLE QString getImageBase64() const;

    // Load an image file and return as base64 string
    Q_INVOKABLE QString loadImageFileBase64(const QString &filePath) const;
};

#endif // CLIPBOARDHELPER_H
