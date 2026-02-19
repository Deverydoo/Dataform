#include "clipboardhelper.h"
#include <QGuiApplication>
#include <QClipboard>
#include <QMimeData>
#include <QImage>
#include <QBuffer>
#include <QFile>
#include <QFileInfo>
#include <QDebug>

ClipboardHelper::ClipboardHelper(QObject *parent)
    : QObject(parent)
{
}

bool ClipboardHelper::hasImage() const
{
    QClipboard *clipboard = QGuiApplication::clipboard();
    if (!clipboard) return false;
    const QMimeData *mimeData = clipboard->mimeData();
    return mimeData && mimeData->hasImage();
}

QString ClipboardHelper::getImageBase64() const
{
    QClipboard *clipboard = QGuiApplication::clipboard();
    if (!clipboard) return QString();

    const QMimeData *mimeData = clipboard->mimeData();
    if (!mimeData || !mimeData->hasImage()) return QString();

    QImage image = qvariant_cast<QImage>(mimeData->imageData());
    if (image.isNull()) return QString();

    // Scale down if very large (max 2048px on longest side)
    if (image.width() > 2048 || image.height() > 2048) {
        image = image.scaled(2048, 2048, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    QByteArray byteArray;
    QBuffer buffer(&byteArray);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "PNG");

    QString base64 = byteArray.toBase64();
    qDebug() << "ClipboardHelper: captured image"
             << image.width() << "x" << image.height()
             << "base64 size:" << base64.length();
    return base64;
}

QString ClipboardHelper::loadImageFileBase64(const QString &filePath) const
{
    if (filePath.isEmpty()) return QString();

    // Handle file:/// URLs
    QString path = filePath;
    if (path.startsWith("file:///")) {
        path = path.mid(8);  // Strip file:///
    } else if (path.startsWith("file://")) {
        path = path.mid(7);
    }

    QFileInfo fileInfo(path);
    if (!fileInfo.exists()) {
        qWarning() << "ClipboardHelper: file not found:" << path;
        return QString();
    }

    // Check file extension
    QString suffix = fileInfo.suffix().toLower();
    if (suffix != "png" && suffix != "jpg" && suffix != "jpeg" &&
        suffix != "gif" && suffix != "bmp" && suffix != "webp") {
        qWarning() << "ClipboardHelper: unsupported image format:" << suffix;
        return QString();
    }

    // Load and optionally resize
    QImage image(path);
    if (image.isNull()) {
        qWarning() << "ClipboardHelper: failed to load image:" << path;
        return QString();
    }

    // Scale down if very large
    if (image.width() > 2048 || image.height() > 2048) {
        image = image.scaled(2048, 2048, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    QByteArray byteArray;
    QBuffer buffer(&byteArray);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "PNG");

    QString base64 = byteArray.toBase64();
    qDebug() << "ClipboardHelper: loaded image from" << path
             << image.width() << "x" << image.height()
             << "base64 size:" << base64.length();
    return base64;
}
