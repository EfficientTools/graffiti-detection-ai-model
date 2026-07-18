import CoreGraphics

enum DetectionOverlayLayout {
    static func aspectFitRect(imageSize: CGSize, containerSize: CGSize) -> CGRect {
        guard imageSize.width > 0,
            imageSize.height > 0,
            containerSize.width > 0,
            containerSize.height > 0
        else {
            return .zero
        }

        let scale = min(
            containerSize.width / imageSize.width,
            containerSize.height / imageSize.height
        )
        let fittedSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)

        return CGRect(
            x: (containerSize.width - fittedSize.width) / 2,
            y: (containerSize.height - fittedSize.height) / 2,
            width: fittedSize.width,
            height: fittedSize.height
        )
    }

    static func map(
        box: CGRect,
        imageSize: CGSize,
        containerSize: CGSize
    ) -> CGRect? {
        let fittedRect = aspectFitRect(imageSize: imageSize, containerSize: containerSize)
        guard !fittedRect.isEmpty else { return nil }

        let clippedBox = box.standardized.intersection(CGRect(origin: .zero, size: imageSize))
        guard !clippedBox.isNull, !clippedBox.isEmpty else { return nil }

        let scaleX = fittedRect.width / imageSize.width
        let scaleY = fittedRect.height / imageSize.height

        return CGRect(
            x: fittedRect.minX + clippedBox.minX * scaleX,
            y: fittedRect.minY + clippedBox.minY * scaleY,
            width: clippedBox.width * scaleX,
            height: clippedBox.height * scaleY
        )
    }
}
