import ImageIO
import UIKit

struct PreparedImage {
    let image: UIImage
    let cgImage: CGImage
}

@MainActor
enum ImagePreparer {
    static func prepare(data: Data, maxDimension: Int = 2_048) async -> PreparedImage? {
        guard maxDimension > 0 else { return nil }

        let decoded = await Task.detached(priority: .userInitiated) {
            guard
                let source = CGImageSourceCreateWithData(
                    data as CFData,
                    [kCGImageSourceShouldCache: false] as CFDictionary
                ),
                let image = CGImageSourceCreateThumbnailAtIndex(
                    source,
                    0,
                    [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceThumbnailMaxPixelSize: maxDimension,
                        kCGImageSourceShouldCacheImmediately: true,
                    ] as CFDictionary
                )
            else {
                return nil as SendableCGImage?
            }
            return SendableCGImage(value: image)
        }.value

        guard let cgImage = decoded?.value else { return nil }
        return PreparedImage(image: UIImage(cgImage: cgImage), cgImage: cgImage)
    }

    static func prepare(_ image: UIImage, maxDimension: CGFloat = 2_048) async -> PreparedImage? {
        guard image.size.width > 0, image.size.height > 0 else { return nil }

        let source = SendableUIImage(value: image)
        let rendered = await Task.detached(priority: .userInitiated) {
            let scale = min(1, maxDimension / max(source.value.size.width, source.value.size.height))
            let targetSize = CGSize(
                width: max(1, (source.value.size.width * scale).rounded()),
                height: max(1, (source.value.size.height * scale).rounded())
            )

            let format = UIGraphicsImageRendererFormat()
            format.opaque = true
            format.scale = 1
            format.preferredRange = .standard

            let image = UIGraphicsImageRenderer(size: targetSize, format: format).image { context in
                UIColor.black.setFill()
                context.fill(CGRect(origin: .zero, size: targetSize))
                source.value.draw(in: CGRect(origin: .zero, size: targetSize))
            }
            guard let cgImage = image.cgImage else { return nil as SendableCGImage? }
            return SendableCGImage(value: cgImage)
        }.value

        guard let cgImage = rendered?.value else { return nil }

        return PreparedImage(image: UIImage(cgImage: cgImage), cgImage: cgImage)
    }
}

private struct SendableCGImage: @unchecked Sendable {
    let value: CGImage
}

private struct SendableUIImage: @unchecked Sendable {
    let value: UIImage
}
