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

    static func prepare(_ image: UIImage, maxDimension: CGFloat = 2_048) -> PreparedImage? {
        guard image.size.width > 0, image.size.height > 0 else { return nil }

        let scale = min(1, maxDimension / max(image.size.width, image.size.height))
        let targetSize = CGSize(
            width: max(1, (image.size.width * scale).rounded()),
            height: max(1, (image.size.height * scale).rounded())
        )

        let format = UIGraphicsImageRendererFormat()
        format.opaque = true
        format.scale = 1
        format.preferredRange = .standard

        let renderedImage = UIGraphicsImageRenderer(size: targetSize, format: format).image { context in
            UIColor.black.setFill()
            context.fill(CGRect(origin: .zero, size: targetSize))
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        guard let cgImage = renderedImage.cgImage else {
            return nil
        }

        return PreparedImage(image: renderedImage, cgImage: cgImage)
    }
}

private struct SendableCGImage: @unchecked Sendable {
    let value: CGImage
}
