import UIKit

struct PreparedImage {
    let image: UIImage
    let cgImage: CGImage
}

@MainActor
enum ImagePreparer {
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
