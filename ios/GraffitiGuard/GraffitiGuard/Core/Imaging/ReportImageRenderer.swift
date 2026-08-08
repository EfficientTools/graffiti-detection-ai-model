import UIKit

@MainActor
enum ReportImageRenderer {
    static func render(image: UIImage, report: DetectionReport) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.opaque = true
        format.scale = 1
        format.preferredRange = .standard

        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        return renderer.image { context in
            image.draw(in: CGRect(origin: .zero, size: image.size))

            guard report.imageSize.width > 0, report.imageSize.height > 0 else { return }
            let scaleX = image.size.width / report.imageSize.width
            let scaleY = image.size.height / report.imageSize.height
            let lineWidth = max(3, min(image.size.width, image.size.height) * 0.005)
            let fontSize = max(14, min(28, image.size.width * 0.026))

            for detection in report.items {
                let box = CGRect(
                    x: detection.box.minX * scaleX,
                    y: detection.box.minY * scaleY,
                    width: detection.box.width * scaleX,
                    height: detection.box.height * scaleY
                ).intersection(CGRect(origin: .zero, size: image.size))
                guard !box.isNull, !box.isEmpty else { continue }

                let color: UIColor = detection.confidence >= 0.7 ? .systemGreen : .systemOrange
                context.cgContext.setStrokeColor(color.cgColor)
                context.cgContext.setLineWidth(lineWidth)
                context.cgContext.stroke(box.insetBy(dx: lineWidth / 2, dy: lineWidth / 2))

                let label =
                    "Graffiti  "
                    + detection.confidence.formatted(
                        .percent.precision(.fractionLength(0))
                    )
                let attributes: [NSAttributedString.Key: Any] = [
                    .font: UIFont.systemFont(ofSize: fontSize, weight: .bold),
                    .foregroundColor: UIColor.black,
                ]
                let textSize = label.size(withAttributes: attributes)
                let labelRect = CGRect(
                    x: box.minX,
                    y: max(0, box.minY - textSize.height - 12),
                    width: min(textSize.width + 16, image.size.width - box.minX),
                    height: textSize.height + 8
                )
                context.cgContext.setFillColor(color.cgColor)
                context.cgContext.fill(labelRect)
                label.draw(
                    at: CGPoint(x: labelRect.minX + 8, y: labelRect.minY + 4),
                    withAttributes: attributes
                )
            }
        }
    }
}
