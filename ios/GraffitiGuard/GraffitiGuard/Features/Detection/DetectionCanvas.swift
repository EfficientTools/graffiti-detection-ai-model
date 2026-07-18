import SwiftUI
import UIKit

struct DetectionCanvas: View {
    let image: UIImage?
    let report: DetectionReport?
    let onChooseImage: () -> Void

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topLeading) {
                Color.guardInk

                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: geometry.size.width, height: geometry.size.height)

                    if let report {
                        ForEach(report.items) { detection in
                            if let rect = DetectionOverlayLayout.map(
                                box: detection.box,
                                imageSize: report.imageSize,
                                containerSize: geometry.size
                            ) {
                                DetectionBox(rect: rect, confidence: detection.confidence)
                            }
                        }
                    }
                } else {
                    Button(action: onChooseImage) {
                        emptyState
                            .frame(width: geometry.size.width, height: geometry.size.height)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Choose a street image")
                    .accessibilityHint("Choose a photo or file, or take a photo")
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        }
        .accessibilityElement(children: .contain)
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Image(systemName: "viewfinder.rectangular")
                .font(.system(size: 54, weight: .light))
                .foregroundStyle(Color.guardGreen)

            Text("Choose a street image")
                .font(.headline)
                .foregroundStyle(.white)

            Text("Choose from Photos or Files, or take a photo")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.60))
        }
    }
}

private struct DetectionBox: View {
    let rect: CGRect
    let confidence: Double

    private var color: Color {
        confidence >= 0.7 ? .guardGreen : .guardAmber
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(color.opacity(0.08))
                .overlay {
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .stroke(color, lineWidth: 3)
                }
                .frame(width: rect.width, height: rect.height)
                .offset(x: rect.minX, y: rect.minY)

            Text("Graffiti  \(confidence.formatted(.percent.precision(.fractionLength(0))))")
                .font(.caption2.weight(.bold))
                .foregroundStyle(Color.guardInk)
                .padding(.horizontal, 7)
                .padding(.vertical, 4)
                .background(color, in: Capsule())
                .offset(x: rect.minX, y: max(4, rect.minY - 25))
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Graffiti detected")
        .accessibilityValue(confidence.formatted(.percent.precision(.fractionLength(0))))
    }
}
