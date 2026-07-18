import CoreGraphics
import Foundation

struct DetectionResult: Equatable, Sendable {
    let items: [GraffitiDetection]
    let processingTimeMs: Double

    var count: Int {
        items.count
    }

    var strongestConfidence: Double? {
        items.map(\.confidence).max()
    }
}

struct GraffitiDetection: Identifiable, Equatable, Sendable {
    let id: Int
    let confidence: Double
    let box: CGRect
}

struct DetectionReport: Equatable, Sendable {
    let result: DetectionResult
    let imageSize: CGSize
    let threshold: Double
    let completedAt: Date
    let reference: String

    init(
        result: DetectionResult,
        imageSize: CGSize,
        threshold: Double,
        completedAt: Date = Date(),
        reference: String? = nil
    ) {
        self.result = result
        self.imageSize = imageSize
        self.threshold = threshold
        self.completedAt = completedAt
        self.reference = reference ?? Self.makeReference()
    }

    var items: [GraffitiDetection] {
        result.items
    }

    var shareText: String {
        let outcome = result.count > 0 ? "Likely graffiti detected" : "No likely graffiti detected"
        let highestConfidence =
            result.strongestConfidence?.formatted(
                .percent.precision(.fractionLength(0))
            ) ?? "Not applicable"
        let processingTime = result.processingTimeMs.formatted(
            .number.precision(.fractionLength(0))
        )

        return """
            Graffiti Guard inspection
            Reference: \(reference)
            Completed: \(completedAt.formatted(date: .abbreviated, time: .shortened))
            Result: \(outcome)
            Detected regions: \(result.count)
            Highest confidence: \(highestConfidence)
            Minimum confidence: \(threshold.formatted(.percent.precision(.fractionLength(0))))
            Model processing: \(processingTime) ms

            Processed privately on this device. Human verification is required before any response or enforcement decision.
            """
    }

    private static func makeReference() -> String {
        "GG-" + UUID().uuidString.prefix(6).uppercased()
    }
}

enum DetectionError: LocalizedError, Equatable, Sendable {
    case invalidImage
    case modelUnavailable
    case incompatibleModel
    case modelLoadFailed
    case inferenceFailed

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            "The selected image could not be prepared for detection."
        case .modelUnavailable:
            "Add GraffitiDetector.mlpackage to the app before running detection."
        case .incompatibleModel:
            "The bundled model does not match the Graffiti Guard Core ML contract."
        case .modelLoadFailed:
            "The bundled Core ML model could not be loaded."
        case .inferenceFailed:
            "On-device graffiti detection failed."
        }
    }
}
