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

    var items: [GraffitiDetection] {
        result.items
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
