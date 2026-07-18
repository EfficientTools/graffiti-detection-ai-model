import CoreGraphics
import CoreImage
@preconcurrency import CoreML
import CoreVideo
import Foundation

protocol GraffitiDetecting: Sendable {
    func prepare() async throws
    func detect(image: CGImage, threshold: Double) async throws -> DetectionResult
}

extension GraffitiDetecting {
    func prepare() async throws {}
}

actor OnDeviceGraffitiDetector: GraffitiDetecting {
    static let modelResourceName = "GraffitiDetector"

    private let modelURL: URL?
    private let imageContext = CIContext(options: [.cacheIntermediates: false])
    private var model: MLModel?
    private var inputBuffer: CVPixelBuffer?
    private var inputBufferSize: CGSize = .zero
    private var predictionInFlight = false
    private var predictionWaiters: [CheckedContinuation<Void, Never>] = []

    init(bundle: Bundle = .main) {
        modelURL = Self.modelURL(in: bundle)
    }

    static var isModelBundled: Bool {
        modelURL(in: .main) != nil
    }

    func prepare() async throws {
        _ = try await loadModel()
    }

    func detect(image: CGImage, threshold: Double) async throws -> DetectionResult {
        await acquirePredictionSlot()
        defer { releasePredictionSlot() }
        try Task.checkCancellation()
        guard threshold.isFinite else {
            throw DetectionError.inferenceFailed
        }
        let effectiveThreshold = min(max(threshold, 0.1), 0.9)

        let clock = ContinuousClock()
        let startedAt = clock.now
        let model = try await loadModel()
        let modelInput = try makeInput(for: image, model: model)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: modelInput.pixelBuffer),
            "confidenceThreshold": MLFeatureValue(double: effectiveThreshold),
            "iouThreshold": MLFeatureValue(double: 0.45),
        ])

        let prediction: any MLFeatureProvider
        do {
            prediction = try await model.prediction(from: provider)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw DetectionError.inferenceFailed
        }
        try Task.checkCancellation()

        guard
            let coordinates = prediction.featureValue(for: "coordinates")?.multiArrayValue,
            let confidence = prediction.featureValue(for: "confidence")?.multiArrayValue
        else {
            throw DetectionError.incompatibleModel
        }

        let items = try CoreMLDetectionDecoder.decode(
            coordinates: coordinates,
            confidence: confidence,
            transform: modelInput.transform,
            threshold: effectiveThreshold
        )
        let elapsed = startedAt.duration(to: clock.now)
        let milliseconds =
            Double(elapsed.components.seconds) * 1_000
            + Double(elapsed.components.attoseconds) / 1e15

        return DetectionResult(items: items, processingTimeMs: milliseconds)
    }

    private static func modelURL(in bundle: Bundle) -> URL? {
        bundle.url(forResource: modelResourceName, withExtension: "mlmodelc")
    }

    private func acquirePredictionSlot() async {
        guard predictionInFlight else {
            predictionInFlight = true
            return
        }

        await withCheckedContinuation { continuation in
            predictionWaiters.append(continuation)
        }
    }

    private func releasePredictionSlot() {
        guard !predictionWaiters.isEmpty else {
            predictionInFlight = false
            return
        }

        predictionWaiters.removeFirst().resume()
    }

    private func loadModel() async throws -> MLModel {
        if let model {
            return model
        }
        guard let modelURL else {
            throw DetectionError.modelUnavailable
        }

        let configuration = MLModelConfiguration()
        #if targetEnvironment(simulator)
            configuration.computeUnits = .cpuAndNeuralEngine
        #else
            configuration.computeUnits = .all
        #endif

        do {
            let loadedModel = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
            model = loadedModel
            return loadedModel
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw DetectionError.modelLoadFailed
        }
    }

    private func makeInput(for image: CGImage, model: MLModel) throws -> CoreMLModelInput {
        guard
            let constraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint,
            constraint.pixelsWide > 0,
            constraint.pixelsHigh > 0
        else {
            throw DetectionError.incompatibleModel
        }

        let transform = LetterboxTransform(
            sourceSize: CGSize(width: image.width, height: image.height),
            targetSize: CGSize(width: constraint.pixelsWide, height: constraint.pixelsHigh)
        )
        let pixelBuffer = try reusablePixelBuffer(
            width: constraint.pixelsWide,
            height: constraint.pixelsHigh
        )
        render(
            image: image,
            to: pixelBuffer,
            transform: transform
        )

        return CoreMLModelInput(pixelBuffer: pixelBuffer, transform: transform)
    }

    private func reusablePixelBuffer(
        width: Int,
        height: Int
    ) throws -> CVPixelBuffer {
        let requestedSize = CGSize(width: width, height: height)
        if inputBufferSize == requestedSize, let inputBuffer {
            return inputBuffer
        }

        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
        ]
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let buffer else {
            throw DetectionError.inferenceFailed
        }

        inputBuffer = buffer
        inputBufferSize = requestedSize
        return buffer
    }

    private func render(
        image: CGImage,
        to buffer: CVPixelBuffer,
        transform: LetterboxTransform
    ) {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let targetRect = CGRect(x: 0, y: 0, width: width, height: height)
        let background = CIImage(color: CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255))
            .cropped(to: targetRect)
        let source = CIImage(cgImage: image)
            .transformed(by: CGAffineTransform(scaleX: transform.scale, y: transform.scale))
            .transformed(
                by: CGAffineTransform(
                    translationX: transform.padding.width,
                    y: transform.padding.height
                )
            )
        imageContext.render(
            source.composited(over: background),
            to: buffer,
            bounds: targetRect,
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
    }
}

struct LetterboxTransform: Equatable, Sendable {
    let sourceSize: CGSize
    let targetSize: CGSize
    let scale: CGFloat
    let padding: CGSize

    init(sourceSize: CGSize, targetSize: CGSize) {
        self.sourceSize = sourceSize
        self.targetSize = targetSize
        scale = min(targetSize.width / sourceSize.width, targetSize.height / sourceSize.height)
        padding = CGSize(
            width: (targetSize.width - sourceSize.width * scale) / 2,
            height: (targetSize.height - sourceSize.height * scale) / 2
        )
    }

    func sourceRect(normalizedXYWH values: [Double]) -> CGRect? {
        guard values.count == 4, values.allSatisfy(\.isFinite) else { return nil }

        let centerX = CGFloat(values[0]) * targetSize.width
        let centerY = CGFloat(values[1]) * targetSize.height
        let width = CGFloat(values[2]) * targetSize.width
        let height = CGFloat(values[3]) * targetSize.height
        guard width > 0, height > 0 else { return nil }

        let modelRect = CGRect(
            x: centerX - width / 2,
            y: centerY - height / 2,
            width: width,
            height: height
        )
        let sourceRect = CGRect(
            x: (modelRect.minX - padding.width) / scale,
            y: (modelRect.minY - padding.height) / scale,
            width: modelRect.width / scale,
            height: modelRect.height / scale
        )
        let clipped = sourceRect.intersection(CGRect(origin: .zero, size: sourceSize))
        return clipped.isNull || clipped.isEmpty ? nil : clipped
    }
}

enum CoreMLDetectionDecoder {
    static func decode(
        coordinates: MLMultiArray,
        confidence: MLMultiArray,
        transform: LetterboxTransform,
        threshold: Double
    ) throws -> [GraffitiDetection] {
        guard coordinates.count.isMultiple(of: 4) else {
            throw DetectionError.incompatibleModel
        }

        let detectionCount = coordinates.count / 4
        guard detectionCount > 0 else { return [] }
        guard confidence.count.isMultiple(of: detectionCount) else {
            throw DetectionError.incompatibleModel
        }

        let classCount = confidence.count / detectionCount
        guard classCount > 0 else {
            throw DetectionError.incompatibleModel
        }

        let candidates = (0..<detectionCount).compactMap { row -> (Double, CGRect)? in
            let score = confidence[row * classCount].doubleValue
            guard score.isFinite, score >= threshold else { return nil }

            let values = (0..<4).map { column in
                coordinates[row * 4 + column].doubleValue
            }
            guard let box = transform.sourceRect(normalizedXYWH: values) else { return nil }
            return (score, box)
        }

        return
            candidates
            .sorted { $0.0 > $1.0 }
            .enumerated()
            .map { index, candidate in
                GraffitiDetection(id: index, confidence: candidate.0, box: candidate.1)
            }
    }
}

private struct CoreMLModelInput {
    let pixelBuffer: CVPixelBuffer
    let transform: LetterboxTransform
}
