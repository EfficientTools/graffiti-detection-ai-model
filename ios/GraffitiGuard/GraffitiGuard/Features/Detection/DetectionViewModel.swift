import Combine
import UIKit

@MainActor
final class DetectionViewModel: ObservableObject {
    enum Phase: Equatable {
        case preparingModel
        case empty
        case modelUnavailable
        case ready
        case detecting
        case complete
        case failed
    }

    @Published private(set) var image: UIImage?
    @Published private(set) var report: DetectionReport?
    @Published private(set) var phase: Phase = .empty
    @Published private(set) var errorMessage: String?

    let modelIsAvailable: Bool

    private let detector: any GraffitiDetecting
    private var cgImage: CGImage?
    private var selectionID = UUID()
    private var detectionTask: Task<DetectionResult, any Error>?
    private var activeDetectionID: UUID?

    init(
        detector: any GraffitiDetecting = OnDeviceGraffitiDetector(),
        modelIsAvailable: Bool = OnDeviceGraffitiDetector.isModelBundled
    ) {
        self.detector = detector
        self.modelIsAvailable = modelIsAvailable
        phase = modelIsAvailable ? .preparingModel : .modelUnavailable
    }

    var canDetect: Bool {
        modelIsAvailable && image != nil && phase != .detecting
    }

    var canReset: Bool {
        image != nil || report != nil || errorMessage != nil
    }

    func prepareModel() async {
        guard modelIsAvailable else { return }

        do {
            try await detector.prepare()
            if phase == .preparingModel {
                phase = image == nil ? .empty : .ready
            }
        } catch is CancellationError {
            return
        } catch {
            showError(error.localizedDescription)
        }
    }

    @discardableResult
    func loadImageData(_ data: Data) async -> Bool {
        guard let prepared = await ImagePreparer.prepare(data: data) else {
            showError(DetectionError.invalidImage.localizedDescription)
            return false
        }
        apply(prepared)
        return true
    }

    @discardableResult
    func loadSample() async -> Bool {
        guard let image = UIImage(named: "DemoStreet") else {
            showError("The sample scene could not be loaded.")
            return false
        }
        return await select(image)
    }

    @discardableResult
    func select(_ image: UIImage) async -> Bool {
        guard let prepared = await ImagePreparer.prepare(image) else {
            showError(DetectionError.invalidImage.localizedDescription)
            return false
        }

        apply(prepared)
        return true
    }

    private func apply(_ prepared: PreparedImage) {
        cancelDetection()
        selectionID = UUID()
        self.image = prepared.image
        cgImage = prepared.cgImage
        report = nil
        errorMessage = nil
        phase = modelIsAvailable ? .ready : .modelUnavailable
    }

    func reset() {
        cancelDetection()
        selectionID = UUID()
        image = nil
        cgImage = nil
        report = nil
        errorMessage = nil
        phase = modelIsAvailable ? .empty : .modelUnavailable
    }

    func detect(threshold: Double) async {
        guard modelIsAvailable else {
            showError(DetectionError.modelUnavailable.localizedDescription)
            return
        }
        guard let image, let cgImage else {
            showError(DetectionError.invalidImage.localizedDescription)
            return
        }

        let activeSelection = selectionID
        let detectionID = UUID()
        phase = .detecting
        errorMessage = nil
        activeDetectionID = detectionID

        detectionTask?.cancel()
        let detector = detector
        let task = Task {
            try await detector.detect(image: cgImage, threshold: threshold)
        }
        detectionTask = task

        do {
            let result = try await task.value
            guard activeSelection == selectionID, activeDetectionID == detectionID else { return }

            report = DetectionReport(
                result: result,
                imageSize: image.size,
                threshold: threshold
            )
            phase = .complete
            detectionTask = nil
            activeDetectionID = nil
        } catch is CancellationError {
            guard activeSelection == selectionID, activeDetectionID == detectionID else { return }
            phase = modelIsAvailable ? .ready : .modelUnavailable
            detectionTask = nil
            activeDetectionID = nil
        } catch {
            guard activeSelection == selectionID, activeDetectionID == detectionID else { return }
            detectionTask = nil
            activeDetectionID = nil
            showError(error.localizedDescription)
        }
    }

    func cancelDetection() {
        activeDetectionID = nil
        detectionTask?.cancel()
        detectionTask = nil

        guard phase == .detecting else { return }
        if report != nil {
            phase = .complete
        } else if image != nil {
            phase = .ready
        } else {
            phase = modelIsAvailable ? .empty : .modelUnavailable
        }
    }

    func showError(_ message: String) {
        errorMessage = message
        phase = .failed
    }
}
