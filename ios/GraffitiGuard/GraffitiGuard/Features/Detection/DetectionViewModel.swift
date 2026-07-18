import Combine
import UIKit

@MainActor
final class DetectionViewModel: ObservableObject {
    enum Phase: Equatable {
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

    init(
        detector: any GraffitiDetecting = OnDeviceGraffitiDetector(),
        modelIsAvailable: Bool = OnDeviceGraffitiDetector.isModelBundled
    ) {
        self.detector = detector
        self.modelIsAvailable = modelIsAvailable
        phase = modelIsAvailable ? .empty : .modelUnavailable
    }

    var canDetect: Bool {
        modelIsAvailable && image != nil && phase != .detecting
    }

    var canReset: Bool {
        image != nil || report != nil || errorMessage != nil
    }

    func loadImageData(_ data: Data) {
        guard let image = UIImage(data: data) else {
            showError(DetectionError.invalidImage.localizedDescription)
            return
        }
        select(image)
    }

    func loadSample() {
        guard let image = UIImage(named: "DemoStreet") else {
            showError("The sample scene could not be loaded.")
            return
        }
        select(image)
    }

    func select(_ image: UIImage) {
        guard let prepared = ImagePreparer.prepare(image) else {
            showError(DetectionError.invalidImage.localizedDescription)
            return
        }

        selectionID = UUID()
        self.image = prepared.image
        cgImage = prepared.cgImage
        report = nil
        errorMessage = nil
        phase = modelIsAvailable ? .ready : .modelUnavailable
    }

    func reset() {
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
        phase = .detecting
        errorMessage = nil

        do {
            let result = try await detector.detect(image: cgImage, threshold: threshold)
            guard activeSelection == selectionID else { return }

            report = DetectionReport(
                result: result,
                imageSize: image.size,
                threshold: threshold
            )
            phase = .complete
        } catch is CancellationError {
            guard activeSelection == selectionID else { return }
            phase = modelIsAvailable ? .ready : .modelUnavailable
        } catch {
            guard activeSelection == selectionID else { return }
            showError(error.localizedDescription)
        }
    }

    func showError(_ message: String) {
        errorMessage = message
        phase = .failed
    }
}
