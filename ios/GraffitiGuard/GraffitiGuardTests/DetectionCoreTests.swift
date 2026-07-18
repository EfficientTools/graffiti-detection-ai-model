import CoreML
import Foundation
import UIKit
import XCTest

@testable import GraffitiGuard

final class DetectionCoreTests: XCTestCase {
    func testSummarizesLocalDetectionResult() {
        let result = DetectionResult(
            items: [
                GraffitiDetection(
                    id: 0,
                    confidence: 0.87,
                    box: CGRect(x: 100, y: 20, width: 200, height: 200)
                )
            ],
            processingTimeMs: 42.4
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result.strongestConfidence, 0.87)
    }

    func testMapsAndClipsBoxesIntoAspectFitImage() throws {
        let mapped = try XCTUnwrap(
            DetectionOverlayLayout.map(
                box: CGRect(x: -20, y: 25, width: 120, height: 75),
                imageSize: CGSize(width: 200, height: 100),
                containerSize: CGSize(width: 300, height: 300)
            )
        )

        XCTAssertEqual(mapped.minX, 0, accuracy: 0.001)
        XCTAssertEqual(mapped.minY, 112.5, accuracy: 0.001)
        XCTAssertEqual(mapped.width, 150, accuracy: 0.001)
        XCTAssertEqual(mapped.height, 112.5, accuracy: 0.001)
    }

    func testRejectsBoxOutsideImage() {
        let mapped = DetectionOverlayLayout.map(
            box: CGRect(x: 300, y: 300, width: 50, height: 50),
            imageSize: CGSize(width: 200, height: 100),
            containerSize: CGSize(width: 300, height: 300)
        )

        XCTAssertNil(mapped)
    }

    func testMapsLetterboxedModelCoordinatesBackToSourceImage() throws {
        let transform = LetterboxTransform(
            sourceSize: CGSize(width: 1_280, height: 720),
            targetSize: CGSize(width: 640, height: 640)
        )

        let rect = try XCTUnwrap(
            transform.sourceRect(normalizedXYWH: [0.5, 0.5, 0.5, 0.28125])
        )

        XCTAssertEqual(transform.scale, 0.5, accuracy: 0.001)
        XCTAssertEqual(transform.padding.height, 140, accuracy: 0.001)
        XCTAssertEqual(rect, CGRect(x: 320, y: 180, width: 640, height: 360))
    }

    func testDecodesPaddedCoreMLOutputAtSelectedThreshold() throws {
        let coordinates = try MLMultiArray(shape: [2, 4], dataType: .double)
        let confidence = try MLMultiArray(shape: [2, 80], dataType: .double)
        for index in 0..<confidence.count {
            confidence[index] = 0
        }

        let coordinateValues = [
            0.5, 0.5, 0.5, 0.5,
            0.25, 0.25, 0.2, 0.2,
        ]
        for (index, value) in coordinateValues.enumerated() {
            coordinates[index] = NSNumber(value: value)
        }
        confidence[0] = 0.82
        confidence[80] = 0.15

        let detections = try CoreMLDetectionDecoder.decode(
            coordinates: coordinates,
            confidence: confidence,
            transform: LetterboxTransform(
                sourceSize: CGSize(width: 640, height: 640),
                targetSize: CGSize(width: 640, height: 640)
            ),
            threshold: 0.25
        )

        XCTAssertEqual(detections.count, 1)
        XCTAssertEqual(detections[0].confidence, 0.82, accuracy: 0.001)
        XCTAssertEqual(detections[0].box, CGRect(x: 160, y: 160, width: 320, height: 320))
    }

    @MainActor
    func testViewModelDisablesDetectionWithoutBundledModel() {
        let viewModel = DetectionViewModel(
            detector: StubGraffitiDetector(),
            modelIsAvailable: false
        )

        XCTAssertEqual(viewModel.phase, .modelUnavailable)
        XCTAssertFalse(viewModel.canDetect)
    }

    func testBuildsShareableInspectionSummary() {
        let report = DetectionReport(
            result: DetectionResult(
                items: [
                    GraffitiDetection(
                        id: 0,
                        confidence: 0.87,
                        box: CGRect(x: 10, y: 20, width: 30, height: 40)
                    )
                ],
                processingTimeMs: 42
            ),
            imageSize: CGSize(width: 640, height: 480),
            threshold: 0.25,
            completedAt: Date(timeIntervalSince1970: 0),
            reference: "GG-TEST01"
        )

        XCTAssertTrue(report.shareText.contains("Reference: GG-TEST01"))
        XCTAssertTrue(report.shareText.contains("Result: Likely graffiti detected"))
        XCTAssertTrue(report.shareText.contains("Highest confidence: 87%"))
        XCTAssertTrue(report.shareText.contains("Minimum confidence: 25%"))
        XCTAssertTrue(report.shareText.contains("Next step: Verify each highlighted region"))
        XCTAssertTrue(report.shareText.contains("Human verification is required"))
    }

    func testReportRecommendsManualReviewForEmptyResult() {
        let report = DetectionReport(
            result: DetectionResult(items: [], processingTimeMs: 12),
            imageSize: CGSize(width: 640, height: 480),
            threshold: 0.25,
            reference: "GG-EMPTY"
        )

        XCTAssertTrue(report.nextStep.contains("review the image manually"))
        XCTAssertTrue(report.shareText.contains("Result: No likely graffiti detected"))
        XCTAssertTrue(report.shareText.contains("Next step: If graffiti is still visible"))
    }

    @MainActor
    func testViewModelResetClearsInspection() {
        let viewModel = DetectionViewModel(
            detector: StubGraffitiDetector(),
            modelIsAvailable: true
        )

        viewModel.loadSample()
        XCTAssertTrue(viewModel.canReset)

        viewModel.reset()

        XCTAssertNil(viewModel.image)
        XCTAssertNil(viewModel.report)
        XCTAssertEqual(viewModel.phase, .empty)
        XCTAssertFalse(viewModel.canReset)
    }

    @MainActor
    func testBundledModelDetectsSampleScene() async throws {
        XCTAssertTrue(OnDeviceGraffitiDetector.isModelBundled)

        let image = try XCTUnwrap(UIImage(named: "DemoStreet"))
        let prepared = try XCTUnwrap(ImagePreparer.prepare(image))
        let detector = OnDeviceGraffitiDetector()

        let clock = ContinuousClock()
        let preparationStartedAt = clock.now
        try await detector.prepare()
        let preparationDuration = preparationStartedAt.duration(to: clock.now)
        let preparationMs =
            Double(preparationDuration.components.seconds) * 1_000
            + Double(preparationDuration.components.attoseconds) / 1e15

        let result = try await detector.detect(
            image: prepared.cgImage,
            threshold: 0.25
        )
        let warmedResult = try await detector.detect(
            image: prepared.cgImage,
            threshold: 0.25
        )

        XCTAssertFalse(result.items.isEmpty)
        XCTAssertFalse(warmedResult.items.isEmpty)
        XCTAssertGreaterThan(preparationMs, 0)
        XCTAssertLessThan(preparationMs, 10_000)
        XCTAssertTrue(result.processingTimeMs.isFinite)
        XCTAssertGreaterThan(result.processingTimeMs, 0)
        XCTAssertLessThan(result.processingTimeMs, 10_000)
        XCTAssertLessThan(warmedResult.processingTimeMs, 5_000)

        let imageBounds = CGRect(origin: .zero, size: prepared.image.size)
        for detection in result.items {
            XCTAssertTrue(detection.confidence.isFinite)
            XCTAssertGreaterThanOrEqual(detection.confidence, 0.25)
            XCTAssertLessThanOrEqual(detection.confidence, 1)
            XCTAssertGreaterThan(detection.box.width, 0)
            XCTAssertGreaterThan(detection.box.height, 0)
            XCTAssertTrue(imageBounds.contains(detection.box))
        }

        print(
            "Bundled Core ML sample: \(result.count) region(s), "
                + "\(preparationMs.formatted(.number.precision(.fractionLength(1)))) ms preparation, "
                + "\(result.processingTimeMs.formatted(.number.precision(.fractionLength(1)))) ms first, "
                + "\(warmedResult.processingTimeMs.formatted(.number.precision(.fractionLength(1)))) ms repeat"
        )
    }
}

private struct StubGraffitiDetector: GraffitiDetecting {
    func detect(image: CGImage, threshold: Double) async throws -> DetectionResult {
        DetectionResult(items: [], processingTimeMs: 0)
    }
}
