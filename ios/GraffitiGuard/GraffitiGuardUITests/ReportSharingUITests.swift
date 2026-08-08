import XCTest

final class ReportSharingUITests: XCTestCase {
    @MainActor
    func testImageChooserOffersPhotoAndFileSources() {
        continueAfterFailure = false

        let app = XCUIApplication()
        app.launchArguments = [
            "-AppleLanguages", "(en)",
            "-AppleLocale", "en_US",
        ]
        app.launch()

        let chooseImage = app.buttons["choose-image-button"]
        XCTAssertTrue(chooseImage.waitForExistence(timeout: 15))
        chooseImage.tap()

        XCTAssertTrue(app.buttons["Choose Photo"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["Choose File"].exists)
    }

    @MainActor
    func testShareReportPresentsSystemActivityView() {
        continueAfterFailure = false

        let app = XCUIApplication()
        app.launchArguments = [
            "-screenshotMode",
            "-AppleLanguages", "(en)",
            "-AppleLocale", "en_US",
        ]
        app.launch()

        let shareButton = app.buttons.matching(identifier: "share-report-button").firstMatch
        XCTAssertTrue(
            shareButton.waitForExistence(timeout: 45),
            "A completed inspection should expose Share report"
        )

        shareButton.tap()

        let closeButton = app.buttons["header.closeButton"]
        XCTAssertTrue(
            closeButton.waitForExistence(timeout: 15),
            "Share report should present the system activity view"
        )
        XCTAssertTrue(closeButton.isHittable)
        closeButton.tap()
    }
}
