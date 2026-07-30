import XCTest

final class ReportSharingUITests: XCTestCase {
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

        let closeButton = app.buttons["Close"]
        XCTAssertTrue(
            closeButton.waitForExistence(timeout: 10),
            "Share report should present the system activity view"
        )

        let copyAction = app.cells["Copy"]
        XCTAssertTrue(
            copyAction.waitForExistence(timeout: 10),
            "The shared report should expose an actionable Copy destination"
        )
        XCTAssertTrue(copyAction.isHittable)
        copyAction.tap()
    }
}
