import SwiftUI
import UIKit

struct ReportSharePayload: Identifiable {
    let id = UUID()
    let subject: String
    let text: String
    let image: UIImage

    @MainActor
    init(report: DetectionReport, image: UIImage) {
        subject = "Graffiti Guard inspection \(report.reference)"
        text = report.shareText
        self.image = ReportImageRenderer.render(image: image, report: report)
    }
}

struct ReportActivityView: UIViewControllerRepresentable {
    let payload: ReportSharePayload

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let itemSource = ReportActivityItemSource(payload: payload)
        let controller = UIActivityViewController(
            activityItems: [itemSource, payload.image],
            applicationActivities: nil
        )
        return controller
    }

    func updateUIViewController(
        _ uiViewController: UIActivityViewController,
        context: Context
    ) {}
}

private final class ReportActivityItemSource: NSObject, UIActivityItemSource {
    private let payload: ReportSharePayload

    init(payload: ReportSharePayload) {
        self.payload = payload
    }

    func activityViewControllerPlaceholderItem(
        _ activityViewController: UIActivityViewController
    ) -> Any {
        payload.text
    }

    func activityViewController(
        _ activityViewController: UIActivityViewController,
        itemForActivityType activityType: UIActivity.ActivityType?
    ) -> Any? {
        payload.text
    }

    func activityViewController(
        _ activityViewController: UIActivityViewController,
        subjectForActivityType activityType: UIActivity.ActivityType?
    ) -> String {
        payload.subject
    }
}
