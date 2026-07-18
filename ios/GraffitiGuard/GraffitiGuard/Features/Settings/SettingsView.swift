import SwiftUI

struct SettingsView: View {
    @AppStorage("confidenceThreshold") private var confidenceThreshold = 0.25
    @Environment(\.dismiss) private var dismiss
    @State private var draftConfidenceThreshold: Double

    init() {
        let defaults = UserDefaults.standard
        let storedConfidence = defaults.object(forKey: "confidenceThreshold") as? Double ?? 0.25
        _draftConfidenceThreshold = State(
            initialValue: min(max(storedConfidence, 0.1), 0.9)
        )
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Label(
                        OnDeviceGraffitiDetector.isModelBundled ? "GraffitiDetector ready" : "GraffitiDetector missing",
                        systemImage: OnDeviceGraffitiDetector.isModelBundled
                            ? "checkmark.seal.fill"
                            : "exclamationmark.triangle.fill"
                    )
                    .foregroundStyle(
                        OnDeviceGraffitiDetector.isModelBundled ? Color.primary : Color.orange
                    )

                    Text("Graffiti Guard uses a bundled Core ML model. No inference server or account is required.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    LabeledContent("Model license", value: "MIT")
                    Text(
                        "One-class graffiti detector published by khoaliamle and converted to Core ML for offline use."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                }

                Section("Confidence") {
                    Slider(value: $draftConfidenceThreshold, in: 0.1...0.9, step: 0.05) {
                        Text("Minimum confidence")
                    }

                    LabeledContent(
                        "Minimum confidence",
                        value: draftConfidenceThreshold.formatted(.percent.precision(.fractionLength(0)))
                    )
                }

                Section("Privacy") {
                    Label("Images never leave this device.", systemImage: "hand.raised.fill")
                    Text("Selection, preprocessing, inference, and detection overlays all run locally.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        confidenceThreshold = draftConfidenceThreshold
                        dismiss()
                    }
                }
            }
        }
    }
}
