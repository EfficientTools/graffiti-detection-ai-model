import SwiftUI

struct SettingsView: View {
    @AppStorage("confidenceThreshold") private var confidenceThreshold = 0.25
    @AppStorage("automaticallyDetect") private var automaticallyDetect = true
    @Environment(\.dismiss) private var dismiss
    @State private var draftConfidenceThreshold: Double
    @State private var draftAutomaticallyDetect: Bool

    init() {
        let defaults = UserDefaults.standard
        let storedConfidence = defaults.object(forKey: "confidenceThreshold") as? Double ?? 0.25
        _draftConfidenceThreshold = State(
            initialValue: min(max(storedConfidence, 0.1), 0.9)
        )
        _draftAutomaticallyDetect = State(
            initialValue: defaults.object(forKey: "automaticallyDetect") as? Bool ?? true
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

                Section("Workflow") {
                    Toggle("Analyse new images automatically", isOn: $draftAutomaticallyDetect)
                    Text("Runs the offline detector as soon as a photo or file is ready.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Privacy") {
                    Label("Images never leave this device.", systemImage: "hand.raised.fill")
                    Text("Selection, preprocessing, inference, and detection overlays all run locally.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Purpose") {
                    Label("Made for cleaner public spaces", systemImage: "building.2.fill")
                    Text(
                        "Motivated by recurring graffiti cleanup challenges visible in Melbourne and cities worldwide, Graffiti Guard helps residents and field teams review images earlier and coordinate maintenance."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                    Text("The app detects likely graffiti in images; it does not predict future vandalism.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Credits") {
                    Link(destination: URL(string: "https://pierrehenry.dev")!) {
                        Label("Pierre-Henry Soria", systemImage: "person.crop.square.filled.and.at.rectangle")
                    }

                    Text(
                        "Creator, AI software engineer, and consultant specialising in computer vision, on-device machine learning, privacy-conscious systems, and production AI deployment."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                    Link(destination: URL(string: "https://github.com/EfficientTools/graffiti-detection-ai-model")!) {
                        Label("Python library and reusable patterns", systemImage: "shippingbox.fill")
                    }
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
                        automaticallyDetect = draftAutomaticallyDetect
                        dismiss()
                    }
                }
            }
        }
    }
}
