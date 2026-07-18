import PhotosUI
import SwiftUI
import UIKit

struct DetectionScreen: View {
    @StateObject private var viewModel = DetectionViewModel()
    @AppStorage("confidenceThreshold") private var confidenceThreshold = 0.25
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var showsCamera = false
    @State private var showsSettings = false

    private var cameraIsAvailable: Bool {
        UIImagePickerController.isSourceTypeAvailable(.camera)
    }

    var body: some View {
        NavigationStack {
            ZStack {
                UrbanBackdrop()

                ScrollView {
                    VStack(spacing: 24) {
                        header

                        ViewThatFits(in: .horizontal) {
                            HStack(alignment: .top, spacing: 22) {
                                scannerCard
                                    .frame(minWidth: 520)
                                controlCard
                                    .frame(width: 330)
                            }

                            VStack(spacing: 18) {
                                scannerCard
                                controlCard
                            }
                        }
                    }
                    .frame(maxWidth: 1_040)
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    .padding(.bottom, 38)
                    .frame(maxWidth: .infinity)
                }
                .scrollBounceBehavior(.basedOnSize)
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showsSettings = true
                    } label: {
                        Label("Settings", systemImage: "slider.horizontal.3")
                    }
                    .buttonStyle(.bordered)
                }
            }
            .sheet(isPresented: $showsSettings) {
                SettingsView()
                    .presentationDetents([.medium, .large])
            }
            .fullScreenCover(isPresented: $showsCamera) {
                CameraPicker { image in
                    viewModel.select(image)
                }
                .ignoresSafeArea()
            }
            .onChange(of: selectedPhoto) { _, item in
                guard let item else { return }
                Task {
                    do {
                        guard let data = try await item.loadTransferable(type: Data.self) else {
                            viewModel.showError("The selected photo could not be loaded.")
                            return
                        }
                        viewModel.loadImageData(data)
                    } catch {
                        viewModel.showError("The selected photo could not be loaded: \(error.localizedDescription)")
                    }
                    selectedPhoto = nil
                }
            }
        }
        .tint(.guardGreen)
    }

    private var header: some View {
        HStack(spacing: 16) {
            BrandMark()
                .frame(width: 64, height: 64)

            VStack(alignment: .leading, spacing: 3) {
                Text("GRAFFITI GUARD")
                    .font(.system(.largeTitle, design: .rounded, weight: .black))
                    .tracking(0.8)

                Text("Private, on-device graffiti detection.")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
    }

    private var scannerCard: some View {
        VStack(spacing: 14) {
            DetectionCanvas(image: viewModel.image, report: viewModel.report)
                .aspectRatio(4 / 3, contentMode: .fit)

            HStack(spacing: 12) {
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Label("Photo", systemImage: "photo.on.rectangle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(SourceButtonStyle())

                Button {
                    showsCamera = true
                } label: {
                    Label("Camera", systemImage: "camera.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(SourceButtonStyle())
                .disabled(!cameraIsAvailable)
                .accessibilityHint(cameraIsAvailable ? "Capture a photo" : "Camera unavailable on this device")
            }
        }
        .padding(14)
        .guardCard()
    }

    private var controlCard: some View {
        VStack(alignment: .leading, spacing: 20) {
            statusSummary

            Button {
                Task {
                    await viewModel.detect(threshold: confidenceThreshold)
                }
            } label: {
                HStack {
                    if viewModel.phase == .detecting {
                        ProgressView()
                            .tint(.guardInk)
                    } else {
                        Image(systemName: "scope")
                    }
                    Text(viewModel.phase == .detecting ? "Detecting" : "Detect graffiti")
                    Spacer()
                    if viewModel.phase != .detecting {
                        Image(systemName: "arrow.right")
                    }
                }
                .font(.headline)
                .padding(.horizontal, 18)
                .frame(minHeight: 54)
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(PrimaryActionButtonStyle())
            .disabled(!viewModel.canDetect)

            Divider()

            VStack(alignment: .leading, spacing: 11) {
                Label(
                    confidenceThreshold.formatted(.percent.precision(.fractionLength(0))) + " minimum confidence",
                    systemImage: "dial.medium"
                )

                Label("On-device Core ML", systemImage: "cpu")

                Button("Change settings") {
                    showsSettings = true
                }
                .font(.subheadline.weight(.semibold))
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)

            Label(
                "Detection supports maintenance decisions; it should not replace human review.",
                systemImage: "person.crop.circle.badge.checkmark"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(22)
        .guardCard()
    }

    @ViewBuilder
    private var statusSummary: some View {
        switch viewModel.phase {
        case .empty:
            StatusHeader(
                icon: "viewfinder",
                color: .guardCyan,
                title: "Ready when you are",
                message: "Choose one image to begin."
            )
        case .modelUnavailable:
            StatusHeader(
                icon: "square.stack.3d.up.slash.fill",
                color: .guardAmber,
                title: "Model required",
                message: "Bundle a trained GraffitiDetector model to enable offline detection."
            )
        case .ready:
            StatusHeader(
                icon: "checkmark.circle.fill",
                color: .guardGreen,
                title: "Image ready",
                message: "Tap Detect to run the model."
            )
        case .detecting:
            StatusHeader(
                icon: "waveform.path.ecg",
                color: .guardCyan,
                title: "Analysing scene",
                message: "Running Core ML entirely on this device."
            )
        case .failed:
            StatusHeader(
                icon: "exclamationmark.triangle.fill",
                color: .guardAmber,
                title: "Detection unavailable",
                message: viewModel.errorMessage ?? "Try again."
            )
        case .complete:
            if let report = viewModel.report {
                ResultSummary(report: report)
            }
        }
    }

}

private struct StatusHeader: View {
    let icon: String
    let color: Color
    let title: String
    let message: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Image(systemName: icon)
                .font(.title2.weight(.bold))
                .foregroundStyle(color)

            Text(title)
                .font(.title3.weight(.bold))

            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }
}

private struct ResultSummary: View {
    let report: DetectionReport

    private var foundGraffiti: Bool {
        report.result.count > 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            StatusHeader(
                icon: foundGraffiti ? "exclamationmark.viewfinder" : "checkmark.shield.fill",
                color: foundGraffiti ? .guardAmber : .guardGreen,
                title: foundGraffiti ? "Graffiti detected" : "No graffiti detected",
                message: foundGraffiti
                    ? "Review the highlighted \(report.result.count == 1 ? "region" : "regions")."
                    : "No regions passed the selected threshold."
            )

            HStack(spacing: 10) {
                MetricPill(
                    value: String(report.result.count),
                    label: report.result.count == 1 ? "region" : "regions"
                )

                MetricPill(
                    value: report.result.strongestConfidence?.formatted(
                        .percent.precision(.fractionLength(0))
                    ) ?? "--",
                    label: "highest"
                )

                MetricPill(
                    value: report.result.processingTimeMs.formatted(
                        .number.precision(.fractionLength(0))
                    ) + "ms",
                    label: "model"
                )
            }
        }
    }
}

private struct MetricPill: View {
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.subheadline.weight(.bold))
                .lineLimit(1)
                .minimumScaleFactor(0.75)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 9)
        .background(Color.primary.opacity(0.055), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
    }
}

private struct SourceButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .padding(.horizontal, 14)
            .frame(minHeight: 46)
            .foregroundStyle(.primary)
            .background(Color.primary.opacity(configuration.isPressed ? 0.11 : 0.06))
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .opacity(isEnabled ? (configuration.isPressed ? 0.78 : 1) : 0.35)
    }
}

private struct PrimaryActionButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color.guardInk)
            .background(
                LinearGradient(
                    colors: [.guardGreen, .guardCyan],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .opacity(isEnabled ? (configuration.isPressed ? 0.72 : 1) : 0.35)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
    }
}
