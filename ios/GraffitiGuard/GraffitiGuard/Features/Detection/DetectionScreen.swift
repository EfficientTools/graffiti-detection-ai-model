import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct DetectionScreen: View {
    @StateObject private var viewModel = DetectionViewModel()
    @AppStorage("confidenceThreshold") private var confidenceThreshold = 0.25
    @AppStorage("automaticallyDetect") private var automaticallyDetect = true
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var showsImageSourceChooser = false
    @State private var showsPhotoLibrary = false
    @State private var showsFileImporter = false
    @State private var showsCamera = false
    @State private var showsSettings = false
    @State private var didLoadStorePreview = false
    @State private var isDropTargeted = false
    @State private var isLoadingImage = false

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
                                    .frame(minWidth: 600)
                                controlCard
                                    .frame(width: 370)
                            }

                            VStack(spacing: 18) {
                                scannerCard
                                controlCard
                            }
                        }
                    }
                    .frame(maxWidth: 1_260)
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    .padding(.bottom, 38)
                    .frame(maxWidth: .infinity)
                }
                .scrollBounceBehavior(.basedOnSize)
            }
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    if let report = viewModel.report {
                        ShareLink(
                            item: report.shareText,
                            subject: Text("Graffiti Guard inspection \(report.reference)")
                        ) {
                            Label("Share report", systemImage: "square.and.arrow.up")
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoadingImage)
                    }

                    if viewModel.canReset {
                        Button {
                            viewModel.reset()
                        } label: {
                            Label("New inspection", systemImage: "plus")
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoadingImage)
                        .keyboardShortcut("n", modifiers: .command)
                    }

                    Button {
                        showsSettings = true
                    } label: {
                        Label("Settings", systemImage: "slider.horizontal.3")
                    }
                    .buttonStyle(.bordered)
                    .keyboardShortcut(",", modifiers: .command)
                }
            }
            .sheet(isPresented: $showsSettings) {
                SettingsView()
                    .presentationDetents([.medium, .large])
            }
            .confirmationDialog(
                "Choose a street image",
                isPresented: $showsImageSourceChooser,
                titleVisibility: .visible
            ) {
                Button("Choose Photo") {
                    showsPhotoLibrary = true
                }
                .keyboardShortcut(.defaultAction)

                if cameraIsAvailable {
                    Button("Take Photo") {
                        showsCamera = true
                    }
                }

                Button("Choose File") {
                    showsFileImporter = true
                }

                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Select the image source for this inspection.")
            }
            .photosPicker(
                isPresented: $showsPhotoLibrary,
                selection: $selectedPhoto,
                matching: .images
            )
            .fileImporter(
                isPresented: $showsFileImporter,
                allowedContentTypes: [.image]
            ) { result in
                loadImageFile(result)
            }
            .fullScreenCover(isPresented: $showsCamera) {
                CameraPicker { image in
                    Task {
                        await prepareImage {
                            viewModel.select(image)
                        }
                    }
                }
                .ignoresSafeArea()
            }
            .onChange(of: selectedPhoto) { _, item in
                guard let item else { return }
                Task {
                    isLoadingImage = true
                    defer {
                        isLoadingImage = false
                        selectedPhoto = nil
                    }

                    do {
                        guard let data = try await item.loadTransferable(type: Data.self) else {
                            viewModel.showError("The selected photo could not be loaded.")
                            return
                        }
                        let selected = await viewModel.loadImageData(data)
                        isLoadingImage = false
                        if selected {
                            await detectAutomaticallyIfNeeded()
                        }
                    } catch {
                        viewModel.showError("The selected photo could not be loaded: \(error.localizedDescription)")
                    }
                }
            }
            .task {
                await viewModel.prepareModel()
                guard !didLoadStorePreview else { return }

                let arguments = ProcessInfo.processInfo.arguments
                if arguments.contains("-settingsScreenshotMode") {
                    didLoadStorePreview = true
                    try? await Task.sleep(for: .milliseconds(500))
                    showsSettings = true
                    return
                }

                guard
                    arguments.contains("-screenshotMode")
                        || arguments.contains("-readyScreenshotMode")
                else { return }

                didLoadStorePreview = true
                viewModel.loadSample()
                guard arguments.contains("-screenshotMode") else { return }
                await viewModel.detect(threshold: confidenceThreshold)
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

                Text("Private field inspections, entirely on-device.")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
    }

    private var scannerCard: some View {
        let imageIsLoading = isLoadingImage

        return VStack(spacing: 14) {
            ZStack {
                DetectionCanvas(
                    image: viewModel.image,
                    report: viewModel.report,
                    onChooseImage: {
                        showsImageSourceChooser = true
                    }
                )
                .disabled(isLoadingImage)

                if isLoadingImage {
                    ImageLoadingOverlay()
                        .transition(.opacity.combined(with: .scale(scale: 0.98)))
                }
            }
            .aspectRatio(4 / 3, contentMode: .fit)
            .animation(.easeOut(duration: 0.2), value: isLoadingImage)

            HStack(spacing: 12) {
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    HStack(spacing: 8) {
                        if imageIsLoading {
                            ProgressView()
                                .controlSize(.small)
                            Text("Loading")
                        } else {
                            Label("Photo", systemImage: "photo.on.rectangle")
                        }
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(SourceButtonStyle())
                .disabled(isLoadingImage)

                Button {
                    showsCamera = true
                } label: {
                    Label("Camera", systemImage: "camera.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(SourceButtonStyle())
                .disabled(!cameraIsAvailable || isLoadingImage)
                .accessibilityHint(cameraIsAvailable ? "Capture a photo" : "Camera unavailable on this device")
            }

            if horizontalSizeClass == .regular {
                Label("On iPad, you can also drop a photo directly into this panel.", systemImage: "hand.draw")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Button {
                Task {
                    await prepareImage {
                        viewModel.loadSample()
                    }
                }
            } label: {
                Label("Try sample scene", systemImage: "building.2.crop.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(isLoadingImage)
        }
        .padding(14)
        .guardCard()
        .overlay {
            if isDropTargeted {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .stroke(Color.guardCyan, style: StrokeStyle(lineWidth: 3, dash: [10, 6]))
                    .padding(3)
            }
        }
        .dropDestination(for: Data.self) { items, _ in
            guard !isLoadingImage, let data = items.first else { return false }
            Task {
                await prepareImage {
                    await viewModel.loadImageData(data)
                }
            }
            return true
        } isTargeted: {
            isDropTargeted = $0
        }
    }

    private func loadImageFile(_ result: Result<URL, any Error>) {
        guard !isLoadingImage else { return }

        Task {
            isLoadingImage = true
            defer { isLoadingImage = false }

            do {
                let url = try result.get()
                let data = try await ImportedImageDataLoader.load(from: url)
                let selected = await viewModel.loadImageData(data)
                isLoadingImage = false
                if selected {
                    await detectAutomaticallyIfNeeded()
                }
            } catch {
                viewModel.showError("The selected image file could not be loaded: \(error.localizedDescription)")
            }
        }
    }

    @MainActor
    private func prepareImage(_ selection: @MainActor () async -> Bool) async {
        guard !isLoadingImage else { return }
        isLoadingImage = true
        await Task.yield()
        let selected = await selection()
        isLoadingImage = false
        if selected {
            await detectAutomaticallyIfNeeded()
        }
    }

    @MainActor
    private func detectAutomaticallyIfNeeded() async {
        guard automaticallyDetect, viewModel.canDetect else { return }
        await viewModel.detect(threshold: confidenceThreshold)
    }

    private var controlCard: some View {
        VStack(alignment: .leading, spacing: 20) {
            statusSummary

            if viewModel.phase == .complete, let report = viewModel.report {
                HStack(spacing: 10) {
                    ShareLink(
                        item: report.shareText,
                        subject: Text("Graffiti Guard inspection \(report.reference)")
                    ) {
                        Label("Share report", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)

                    Button {
                        viewModel.reset()
                    } label: {
                        Label("New", systemImage: "plus")
                    }
                    .buttonStyle(.bordered)
                }

                Button {
                    Task {
                        await viewModel.detect(threshold: confidenceThreshold)
                    }
                } label: {
                    Label("Run detection again", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .keyboardShortcut(.return, modifiers: .command)
            } else {
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
                .disabled(!viewModel.canDetect || isLoadingImage)
                .keyboardShortcut(.return, modifiers: .command)
            }

            Divider()

            VStack(alignment: .leading, spacing: 11) {
                Label(
                    confidenceThreshold.formatted(.percent.precision(.fractionLength(0))) + " minimum confidence",
                    systemImage: "dial.medium"
                )

                Slider(value: $confidenceThreshold, in: 0.1...0.9, step: 0.05) {
                    Text("Minimum confidence")
                }
                .tint(.guardCyan)

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
                message: "Choose an image or try the sample scene."
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
                message: automaticallyDetect
                    ? "Starting private on-device analysis."
                    : "Tap Detect to run the model."
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

enum ImportedImageDataLoader {
    static func load(from url: URL) async throws -> Data {
        try await Task.detached(priority: .userInitiated) {
            let hasSecurityAccess = url.startAccessingSecurityScopedResource()
            defer {
                if hasSecurityAccess {
                    url.stopAccessingSecurityScopedResource()
                }
            }
            return try Data(contentsOf: url, options: .mappedIfSafe)
        }.value
    }
}

private struct ImageLoadingOverlay: View {
    var body: some View {
        ZStack {
            Color.guardInk.opacity(0.88)

            VStack(spacing: 13) {
                ProgressView()
                    .controlSize(.large)
                    .tint(.guardGreen)

                Text("Loading image")
                    .font(.headline)
                    .foregroundStyle(.white)

                Text("Retrieving from iCloud if needed.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.68))
            }
            .padding(24)
        }
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Loading selected image")
        .accessibilityHint("Retrieving from iCloud if needed")
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
                title: foundGraffiti ? "Likely graffiti detected" : "No likely graffiti found",
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

            VStack(alignment: .leading, spacing: 7) {
                Label("Recommended next step", systemImage: "arrow.turn.down.right")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(foundGraffiti ? Color.guardAmber : Color.guardGreen)

                Text(report.nextStep)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 13))

            HStack {
                Label(report.reference, systemImage: "number")
                Spacer()
                Text(report.completedAt, format: .dateTime.hour().minute())
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
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
