# Graffiti Guard for iPhone and iPad

Graffiti Guard is a universal SwiftUI app that checks a camera capture or selected photo with a bundled Core ML model, then overlays likely graffiti regions. Inference is private, offline, and optimized for Apple silicon; the app has no server setting or network dependency.

Completed inspections include a context-aware next step and a shareable reference for maintenance handoff. Apple CI exercises the packaged model itself, not only mocked inference code.

On iPad, field teams can drop in a photo, tune the confidence threshold, run detection with a hardware-keyboard shortcut, and share a referenced inspection summary. Results support human review; the app does not predict future vandalism.

## Run

Requirements: macOS, Xcode 26, XcodeGen, and iOS or iPadOS 17 or newer.

The App Store target includes an MIT-licensed one-class Core ML detector. To replace it with your own Ultralytics model, export the weights, regenerate the project, then run the `GraffitiGuard` scheme:

```bash
python -m pip install -e ".[apple]"
python scripts/export_coreml.py --weights models/best.pt
xcodegen generate --spec ios/GraffitiGuard/project.yml
open ios/GraffitiGuard/GraffitiGuard.xcodeproj
```

Replacement models must contain exactly one class named `graffiti`. The export creates `GraffitiDetector.mlpackage` with 640-pixel letterbox input and Core ML non-maximum suppression. See the repository's third-party notices for the bundled detector's source and license.

## Test

```bash
xcodebuild \
  -project ios/GraffitiGuard/GraffitiGuard.xcodeproj \
  -scheme GraffitiGuard \
  -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

GitHub Actions builds Debug and Release configurations and runs the Core ML contract tests on an iPhone simulator.

## Publish

The project uses bundle ID `dev.pierrehenry.GraffitiGuard` and Apple Team `2V8LZ2444Y` with automatic signing.

1. Export and validate the trained model on representative street images.
2. Create the matching app record in App Store Connect.
3. In Xcode, choose **Product > Archive**, then upload through **Distribute App > App Store Connect**.
4. Start with TestFlight, then add the required screenshots, support URL, privacy URL, review notes, and App Privacy answers before App Review.

All image processing stays on-device and the privacy manifest declares no tracking or collected data. Recheck that disclosure whenever analytics, crash reporting, or any network feature is added.

## Embedded Use

`GraffitiDetecting` isolates inference from the SwiftUI interface. A later embedded target can preserve the same normalized image and bounding-box contract while replacing the Core ML implementation with its platform runtime.
