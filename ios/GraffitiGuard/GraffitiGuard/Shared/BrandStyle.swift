import SwiftUI

extension Color {
    static let guardInk = Color(red: 0.06, green: 0.10, blue: 0.13)
    static let guardCanvas = Color(red: 0.94, green: 0.96, blue: 0.93)
    static let guardGreen = Color(red: 0.22, green: 0.78, blue: 0.31)
    static let guardCyan = Color(red: 0.04, green: 0.69, blue: 0.80)
    static let guardAmber = Color(red: 0.96, green: 0.58, blue: 0.06)
}

struct BrandMark: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.guardInk)

            Image(systemName: "viewfinder")
                .font(.system(size: 33, weight: .bold))
                .foregroundStyle(Color.white)

            Circle()
                .fill(Color.guardGreen)
                .frame(width: 10, height: 10)
                .offset(x: 19, y: -20)

            Circle()
                .fill(Color.guardAmber)
                .frame(width: 7, height: 7)
                .offset(x: 27, y: -30)
        }
        .accessibilityHidden(true)
    }
}

struct UrbanBackdrop: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        ZStack {
            (colorScheme == .dark ? Color.guardInk : Color.guardCanvas)
                .ignoresSafeArea()

            LinearGradient(
                colors: [
                    Color.guardGreen.opacity(colorScheme == .dark ? 0.13 : 0.10),
                    Color.clear,
                    Color.guardCyan.opacity(colorScheme == .dark ? 0.10 : 0.07),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            Canvas { context, size in
                var path = Path()
                let spacing: CGFloat = 72
                for y in stride(from: 0, through: size.height, by: spacing) {
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: size.width, y: y))
                }
                for x in stride(from: 0, through: size.width, by: spacing * 1.6) {
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: size.height))
                }
                let lineColor =
                    colorScheme == .dark
                    ? Color.white.opacity(0.025)
                    : Color.guardInk.opacity(0.035)
                context.stroke(path, with: .color(lineColor), lineWidth: 1)
            }
            .ignoresSafeArea()
        }
    }
}

private struct GuardCard: ViewModifier {
    @Environment(\.colorScheme) private var colorScheme

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(colorScheme == .dark ? Color.white.opacity(0.07) : Color.white.opacity(0.82))
                    .shadow(color: Color.black.opacity(colorScheme == .dark ? 0.20 : 0.08), radius: 22, y: 10)
            )
            .overlay {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .stroke(colorScheme == .dark ? Color.white.opacity(0.10) : Color.white, lineWidth: 1)
            }
    }
}

extension View {
    func guardCard() -> some View {
        modifier(GuardCard())
    }
}
