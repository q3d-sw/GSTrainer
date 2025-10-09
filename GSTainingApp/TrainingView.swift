//
//  TrainingView.swift
//  GSTainingApp
//
//  Simple UI to start/pause training and visualize progress.
//

import SwiftUI

struct TrainingView: View {
    @StateObject private var trainer = Trainer()!
    @State private var selectedScale: Int = 0 // 0: original, else long-side value
    @State private var showAlert = false
    @State private var alertMessage = ""

    var body: some View {
        ScrollView {
        VStack(spacing: 12) {
            HStack {
                Button(trainer.isTraining ? "Pause Training" : "Start Training") {
                    if trainer.isTraining { trainer.pause() } else { trainer.start() }
                }
                .buttonStyle(.borderedProminent)

                Text("Iter: \(trainer.iteration)")
                Spacer()
            }

        ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 12) {
            Button("Gen dataset.json") {
                    Task {
                        do {
                            let res = try AutoDatasetBuilder.build()
                                alertMessage = "Done: frames=\(res.frames), \(res.width)x\(res.height), EXIF=\(res.usedEXIF ? "yes" : "no").\n\nPath: \(res.outputURL.lastPathComponent)"
                            // After generation — reload dataset in Trainer with current scale
                            await MainActor.run {
                                if selectedScale == 0 { trainer.setDownscale(longSide: nil) }
                                else { trainer.setDownscale(longSide: selectedScale) }
                            }
                            showAlert = true
                        } catch {
                            alertMessage = "Generation error: \(error.localizedDescription)\nMake sure images are in Documents/images."
                            showAlert = true
                        }
                    }
            }
            .buttonStyle(.bordered)
            Button("Auto-align") {
                    Task { trainer.autoAlign() }
            }
            .buttonStyle(.bordered)
            Button("Sim3 Refine") {
                    Task { trainer.sim3Refine() }
            }
            .buttonStyle(.bordered)
            Button("Sim3 Refine (GN)") {
                    Task { trainer.sim3RefineGN() }
            }
            .buttonStyle(.bordered)
            Button("Reseed colors") {
                    Task { trainer.reseedColors() }
            }
            .buttonStyle(.bordered)
            Divider()
            Picker("Resolution", selection: $selectedScale) {
                    Text("Original").tag(0)
                    Text("Long side 512").tag(512)
                    Text("Long side 768").tag(768)
                    Text("Long side 1024").tag(1024)
            }
            .pickerStyle(.menu)
            .onChange(of: selectedScale) { oldVal, newVal in
                    if newVal == 0 { trainer.setDownscale(longSide: nil) }
                    else { trainer.setDownscale(longSide: newVal) }
            }
            Divider()
            Menu("Gaussians") {
                    Button("512") { trainer.setGaussianCount(512) }
                    Button("1k") { trainer.setGaussianCount(1024) }
                    Button("2k") { trainer.setGaussianCount(2048) }
                    Button("4k") { trainer.setGaussianCount(4096) }
                    Button("8k") { trainer.setGaussianCount(8192) }
                    Button("16k") { trainer.setGaussianCount(16384) }
            }
            Spacer(minLength: 0)
        }
    .lineLimit(1)
        }

            // Auto-config summary and options
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Overlay projections", isOn: $trainer.overlayEnabled)
                Toggle("Geometry fine-tuning (positions)", isOn: $trainer.geoOptimizePositions)
                HStack {
                    Text(trainer.autoConfigSummary.isEmpty ? "Auto-configuring..." : trainer.autoConfigSummary)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(trainer.autoConfigSummary.isEmpty ? .secondary : .primary)
                    if !trainer.dataSourceLabel.isEmpty {
                        Text(trainer.dataSourceLabel)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .padding(.leading, 6)
                    }
                    Spacer()
                }
            }
            .padding(8)
            .background(Color.black.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            if let data = trainer.previewImageData, let ui = UIImage(data: data) {
                Image(uiImage: ui)
                    .resizable()
                    .interpolation(.none)
                    .scaledToFit()
                    .frame(maxHeight: 360)
                    .border(.gray)
            } else {
                Rectangle().fill(Color.black.opacity(0.05)).frame(height: 300).overlay(Text("Preview"))
            }

            if trainer.overlayEnabled, let dataOv = trainer.overlayImageData, let uiOv = UIImage(data: dataOv) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Overlay: GT + splat centers")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Image(uiImage: uiOv)
                        .resizable()
                        .interpolation(.none)
                        .scaledToFit()
                        .frame(maxHeight: 320)
                        .border(.red)
                }
            }

            LossChart(losses: trainer.lossHistory)
                .frame(height: 120)

            VStack(alignment: .leading, spacing: 6) {
                Text("Logs").font(.headline)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(trainer.lastLogLines.enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(size: 12, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .frame(maxHeight: 280)
                .background(Color.black.opacity(0.04))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            Spacer()
        }
        .padding()
        }
        .navigationTitle("Gaussian Splat Trainer")
        .alert("Auto-generate dataset.json", isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(alertMessage)
        }
    }
}

struct LossChart: View {
    let losses: [Float]
    var body: some View {
        GeometryReader { geo in
            let maxLoss = max(1e-6, losses.max() ?? 1)
            let minLoss: Float = 0
            Path { p in
                for (i, l) in losses.enumerated() {
                    let x = CGFloat(i) / CGFloat(max(1, losses.count - 1)) * geo.size.width
                    let yNorm = (CGFloat((l - minLoss) / (maxLoss - minLoss)))
                    let y = geo.size.height * (1 - yNorm)
                    if i == 0 { p.move(to: CGPoint(x: x, y: y)) }
                    else { p.addLine(to: CGPoint(x: x, y: y)) }
                }
            }
            .stroke(Color.accentColor, lineWidth: 2)
            .background(Color.black.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}

#Preview {
    NavigationStack { TrainingView() }
}
