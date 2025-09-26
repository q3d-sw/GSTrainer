//
//  Trainer.swift
//  GSTainingApp
//
//  Baseline on-device training loop with manual Adam updates.
//

import Foundation
import Combine
import Metal
import simd
import UIKit

final class Trainer: ObservableObject {
    // Public observables
    @Published var isTraining: Bool = false
    @Published var iteration: Int = 0
    @Published var lossHistory: [Float] = []
    @Published var previewImageData: Data? // PNG data for SwiftUI Image
    @Published var overlayImageData: Data? // PNG overlay of projected points over GT
    @Published var lastLogLines: [String] = []
    @Published var dataSourceLabel: String = ""
    // Tracks last iteration a color reseed happened (for variance-based reseed throttle)
    private var lastReseedIteration: Int = 0
    // One-time bootstrap multi-frame reseed flag
    private var bootstrapReseedDone: Bool = false

    // Core components
    private var dataset: DatasetManager
    private let renderer: MetalRenderer
    private let device: MTLDevice
    private var targetTexture: MTLTexture
    private var residualTexture: MTLTexture
    private var gaussians: [GaussianParam]
    private var gaussianBuffer: MTLBuffer
    private var gradBuffers: MetalRenderer.GradBuffers?
    private var originalPositions: [SIMD3<Float>] = [] // store unscaled PLY positions
    // SH L1 linear coefficients per gaussian (RGB each has [x,y,z])
    private var shR1: [SIMD3<Float>] = []
    private var shG1: [SIMD3<Float>] = []
    private var shB1: [SIMD3<Float>] = []
    // SH L2 extra coefficients per gaussian per channel (we pack 6 coeffs as two float4 blocks on GPU).
    // We store them as raw SIMD4 arrays for parity with GPU layout (ex1: c2_0..c2_3, ex2: c2_4,c2_5,0,0)
    private var shR2_ex1: [SIMD4<Float>] = []
    private var shR2_ex2: [SIMD4<Float>] = []
    private var shG2_ex1: [SIMD4<Float>] = []
    private var shG2_ex2: [SIMD4<Float>] = []
    private var shB2_ex1: [SIMD4<Float>] = []
    private var shB2_ex2: [SIMD4<Float>] = []
    // Anisotropic scales per gaussian (meters) and their Adam moments
    private var scaleX: [Float] = []
    private var scaleY: [Float] = []
    private var scaleZ: [Float] = []
    private var mScaleX: [Float] = []
    private var mScaleY: [Float] = []
    private var mScaleZ: [Float] = []
    private var vScaleX: [Float] = []
    private var vScaleY: [Float] = []
    private var vScaleZ: [Float] = []

    // Optimizer state (Adam)
    private var m: [GaussianParam] // first moment per param (only for color, opacity, sigma, position simplified)
    private var v: [GaussianParam] // second moment
    // Adam moments for SH L1 coefficients
    private var mShR1: [SIMD3<Float>] = []
    private var vShR1: [SIMD3<Float>] = []
    private var mShG1: [SIMD3<Float>] = []
    private var vShG1: [SIMD3<Float>] = []
    private var mShB1: [SIMD3<Float>] = []
    private var vShB1: [SIMD3<Float>] = []
    // Adam moments for SH L2 (store per SIMD4 block)
    private var mShR2_ex1: [SIMD4<Float>] = []
    private var mShR2_ex2: [SIMD4<Float>] = []
    private var vShR2_ex1: [SIMD4<Float>] = []
    private var vShR2_ex2: [SIMD4<Float>] = []
    private var mShG2_ex1: [SIMD4<Float>] = []
    private var mShG2_ex2: [SIMD4<Float>] = []
    private var vShG2_ex1: [SIMD4<Float>] = []
    private var vShG2_ex2: [SIMD4<Float>] = []
    private var mShB2_ex1: [SIMD4<Float>] = []
    private var mShB2_ex2: [SIMD4<Float>] = []
    private var vShB2_ex1: [SIMD4<Float>] = []
    private var vShB2_ex2: [SIMD4<Float>] = []

    // Hyper-params
    private var lr: Float = 1e-3
    private let beta1: Float = 0.9
    private let beta2: Float = 0.999
    private let eps: Float = 1e-8
    private let opacityPrior: Float = 0.35
    private let opacityReg: Float = 0.05 // weight for (opacity - prior)^2
    private let maxOpacity: Float = 0.85
    private let maxStepColor: Float = 0.05
    private let maxStepOpacity: Float = 0.02
    private let frameHold: Int = 10
    private let maxKernelSigmaPixels: Float = 8.0 // limit kernel radius in pixels for local gradient
    // Optional auto-alignment on startup (sweep flips/zSign/scale)
    private let autoAlignOnStartup: Bool = false
    // Geometry fine-tuning (positions)
    @Published var geoOptimizePositions: Bool = false
    private let posLr: Float = 1e-4
    private let posStepClip: Float = 1e-3
    // Auto-alignment summary for UI
    @Published var autoConfigSummary: String = ""
    @Published var autoInBoundsPct: Float = 0
    @Published var autoFrontPct: Float = 0

    private var timer: Timer?
    private var zSign: Float = -1.0
    private var zeroFramesStreak: Int = 0
    private var frameHoldCounter: Int = 0
    private var currentFrameIndex: Int = 0
    private var lossEMA: Float = 0
    @Published var overlayEnabled: Bool = true
    @Published var plyScale: Float = 1.0
    @Published var flipX: Bool = false
    @Published var flipY: Bool = false
    @Published var flipZ: Bool = false
    // Sim3 refinement state
    private var sim3Scale: Float = 1.0
    private var sim3Rot: simd_quatf = simd_quaternion(0, SIMD3<Float>(0,0,1)) // identity
    private var sim3Trans: SIMD3<Float> = .zero
    @Published var enableSHL2: Bool = false
    private var didLogSHL2Enable: Bool = false
    private var globalGradClipNorm: Float = 5.0
    // Read-only view for UI to know current camera forward sign
    var isCameraForwardPlusZ: Bool { zSign > 0 }
    private func log(_ msg: String) {
        print("[Trainer] \(msg)")
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.lastLogLines.append(msg)
            if self.lastLogLines.count > 200 { self.lastLogLines.removeFirst(self.lastLogLines.count - 200) }
        }
    }

    private func checkSHL2ActivationLog() {
        if enableSHL2 && !didLogSHL2Enable {
            didLogSHL2Enable = true
            log("SH L2 включён: 9 гармоник на канал (27 цветовых коэффициентов, было 4/канал)")
        }
    }

    @inline(__always)
    private func applyGlobalClip(color: inout SIMD3<Float>, opacity: inout Float, sigma: inout Float, shR1: inout SIMD3<Float>, shG1: inout SIMD3<Float>, shB1: inout SIMD3<Float>) {
        let sum = color.x*color.x + color.y*color.y + color.z*color.z + opacity*opacity + sigma*sigma +
        shR1.x*shR1.x + shR1.y*shR1.y + shR1.z*shR1.z +
        shG1.x*shG1.x + shG1.y*shG1.y + shG1.z*shG1.z +
        shB1.x*shB1.x + shB1.y*shB1.y + shB1.z*shB1.z
        let norm = sqrt(sum)
        if norm > globalGradClipNorm && norm > 0 {
            let scale = globalGradClipNorm / norm
            color *= scale; opacity *= scale; sigma *= scale
            shR1 *= scale; shG1 *= scale; shB1 *= scale
        }
    }
    private func applyGlobalClipExtended(color: inout SIMD3<Float>, opacity: inout Float, sigma: inout Float,
                                         shR1: inout SIMD3<Float>, shG1: inout SIMD3<Float>, shB1: inout SIMD3<Float>,
                                         r2e1: inout SIMD4<Float>, r2e2: inout SIMD4<Float>,
                                         g2e1: inout SIMD4<Float>, g2e2: inout SIMD4<Float>,
                                         b2e1: inout SIMD4<Float>, b2e2: inout SIMD4<Float>) {
        var sum: Float = color.x*color.x + color.y*color.y + color.z*color.z + opacity*opacity + sigma*sigma
        func add3(_ v: SIMD3<Float>) { sum += v.x*v.x + v.y*v.y + v.z*v.z }
        func add4(_ v: SIMD4<Float>) { sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w }
        add3(shR1); add3(shG1); add3(shB1)
        add4(r2e1); add4(r2e2); add4(g2e1); add4(g2e2); add4(b2e1); add4(b2e2)
        let norm = sqrt(sum)
        if norm > globalGradClipNorm && norm > 0 {
            let scale = globalGradClipNorm / norm
            color *= scale; opacity *= scale; sigma *= scale
            shR1 *= scale; shG1 *= scale; shB1 *= scale
            r2e1 *= scale; r2e2 *= scale; g2e1 *= scale; g2e2 *= scale; b2e1 *= scale; b2e2 *= scale
        }
    }

    // MARK: - Optimizer helpers (extracted for clarity)
    @inline(__always)
    private func adamUpdateSIMD4(beta1: Float, beta2: Float, eps: Float, grad: SIMD4<Float>, m: inout SIMD4<Float>, v: inout SIMD4<Float>, param: inout SIMD4<Float>, lr: Float, t: Float, clip: Float) {
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        let mhat = m / (1 - powf(beta1, t))
        let vhat = v / (1 - powf(beta2, t))
        let denom = SIMD4<Float>( sqrt(vhat.x)+eps, sqrt(vhat.y)+eps, sqrt(vhat.z)+eps, sqrt(vhat.w)+eps )
        var step = lr * mhat / denom
        // component clip
        step.x = max(-clip, min(clip, step.x))
        step.y = max(-clip, min(clip, step.y))
        step.z = max(-clip, min(clip, step.z))
        step.w = max(-clip, min(clip, step.w))
        param -= step
    }

    @inline(__always)
    private func clamp3(_ v: SIMD3<Float>, _ lo: Float, _ hi: Float) -> SIMD3<Float> {
        SIMD3<Float>(max(lo, min(hi, v.x)), max(lo, min(hi, v.y)), max(lo, min(hi, v.z)))
    }

    // Lightweight struct for self-test decode (mirror of CheckpointData used internally)
    private struct CheckpointData: Codable {
        let iteration: Int
        let gaussians: [GaussianParam]
        let learningRate: Float
        let shR1: [SIMD3<Float>]?
        let shG1: [SIMD3<Float>]?
        let shB1: [SIMD3<Float>]?
        let scaleX: [Float]?
        let scaleY: [Float]?
        let scaleZ: [Float]?
    }

    init?(imageSize: Int = 256, gaussianCount: Int = 8192) {
        guard let renderer = MetalRenderer(), let device = renderer.device as MTLDevice? else { return nil }
        self.renderer = renderer
        self.device = device

    // Dataset (use original resolution without resize)
        self.dataset = DatasetManager(width: imageSize, height: imageSize, targetSize: nil)
    // Indicate data source for UI
        if DatasetManager.loadTransformsSpec() != nil {
            self.dataSourceLabel = "source=transforms"
        } else if DatasetManager.loadDatasetSpec() != nil {
            self.dataSourceLabel = "source=dataset"
        } else {
            self.dataSourceLabel = "source=synthetic"
        }

        // Initialize gaussians: try points.ply first
        var gs: [GaussianParam] = []
        if let pts = PointsPLYParser.loadFromDocuments(limit: gaussianCount), !pts.isEmpty {
            gs.reserveCapacity(pts.count)
            for p in pts {
                let pos = p.position
                let sigma: Float = p.sigma ?? 0.03
                let color = p.color ?? SIMD3<Float>(repeating: 0.7)
                let opacity: Float = p.opacity ?? 0.3
                gs.append(GaussianParam(position: pos, sigma: sigma, color: color, opacity: opacity))
            }
        } else {
            // Fallback: random cloud near origin
            gs.reserveCapacity(gaussianCount)
            var rng = SystemRandomNumberGenerator()
            for _ in 0..<gaussianCount {
                let pos = SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                       Float.random(in: -1...1, using: &rng),
                                       Float.random(in: -1...1, using: &rng))
                let sigma: Float = Float.random(in: 0.02...0.08, using: &rng)
                let color = SIMD3<Float>(Float.random(in: 0...1, using: &rng), Float.random(in: 0...1, using: &rng), Float.random(in: 0...1, using: &rng))
                let opacity: Float = Float.random(in: 0.05...0.5, using: &rng)
                gs.append(GaussianParam(position: pos, sigma: sigma, color: color, opacity: opacity))
            }
        }
        self.gaussians = gs
    self.originalPositions = gs.map { $0.position }
        self.m = Array(repeating: GaussianParam(position: .zero, sigma: 0, color: .zero, opacity: 0), count: gs.count)
        self.v = self.m
        // Initialize SH L1 arrays and their optimizer moments
        self.shR1 = Array(repeating: SIMD3<Float>(repeating: 0), count: gs.count)
        self.shG1 = self.shR1
        self.shB1 = self.shR1
        self.mShR1 = self.shR1
        self.vShR1 = self.shR1
        self.mShG1 = self.shR1
        self.vShG1 = self.shR1

    // Initialize anisotropic scales from isotropic sigma
    self.scaleX = gs.map { $0.sigma }
    self.scaleY = self.scaleX
    self.scaleZ = self.scaleX
    self.mScaleX = Array(repeating: 0, count: gs.count)
    self.mScaleY = self.mScaleX
    self.mScaleZ = self.mScaleX
    self.vScaleX = self.mScaleX
    self.vScaleY = self.mScaleX
    self.vScaleZ = self.mScaleX
        self.mShB1 = self.shR1
        self.vShB1 = self.shR1
    // Initialize SH L2 arrays (zero) & moments
    self.shR2_ex1 = Array(repeating: .zero, count: gs.count)
    self.shR2_ex2 = self.shR2_ex1
    self.shG2_ex1 = self.shR2_ex1
    self.shG2_ex2 = self.shR2_ex1
    self.shB2_ex1 = self.shR2_ex1
    self.shB2_ex2 = self.shR2_ex1
    self.mShR2_ex1 = self.shR2_ex1; self.mShR2_ex2 = self.shR2_ex2
    self.mShG2_ex1 = self.shG2_ex1; self.mShG2_ex2 = self.shG2_ex2
    self.mShB2_ex1 = self.shB2_ex1; self.mShB2_ex2 = self.shB2_ex2
    self.vShR2_ex1 = self.shR2_ex1; self.vShR2_ex2 = self.shR2_ex2
    self.vShG2_ex1 = self.shG2_ex1; self.vShG2_ex2 = self.shG2_ex2
    self.vShB2_ex1 = self.shB2_ex1; self.vShB2_ex2 = self.shB2_ex2

        // GPU buffer
        let gpuArray = gs.map { GaussianParamGPU($0) }
        guard let buf = renderer.makeBuffer(array: gpuArray) else { return nil }
        self.gaussianBuffer = buf

    // Target texture for rendering previews/training
    guard let tex = renderer.makeTexture(width: dataset.width, height: dataset.height),
        let rtex = renderer.makeTexture(width: dataset.width, height: dataset.height) else { return nil }
    self.targetTexture = tex
    self.residualTexture = rtex
    self.gradBuffers = renderer.makeGradBuffers(count: gs.count)

        // Try load checkpoint
        if let ckpt = CheckpointManager.load() {
            self.iteration = ckpt.iteration
            self.gaussians = ckpt.gaussians
            self.lr = ckpt.learningRate
            self.originalPositions = self.gaussians.map { $0.position }
            // Resize SH arrays to match loaded gaussians
            let n = self.gaussians.count
            if let r1 = ckpt.shR1, r1.count == n { self.shR1 = r1 } else { self.shR1 = Array(repeating: .zero, count: n) }
            if let g1 = ckpt.shG1, g1.count == n { self.shG1 = g1 } else { self.shG1 = Array(repeating: .zero, count: n) }
            if let b1 = ckpt.shB1, b1.count == n { self.shB1 = b1 } else { self.shB1 = Array(repeating: .zero, count: n) }
            self.mShR1 = self.shR1
            self.vShR1 = self.shR1
            self.mShG1 = self.shR1
            self.vShG1 = self.shR1
            self.mShB1 = self.shR1
            self.vShB1 = self.shR1
            // Load anisotropic scales if present
            if let sx = ckpt.scaleX, let sy = ckpt.scaleY, let sz = ckpt.scaleZ, sx.count == n, sy.count == n, sz.count == n {
                self.scaleX = sx; self.scaleY = sy; self.scaleZ = sz
            } else {
                self.scaleX = self.gaussians.map { $0.sigma }
                self.scaleY = self.scaleX
                self.scaleZ = self.scaleX
            }
            // Load SH L2 if present (optional backward compatibility)
            if let r2e1 = ckpt.shR2_ex1, r2e1.count == n { self.shR2_ex1 = r2e1 } else { self.shR2_ex1 = Array(repeating: .zero, count: n) }
            if let r2e2 = ckpt.shR2_ex2, r2e2.count == n { self.shR2_ex2 = r2e2 } else { self.shR2_ex2 = Array(repeating: .zero, count: n) }
            if let g2e1 = ckpt.shG2_ex1, g2e1.count == n { self.shG2_ex1 = g2e1 } else { self.shG2_ex1 = Array(repeating: .zero, count: n) }
            if let g2e2 = ckpt.shG2_ex2, g2e2.count == n { self.shG2_ex2 = g2e2 } else { self.shG2_ex2 = Array(repeating: .zero, count: n) }
            if let b2e1 = ckpt.shB2_ex1, b2e1.count == n { self.shB2_ex1 = b2e1 } else { self.shB2_ex1 = Array(repeating: .zero, count: n) }
            if let b2e2 = ckpt.shB2_ex2, b2e2.count == n { self.shB2_ex2 = b2e2 } else { self.shB2_ex2 = Array(repeating: .zero, count: n) }
            self.mShR2_ex1 = self.shR2_ex1; self.vShR2_ex1 = self.shR2_ex1
            self.mShR2_ex2 = self.shR2_ex2; self.vShR2_ex2 = self.shR2_ex2
            self.mShG2_ex1 = self.shG2_ex1; self.vShG2_ex1 = self.shG2_ex1
            self.mShG2_ex2 = self.shG2_ex2; self.vShG2_ex2 = self.shG2_ex2
            self.mShB2_ex1 = self.shB2_ex1; self.vShB2_ex1 = self.shB2_ex1
            self.mShB2_ex2 = self.shB2_ex2; self.vShB2_ex2 = self.shB2_ex2
            self.mScaleX = Array(repeating: 0, count: n)
            self.mScaleY = self.mScaleX
            self.mScaleZ = self.mScaleX
            self.vScaleX = self.mScaleX
            self.vScaleY = self.mScaleX
            self.vScaleZ = self.mScaleX
            self._syncGaussiansToGPU()
            log("Loaded checkpoint at iter=\(iteration), gaussians=\(gaussians.count)")
        } else {
            // Apply a fixed default configuration: flips=0/0/0, forward=-Z, scale=1.0
            self.plyScale = 1.0
            self.flipX = false
            self.flipY = false
            self.flipZ = false
            self.applyPointTransform()
            self.setCameraForward(signPlusZ: false)
            // Compute diagnostics on a few frames
            let frameCount = dataset.samples.count
            var framesToTest: [Int] = [0]
            if frameCount > 2 { framesToTest.append(frameCount/2) }
            if frameCount > 1 { framesToTest.append(frameCount-1) }
            framesToTest = Array(Set(framesToTest)).sorted()
            var total: Int = 0
            var front: Int = 0
            var inBounds: Int = 0
            for fi in framesToTest {
                let diag = self.reprojectionDiagnostics(for: fi)
                total += diag.total
                front += diag.front
                inBounds += diag.inBounds
            }
            self.autoFrontPct = (total > 0) ? (Float(front) / Float(total) * 100.0) : 0
            self.autoInBoundsPct = (total > 0) ? (Float(inBounds) / Float(total) * 100.0) : 0
            self.autoConfigSummary = String(
                format: "fixed: scale=%.2f, flips=%d/%d/%d, forward=%@, inBounds=%.1f%%, front=%.1f%%, %@",
                self.plyScale, self.flipX ? 1:0, self.flipY ? 1:0, self.flipZ ? 1:0,
                (self.zSign > 0 ? "+Z" : "-Z"), self.autoInBoundsPct, self.autoFrontPct, self.dataSourceLabel
            )
            log("Applied fixed orientation -> " + self.autoConfigSummary)
        }
    log("Dataset loaded: frames=\(dataset.samples.count), size=\(dataset.width)x\(dataset.height) (\(dataSourceLabel))")
        log("Gaussians initialized: \(gaussians.count)")
    // Optional auto-align sweep at startup
        if autoAlignOnStartup {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.autoAlign()
            }
        }
    }

    // MARK: - Resolution control
    // Compute original (source) dataset dimensions from specs (transforms.json or dataset.json)
    private func originalDatasetDims() -> (w: Int, h: Int)? {
        if let spec = DatasetManager.loadTransformsSpec() { return (spec.width, spec.height) }
        if let spec = DatasetManager.loadDatasetSpec() { return (spec.width, spec.height) }
        return nil
    }

    // Public API: set downscale as desired long-side resolution. Pass nil to use originals.
    func setDownscale(longSide: Int?) {
        let target: CGSize?
        if let L = longSide, L > 0 {
            // Use original source dims when possible to avoid compounding scales
            let base = originalDatasetDims() ?? (w: dataset.width, h: dataset.height)
            let w0 = CGFloat(base.w), h0 = CGFloat(base.h)
            let s = CGFloat(L) / max(w0, h0)
            let tw = Int(round(w0 * s))
            let th = Int(round(h0 * s))
            target = CGSize(width: max(1, tw), height: max(1, th))
        } else {
            target = nil
        }
        reconfigureDataset(targetSize: target)
    }

    private func reconfigureDataset(targetSize: CGSize?) {
        let wasTraining = isTraining
        if wasTraining { pause() }
    // Rebuild dataset and render target
        self.dataset = DatasetManager(width: dataset.width, height: dataset.height, targetSize: targetSize)
        if DatasetManager.loadTransformsSpec() != nil {
            self.dataSourceLabel = "source=transforms"
        } else if DatasetManager.loadDatasetSpec() != nil {
            self.dataSourceLabel = "source=dataset"
        } else {
            self.dataSourceLabel = "source=synthetic"
        }
        if let newTex = renderer.makeTexture(width: dataset.width, height: dataset.height) {
            self.targetTexture = newTex
        }
        if let newR = renderer.makeTexture(width: dataset.width, height: dataset.height) {
            self.residualTexture = newR
        }
    // Recompute overlay/metrics summary for current fixed orientation
        let frameCount = dataset.samples.count
        var framesToTest: [Int] = [0]
        if frameCount > 2 { framesToTest.append(frameCount/2) }
        if frameCount > 1 { framesToTest.append(frameCount-1) }
        framesToTest = Array(Set(framesToTest)).sorted()
        var total: Int = 0, front: Int = 0, inBounds: Int = 0
        for fi in framesToTest {
            let d = reprojectionDiagnostics(for: fi)
            total += d.total; front += d.front; inBounds += d.inBounds
        }
        self.autoFrontPct = (total > 0) ? (Float(front) / Float(total) * 100.0) : 0
        self.autoInBoundsPct = (total > 0) ? (Float(inBounds) / Float(total) * 100.0) : 0
        self.autoConfigSummary = String(
            format: "fixed: scale=%.2f, flips=%d/%d/%d, forward=%@, inBounds=%.1f%%, front=%.1f%%, size=%dx%d, %@",
            self.plyScale, self.flipX ? 1:0, self.flipY ? 1:0, self.flipZ ? 1:0,
            (self.zSign > 0 ? "+Z" : "-Z"), self.autoInBoundsPct, self.autoFrontPct,
            self.dataset.width, self.dataset.height, self.dataSourceLabel
        )
        log("Reconfigured dataset -> \(dataset.width)x\(dataset.height)")
        if wasTraining { start() }
    }

    // Apply a scale factor to original PLY positions and sync to GPU
    func setPlyScale(_ scale: Float) {
        let s = max(0.01, min(100.0, scale))
        plyScale = s
        applyPointTransform()
        log(String(format: "Applied PLY scale: %.4f", plyScale))
        // Force overlay refresh on next log tick
    }

    func setFlip(x: Bool? = nil, y: Bool? = nil, z: Bool? = nil) {
        if let x = x { flipX = x }
        if let y = y { flipY = y }
        if let z = z { flipZ = z }
        applyPointTransform()
        log("Applied flips: X=\(flipX ? 1 : 0), Y=\(flipY ? 1 : 0), Z=\(flipZ ? 1 : 0)")
    }

    func setCameraForward(signPlusZ: Bool) {
        zSign = signPlusZ ? 1.0 : -1.0
        log("Camera forward set to \(signPlusZ ? "+Z" : "-Z")")
    }

    private func applyPointTransform() {
        let sx: Float = flipX ? -plyScale : plyScale
        let sy: Float = flipY ? -plyScale : plyScale
        let sz: Float = flipZ ? -plyScale : plyScale
        for i in 0..<gaussians.count {
            let p0 = originalPositions[i]
            let p1 = SIMD3<Float>(p0.x * sx, p0.y * sy, p0.z * sz)
            let pr = sim3Rot.act(p1)
            gaussians[i].position = sim3Scale * pr + sim3Trans
        }
        _syncGaussiansToGPU()
    }

    // MARK: - Sim3 refine (global s, R, t)
    func sim3Refine(maxPoints: Int = 4000) {
        let wasTraining = isTraining
        if wasTraining { pause() }
        let frameCount = dataset.samples.count
        guard frameCount > 0, !gaussians.isEmpty else { return }
        // Sample frames: first, middle, last
        var framesToUse: [Int] = [0]
        if frameCount > 2 { framesToUse.append(frameCount/2) }
        if frameCount > 1 { framesToUse.append(frameCount-1) }
        framesToUse = Array(Set(framesToUse)).sorted()

        // Precompute base-flipped-scaled points (without Sim3) for speed
        let sx: Float = flipX ? -plyScale : plyScale
        let sy: Float = flipY ? -plyScale : plyScale
        let sz: Float = flipZ ? -plyScale : plyScale
        let step = max(1, gaussians.count / maxPoints)
        var basePts: [SIMD3<Float>] = []
        basePts.reserveCapacity((gaussians.count + step - 1)/step)
        var idx = 0
        var bboxMin = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var bboxMax = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        while idx < gaussians.count {
            let p0 = originalPositions[idx]
            let p1 = SIMD3<Float>(p0.x * sx, p0.y * sy, p0.z * sz)
            basePts.append(p1)
            bboxMin = simd_min(bboxMin, p1)
            bboxMax = simd_max(bboxMax, p1)
            idx += step
        }
        let extent = bboxMax - bboxMin
        let radius = max(extent.x, max(extent.y, extent.z))
        // Initial parameters
    let s0 = sim3Scale
        var rx0: Float = 0, ry0: Float = 0, rz0: Float = 0 // current sim3Rot in euler (approx)
        // Extract approximate Euler from quaternion (ZYX order)
        func quatToEuler(_ q: simd_quatf) -> (Float,Float,Float) {
            let m = simd_float3x3(q)
            let sy = -m[2,0]
            let _ = sqrtf(max(0, 1 - sy*sy))
            let x = atan2f(m[2,1], m[2,2])
            let y = asinf(max(-1, min(1, sy)))
            let z = atan2f(m[1,0], m[0,0])
            return (x,y,z)
        }
        (rx0, ry0, rz0) = quatToEuler(sim3Rot)
    let t0 = sim3Trans

        struct FramePack { let W2C: simd_float4x4; let intr: CameraIntrinsics; let w: Int; let h: Int }
        var packs: [FramePack] = []
        packs.reserveCapacity(framesToUse.count)
        for fi in framesToUse {
            let s = dataset.samples[fi]
            packs.append(FramePack(W2C: worldToCamMatrix(extr: s.extr), intr: s.intr, w: dataset.width, h: dataset.height))
        }
        @inline(__always)
    func score(s: Float, rx: Float, ry: Float, rz: Float, t: SIMD3<Float>) -> Float {
            // Build rotation
            let qx = simd_quaternion(rx, SIMD3<Float>(1,0,0))
            let qy = simd_quaternion(ry, SIMD3<Float>(0,1,0))
            let qz = simd_quaternion(rz, SIMD3<Float>(0,0,1))
            let q = qz * qy * qx
            var total: Float = 0
            for p in basePts {
                let pr = q.act(p)
                let pw = s * pr + t
                for pk in packs {
                    let cam4 = pk.W2C * SIMD4<Float>(pw.x, pw.y, pw.z, 1)
                    let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
                    let zf: Float = (zSign < 0) ? (-cam.z) : (cam.z)
                    if zf <= 0 { total -= 1; continue }
                    let u = pk.intr.fx * (cam.x / zf) + pk.intr.cx
                    let v = pk.intr.fy * (cam.y / zf) + pk.intr.cy
                    if u >= 0, u < Float(pk.w), v >= 0, v < Float(pk.h) {
                        total += 2
                    } else {
                        // Penalty by out-of-bounds distance
                        let dx: Float = (u < 0) ? -u : (u >= Float(pk.w) ? (u - Float(pk.w-1)) : 0)
                        let dy: Float = (v < 0) ? -v : (v >= Float(pk.h) ? (v - Float(pk.h-1)) : 0)
                        let norm = sqrtf(dx*dx + dy*dy) / Float(max(pk.w, pk.h))
                        total -= min(2.0, norm * 4.0)
                    }
                }
            }
            return total
        }

        // Coordinate-descent coarse-to-fine
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            var s = s0
            var rx = rx0, ry = ry0, rz = rz0
            var tx = t0.x, ty = t0.y, tz = t0.z
            var dS: Float = 0.25 // 25%
            var dA: Float = Float(15.0 * .pi / 180.0) // 15 deg
            var dT: Float = max(0.01, 0.25 * radius)
            var bestScore = score(s: s, rx: rx, ry: ry, rz: rz, t: SIMD3<Float>(tx,ty,tz))
            for _ in 0..<3 { // 3 coarse levels
                // scale
                let sCands = [s * (1 - dS), s, s * (1 + dS)]
                var bestLocal = (val: s, scr: bestScore)
                for sv in sCands {
                    let sc = score(s: sv, rx: rx, ry: ry, rz: rz, t: SIMD3<Float>(tx,ty,tz))
                    if sc > bestLocal.scr { bestLocal = (sv, sc) }
                }
                s = bestLocal.val; bestScore = bestLocal.scr
                // rx, ry, rz
                func updateAngle(_ a: Float, axis: Int) -> (val: Float, scr: Float) {
                    let cands = [a - dA, a, a + dA]
                    var best = (val: a, scr: bestScore)
                    for av in cands {
                        let sc = axis == 0 ? score(s: s, rx: av, ry: ry, rz: rz, t: SIMD3<Float>(tx,ty,tz))
                              : axis == 1 ? score(s: s, rx: rx, ry: av, rz: rz, t: SIMD3<Float>(tx,ty,tz))
                                           : score(s: s, rx: rx, ry: ry, rz: av, t: SIMD3<Float>(tx,ty,tz))
                        if sc > best.scr { best = (av, sc) }
                    }
                    return best
                }
                var upd = updateAngle(rx, axis: 0); rx = upd.val; bestScore = upd.scr
                upd = updateAngle(ry, axis: 1); ry = upd.val; bestScore = upd.scr
                upd = updateAngle(rz, axis: 2); rz = upd.val; bestScore = upd.scr
                // tx, ty, tz
                func updateTrans(current: Float, axis: Int) -> (val: Float, scr: Float) {
                    let cands = [current - dT, current, current + dT]
                    var best = (val: current, scr: bestScore)
                    for tv in cands {
                        var tvec = SIMD3<Float>(tx,ty,tz)
                        if axis == 0 { tvec.x = tv } else if axis == 1 { tvec.y = tv } else { tvec.z = tv }
                        let sc = score(s: s, rx: rx, ry: ry, rz: rz, t: tvec)
                        if sc > best.scr { best = (tv, sc) }
                    }
                    return best
                }
                var updt = updateTrans(current: tx, axis: 0); tx = updt.val; bestScore = updt.scr
                updt = updateTrans(current: ty, axis: 1); ty = updt.val; bestScore = updt.scr
                updt = updateTrans(current: tz, axis: 2); tz = updt.val; bestScore = updt.scr
                dS *= 0.5; dA *= 0.5; dT *= 0.5
            }
            // Apply on main
            let qx = simd_quaternion(rx, SIMD3<Float>(1,0,0))
            let qy = simd_quaternion(ry, SIMD3<Float>(0,1,0))
            let qz = simd_quaternion(rz, SIMD3<Float>(0,0,1))
            let q = qz * qy * qx
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.sim3Scale = s
                self.sim3Rot = q
                self.sim3Trans = SIMD3<Float>(tx,ty,tz)
                self.applyPointTransform()
                // Update summary
                let frameCount = self.dataset.samples.count
                var framesToTest: [Int] = [0]
                if frameCount > 2 { framesToTest.append(frameCount/2) }
                if frameCount > 1 { framesToTest.append(frameCount-1) }
                framesToTest = Array(Set(framesToTest)).sorted()
                var total: Int = 0, front: Int = 0, inBounds: Int = 0
                for fi in framesToTest {
                    let d = self.reprojectionDiagnostics(for: fi)
                    total += d.total; front += d.front; inBounds += d.inBounds
                }
                self.autoFrontPct = (total > 0) ? (Float(front) / Float(total) * 100.0) : 0
                self.autoInBoundsPct = (total > 0) ? (Float(inBounds) / Float(total) * 100.0) : 0
                self.autoConfigSummary = String(
                    format: "refined(sim3): s=%.3f, t=(%.3f,%.3f,%.3f), inBounds=%.1f%%, front=%.1f%%, size=%dx%d, %@",
                    self.sim3Scale, self.sim3Trans.x, self.sim3Trans.y, self.sim3Trans.z,
                    self.autoInBoundsPct, self.autoFrontPct, self.dataset.width, self.dataset.height, self.dataSourceLabel
                )
                self.log(String(format: "Sim3 refined: s=%.4f, rx=%.2f°", ry*180/Float.pi, rz*180/Float.pi, tx, ty, tz))
                if wasTraining { self.start() }
            }
        }
    }

    // Reseed gaussians colors from a GT frame (sRGB->linear)
    func reseedColors(frameIndex: Int? = nil, blend: Float = 0.0) {
        let idx = frameIndex ?? 0
        guard dataset.samples.indices.contains(idx) else { return }
        let s = dataset.samples[idx]
        guard let gtCG = s.image.cgImage else { return }
        let w = gtCG.width, h = gtCG.height
        var buf = [UInt8](repeating: 0, count: w*h*4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let info: CGBitmapInfo = [ .byteOrder32Big, CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue) ]
        if let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: cs, bitmapInfo: info.rawValue) {
            ctx.draw(gtCG, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        @inline(__always)
        func srgbToLinear(_ c: Float) -> Float { return (c <= 0.04045) ? c/12.92 : powf((c+0.055)/1.055, 2.4) }
        let W2C = worldToCamMatrix(extr: s.extr)
        let alpha = max(0, min(1, blend))
        for i in 0..<gaussians.count {
            let g = gaussians[i]
            let cam4 = W2C * SIMD4<Float>(g.position, 1)
            let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
            let zf: Float = (zSign < 0) ? (-cam.z) : (cam.z)
            if zf <= 0 { continue }
            let u = s.intr.fx * (cam.x / zf) + s.intr.cx
            let v = s.intr.fy * (cam.y / zf) + s.intr.cy
            let xi = Int(round(u)), yi = Int(round(v))
            if xi < 0 || yi < 0 || xi >= w || yi >= h { continue }
            let pidx = (yi * w + xi) * 4
            let r = srgbToLinear(Float(buf[pidx+0]) / 255.0)
            let gcol = srgbToLinear(Float(buf[pidx+1]) / 255.0)
            let b = srgbToLinear(Float(buf[pidx+2]) / 255.0)
            var cur = gaussians[i].color
            let newC = SIMD3<Float>(r, gcol, b)
            cur = alpha * cur + (1 - alpha) * newC
            gaussians[i].color = simd_clamp(cur, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
        }
        _syncGaussiansToGPU()
        log("Reseeded colors from frame=\(idx), blend=\(alpha)")
    }

    // Change number of gaussians and rebuild from PLY (or fallback), preserving current orientation settings
    func setGaussianCount(_ newCount: Int) {
        let count = max(32, min(131072, newCount))
        let wasTraining = isTraining
        if wasTraining { pause() }
        var gs: [GaussianParam] = []
        if let pts = PointsPLYParser.loadFromDocuments(limit: count), !pts.isEmpty {
            gs.reserveCapacity(max(count, pts.count))
            for p in pts {
                let pos = p.position
                let sigma: Float = p.sigma ?? 0.03
                let color = p.color ?? SIMD3<Float>(repeating: 0.7)
                let opacity: Float = p.opacity ?? 0.3
                gs.append(GaussianParam(position: pos, sigma: sigma, color: color, opacity: opacity))
            }
            if gs.count < count {
                // Pad with random gaussians if PLY has fewer points than requested
                var rng = SystemRandomNumberGenerator()
                for _ in gs.count..<count {
                    let pos = SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                           Float.random(in: -1...1, using: &rng),
                                           Float.random(in: -1...1, using: &rng))
                    let sigma: Float = Float.random(in: 0.02...0.08, using: &rng)
                    let color = SIMD3<Float>(Float.random(in: 0...1, using: &rng),
                                              Float.random(in: 0...1, using: &rng),
                                              Float.random(in: 0...1, using: &rng))
                    let opacity: Float = Float.random(in: 0.05...0.5, using: &rng)
                    gs.append(GaussianParam(position: pos, sigma: sigma, color: color, opacity: opacity))
                }
                log("PLY had fewer points (\(pts.count)) than requested (\(count)); filled with random to reach target count.")
            }
        } else {
            // Fully random fallback
            gs.reserveCapacity(count)
            var rng = SystemRandomNumberGenerator()
            for _ in 0..<count {
                let pos = SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                       Float.random(in: -1...1, using: &rng),
                                       Float.random(in: -1...1, using: &rng))
                let sigma: Float = Float.random(in: 0.02...0.08, using: &rng)
                let color = SIMD3<Float>(Float.random(in: 0...1, using: &rng),
                                          Float.random(in: 0...1, using: &rng),
                                          Float.random(in: 0...1, using: &rng))
                let opacity: Float = Float.random(in: 0.05...0.5, using: &rng)
                gs.append(GaussianParam(position: pos, sigma: sigma, color: color, opacity: opacity))
            }
            log("No PLY found; created random cloud of \(count) gaussians.")
        }

        self.gaussians = gs
        self.originalPositions = gs.map { $0.position }
        // reset optimizer states
        self.m = Array(repeating: GaussianParam(position: .zero, sigma: 0, color: .zero, opacity: 0), count: gs.count)
        self.v = self.m
    // reset SH L1 and moments
    self.shR1 = Array(repeating: SIMD3<Float>(repeating: 0), count: gs.count)
    self.shG1 = self.shR1
    self.shB1 = self.shR1
    self.mShR1 = self.shR1
    self.vShR1 = self.shR1
    self.mShG1 = self.shR1
    self.vShG1 = self.shR1
    self.mShB1 = self.shR1
    self.vShB1 = self.shR1
        // (Re)allocate GPU buffer
        let gpuArray = gs.map { GaussianParamGPU($0) }
        if let newBuf = renderer.makeBuffer(array: gpuArray) {
            self.gaussianBuffer = newBuf
        }
        // Re-apply current flips/scale to positions
        applyPointTransform()
        // Recompute diagnostics summary
        let frameCount = dataset.samples.count
        var framesToTest: [Int] = [0]
        if frameCount > 2 { framesToTest.append(frameCount/2) }
        if frameCount > 1 { framesToTest.append(frameCount-1) }
        framesToTest = Array(Set(framesToTest)).sorted()
        var total: Int = 0, front: Int = 0, inBounds: Int = 0
        for fi in framesToTest {
            let d = reprojectionDiagnostics(for: fi)
            total += d.total; front += d.front; inBounds += d.inBounds
        }
        self.autoFrontPct = (total > 0) ? (Float(front) / Float(total) * 100.0) : 0
        self.autoInBoundsPct = (total > 0) ? (Float(inBounds) / Float(total) * 100.0) : 0
        self.autoConfigSummary = String(
            format: "fixed: scale=%.2f, flips=%d/%d/%d, forward=%@, inBounds=%.1f%%, front=%.1f%%, size=%dx%d, %@",
            self.plyScale, self.flipX ? 1:0, self.flipY ? 1:0, self.flipZ ? 1:0,
            (self.zSign > 0 ? "+Z" : "-Z"), self.autoInBoundsPct, self.autoFrontPct,
            self.dataset.width, self.dataset.height, self.dataSourceLabel
        )
        log("Rebuilt gaussians: count=\(gaussians.count)")
        if wasTraining { start() }
    }

    func start() {
        guard !isTraining else { return }
        isTraining = true
        log("Training started")
        frameHoldCounter = frameHold
        currentFrameIndex = 0
        lossEMA = 0
        scheduleTick()
    }

    func pause() {
        isTraining = false
        timer?.invalidate()
        timer = nil
        saveCheckpoint()
        log("Training paused at iter=\(iteration)")
    }

    private func scheduleTick() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { [weak self] _ in
            self?.step()
        }
    }

    private func _syncGaussiansToGPU() {
        // Build a temporary array to include SH L1 coefficients
        var arr = gaussians.map { GaussianParamGPU($0) }
        let n = arr.count
        if shR1.count != n { shR1 = Array(repeating: .zero, count: n) }
        if shG1.count != n { shG1 = Array(repeating: .zero, count: n) }
        if shB1.count != n { shB1 = Array(repeating: .zero, count: n) }
        // Ensure SH L2 arrays sized
        func ensureL2Arrays() {
            if shR2_ex1.count != n { shR2_ex1 = Array(repeating: .zero, count: n) }
            if shR2_ex2.count != n { shR2_ex2 = Array(repeating: .zero, count: n) }
            if shG2_ex1.count != n { shG2_ex1 = Array(repeating: .zero, count: n) }
            if shG2_ex2.count != n { shG2_ex2 = Array(repeating: .zero, count: n) }
            if shB2_ex1.count != n { shB2_ex1 = Array(repeating: .zero, count: n) }
            if shB2_ex2.count != n { shB2_ex2 = Array(repeating: .zero, count: n) }
        }
        ensureL2Arrays()
        if scaleX.count != n { scaleX = Array(repeating: 0.03, count: n) }
        if scaleY.count != n { scaleY = scaleX }
        if scaleZ.count != n { scaleZ = scaleX }
        for i in 0..<n {
            let r = shR1[i], g = shG1[i], b = shB1[i]
            arr[i].shR.y = r.x; arr[i].shR.z = r.y; arr[i].shR.w = r.z
            arr[i].shG.y = g.x; arr[i].shG.z = g.y; arr[i].shG.w = g.z
            arr[i].shB.y = b.x; arr[i].shB.z = b.y; arr[i].shB.w = b.z
            if enableSHL2 {
                arr[i].shR_ex1 = shR2_ex1[i]; arr[i].shR_ex2 = shR2_ex2[i]
                arr[i].shG_ex1 = shG2_ex1[i]; arr[i].shG_ex2 = shG2_ex2[i]
                arr[i].shB_ex1 = shB2_ex1[i]; arr[i].shB_ex2 = shB2_ex2[i]
            } else {
                arr[i].shR_ex1 = .zero; arr[i].shR_ex2 = .zero
                arr[i].shG_ex1 = .zero; arr[i].shG_ex2 = .zero
                arr[i].shB_ex1 = .zero; arr[i].shB_ex2 = .zero
            }
            // write anisotropic scales
            arr[i].scale = SIMD3<Float>(scaleX[i], scaleY[i], scaleZ[i])
            arr[i].opacity = gaussians[i].opacity
            // keep DC color in arr[i].sh* .x already set by init
        }
        let len = MemoryLayout<GaussianParamGPU>.stride * n
        if gaussianBuffer.length < len {
            if let new = renderer.makeBuffer(array: arr) { gaussianBuffer = new }
        } else if len > 0 {
            arr.withUnsafeBytes { rawBuf in
                if let base = rawBuf.baseAddress { memcpy(gaussianBuffer.contents(), base, len) }
            }
        }
    }

    private func worldToCamMatrix(extr: CameraExtrinsics) -> simd_float4x4 {
        // extr.matrix4x4: camera-to-world; we need world-to-camera (inverse)
        let camToWorld = extr.matrix4x4
        let worldToCam = camToWorld.inverse
        return worldToCam
    }

    private func renderPrediction(for sampleIndex: Int) -> CGImage? {
        let s = dataset.samples[sampleIndex]
        let W2C = worldToCamMatrix(extr: s.extr)
    // Sort a copy of gaussians by depth (front-to-back) to make compositing more stable
        struct DepthItem { let zf: Float; let g: GaussianParam }
        var items: [DepthItem] = []
        items.reserveCapacity(gaussians.count)
        for g in gaussians {
            let cam4 = W2C * SIMD4<Float>(g.position, 1)
            let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
            var zf = (zSign < 0) ? (-cam.z) : (cam.z)
            if zf <= 0 { zf = Float.greatestFiniteMagnitude }
            items.append(DepthItem(zf: zf, g: g))
        }
        items.sort { $0.zf < $1.zf }
        let sortedGPU = items.map { GaussianParamGPU($0.g) }
    // Create a temporary buffer for rendering only
        guard let tmpBuf = renderer.makeBuffer(array: sortedGPU) else { return nil }
        renderer.forward(gaussiansBuffer: tmpBuf, gaussianCount: gaussians.count, worldToCam: W2C, intrinsics: s.intr, target: targetTexture, pointScale: 800.0, zSign: zSign, enableSHL2: enableSHL2)
        return renderer.cgImage(from: targetTexture)
    }

    // Compute diagnostics: fraction of points in front of camera and within image bounds
    private func reprojectionDiagnostics(for sampleIndex: Int) -> (front: Int, inBounds: Int, total: Int) {
        let s = dataset.samples[sampleIndex]
        let w = dataset.width, h = dataset.height
        let W2C = worldToCamMatrix(extr: s.extr)
        var front = 0, inBounds = 0
        for g in gaussians {
            let cam4 = W2C * SIMD4<Float>(g.position, 1)
            let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
            let zf: Float = (zSign < 0) ? (-cam.z) : (cam.z)
            if zf <= 0 { continue }
            front += 1
            let u = s.intr.fx * (cam.x / zf) + s.intr.cx
            let v = s.intr.fy * (cam.y / zf) + s.intr.cy
            if u >= 0, u < Float(w), v >= 0, v < Float(h) { inBounds += 1 }
        }
        return (front, inBounds, gaussians.count)
    }

    // Build an overlay image: GT with projected point centers (red dots)
    private func makeOverlay(for sampleIndex: Int) -> CGImage? {
        let s = dataset.samples[sampleIndex]
        guard let gtCG = s.image.cgImage else { return nil }
        let w = gtCG.width, h = gtCG.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let info: CGBitmapInfo = [
            .byteOrder32Big,
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        ]
        var buffer = [UInt8](repeating: 0, count: w*h*4)
        guard let ctx = CGContext(data: &buffer, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: colorSpace, bitmapInfo: info.rawValue) else { return nil }
        // Draw GT
        ctx.draw(gtCG, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Draw points
        let W2C = worldToCamMatrix(extr: s.extr)
        let dotSize = 2
        func drawDot(x: Int, y: Int) {
            for dy in -dotSize...dotSize {
                let py = y + dy
                if py < 0 || py >= h { continue }
                for dx in -dotSize...dotSize {
                    let px = x + dx
                    if px < 0 || px >= w { continue }
                    let idx = (py * w + px) * 4
                    buffer[idx+0] = 255 // R
                    buffer[idx+1] = 0   // G
                    buffer[idx+2] = 0   // B
                    buffer[idx+3] = 255 // A
                }
            }
        }
        var countDrawn = 0
        for g in gaussians {
            let cam4 = W2C * SIMD4<Float>(g.position, 1)
            let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
            let zf: Float = (zSign < 0) ? (-cam.z) : (cam.z)
            if zf <= 0 { continue }
            let u = s.intr.fx * (cam.x / zf) + s.intr.cx
            let v = s.intr.fy * (cam.y / zf) + s.intr.cy
            let xi = Int(round(u))
            let yi = Int(round(v))
            if xi < 0 || xi >= w || yi < 0 || yi >= h { continue }
            drawDot(x: xi, y: yi)
            countDrawn += 1
            if countDrawn > 2000 { break } // safety cap for very dense clouds
        }
        guard let outCG = ctx.makeImage() else { return nil }
        return outCG
    }

    // Try flips, zSign, and a small set of PLY scales; pick the one maximizing in-bounds across sample frames
    func autoAlign() {
        let frameCount = dataset.samples.count
        if frameCount == 0 || gaussians.isEmpty { return }
        // pick up to 3 frames: first, middle, last
        var framesToTest: [Int] = [0]
        if frameCount > 2 { framesToTest.append(frameCount/2) }
        if frameCount > 1 { framesToTest.append(frameCount-1) }
        framesToTest = Array(Set(framesToTest)).sorted()
        let orig = originalPositions
        var bestKey: (inb: Float, fr: Float) = (-1, -1)
        var bestCfg: (bx: Bool, by: Bool, bz: Bool, signPlusZ: Bool, scale: Float) = (false, false, false, false, 1.0)
        let candidates: [(Bool,Bool,Bool)] = [
            (false,false,false),(true,false,false),(false,true,false),(false,false,true),
            (true,true,false),(true,false,true),(false,true,true),(true,true,true)
        ]
        let scales: [Float] = [0.5, 0.75, 1.0, 1.25, 1.5]
        for (bx,by,bz) in candidates {
            for signPlusZ in [false,true] {
                for scale in scales {
                    var total: Int = 0
                    var front: Int = 0
                    var inBounds: Int = 0
                    let sx: Float = (bx ? -scale : scale)
                    let sy: Float = (by ? -scale : scale)
                    let sz: Float = (bz ? -scale : scale)
                    let zS: Float = signPlusZ ? 1.0 : -1.0
                    let step = max(1, gaussians.count / 3000) // sample up to ~3000 points
                    for fi in framesToTest {
                        let s = dataset.samples[fi]
                        let W2C = worldToCamMatrix(extr: s.extr)
                        let w = dataset.width, h = dataset.height
                        var idx = 0
                        while idx < gaussians.count {
                            let p0 = orig[idx]
                            let p = SIMD4<Float>(p0.x * sx, p0.y * sy, p0.z * sz, 1)
                            let cam4 = W2C * p
                            let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
                            total += 1
                            let zf: Float = (zS < 0) ? (-cam.z) : (cam.z)
                            if zf > 0 {
                                front += 1
                                let u = s.intr.fx * (cam.x / zf) + s.intr.cx
                                let v = s.intr.fy * (cam.y / zf) + s.intr.cy
                                if u >= 0, u < Float(w), v >= 0, v < Float(h) { inBounds += 1 }
                            }
                            idx += step
                        }
                    }
                    let frontPct = (total > 0) ? (Float(front) / Float(total) * 100.0) : 0
                    let inbPct = (total > 0) ? (Float(inBounds) / Float(total) * 100.0) : 0
                    let key = (inb: round(inbPct * 1000)/1000, fr: round(frontPct * 1000)/1000)
                    if key.inb > bestKey.inb || (key.inb == bestKey.inb && key.fr > bestKey.fr) {
                        bestKey = key
                        bestCfg = (bx,by,bz,signPlusZ,scale)
                    }
                }
            }
        }
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            // Apply chosen config
            self.flipX = bestCfg.bx
            self.flipY = bestCfg.by
            self.flipZ = bestCfg.bz
            self.plyScale = bestCfg.scale
            self.applyPointTransform()
            self.setCameraForward(signPlusZ: bestCfg.signPlusZ)
            self.autoInBoundsPct = bestKey.inb
            self.autoFrontPct = bestKey.fr
            self.autoConfigSummary = String(
                format: "auto-align: scale=%.2f, flips=%d/%d/%d, forward=%@, inBounds=%.1f%%, front=%.1f%%",
                self.plyScale,
                self.flipX ? 1:0, self.flipY ? 1:0, self.flipZ ? 1:0,
                (self.zSign > 0 ? "+Z" : "-Z"),
                self.autoInBoundsPct, self.autoFrontPct
            )
            self.log("AutoAlign chosen -> " + self.autoConfigSummary)
        }
    }

    private func computeLossAndGradients(pred: CGImage, gt: UIImage) -> Float {
    // L2 loss between pred and GT + local gradients for color/opacity (and optional position cues)
        guard let gtCG = gt.cgImage else { return 0 }
        let w = pred.width, h = pred.height
        let bytes = w * h * 4
        var predData = [UInt8](repeating: 0, count: bytes)
        var gtData = [UInt8](repeating: 0, count: bytes)
        let cs = CGColorSpaceCreateDeviceRGB()
        let info: CGBitmapInfo = [
            .byteOrder32Big,
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        ]
        if let ctxP = CGContext(data: &predData, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: cs, bitmapInfo: info.rawValue) {
            ctxP.draw(pred, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        if let ctxG = CGContext(data: &gtData, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: cs, bitmapInfo: info.rawValue) {
            ctxG.draw(gtCG, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        // sRGB -> Linear helper
        @inline(__always)
        func srgbToLinear(_ c: Float) -> Float {
            if c <= 0.04045 { return c / 12.92 }
            return powf((c + 0.055) / 1.055, 2.4)
        }
    // Prepare linear-color buffers for loss
        var predLin = [SIMD3<Float>](repeating: .zero, count: w*h)
        var gtLin = [SIMD3<Float>](repeating: .zero, count: w*h)
        var predLum = [Float](repeating: 0, count: w*h)
        var gtLum = [Float](repeating: 0, count: w*h)
        let lumW = SIMD3<Float>(0.2126, 0.7152, 0.0722)
        for y in 0..<h {
            for x in 0..<w {
                let idx = (y*w + x) * 4
                let pr = srgbToLinear(Float(predData[idx+0]) / 255.0)
                let pg = srgbToLinear(Float(predData[idx+1]) / 255.0)
                let pb = srgbToLinear(Float(predData[idx+2]) / 255.0)
                let gr = srgbToLinear(Float(gtData[idx+0]) / 255.0)
                let gg = srgbToLinear(Float(gtData[idx+1]) / 255.0)
                let gb = srgbToLinear(Float(gtData[idx+2]) / 255.0)
                let p = SIMD3<Float>(pr, pg, pb)
                let g = SIMD3<Float>(gr, gg, gb)
                predLin[y*w + x] = p
                gtLin[y*w + x] = g
                predLum[y*w + x] = simd_dot(p, lumW)
                gtLum[y*w + x] = simd_dot(g, lumW)
            }
        }
    // Per-frame exposure normalization: scalar gain to align brightness
        var num: Float = 0
        var den: Float = 0
        for i in 0..<(w*h) {
            num += predLum[i] * gtLum[i]
            den += predLum[i] * predLum[i]
        }
        var gain: Float = (den > 1e-8) ? (num / den) : 1.0
        gain = max(0.5, min(2.0, gain))
    // Compute loss in linear space using the exposure gain
        var loss: Float = 0
        for i in 0..<(w*h) {
            let e = gain * predLin[i] - gtLin[i]
            loss += simd_dot(e, e)
        }
        loss /= Float(w * h)

        // Build residual (linear): r = gain*pred - gt and upload to residualTexture as half
        // Also prepare arrays for GPU gradient readback
        var residualLinear = [SIMD3<Float>](repeating: .zero, count: w*h)
        for i in 0..<(w*h) { residualLinear[i] = gain * predLin[i] - gtLin[i] }
        // Upload residual to RGBA16F texture (alpha unused)
        let rowBytes = w * MemoryLayout<UInt16>.stride * 4
        let raw = UnsafeMutableRawPointer.allocate(byteCount: rowBytes * h, alignment: 0x1000)
        defer { raw.deallocate() }
        func floatToHalf(_ x: Float) -> UInt16 {
            // very small scalar converter (not IEEE-perfect but adequate here)
            var f = x
            let sign: UInt16 = (f < 0) ? 0x8000 : 0
            f = max(-65504.0, min(65504.0, f))
            let absf = fabsf(f)
            var e: Int32 = 0
            let m = frexpf(absf, &e) // absf = m * 2^e, m in [0.5,1)
            // Map to half range: exponent bias 15
            let he = UInt16(max(0, min(31, Int(e) + 14)))
            let hm = UInt16(max(0, min(1023, Int((m * 2 - 1) * 1024))))
            return sign | (he << 10) | hm
        }
        let dst = raw.bindMemory(to: UInt16.self, capacity: w*h*4)
        for i in 0..<(w*h) {
            let c = residualLinear[i]
            dst[i*4+0] = floatToHalf(c.x)
            dst[i*4+1] = floatToHalf(c.y)
            dst[i*4+2] = floatToHalf(c.z)
            dst[i*4+3] = 0
        }
        let region = MTLRegionMake2D(0, 0, w, h)
        residualTexture.replace(region: region, mipmapLevel: 0, withBytes: raw, bytesPerRow: rowBytes)

        // GPU accumulate gradients (color DC via SH c0 and opacity)
        let s = dataset.samples[currentFrameIndex]
        let W2C = worldToCamMatrix(extr: s.extr)
        if gradBuffers == nil || (gradBuffers!.c0R.length / MemoryLayout<UInt32>.stride) != gaussians.count {
            gradBuffers = renderer.makeGradBuffers(count: gaussians.count)
        }
        if let gbufs = gradBuffers {
            // zero grad buffers
            memset(gbufs.c0R.contents(), 0, gbufs.c0R.length)
            memset(gbufs.c0G.contents(), 0, gbufs.c0G.length)
            memset(gbufs.c0B.contents(), 0, gbufs.c0B.length)
            memset(gbufs.r1x.contents(), 0, gbufs.r1x.length)
            memset(gbufs.r1y.contents(), 0, gbufs.r1y.length)
            memset(gbufs.r1z.contents(), 0, gbufs.r1z.length)
            memset(gbufs.g1x.contents(), 0, gbufs.g1x.length)
            memset(gbufs.g1y.contents(), 0, gbufs.g1y.length)
            memset(gbufs.g1z.contents(), 0, gbufs.g1z.length)
            memset(gbufs.b1x.contents(), 0, gbufs.b1x.length)
            memset(gbufs.b1y.contents(), 0, gbufs.b1y.length)
            memset(gbufs.b1z.contents(), 0, gbufs.b1z.length)
            // L2 per-coefficient buffers now 6 per channel
            for buf in [gbufs.r2, gbufs.g2, gbufs.b2] { memset(buf.contents(), 0, buf.length) }
            memset(gbufs.opacity.contents(), 0, gbufs.opacity.length)
            memset(gbufs.sigma.contents(), 0, gbufs.sigma.length)
            memset(gbufs.posX.contents(), 0, gbufs.posX.length)
            memset(gbufs.posY.contents(), 0, gbufs.posY.length)
            memset(gbufs.posZ.contents(), 0, gbufs.posZ.length)
            memset(gbufs.sX.contents(), 0, gbufs.sX.length)
            memset(gbufs.sY.contents(), 0, gbufs.sY.length)
            memset(gbufs.sZ.contents(), 0, gbufs.sZ.length)
            renderer.backward(gaussiansBuffer: gaussianBuffer, gaussianCount: gaussians.count, worldToCam: W2C, intrinsics: s.intr, residual: residualTexture, grads: gbufs, zSign: zSign, enableSHL2: enableSHL2)
        }

        // Read back grads (fixed-point scaled by 1024) and normalize
        var colorGrads = Array(repeating: SIMD3<Float>(repeating: 0), count: gaussians.count)
        var shR1Grads = Array(repeating: SIMD3<Float>(repeating: 0), count: gaussians.count)
        var shG1Grads = shR1Grads
        var shB1Grads = shR1Grads
    var shR2_ex1_Grads = Array(repeating: SIMD4<Float>(repeating: 0), count: gaussians.count)
    var shR2_ex2_Grads = shR2_ex1_Grads
    var shG2_ex1_Grads = shR2_ex1_Grads
    var shG2_ex2_Grads = shR2_ex1_Grads
    var shB2_ex1_Grads = shR2_ex1_Grads
    var shB2_ex2_Grads = shR2_ex1_Grads
        var opacityGrads = Array(repeating: Float(0), count: gaussians.count)
        var sigmaGrads = Array(repeating: Float(0), count: gaussians.count)
        var posGradsGPU = Array(repeating: SIMD3<Float>(repeating: 0), count: gaussians.count)
        var sXGrads = Array(repeating: Float(0), count: gaussians.count)
        var sYGrads = Array(repeating: Float(0), count: gaussians.count)
        var sZGrads = Array(repeating: Float(0), count: gaussians.count)
        let norm: Float = 1.0 / 1024.0 / Float(max(1, w*h))
        if let gbufs = gradBuffers {
            let n = gaussians.count
            let gr = gbufs.c0R.contents().bindMemory(to: Int32.self, capacity: n)
            let gg = gbufs.c0G.contents().bindMemory(to: Int32.self, capacity: n)
            let gb = gbufs.c0B.contents().bindMemory(to: Int32.self, capacity: n)
            let rr1x = gbufs.r1x.contents().bindMemory(to: Int32.self, capacity: n)
            let rr1y = gbufs.r1y.contents().bindMemory(to: Int32.self, capacity: n)
            let rr1z = gbufs.r1z.contents().bindMemory(to: Int32.self, capacity: n)
            let rg1x = gbufs.g1x.contents().bindMemory(to: Int32.self, capacity: n)
            let rg1y = gbufs.g1y.contents().bindMemory(to: Int32.self, capacity: n)
            let rg1z = gbufs.g1z.contents().bindMemory(to: Int32.self, capacity: n)
            let rb1x = gbufs.b1x.contents().bindMemory(to: Int32.self, capacity: n)
            let rb1y = gbufs.b1y.contents().bindMemory(to: Int32.self, capacity: n)
            let rb1z = gbufs.b1z.contents().bindMemory(to: Int32.self, capacity: n)
            // Ensure SH L2 activation log once if enabled
            checkSHL2ActivationLog()
            // Coefficient-major stride equals gaussian count
            let stride = n
                let r2ptr = gbufs.r2.contents().bindMemory(to: Int32.self, capacity: stride*6)
                let g2ptr = gbufs.g2.contents().bindMemory(to: Int32.self, capacity: stride*6)
                let b2ptr = gbufs.b2.contents().bindMemory(to: Int32.self, capacity: stride*6)
            let go = gbufs.opacity.contents().bindMemory(to: Int32.self, capacity: n)
            let gs = gbufs.sigma.contents().bindMemory(to: Int32.self, capacity: n)
            let gpx = gbufs.posX.contents().bindMemory(to: Int32.self, capacity: n)
            let gpy = gbufs.posY.contents().bindMemory(to: Int32.self, capacity: n)
            let gpz = gbufs.posZ.contents().bindMemory(to: Int32.self, capacity: n)
            let gsx = gbufs.sX.contents().bindMemory(to: Int32.self, capacity: n)
            let gsy = gbufs.sY.contents().bindMemory(to: Int32.self, capacity: n)
            let gsz = gbufs.sZ.contents().bindMemory(to: Int32.self, capacity: n)
            // Note: anisotropic scale grads (sX/sY/sZ) are produced but unused here until we parameterize anisotropic scales on CPU.
            for i in 0..<n {
                colorGrads[i].x = Float(gr[i]) * norm
                colorGrads[i].y = Float(gg[i]) * norm
                colorGrads[i].z = Float(gb[i]) * norm
                shR1Grads[i] = SIMD3<Float>(Float(rr1x[i]) * norm, Float(rr1y[i]) * norm, Float(rr1z[i]) * norm)
                shG1Grads[i] = SIMD3<Float>(Float(rg1x[i]) * norm, Float(rg1y[i]) * norm, Float(rg1z[i]) * norm)
                shB1Grads[i] = SIMD3<Float>(Float(rb1x[i]) * norm, Float(rb1y[i]) * norm, Float(rb1z[i]) * norm)
                // SH L2: interpret packed additions (we stored all coeff contributions into same atomic integers sequentially)
                // Map 6 coeff grads into two SIMD4 blocks (ex1: 0..3, ex2: 4..5, pad..pad)
                func coeff(_ base: UnsafePointer<Int32>, _ c: Int, _ i: Int) -> Float { Float(base[c*stride + i]) * norm }
                shR2_ex1_Grads[i] = SIMD4<Float>(coeff(r2ptr,0,i), coeff(r2ptr,1,i), coeff(r2ptr,2,i), coeff(r2ptr,3,i))
                shR2_ex2_Grads[i] = SIMD4<Float>(coeff(r2ptr,4,i), coeff(r2ptr,5,i), 0, 0)
                shG2_ex1_Grads[i] = SIMD4<Float>(coeff(g2ptr,0,i), coeff(g2ptr,1,i), coeff(g2ptr,2,i), coeff(g2ptr,3,i))
                shG2_ex2_Grads[i] = SIMD4<Float>(coeff(g2ptr,4,i), coeff(g2ptr,5,i), 0, 0)
                shB2_ex1_Grads[i] = SIMD4<Float>(coeff(b2ptr,0,i), coeff(b2ptr,1,i), coeff(b2ptr,2,i), coeff(b2ptr,3,i))
                shB2_ex2_Grads[i] = SIMD4<Float>(coeff(b2ptr,4,i), coeff(b2ptr,5,i), 0, 0)
                opacityGrads[i] = Float(go[i]) * norm
                sigmaGrads[i] = Float(gs[i]) * norm
                posGradsGPU[i] = SIMD3<Float>(Float(gpx[i]) * norm, Float(gpy[i]) * norm, Float(gpz[i]) * norm)
                sXGrads[i] = Float(gsx[i]) * norm
                sYGrads[i] = Float(gsy[i]) * norm
                sZGrads[i] = Float(gsz[i]) * norm
            }
        }

        // Position gradients: prefer GPU-computed grads when enabled
        var posGrads = Array(repeating: SIMD3<Float>(repeating: 0), count: gaussians.count)
        if geoOptimizePositions {
            posGrads = posGradsGPU
        }
        // removed legacy CPU color/opacity gradient integration (now computed on GPU)

    let scaleReg: Float = 1e-3
    let scaleMin: Float = 0.003
    let scaleMax: Float = 0.3

        // Adam update per gaussian (with opacity prior and step clipping)
        let t = Float(iteration + 1)
        for i in 0..<gaussians.count {
            var g = gaussians[i]
            // Gradients
            var gradColor = colorGrads[i]
            var gradOpacity = opacityGrads[i] + 2.0 * opacityReg * (g.opacity - opacityPrior)
            var gradSigma = sigmaGrads[i] + scaleReg * g.sigma
            var gradR1 = shR1Grads[i]
            var gradG1 = shG1Grads[i]
            var gradB1 = shB1Grads[i]
            // Global clip (in-place scale if norm exceeds)
            if enableSHL2 {
                var r2e1g = shR2_ex1_Grads[i]
                var r2e2g = shR2_ex2_Grads[i]
                var g2e1g = shG2_ex1_Grads[i]
                var g2e2g = shG2_ex2_Grads[i]
                var b2e1g = shB2_ex1_Grads[i]
                var b2e2g = shB2_ex2_Grads[i]
                applyGlobalClipExtended(color: &gradColor, opacity: &gradOpacity, sigma: &gradSigma, shR1: &gradR1, shG1: &gradG1, shB1: &gradB1, r2e1: &r2e1g, r2e2: &r2e2g, g2e1: &g2e1g, g2e2: &g2e2g, b2e1: &b2e1g, b2e2: &b2e2g)
                shR2_ex1_Grads[i] = r2e1g; shR2_ex2_Grads[i] = r2e2g
                shG2_ex1_Grads[i] = g2e1g; shG2_ex2_Grads[i] = g2e2g
                shB2_ex1_Grads[i] = b2e1g; shB2_ex2_Grads[i] = b2e2g
            } else {
                applyGlobalClip(color: &gradColor, opacity: &gradOpacity, sigma: &gradSigma, shR1: &gradR1, shG1: &gradG1, shB1: &gradB1)
            }
            // Replace original references for optimizer
            colorGrads[i] = gradColor
            opacityGrads[i] = gradOpacity
            sigmaGrads[i] = gradSigma
            shR1Grads[i] = gradR1
            shG1Grads[i] = gradG1
            shB1Grads[i] = gradB1
            // Moments
            m[i].color = beta1 * m[i].color + (1 - beta1) * gradColor
            v[i].color = beta2 * v[i].color + (1 - beta2) * (gradColor * gradColor)
            m[i].opacity = beta1 * m[i].opacity + (1 - beta1) * gradOpacity
            v[i].opacity = beta2 * v[i].opacity + (1 - beta2) * (gradOpacity * gradOpacity)
            m[i].sigma = beta1 * m[i].sigma + (1 - beta1) * gradSigma
            v[i].sigma = beta2 * v[i].sigma + (1 - beta2) * (gradSigma * gradSigma)
            // SH L1 moments (and optionally L2 below)
            let gcR1 = gradR1
            let gcG1 = gradG1
            let gcB1 = gradB1
            mShR1[i] = beta1 * mShR1[i] + (1 - beta1) * gcR1
            vShR1[i] = beta2 * vShR1[i] + (1 - beta2) * (gcR1 * gcR1)
            mShG1[i] = beta1 * mShG1[i] + (1 - beta1) * gcG1
            vShG1[i] = beta2 * vShG1[i] + (1 - beta2) * (gcG1 * gcG1)
            mShB1[i] = beta1 * mShB1[i] + (1 - beta1) * gcB1
            vShB1[i] = beta2 * vShB1[i] + (1 - beta2) * (gcB1 * gcB1)

            let mhatColor = m[i].color / (1 - pow(beta1, t))
            let vhatColor = v[i].color / (1 - pow(beta2, t))
            let mhatOpacity = m[i].opacity / (1 - pow(beta1, t))
            let vhatOpacity = v[i].opacity / (1 - pow(beta2, t))
            let mhatSigma = m[i].sigma / (1 - pow(beta1, t))
            let vhatSigma = v[i].sigma / (1 - pow(beta2, t))

            let denomColor = SIMD3<Float>(
                sqrt(vhatColor.x) + eps,
                sqrt(vhatColor.y) + eps,
                sqrt(vhatColor.z) + eps
            )
            // Small weight decay for stability
            let wd: Float = 1e-4
            var stepColor = lr * (mhatColor / denomColor + wd * g.color)
            // per-component clipping
            stepColor.x = max(-maxStepColor, min(maxStepColor, stepColor.x))
            stepColor.y = max(-maxStepColor, min(maxStepColor, stepColor.y))
            stepColor.z = max(-maxStepColor, min(maxStepColor, stepColor.z))
            g.color -= stepColor

            let denomOpacity = sqrt(vhatOpacity) + eps
            var stepOpacity = lr * (mhatOpacity / denomOpacity + wd * g.opacity)
            stepOpacity = max(-maxStepOpacity, min(maxStepOpacity, stepOpacity))
            g.opacity -= stepOpacity

            g.sigma -= lr * (mhatSigma / (sqrt(vhatSigma) + eps))

            g.color = simd_clamp(g.color, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
            // Clamp opacity to avoid saturating to 0 or 1
            g.opacity = max(0.02, min(maxOpacity, g.opacity))
            // Update anisotropic scales using Adam grads (proxy); keep sigma as mean(scale)
            // Moments for scales
            mScaleX[i] = beta1 * mScaleX[i] + (1 - beta1) * (sXGrads[i] + scaleReg * scaleX[i])
            mScaleY[i] = beta1 * mScaleY[i] + (1 - beta1) * (sYGrads[i] + scaleReg * scaleY[i])
            mScaleZ[i] = beta1 * mScaleZ[i] + (1 - beta1) * (sZGrads[i] + scaleReg * scaleZ[i])
            vScaleX[i] = beta2 * vScaleX[i] + (1 - beta2) * pow(sXGrads[i] + scaleReg * scaleX[i], 2)
            vScaleY[i] = beta2 * vScaleY[i] + (1 - beta2) * pow(sYGrads[i] + scaleReg * scaleY[i], 2)
            vScaleZ[i] = beta2 * vScaleZ[i] + (1 - beta2) * pow(sZGrads[i] + scaleReg * scaleZ[i], 2)
            let mhatSX = mScaleX[i] / (1 - pow(beta1, t)); let vhatSX = vScaleX[i] / (1 - pow(beta2, t))
            let mhatSY = mScaleY[i] / (1 - pow(beta1, t)); let vhatSY = vScaleY[i] / (1 - pow(beta2, t))
            let mhatSZ = mScaleZ[i] / (1 - pow(beta1, t)); let vhatSZ = vScaleZ[i] / (1 - pow(beta2, t))
            var stepSX = lr * (mhatSX / (sqrt(vhatSX) + eps))
            var stepSY = lr * (mhatSY / (sqrt(vhatSY) + eps))
            var stepSZ = lr * (mhatSZ / (sqrt(vhatSZ) + eps))
            let maxStepScale: Float = 0.01
            stepSX = max(-maxStepScale, min(maxStepScale, stepSX))
            stepSY = max(-maxStepScale, min(maxStepScale, stepSY))
            stepSZ = max(-maxStepScale, min(maxStepScale, stepSZ))
            scaleX[i] = max(scaleMin, min(scaleMax, scaleX[i] - stepSX))
            scaleY[i] = max(scaleMin, min(scaleMax, scaleY[i] - stepSY))
            scaleZ[i] = max(scaleMin, min(scaleMax, scaleZ[i] - stepSZ))
            g.sigma = (scaleX[i] + scaleY[i] + scaleZ[i]) / 3.0
            gaussians[i] = g
            // Apply SH L1 updates (Adam)
            let mhatR1 = mShR1[i] / (1 - pow(beta1, t))
            let vhatR1 = vShR1[i] / (1 - pow(beta2, t))
            let mhatG1 = mShG1[i] / (1 - pow(beta1, t))
            let vhatG1 = vShG1[i] / (1 - pow(beta2, t))
            let mhatB1 = mShB1[i] / (1 - pow(beta1, t))
            let vhatB1 = vShB1[i] / (1 - pow(beta2, t))
            // decoupled weight decay for SH
            let wdSH: Float = 1e-4
            let denomR1 = SIMD3<Float>(sqrt(vhatR1.x)+eps, sqrt(vhatR1.y)+eps, sqrt(vhatR1.z)+eps)
            let denomG1 = SIMD3<Float>(sqrt(vhatG1.x)+eps, sqrt(vhatG1.y)+eps, sqrt(vhatG1.z)+eps)
            let denomB1 = SIMD3<Float>(sqrt(vhatB1.x)+eps, sqrt(vhatB1.y)+eps, sqrt(vhatB1.z)+eps)
            var stepR1 = lr * (mhatR1 / denomR1 + wdSH * shR1[i])
            var stepG1 = lr * (mhatG1 / denomG1 + wdSH * shG1[i])
            var stepB1 = lr * (mhatB1 / denomB1 + wdSH * shB1[i])
            // clip steps for stability
            let maxStepSH: Float = 0.05
            stepR1.x = max(-maxStepSH, min(maxStepSH, stepR1.x))
            stepR1.y = max(-maxStepSH, min(maxStepSH, stepR1.y))
            stepR1.z = max(-maxStepSH, min(maxStepSH, stepR1.z))
            stepG1.x = max(-maxStepSH, min(maxStepSH, stepG1.x))
            stepG1.y = max(-maxStepSH, min(maxStepSH, stepG1.y))
            stepG1.z = max(-maxStepSH, min(maxStepSH, stepG1.z))
            stepB1.x = max(-maxStepSH, min(maxStepSH, stepB1.x))
            stepB1.y = max(-maxStepSH, min(maxStepSH, stepB1.y))
            stepB1.z = max(-maxStepSH, min(maxStepSH, stepB1.z))
            shR1[i] -= stepR1
            shG1[i] -= stepG1
            shB1[i] -= stepB1
            if enableSHL2 {
                let tt = Float(iteration+1)
                adamUpdateSIMD4(beta1: beta1, beta2: beta2, eps: eps, grad: shR2_ex1_Grads[i], m: &mShR2_ex1[i], v: &vShR2_ex1[i], param: &shR2_ex1[i], lr: lr, t: tt, clip: 0.05)
                adamUpdateSIMD4(beta1: beta1, beta2: beta2, eps: eps, grad: shR2_ex2_Grads[i], m: &mShR2_ex2[i], v: &vShR2_ex2[i], param: &shR2_ex2[i], lr: lr, t: tt, clip: 0.05)
                adamUpdateSIMD4(beta1: beta1, beta2: beta2, eps: eps, grad: shG2_ex1_Grads[i], m: &mShG2_ex1[i], v: &vShG2_ex1[i], param: &shG2_ex1[i], lr: lr, t: tt, clip: 0.05)
                adamUpdateSIMD4(beta1: beta1, beta2: beta2, eps: eps, grad: shG2_ex2_Grads[i], m: &mShG2_ex2[i], v: &vShG2_ex2[i], param: &shG2_ex2[i], lr: lr, t: tt, clip: 0.05)
                adamUpdateSIMD4(beta1: beta1, beta2: beta2, eps: eps, grad: shB2_ex1_Grads[i], m: &mShB2_ex1[i], v: &vShB2_ex1[i], param: &shB2_ex1[i], lr: lr, t: tt, clip: 0.05)
                adamUpdateSIMD4(beta1: beta1, beta2: beta2, eps: eps, grad: shB2_ex2_Grads[i], m: &mShB2_ex2[i], v: &vShB2_ex2[i], param: &shB2_ex2[i], lr: lr, t: tt, clip: 0.05)
            }
            // clamp SH coefficients to a reasonable range
            shR1[i] = clamp3(shR1[i], -1.0, 1.0)

            // Position Adam update (world-space)
            if geoOptimizePositions {
                let gp = posGrads[i]
                m[i].position = beta1 * m[i].position + (1 - beta1) * gp
                v[i].position = beta2 * v[i].position + (1 - beta2) * (gp * gp)
                let mhatP = m[i].position / (1 - pow(beta1, t))
                let vhatP = v[i].position / (1 - pow(beta2, t))
                var stepP = posLr * (mhatP / (SIMD3<Float>(sqrt(vhatP.x), sqrt(vhatP.y), sqrt(vhatP.z)) + SIMD3<Float>(repeating: eps)))
                // clip for stability
                stepP.x = max(-posStepClip, min(posStepClip, stepP.x))
                stepP.y = max(-posStepClip, min(posStepClip, stepP.y))
                stepP.z = max(-posStepClip, min(posStepClip, stepP.z))
                g.position -= stepP
            }
            shG1[i] = clamp3(shG1[i], -1.0, 1.0)
            shB1[i] = clamp3(shB1[i], -1.0, 1.0)
        }

        // Periodic diagnostics (every 100 iterations) for gradient and coefficient health
        if iteration % 100 == 0 {
            let n = max(1, gaussians.count)
            // Helper lambdas
            func rms3(_ arr: [SIMD3<Float>]) -> Float {
                var s: Float = 0; for v in arr { s += v.x*v.x + v.y*v.y + v.z*v.z }; return sqrt(s / (Float(arr.count)*3.0))
            }
            func rms4(_ arr: [SIMD4<Float>]) -> Float {
                var s: Float = 0; for v in arr { s += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w }; return sqrt(s / (Float(arr.count)*4.0))
            }
            func minMax3(_ arr: [SIMD3<Float>]) -> (Float,Float) {
                var mn: Float = .greatestFiniteMagnitude, mx: Float = -.greatestFiniteMagnitude
                for v in arr { mn = min(mn, v.x, v.y, v.z); mx = max(mx, v.x, v.y, v.z) }
                return (mn, mx)
            }
            func minMax4(_ arr: [SIMD4<Float>]) -> (Float,Float) {
                var mn: Float = .greatestFiniteMagnitude, mx: Float = -.greatestFiniteMagnitude
                for v in arr { mn = min(mn, v.x, v.y, v.z, v.w); mx = max(mx, v.x, v.y, v.z, v.w) }
                return (mn, mx)
            }
            // SH L1 coeff stats
            let (r1mn,r1mx) = minMax3(shR1)
            let (g1mn,g1mx) = minMax3(shG1)
            let (b1mn,b1mx) = minMax3(shB1)
            var l2CountActive = 0
            if enableSHL2 {
                for i in 0..<n { // count non-trivial L2 coeffs (abs > 1e-6)
                    let a = shR2_ex1[i]; let b = shR2_ex2[i]
                    if abs(a.x) > 1e-6 { l2CountActive += 1 }
                    if abs(a.y) > 1e-6 { l2CountActive += 1 }
                    if abs(a.z) > 1e-6 { l2CountActive += 1 }
                    if abs(a.w) > 1e-6 { l2CountActive += 1 }
                    if abs(b.x) > 1e-6 { l2CountActive += 1 }
                    if abs(b.y) > 1e-6 { l2CountActive += 1 }
                }
            }
            // Gradient RMS (use last batch grads already clipped)
            let rmsColor = {
                var s: Float = 0; for v in colorGrads { s += v.x*v.x + v.y*v.y + v.z*v.z }; return sqrt(s / (Float(colorGrads.count)*3.0))
            }()
            let rmsOpacity = {
                var s: Float = 0; for v in opacityGrads { s += v*v }; return sqrt(s / Float(opacityGrads.count))
            }()
            let rmsSigma = {
                var s: Float = 0; for v in sigmaGrads { s += v*v }; return sqrt(s / Float(sigmaGrads.count))
            }()
            let rmsR1 = rms3(shR1Grads)
            let rmsG1 = rms3(shG1Grads)
            let rmsB1 = rms3(shB1Grads)
            var rmsR2e1: Float = 0, rmsR2e2: Float = 0, rmsG2e1: Float = 0, rmsG2e2: Float = 0, rmsB2e1: Float = 0, rmsB2e2: Float = 0
            if enableSHL2 {
                rmsR2e1 = rms4(shR2_ex1_Grads); rmsR2e2 = rms4(shR2_ex2_Grads)
                rmsG2e1 = rms4(shG2_ex1_Grads); rmsG2e2 = rms4(shG2_ex2_Grads)
                rmsB2e1 = rms4(shB2_ex1_Grads); rmsB2e2 = rms4(shB2_ex2_Grads)
            }
            log(String(format: "diag iter %d | colorRMS %.3e opacRMS %.3e sigRMS %.3e | R1[%.2f,%.2f] rms %.2e | G1[%.2f,%.2f] rms %.2e | B1[%.2f,%.2f] rms %.2e", iteration, rmsColor, rmsOpacity, rmsSigma, r1mn, r1mx, rmsR1, g1mn, g1mx, rmsG1, b1mn, b1mx, rmsB1))
            if enableSHL2 {
                log(String(format: "   L2 active coeffs %d / %d | R2e1 %.3e R2e2 %.3e G2e1 %.3e G2e2 %.3e B2e1 %.3e B2e2 %.3e", l2CountActive, n*18, rmsR2e1, rmsR2e2, rmsG2e1, rmsG2e2, rmsB2e1, rmsB2e2))
            }
        }

        // Position updates are applied inline above via Adam when enabled

        return loss
    }

    // MARK: - Precise Gauss-Newton Sim3 with quadratic objective
    func sim3RefineGN(maxPoints: Int = 2000, iterations: Int = 5, lambda: Float = 1e-3) {
        let wasTraining = isTraining
        if wasTraining { pause() }
        let frameCount = dataset.samples.count
        guard frameCount > 0, !gaussians.isEmpty else { return }
        // Use up to 8 evenly spaced frames across the sequence
        var framesToUse: [Int] = []
        let wantedFrames = min(8, frameCount)
        if wantedFrames <= 1 {
            framesToUse = [0]
        } else {
            for i in 0..<wantedFrames {
                let idx = Int(round(Float(i) * Float(frameCount - 1) / Float(wantedFrames - 1)))
                framesToUse.append(idx)
            }
            framesToUse = Array(Set(framesToUse)).sorted()
        }
        // Base points from original positions with flips/scale (no Sim3)
        let sx: Float = flipX ? -plyScale : plyScale
        let sy: Float = flipY ? -plyScale : plyScale
        let sz: Float = flipZ ? -plyScale : plyScale
        let step = max(1, gaussians.count / maxPoints)
        var basePts: [SIMD3<Float>] = []
        basePts.reserveCapacity((gaussians.count + step - 1)/step)
        var idx = 0
        var bboxMin = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var bboxMax = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        while idx < gaussians.count {
            let p0 = originalPositions[idx]
            let p = SIMD3<Float>(p0.x * sx, p0.y * sy, p0.z * sz)
            basePts.append(p)
            bboxMin = simd_min(bboxMin, p)
            bboxMax = simd_max(bboxMax, p)
            idx += step
        }
    let extent = bboxMax - bboxMin
    let radius = max(1e-6, max(extent.x, max(extent.y, extent.z)))
        struct FramePack { let W2C: simd_float4x4; let intr: CameraIntrinsics; let w: Int; let h: Int }
        var packs: [FramePack] = []
        packs.reserveCapacity(framesToUse.count)
        for fi in framesToUse {
            let s = dataset.samples[fi]
            packs.append(FramePack(W2C: worldToCamMatrix(extr: s.extr), intr: s.intr, w: dataset.width, h: dataset.height))
        }
        // Params (Euler ZYX)
    var s = max(1e-6, sim3Scale)
    var rs = logf(s) // log-scale for positivity and stability
        func quatToEuler(_ q: simd_quatf) -> (Float,Float,Float) {
            let m = simd_float3x3(q)
            let sy = -m[2,0]
            let x = atan2f(m[2,1], m[2,2])
            let y = asinf(max(-1, min(1, sy)))
            let z = atan2f(m[1,0], m[0,0])
            return (x,y,z)
        }
        var (rx, ry, rz) = quatToEuler(sim3Rot)
        var tx = sim3Trans.x, ty = sim3Trans.y, tz = sim3Trans.z

        @inline(__always)
        func rotFromEuler(_ rx: Float, _ ry: Float, _ rz: Float) -> simd_quatf {
            simd_quaternion(rz, SIMD3<Float>(0,0,1)) * simd_quaternion(ry, SIMD3<Float>(0,1,0)) * simd_quaternion(rx, SIMD3<Float>(1,0,0))
        }
        @inline(__always)
        func applySim3(_ p: SIMD3<Float>, _ s: Float, _ rx: Float, _ ry: Float, _ rz: Float, _ t: SIMD3<Float>) -> SIMD3<Float> {
            let q = rotFromEuler(rx,ry,rz)
            return s * q.act(p) + t
        }
        @inline(__always)
        func residualsFor(_ p: SIMD3<Float>, _ pack: FramePack, _ s: Float, _ rx: Float, _ ry: Float, _ rz: Float, _ t: SIMD3<Float>) -> (Float,Float,Float,Float,Float) {
            let pw = applySim3(p, s, rx, ry, rz, t)
            let cam4 = pack.W2C * SIMD4<Float>(pw.x, pw.y, pw.z, 1)
            let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
            let zf: Float = (zSign < 0) ? (-cam.z) : (cam.z)
            let u = pack.intr.fx * (cam.x / max(1e-6,zf)) + pack.intr.cx
            let v = pack.intr.fy * (cam.y / max(1e-6,zf)) + pack.intr.cy
            // Normalize residuals to be dimensionless
            let wInv = 1.0 / Float(max(1, pack.w))
            let hInv = 1.0 / Float(max(1, pack.h))
            let ru0 = max(0, -u) * wInv
            let ru1 = max(0, u - Float(pack.w - 1)) * wInv
            let rv0 = max(0, -v) * hInv
            let rv1 = max(0, v - Float(pack.h - 1)) * hInv
            let zMin = 0.02 * radius
            let rzResid = max(0, zMin - zf) / radius
            return (ru0, ru1, rv0, rv1, rzResid)
        }

        @inline(__always)
        func objective(_ rs: Float, _ rx: Float, _ ry: Float, _ rz: Float, _ tx: Float, _ ty: Float, _ tz: Float) -> Float {
            var sum: Float = 0
            for p in basePts {
                for pk in packs {
                    let r = residualsFor(p, pk, expf(rs), rx, ry, rz, SIMD3<Float>(tx,ty,tz))
                    sum += r.0*r.0 + r.1*r.1 + r.2*r.2 + r.3*r.3 + (2*r.4)*(2*r.4) // heavier z-front weight
                }
            }
            return sum
        }

        func solve7(_ A: inout [[Float]], _ b: inout [Float]) -> [Float]? {
            let n = 7
            var M = Array(repeating: Array(repeating: Float(0), count: n+1), count: n)
            for i in 0..<n { for j in 0..<n { M[i][j] = A[i][j] }; M[i][n] = b[i] }
            for k in 0..<n {
                var piv = k
                var maxv = abs(M[k][k])
                for i in (k+1)..<n { if abs(M[i][k]) > maxv { maxv = abs(M[i][k]); piv = i } }
                if maxv < 1e-12 { return nil }
                if piv != k { M.swapAt(piv, k) }
                let diag = M[k][k]
                for j in k..<(n+1) { M[k][j] /= diag }
                for i in 0..<n {
                    if i == k { continue }
                    let factor = M[i][k]
                    if factor == 0 { continue }
                    for j in k..<(n+1) { M[i][j] -= factor * M[k][j] }
                }
            }
            var x = Array(repeating: Float(0), count: n)
            for i in 0..<n { x[i] = M[i][n] }
            return x
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            var lm = lambda
            for _ in 0..<iterations {
                let obj0 = objective(rs, rx, ry, rz, tx, ty, tz)
                var JTJ = Array(repeating: Array(repeating: Float(0), count: 7), count: 7)
                var JTr = Array(repeating: Float(0), count: 7)
                // Parameterization: theta = [rs, rx, ry, rz, tx, ty, tz]
                let baseTheta: [Float] = [rs, rx, ry, rz, tx, ty, tz]
                let epsS: Float = 1e-3
                let epsAng: Float = 1e-2 // ~0.57 deg
                let epsT: Float = max(1e-3, 1e-2 * radius)
                let eps: [Float] = [epsS, epsAng, epsAng, epsAng, epsT, epsT, epsT]
                for p in basePts {
                    for pk in packs {
                        let r0 = residualsFor(p, pk, expf(rs), rx, ry, rz, SIMD3<Float>(tx,ty,tz))
                        let rvec0: [Float] = [r0.0, r0.1, r0.2, r0.3, r0.4]
                        var cols = Array(repeating: Array(repeating: Float(0), count: 5), count: 7)
                        for j in 0..<7 {
                            var th = baseTheta
                            th[j] += eps[j]
                            // Convert rs back to s for j==0
                            let s1 = expf(th[0])
                            let r1 = residualsFor(p, pk, s1, th[1], th[2], th[3], SIMD3<Float>(th[4], th[5], th[6]))
                            let rvec1: [Float] = [r1.0, r1.1, r1.2, r1.3, r1.4]
                            for k in 0..<5 { cols[j][k] = (rvec1[k] - rvec0[k]) / eps[j] }
                        }
                        for j in 0..<7 {
                            var jtr: Float = 0
                            for k in 0..<5 { jtr += cols[j][k] * rvec0[k] }
                            JTr[j] += jtr
                            for j2 in 0..<7 {
                                var val: Float = 0
                                for k in 0..<5 { val += cols[j][k] * cols[j2][k] }
                                JTJ[j][j2] += val
                            }
                        }
                    }
                }
                for d in 0..<7 { JTJ[d][d] += lm }
                var A = JTJ
                var b = JTr.map { -$0 }
                if let dx = solve7(&A, &b) {
                    // Backtracking with trust-region clamps
                    let maxAng: Float = Float(15.0 * .pi / 180.0)
                    let maxT = max(0.01, 0.25 * radius)
                    func clampDx(_ v: [Float], scale: Float) -> (Float,Float,Float,Float,Float,Float,Float) {
                        let dRs = max(-0.5, min(0.5, v[0] * scale))
                        let dRx = max(-maxAng, min(maxAng, v[1] * scale))
                        let dRy = max(-maxAng, min(maxAng, v[2] * scale))
                        let dRz = max(-maxAng, min(maxAng, v[3] * scale))
                        let dTx = max(-maxT, min(maxT, v[4] * scale))
                        let dTy = max(-maxT, min(maxT, v[5] * scale))
                        let dTz = max(-maxT, min(maxT, v[6] * scale))
                        return (dRs,dRx,dRy,dRz,dTx,dTy,dTz)
                    }
                    var scale: Float = 1.0
                    var accepted = false
                    var bestTuple: (Float,Float,Float,Float,Float,Float,Float) = (0,0,0,0,0,0,0)
                    for _ in 0..<5 {
                        let d = clampDx(dx, scale: scale)
                        let rs1 = rs + d.0
                        let rx1 = rx + d.1
                        let ry1 = ry + d.2
                        let rz1 = rz + d.3
                        let tx1 = tx + d.4
                        let ty1 = ty + d.5
                        let tz1 = tz + d.6
                        let obj1 = objective(rs1, rx1, ry1, rz1, tx1, ty1, tz1)
                        if obj1 < obj0 { accepted = true; bestTuple = d; break }
                        scale *= 0.5
                    }
                    if accepted {
                        rs += bestTuple.0; rx += bestTuple.1; ry += bestTuple.2; rz += bestTuple.3
                        tx += bestTuple.4; ty += bestTuple.5; tz += bestTuple.6
                        s = expf(rs)
                        lm = max(1e-6, lm * 0.7)
                    } else {
                        lm *= 5.0
                    }
                } else { break }
            }
            let q = rotFromEuler(rx, ry, rz)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.sim3Scale = s
                self.sim3Rot = q
                self.sim3Trans = SIMD3<Float>(tx,ty,tz)
                self.applyPointTransform()
                let frameCount = self.dataset.samples.count
                var framesToTest: [Int] = [0]
                if frameCount > 2 { framesToTest.append(frameCount/2) }
                if frameCount > 1 { framesToTest.append(frameCount-1) }
                framesToTest = Array(Set(framesToTest)).sorted()
                var total: Int = 0, front: Int = 0, inBounds: Int = 0
                for fi in framesToTest {
                    let d = self.reprojectionDiagnostics(for: fi)
                    total += d.total; front += d.front; inBounds += d.inBounds
                }
                self.autoFrontPct = (total > 0) ? (Float(front) / Float(total) * 100.0) : 0
                self.autoInBoundsPct = (total > 0) ? (Float(inBounds) / Float(total) * 100.0) : 0
                self.autoConfigSummary = String(
                    format: "refined(gn): s=%.3f, t=(%.3f,%.3f,%.3f), inBounds=%.1f%%, front=%.1f%%, size=%dx%d, %@",
                    self.sim3Scale, self.sim3Trans.x, self.sim3Trans.y, self.sim3Trans.z,
                    self.autoInBoundsPct, self.autoFrontPct, self.dataset.width, self.dataset.height, self.dataSourceLabel
                )
                self.log(String(format: "Sim3 GN refined: s=%.4f, rx=%.2f°, ry=%.2f°, rz=%.2f°, t=(%.3f,%.3f,%.3f)", s, rx*180/Float.pi, ry*180/Float.pi, rz*180/Float.pi, tx, ty, tz))
                if wasTraining { self.start() }
            }
        }
    }

    private func saveCheckpoint() {
        let ckpt = Checkpoint(iteration: iteration, gaussians: gaussians, learningRate: lr, shR1: shR1, shG1: shG1, shB1: shB1, shR2_ex1: enableSHL2 ? shR2_ex1 : nil, shR2_ex2: enableSHL2 ? shR2_ex2 : nil, shG2_ex1: enableSHL2 ? shG2_ex1 : nil, shG2_ex2: enableSHL2 ? shG2_ex2 : nil, shB2_ex1: enableSHL2 ? shB2_ex1 : nil, shB2_ex2: enableSHL2 ? shB2_ex2 : nil, scaleX: scaleX, scaleY: scaleY, scaleZ: scaleZ)
        CheckpointManager.save(ckpt)
    }

    // MARK: - Self Tests
    func runSelfTests() {
        log("[SelfTest] Starting checkpoint round-trip test...")
        let beforeIter = iteration
        saveCheckpoint()
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { log("[SelfTest] Documents dir missing"); return }
        guard let files = try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: [.contentModificationDateKey]) else { log("[SelfTest] Unable to list documents"); return }
        let ckpts = files.filter { $0.lastPathComponent.hasPrefix("checkpoint_") && $0.pathExtension == "json" }
        guard !ckpts.isEmpty else { log("[SelfTest] No checkpoint json found"); return }
        let newest = ckpts.max { (a,b) in
            let ad = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let bd = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return ad < bd
        }!
        do {
            let data = try Data(contentsOf: newest)
            let decoded = try JSONDecoder().decode(CheckpointData.self, from: data)
            let sampleN = min(10, gaussians.count, decoded.gaussians.count)
            var mismatches = 0
            for i in 0..<sampleN {
                let g0 = gaussians[i]
                let gd = decoded.gaussians[i]
                if simd_length(g0.position - gd.position) > 1e-5 { mismatches += 1 }
                if let dScaleX = decoded.scaleX, i < dScaleX.count {
                    if abs(scaleX[i] - dScaleX[i]) > 1e-5 { mismatches += 1 }
                }
                if let dShR1 = decoded.shR1, i < dShR1.count {
                    if abs(shR1[i].x - dShR1[i].x) > 1e-5 { mismatches += 1 }
                }
            }
            if mismatches == 0 { log("[SelfTest] PASS checkpoint round-trip (iter=\(beforeIter))") }
            else { log("[SelfTest] WARN checkpoint mismatches=\(mismatches)") }
        } catch {
            log("[SelfTest] ERROR decoding checkpoint: \(error)")
        }
    }

    private func step() {
        guard isTraining else { return }
    // Hold the same frame for a few steps to stabilize gradients
        if frameHoldCounter <= 0 {
            currentFrameIndex = (currentFrameIndex + 1) % dataset.samples.count
            frameHoldCounter = frameHold
        }
        let sampleIndex = currentFrameIndex
        guard let predCG = renderPrediction(for: sampleIndex) else { return }
        let gt = dataset.samples[sampleIndex].image
        let loss = computeLossAndGradients(pred: predCG, gt: gt)
        // Update EMA loss for logging
        lossEMA = (lossEMA == 0) ? loss : (0.95 * lossEMA + 0.05 * loss)

        // Sync updated gaussians to GPU
        _syncGaussiansToGPU()

        iteration += 1
        frameHoldCounter -= 1
        if iteration % 5 == 0 {
            // Average opacity for debugging collapse
            let avgOpacity = gaussians.reduce(0.0) { $0 + $1.opacity } / Float(max(1, gaussians.count))
            let stats = computeImageStats(predCG)
            // NEW: compute color variance (after gamma-encoded preview now matches human perception better)
            var variance: Float = 0
            do {
                let w = predCG.width, h = predCG.height
                let total = max(1, w * h)
                var data = [UInt8](repeating: 0, count: total * 4)
                let cs = CGColorSpaceCreateDeviceRGB()
                let info: CGBitmapInfo = [.byteOrder32Big, CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)]
                if let ctx = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: cs, bitmapInfo: info.rawValue) {
                    ctx.draw(predCG, in: CGRect(x: 0, y: 0, width: w, height: h))
                    var mean: SIMD3<Float> = .zero
                    for i in stride(from: 0, to: data.count, by: 4) {
                        mean.x += Float(data[i])
                        mean.y += Float(data[i+1])
                        mean.z += Float(data[i+2])
                    }
                    mean /= Float(total)
                    var varAcc: Float = 0
                    for i in stride(from: 0, to: data.count, by: 4) {
                        let r = Float(data[i]) - mean.x
                        let g = Float(data[i+1]) - mean.y
                        let b = Float(data[i+2]) - mean.z
                        // simple luminance-based variance accumulation
                        let lum = 0.2126*r + 0.7152*g + 0.0722*b
                        varAcc += lum * lum
                    }
                    variance = varAcc / Float(total) / (255.0 * 255.0)
                }
            }
            // Auto-toggle forward sign if too many black frames
            if stats.nonBlackRatio < 0.005 {
                zeroFramesStreak += 1
                if zeroFramesStreak == 20 { // ~ about every 100 log iterations
                    zSign = (zSign > 0) ? -1.0 : 1.0
                    log("Auto-toggled zSign to \(zSign > 0 ? "+Z" : "-Z") due to black frames")
                    zeroFramesStreak = 0
                }
            } else {
                zeroFramesStreak = 0
            }
            // Compute Gaussian color variance (linear space) as additional signal
            var gaussVar: Float = 0
            if !gaussians.isEmpty {
                var meanC = SIMD3<Float>(repeating: 0)
                for g in gaussians { meanC += g.color }
                meanC /= Float(gaussians.count)
                var accVar: Float = 0
                for g in gaussians {
                    let d = g.color - meanC
                    // luminance projection of difference
                    let lum = 0.2126*d.x + 0.7152*d.y + 0.0722*d.z
                    accVar += lum * lum
                }
                gaussVar = accVar / Float(gaussians.count)
            }
            // Helper: compute gaussian chroma variance & mean saturation
            func gaussianChromaStats() -> (meanSat: Float, chromaVar: Float) {
                if gaussians.isEmpty { return (0,0) }
                var mean = SIMD3<Float>(repeating: 0)
                for g in gaussians { mean += g.color }
                mean /= Float(gaussians.count)
                var varAcc: Float = 0
                var satAcc: Float = 0
                for g in gaussians {
                    let c = g.color
                    let mx = max(c.x, max(c.y, c.z))
                    let mn = min(c.x, min(c.y, c.z))
                    let sat = (mx > 1e-5) ? ( (mx - mn) / mx ) : 0 // value-based saturation
                    satAcc += sat
                    let d = c - mean
                    // luminance-weighted square diff for chroma variation
                    let lum = 0.2126*d.x + 0.7152*d.y + 0.0722*d.z
                    varAcc += lum * lum
                }
                return (satAcc / Float(gaussians.count), varAcc / Float(gaussians.count))
            }
            let chroma = gaussianChromaStats()
            // Bootstrap reseed very early (after some iterations so positions stabilize slightly)
            if !bootstrapReseedDone && iteration >= 10 {
                var candidateFrames: [Int] = [0]
                if dataset.samples.count > 2 { candidateFrames.append(dataset.samples.count/2) }
                if dataset.samples.count > 1 { candidateFrames.append(dataset.samples.count-1) }
                candidateFrames.append(sampleIndex)
                let frames = Array(Set(candidateFrames)).sorted()
                log("[ColorBootstrap] Performing initial multi-frame reseed frames=\(frames)")
                reseedColorsMulti(frameIndices: frames, blend: 0.0) // overwrite colors
                lastReseedIteration = iteration
                bootstrapReseedDone = true
                // Boost saturation slightly after bootstrap
                for i in 0..<gaussians.count {
                    var c = gaussians[i].color
                    let avg = (c.x + c.y + c.z) / 3
                    c = avg + (c - SIMD3<Float>(repeating: avg)) * 1.2
                    gaussians[i].color = simd_clamp(c, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
                }
                _syncGaussiansToGPU()
            } else {
                // Auto-reseed trigger: use both image variance and gaussian color variance with relaxed thresholds
                if iteration > 50 && (variance < 0.001 || gaussVar < 2e-4 || chroma.meanSat < 0.08) {
                    if lastReseedIteration == 0 || (iteration - lastReseedIteration) > 120 {
                        var candidateFrames: [Int] = [0]
                        if dataset.samples.count > 2 { candidateFrames.append(dataset.samples.count/2) }
                        if dataset.samples.count > 1 { candidateFrames.append(dataset.samples.count-1) }
                        candidateFrames.append(sampleIndex)
                        let frames = Array(Set(candidateFrames)).sorted()
                        log(String(format: "[ColorAuto] Low variance/chroma (img=%.5f, g=%.5f, sat=%.3f) -> multi reseed (%@)", variance, gaussVar, chroma.meanSat, frames.map{String($0)}.joined(separator: ",")))
                        reseedColorsMulti(frameIndices: frames, blend: 0.3)
                        lastReseedIteration = iteration
                        // Mild saturation and random jitter to escape gray collapse
                        var rng = SystemRandomNumberGenerator()
                        for i in 0..<gaussians.count {
                            var c = gaussians[i].color
                            let avg = (c.x + c.y + c.z)/3
                            let satScale: Float = 1.15
                            c = avg + (c - SIMD3<Float>(repeating: avg)) * satScale
                            // jitter
                            c += SIMD3<Float>(Float.random(in: -0.02...0.02, using: &rng), Float.random(in: -0.02...0.02, using: &rng), Float.random(in: -0.02...0.02, using: &rng))
                            gaussians[i].color = simd_clamp(c, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
                        }
                        _syncGaussiansToGPU()
                    }
                }
                // Periodic fallback every 50 iterations if still near-gray (gaussVar very low)
                if iteration % 50 == 0 && iteration > 0 && (gaussVar < 5e-5 || chroma.meanSat < 0.05) {
                    var candidateFrames: [Int] = [0]
                    if dataset.samples.count > 2 { candidateFrames.append(dataset.samples.count/2) }
                    if dataset.samples.count > 1 { candidateFrames.append(dataset.samples.count-1) }
                    candidateFrames.append(sampleIndex)
                    let frames = Array(Set(candidateFrames)).sorted()
                    log(String(format: "[ColorFallback] gaussVar=%.5f sat=%.3f forcing reseed (%@)", gaussVar, chroma.meanSat, frames.map{String($0)}.joined(separator: ",")))
                    reseedColorsMulti(frameIndices: frames, blend: 0.15)
                    // Light saturation boost only
                    for i in 0..<gaussians.count {
                        var c = gaussians[i].color
                        let avg = (c.x + c.y + c.z)/3
                        c = avg + (c - SIMD3<Float>(repeating: avg)) * 1.1
                        gaussians[i].color = simd_clamp(c, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
                    }
                    _syncGaussiansToGPU()
                    lastReseedIteration = iteration
                }
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.lossHistory.append(loss)
                if let png = UIImage(cgImage: predCG).pngData() {
                    self.previewImageData = png
                }
                // Overlay preview (optional)
                if self.overlayEnabled, let ov = self.makeOverlay(for: sampleIndex), let pngOv = UIImage(cgImage: ov).pngData() {
                    self.overlayImageData = pngOv
                }
            }
            let diag = reprojectionDiagnostics(for: sampleIndex)
            log(String(format: "step=%d loss=%.6f (ema=%.6f), meanRGB=(%.3f, %.3f, %.3f), nonBlack=%.1f%%, avgOpacity=%.3f, var=%.5f gVar=%.5f sat=%.3f | reproj: front=%d(%.1f%%), inBounds=%d(%.1f%%)",
                       iteration,
                       loss, lossEMA,
                       stats.meanRGB.x, stats.meanRGB.y, stats.meanRGB.z,
                       stats.nonBlackRatio * 100.0,
                       avgOpacity,
                       variance, gaussVar, chroma.meanSat,
                       diag.front, 100.0 * Float(diag.front) / Float(max(1, diag.total)),
                       diag.inBounds, 100.0 * Float(diag.inBounds) / Float(max(1, diag.total))))
        }

        if iteration % 200 == 0 { saveCheckpoint() }
    }

    private func computeImageStats(_ image: CGImage) -> (meanRGB: SIMD3<Float>, nonBlackRatio: Float) {
        let w = image.width, h = image.height
        let bytes = w * h * 4
        var data = [UInt8](repeating: 0, count: bytes)
        let cs = CGColorSpaceCreateDeviceRGB()
        let info: CGBitmapInfo = [
            .byteOrder32Big,
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        ]
        if let ctx = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: cs, bitmapInfo: info.rawValue) {
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
    var sum = SIMD3<Float>(repeating: 0)
        var nonBlack: Int = 0
        for i in stride(from: 0, to: bytes, by: 4) {
            let r = Float(data[i+0])
            let g = Float(data[i+1])
            let b = Float(data[i+2])
            sum += SIMD3<Float>(r, g, b)
            if data[i+0] | data[i+1] | data[i+2] != 0 { nonBlack += 1 }
        }
        let denom = max(1, w * h)
        let mean = sum / Float(denom) / 255.0
        let ratio = Float(nonBlack) / Float(denom)
        return (mean, ratio)
    }

    // Multi-frame color reseed (GT->gaussians)
    func reseedColorsMulti(frameIndices: [Int], blend: Float = 0.0) {
        let frames = frameIndices.filter { dataset.samples.indices.contains($0) }
        guard !frames.isEmpty else { return }
        let alpha = max(0, min(1, blend))
        // Accumulate linear colors
        var acc = Array(repeating: SIMD3<Float>(repeating: 0), count: gaussians.count)
        var hits = Array(repeating: 0, count: gaussians.count)
        @inline(__always) func srgbToLinear(_ c: Float) -> Float { return (c <= 0.04045) ? c/12.92 : powf((c+0.055)/1.055, 2.4) }
        for fi in frames {
            let s = dataset.samples[fi]
            guard let gtCG = s.image.cgImage else { continue }
            let w = gtCG.width, h = gtCG.height
            var buf = [UInt8](repeating: 0, count: w*h*4)
            let cs = CGColorSpaceCreateDeviceRGB()
            let info: CGBitmapInfo = [ .byteOrder32Big, CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue) ]
            if let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: cs, bitmapInfo: info.rawValue) {
                ctx.draw(gtCG, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
            let W2C = worldToCamMatrix(extr: s.extr)
            for i in 0..<gaussians.count {
                let g = gaussians[i]
                let cam4 = W2C * SIMD4<Float>(g.position, 1)
                let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
                let zf: Float = (zSign < 0) ? (-cam.z) : (cam.z)
                if zf <= 0 { continue }
                let u = s.intr.fx * (cam.x / zf) + s.intr.cx
                let v = s.intr.fy * (cam.y / zf) + s.intr.cy
                let xi = Int(round(u)), yi = Int(round(v))
                if xi < 0 || yi < 0 || xi >= w || yi >= h { continue }
                let pidx = (yi * w + xi) * 4
                let r = srgbToLinear(Float(buf[pidx+0]) / 255.0)
                let gcol = srgbToLinear(Float(buf[pidx+1]) / 255.0)
                let b = srgbToLinear(Float(buf[pidx+2]) / 255.0)
                acc[i] += SIMD3<Float>(r,gcol,b)
                hits[i] += 1
            }
        }
        for i in 0..<gaussians.count {
            guard hits[i] > 0 else { continue }
            var cur = gaussians[i].color
            let newC = acc[i] / Float(hits[i])
            cur = alpha * cur + (1 - alpha) * newC
            gaussians[i].color = simd_clamp(cur, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
        }
        _syncGaussiansToGPU()
        log("Reseeded colors from multi-frames count=\(frames.count), blend=\(alpha)")
    }
}
