//  Validation.swift
//  GSTainingApp
//
//  Finite-difference gradient validation and lightweight profiling utilities.
//
//  Provides:
//   - Finite difference check for color DC and position of a single gaussian on a single frame.
//   - Loss recomputation path using existing forward + fused GPU loss (reads residual texture indirectly).
//   - Simple timing helper.
//
//  Usage:
//    let v = TrainerGradientValidator(renderer: renderer)
//    v.validateColorAndPosition(trainer: self, gaussianIndex: 0, frameIndex: 0)
//
//  NOTE: This is an offline diagnostic; do not call every iteration.
//

import Foundation
import Metal
import simd
import UIKit

struct FiniteDiffResult {
    let analytic: Float
    let numeric: Float
    let relError: Float
}

final class TrainerGradientValidator {
    private let renderer: MetalRenderer
    private let device: MTLDevice
    private let epsilonColor: Float = 1e-3
    private let epsilonPos: Float = 1e-3
    private let maxGaussiansForCheck: Int = 1 // only perturb one gaussian

    init?(renderer: MetalRenderer) {
        self.renderer = renderer
        guard let dev = renderer.device as MTLDevice? else { return nil }
        self.device = dev
    }

    // Recompute loss (L2) for a current trainer state on a given frame.
    // We reuse trainer.forward pass + fused GPU loss machinery indirectly via a helper.
    private func renderAndComputeLoss(trainer: Trainer, frameIndex: Int) -> Float {
        // We intentionally access trainer internals via reflection hooks added if needed.
        // For now, we expose a minimal internal API by adding an extension below.
        return trainer._renderAndComputeLoss(frameIndex: frameIndex)
    }

    func validateColorAndPosition(trainer: Trainer, gaussianIndex gi: Int, frameIndex fi: Int) {
        guard gi < trainer.gaussianCountInternal else { print("[Validator] Invalid gaussian index"); return }
        print("[Validator] Finite-diff check gauss=\(gi) frame=\(fi)")
        // 1. Run one normal backward step to populate gradient buffers
        trainer._runSingleBackward(frameIndex: fi)
        guard let analytic = trainer._fetchAnalyticGradients(gaussianIndex: gi) else {
            print("[Validator] Could not fetch analytic grads")
            return
        }
        // 2. Numeric gradient for color DC (R component only for brevity)
        var numColorR: Float = 0
        if true {
            let baseLoss = renderAndComputeLoss(trainer: trainer, frameIndex: fi)
            trainer._perturbColorDC(gaussianIndex: gi, dr: epsilonColor)
            let plus = renderAndComputeLoss(trainer: trainer, frameIndex: fi)
            trainer._perturbColorDC(gaussianIndex: gi, dr: -epsilonColor) // restore
            numColorR = (plus - baseLoss) / epsilonColor
        }
        // 3. Numeric gradient for position.x
        var numPosX: Float = 0
        if true {
            let baseLoss = renderAndComputeLoss(trainer: trainer, frameIndex: fi)
            trainer._perturbPosition(gaussianIndex: gi, dx: epsilonPos)
            let plus = renderAndComputeLoss(trainer: trainer, frameIndex: fi)
            trainer._perturbPosition(gaussianIndex: gi, dx: -epsilonPos) // restore
            numPosX = (plus - baseLoss) / epsilonPos
        }
        // 4. Report
        func relErr(a: Float, n: Float) -> Float {
            let denom = max(1e-5, abs(a) + abs(n))
            return abs(a - n) / denom
        }
        print(String(format: "[Validator] dL/dColorR analytic=%.4e numeric=%.4e relErr=%.3g", analytic.colorR, numColorR, relErr(a: analytic.colorR, n: numColorR)))
        print(String(format: "[Validator] dL/dPosX   analytic=%.4e numeric=%.4e relErr=%.3g", analytic.posX, numPosX, relErr(a: analytic.posX, n: numPosX)))
    }
}

// MARK: - Internal accessors (fileprivate extension to Trainer)
fileprivate extension Trainer {
    var gaussianCountInternal: Int { gaussians.count }

    struct AnalyticSubsetGrads {
        let colorR: Float
        let posX: Float
    }

    func _runSingleBackward(frameIndex: Int) {
        guard frameIndex < dataset.samples.count else { return }
        let sample = dataset.samples[frameIndex]
        // Upload GT (sRGB -> linear RGBA16F) using existing helper
        renderer.uploadGTImage(sample.image, existing: &gtTexture, width: dataset.width, height: dataset.height)
        guard let gtTex = gtTexture else { return }
        // Build worldToCamera from extrinsics (camera-to-world) inverse
        let camToWorld = sample.extr.matrix4x4
        let worldToCam = camToWorld.inverseSafe
        // Forward render
        renderer.forward(gaussiansBuffer: gaussianBuffer, gaussianCount: gaussians.count, worldToCam: worldToCam, intrinsics: sample.intr, target: targetTexture, zSign: zSign, enableSHL2: enableSHL2)
        // Fused exposure + residual + loss
        _ = renderer.fusedExposureResidualLoss(pred: targetTexture, gt: gtTex, residual: residualTexture)
        // Backward
        if let grads = gradBuffers {
            renderer.backward(gaussiansBuffer: gaussianBuffer, gaussianCount: gaussians.count, worldToCam: worldToCam, intrinsics: sample.intr, residual: residualTexture, grads: grads, zSign: zSign, enableSHL2: enableSHL2)
        }
    }

    func _fetchAnalyticGradients(gaussianIndex: Int) -> AnalyticSubsetGrads? {
        guard let grads = gradBuffers else { return nil }
        // grad buffers are Int32 fixed-point scaled by 1024 per Metal kernel comment.
        let scale: Float = 1.0 / 1024.0
        let ptrR = grads.c0R.contents().bindMemory(to: Int32.self, capacity: gaussians.count)
        let ptrPosX = grads.posX.contents().bindMemory(to: Int32.self, capacity: gaussians.count)
        let gColorR = Float(ptrR[gaussianIndex]) * scale
        let gPosX = Float(ptrPosX[gaussianIndex]) * scale
        return AnalyticSubsetGrads(colorR: gColorR, posX: gPosX)
    }

    func _perturbColorDC(gaussianIndex: Int, dr: Float) {
        gaussians[gaussianIndex].color.x += dr
        gaussians[gaussianIndex].color.x = min(max(gaussians[gaussianIndex].color.x, 0), 1)
        // Sync only that gaussian to GPU
        var g = GaussianParamGPU(gaussians[gaussianIndex])
        memcpy(gaussianBuffer.contents().advanced(by: gaussianIndex * MemoryLayout<GaussianParamGPU>.stride), &g, MemoryLayout<GaussianParamGPU>.stride)
    }

    func _perturbPosition(gaussianIndex: Int, dx: Float) {
        gaussians[gaussianIndex].position.x += dx
        var g = GaussianParamGPU(gaussians[gaussianIndex])
        memcpy(gaussianBuffer.contents().advanced(by: gaussianIndex * MemoryLayout<GaussianParamGPU>.stride), &g, MemoryLayout<GaussianParamGPU>.stride)
    }

    func _renderAndComputeLoss(frameIndex: Int) -> Float {
        guard frameIndex < dataset.samples.count else { return 0 }
        let sample = dataset.samples[frameIndex]
        renderer.uploadGTImage(sample.image, existing: &gtTexture, width: dataset.width, height: dataset.height)
        guard let gtTex = gtTexture else { return 0 }
        let camToWorld = sample.extr.matrix4x4
        let worldToCam = camToWorld.inverseSafe
        renderer.forward(gaussiansBuffer: gaussianBuffer, gaussianCount: gaussians.count, worldToCam: worldToCam, intrinsics: sample.intr, target: targetTexture, zSign: zSign, enableSHL2: enableSHL2)
        if let fused = renderer.fusedExposureResidualLoss(pred: targetTexture, gt: gtTex, residual: residualTexture) {
            return fused.loss
        }
        return 0
    }

    // Convert 16-bit float payload to 32-bit (very small helper; could be replaced by simd half if available)
    func halfToFloat(_ h: UInt16) -> Float {
        // IEEE 754 half to float (approx fast path). For validation only; precision adequate.
        let s = (h & 0x8000) >> 15
    var e = Int((h & 0x7C00) >> 10)
    let f = Int(h & 0x03FF)
        if e == 0 {
            if f == 0 { return s == 1 ? -0.0 : 0.0 }
            // subnormal
            let exp: Float = powf(2, -14)
            let mant = Float(f) / 1024.0
            let val = exp * mant
            return s == 1 ? -val : val
        } else if e == 31 {
            return s == 1 ? -Float.infinity : Float.infinity
        }
        e = e - 15 + 127
        let bits = (UInt32(s) << 31) | (UInt32(e) << 23) | (UInt32(f) << 13)
        return Float(bitPattern: bits)
    }
}
