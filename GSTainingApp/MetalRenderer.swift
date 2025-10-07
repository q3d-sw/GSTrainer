//
//  MetalRenderer.swift
//  GSTainingApp
//
//  Baseline Metal compute for Gaussian rendering into RGBA16Float texture.
//

import Foundation
import Metal
import MetalKit
import simd
import UIKit // for UIImage in uploadGTImage

struct RenderUniforms {
    var worldToCam: simd_float4x4
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float
    var zSign: Float
    var imageWidth: UInt32
    var imageHeight: UInt32
    var gaussianCount: UInt32
    var pointScale: Float
    var tilesX: UInt32
    var tilesY: UInt32
    var tileSize: UInt32
    var useCache: UInt32
}

struct GradUniforms {
    var worldToCam: simd_float4x4
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float
    var zSign: Float
    var imageWidth: UInt32
    var imageHeight: UInt32
    var gaussianCount: UInt32
    var tilesX: UInt32
    var tilesY: UInt32
    var tileSize: UInt32
    var useCache: UInt32
    var lossMode: UInt32
    var charbEps: Float
    var _pad0: Float
    var _pad1: Float
}

struct GaussianScreenCache {
    var uv: SIMD2<Float>
    var invz: Float
    var zf: Float
    var cam4: SIMD4<Float>
    var j00: Float; var j01: Float; var j02: Float; var pad0: Float
    var j10: Float; var j11: Float; var j12: Float; var pad1: Float
    var sxx: Float; var sxy: Float; var syy: Float; var pad2: Float
    var a00: Float; var a01: Float; var a11: Float; var pad3: Float
    var viewDir4: SIMD4<Float>
}

// Full CPU mirror of Metal GaussianParamGPU (keep field order & alignment identical!)
// Metal side:
// struct GaussianParamGPU {
//   float3 position; float _pad0;
//   float3 scale;    float opacity;
//   float4 shR; float4 shG; float4 shB;
//   float4 shR_ex1; float4 shR_ex2; float4 shG_ex1; float4 shG_ex2; float4 shB_ex1; float4 shB_ex2;
// };
// NOTE: If you add fields in Metal, update this mirror accordingly.
struct GaussianParamCPU {
    var position: SIMD3<Float>; var _pad0: Float
    var scale: SIMD3<Float>;    var opacity: Float
    var shR: SIMD4<Float>
    var shG: SIMD4<Float>
    var shB: SIMD4<Float>
    var shR_ex1: SIMD4<Float>
    var shR_ex2: SIMD4<Float>
    var shG_ex1: SIMD4<Float>
    var shG_ex2: SIMD4<Float>
    var shB_ex1: SIMD4<Float>
    var shB_ex2: SIMD4<Float>
}

// Swift host mirror of Metal struct ResidualUniforms (float gain; uint width; uint height)
// Alignment: 4 + 4 + 4 = 12 bytes; Metal side packs identically for these scalars.
// Keep field order identical to Metal definition.
struct ResidualUniforms {
    var gain: Float
    var width: UInt32
    var height: UInt32
    var lossMode: UInt32
    var charbEps: Float
}

final class MetalRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let backwardPipeline: MTLComputePipelineState
    private let tiledPipeline: MTLComputePipelineState?
    private let tiledBackwardPipeline: MTLComputePipelineState?
    // Fused loss kernels
    private let exposureSumsPipeline: MTLComputePipelineState?
    private let residualLossPipeline: MTLComputePipelineState?
    // Residual heatmap
    private let heatmapDownsamplePipeline: MTLComputePipelineState?
    // Простая инструментализация (мс) последних запусков
    public private(set) var lastForwardMS: Double = 0
    public private(set) var lastBackwardMS: Double = 0
    public private(set) var lastTileBuildMS: Double = 0
    public private(set) var lastTilesBucketed: Int = 0 // number of tiles where bucket depth ordering used
    public private(set) var lastTilesSorted: Int = 0   // number of tiles where full sort used

    // Cache for tile buffers to reuse when camera/intrinsics unchanged
    private struct TileCacheKey: Hashable {
        let w2c: simd_float4x4
        let fx: Float; let fy: Float; let cx: Float; let cy: Float
        let w: Int; let h: Int; let count: Int; let tileSize: Int; let zSign: Float
        func hash(into hasher: inout Hasher) {
            // Hash a few representative matrix elements for speed
            hasher.combine(w2c.columns.0.x); hasher.combine(w2c.columns.0.y)
            hasher.combine(w2c.columns.1.x); hasher.combine(w2c.columns.1.y)
            hasher.combine(w2c.columns.2.z); hasher.combine(w2c.columns.3.x)
            hasher.combine(fx); hasher.combine(fy); hasher.combine(cx); hasher.combine(cy)
            hasher.combine(w); hasher.combine(h); hasher.combine(count); hasher.combine(tileSize); hasher.combine(zSign)
        }
        static func ==(l: TileCacheKey, r: TileCacheKey) -> Bool {
                return l.w2c.columns.0 == r.w2c.columns.0 &&
                    l.w2c.columns.1 == r.w2c.columns.1 &&
                    l.w2c.columns.2 == r.w2c.columns.2 &&
                    l.w2c.columns.3 == r.w2c.columns.3 &&
                    l.fx == r.fx && l.fy == r.fy && l.cx == r.cx && l.cy == r.cy &&
                    l.w == r.w && l.h == r.h && l.count == r.count && l.tileSize == r.tileSize && l.zSign == r.zSign
        }
    }
    private var tileCache: [TileCacheKey: (offsets: MTLBuffer, list: MTLBuffer, tilesX: Int, tilesY: Int)] = [:]
    private var lastCameraChangeFrame: Int = 0
    // Dummy buffer to satisfy binding when forward cache disabled (index 5 expected by shader)
    private var dummyOneFloatBuffer: MTLBuffer? = nil
    var enableTileReuse: Bool = true
    var enableFrustumCull: Bool = true
    private var useTiledKernel: Bool = false
    var useTiledBackward: Bool = true // experimental shared-memory gradient reduction
    var enableDepthBucketOrdering: Bool = true // enable hybrid bucket+sort depth ordering per tile
    var depthBucketBinCount: Int = 64 // number of discrete depth buckets (power of two preferred)
    // Robust loss
    var enableCharbonnierLoss: Bool = false
    var charbEps: Float = 1e-3

    func depthOrderingStats() -> (bucketed: Int, sorted: Int) {
        return (lastTilesBucketed, lastTilesSorted)
    }

    init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue

        // Load default library (includes MetalRenderer.metal)
        guard let lib = device.makeDefaultLibrary(),
              let fn = lib.makeFunction(name: "renderGaussians"),
              let bwd = lib.makeFunction(name: "accumulateGradients") else {
            return nil
        }
        do {
            self.pipeline = try device.makeComputePipelineState(function: fn)
            self.backwardPipeline = try device.makeComputePipelineState(function: bwd)
            if let tiledFn = lib.makeFunction(name: "tiledRenderGaussians") {
                self.tiledPipeline = try? device.makeComputePipelineState(function: tiledFn)
            } else { self.tiledPipeline = nil }
            if let tiledBwdFn = lib.makeFunction(name: "tiledAccumulateGradients") {
                self.tiledBackwardPipeline = try? device.makeComputePipelineState(function: tiledBwdFn)
            } else { self.tiledBackwardPipeline = nil }
            if let expFn = lib.makeFunction(name: "computeExposureSums") {
                self.exposureSumsPipeline = try? device.makeComputePipelineState(function: expFn)
            } else { self.exposureSumsPipeline = nil }
            if let resFn = lib.makeFunction(name: "computeResidualAndLoss") {
                self.residualLossPipeline = try? device.makeComputePipelineState(function: resFn)
            } else { self.residualLossPipeline = nil }
            if let hmFn = lib.makeFunction(name: "downsampleResidualHeatmap") {
                self.heatmapDownsamplePipeline = try? device.makeComputePipelineState(function: hmFn)
            } else { self.heatmapDownsamplePipeline = nil }
        } catch {
            print("Failed to create pipeline: \(error)")
            return nil
        }
    }

    /// Upload a UIImage (sRGB) into an RGBA16F texture (linear) allocating if needed
    func uploadGTImage(_ image: UIImage, existing: inout MTLTexture?, width: Int, height: Int) {
        if existing == nil || existing!.width != width || existing!.height != height {
            existing = makeTexture(width: width, height: height)
        }
        guard let tex = existing, let cg = image.cgImage else { return }
        let w = cg.width, h = cg.height
        var buf = [UInt8](repeating: 0, count: w*h*4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let info: CGBitmapInfo = [.byteOrder32Big, CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)]
        if let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: cs, bitmapInfo: info.rawValue) {
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        @inline(__always) func srgbToLinear(_ c: Float) -> Float { return (c <= 0.04045) ? c/12.92 : powf((c+0.055)/1.055, 2.4) }
        let rowBytes = w * MemoryLayout<UInt16>.stride * 4
        let raw = UnsafeMutableRawPointer.allocate(byteCount: rowBytes * h, alignment: 0x100)
        defer { raw.deallocate() }
        let dst = raw.bindMemory(to: UInt16.self, capacity: w*h*4)
        func floatToHalf(_ x: Float) -> UInt16 {
            var f = x
            let sign: UInt16 = (f < 0) ? 0x8000 : 0
            f = max(0.0, min(65504.0, f))
            var e: Int32 = 0
            let m = frexpf(f == 0 ? 0 : f, &e)
            if f == 0 { return 0 }
            let he = UInt16(max(0, min(31, Int(e) + 14)))
            let hm = UInt16(max(0, min(1023, Int((m * 2 - 1) * 1024))))
            return sign | (he << 10) | hm
        }
        for i in 0..<(w*h) {
            let r = srgbToLinear(Float(buf[i*4+0]) / 255.0)
            let g = srgbToLinear(Float(buf[i*4+1]) / 255.0)
            let b = srgbToLinear(Float(buf[i*4+2]) / 255.0)
            dst[i*4+0] = floatToHalf(r)
            dst[i*4+1] = floatToHalf(g)
            dst[i*4+2] = floatToHalf(b)
            dst[i*4+3] = 0
        }
        let region = MTLRegionMake2D(0,0,w,h)
        tex.replace(region: region, mipmapLevel: 0, withBytes: raw, bytesPerRow: rowBytes)
    }

    // Build / reuse a heatmap texture (RGBA16F) downsampled by 'factor'. Returns texture or nil.
    func buildResidualHeatmap(residual: MTLTexture, existing: inout MTLTexture?, factor: Int = 4) -> MTLTexture? {
        guard factor > 1, let pipe = heatmapDownsamplePipeline else { return nil }
        let outW = max(1, residual.width / factor)
        let outH = max(1, residual.height / factor)
        if existing == nil || existing!.width != outW || existing!.height != outH {
            existing = makeTexture(width: outW, height: outH)
        }
        guard let heat = existing, let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return nil }
        struct HMUniforms { var inWidth, inHeight, outWidth, outHeight, block, lossMode: UInt32; var charbEps: Float }
        var u = HMUniforms(inWidth: UInt32(residual.width), inHeight: UInt32(residual.height), outWidth: UInt32(outW), outHeight: UInt32(outH), block: UInt32(factor), lossMode: enableCharbonnierLoss ? 1 : 0, charbEps: charbEps)
        enc.setComputePipelineState(pipe)
        enc.setTexture(residual, index: 0)
        enc.setTexture(heat, index: 1)
        enc.setBytes(&u, length: MemoryLayout<HMUniforms>.stride, index: 0)
        let w = pipe.threadExecutionWidth
        let h = max(1, pipe.maxTotalThreadsPerThreadgroup / w)
        let grid = MTLSize(width: outW, height: outH, depth: 1)
        let tg = MTLSize(width: w, height: h, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        return heat
    }

    struct FusedLossOutput { let gain: Float; let loss: Float }

    // Run exposure gain computation and residual+loss on GPU.
    // If overrideGain is provided, skip auto-computed gain (still compute sums for potential diagnostics but ignore result).
    func fusedExposureResidualLoss(pred: MTLTexture, gt: MTLTexture, residual: MTLTexture, overrideGain: Float? = nil) -> FusedLossOutput? {
        guard let expPipe = exposureSumsPipeline, let resPipe = residualLossPipeline else { return nil }
        let w = pred.width, h = pred.height
        // Buffers for exposure sums (num, den) and loss
        guard let sumsBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 2, options: .storageModeShared),
              let lossBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else { return nil }
        memset(sumsBuf.contents(), 0, sumsBuf.length)
        memset(lossBuf.contents(), 0, lossBuf.length)
        // Pass 1
        guard let cmd1 = commandQueue.makeCommandBuffer(), let enc1 = cmd1.makeComputeCommandEncoder() else { return nil }
        enc1.setComputePipelineState(expPipe)
        enc1.setTexture(pred, index: 0)
        enc1.setTexture(gt, index: 1)
        enc1.setBuffer(sumsBuf, offset: 0, index: 0)
        let tgW = expPipe.threadExecutionWidth
        let tgH = max(1, expPipe.maxTotalThreadsPerThreadgroup / tgW)
        let grid = MTLSize(width: w, height: h, depth: 1)
        let tg = MTLSize(width: tgW, height: tgH, depth: 1)
        enc1.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc1.endEncoding(); cmd1.commit(); cmd1.waitUntilCompleted()
        // Read sums and compute gain
        let sums = sumsBuf.contents().bindMemory(to: UInt32.self, capacity: 2)
        let kExposureScale: Float = 4096.0
        let num = Float(sums[0]) / kExposureScale
        let den = Float(sums[1]) / kExposureScale
    var gain: Float = (den > 1e-8) ? (num / den) : 1.0
    gain = max(0.5, min(2.0, gain))
    if let og = overrideGain { gain = og }
        // Pass 2 residual+loss
        guard let cmd2 = commandQueue.makeCommandBuffer(), let enc2 = cmd2.makeComputeCommandEncoder() else { return nil }
        enc2.setComputePipelineState(resPipe)
        enc2.setTexture(pred, index: 0)
        enc2.setTexture(gt, index: 1)
        enc2.setTexture(residual, index: 2)
    var uni = ResidualUniforms(gain: gain, width: UInt32(w), height: UInt32(h), lossMode: enableCharbonnierLoss ? 1 : 0, charbEps: charbEps)
        enc2.setBytes(&uni, length: MemoryLayout<ResidualUniforms>.stride, index: 0)
        enc2.setBuffer(lossBuf, offset: 0, index: 1)
        let tgW2 = resPipe.threadExecutionWidth
        let tgH2 = max(1, resPipe.maxTotalThreadsPerThreadgroup / tgW2)
        let tg2 = MTLSize(width: tgW2, height: tgH2, depth: 1)
        enc2.dispatchThreads(grid, threadsPerThreadgroup: tg2)
        enc2.endEncoding(); cmd2.commit(); cmd2.waitUntilCompleted()
        let lossPtr = lossBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
        let kLossScale: Float = 1024.0
        let sumLoss = Float(lossPtr[0]) / kLossScale
        let meanLoss = sumLoss / Float(w * h)
        return FusedLossOutput(gain: gain, loss: meanLoss)
    }

    // Build screen-space tiling (CPU) for current camera and gaussians
    private func buildTiles(gaussianBuffer: MTLBuffer, count: Int, worldToCam: simd_float4x4, intrinsics: CameraIntrinsics, width: Int, height: Int, zSign: Float, tileSize: Int) -> (offsets: MTLBuffer, list: MTLBuffer, tilesX: Int, tilesY: Int)? {
        let t0 = CFAbsoluteTimeGetCurrent()
        let key = TileCacheKey(w2c: worldToCam, fx: intrinsics.fx, fy: intrinsics.fy, cx: intrinsics.cx, cy: intrinsics.cy, w: width, h: height, count: count, tileSize: tileSize, zSign: zSign)
        if enableTileReuse, let cached = tileCache[key] { return cached }

        // First compute frustum planes if enabled (simple clip in camera space by z>0, x,y within +- z * margin)
        let tilesX = max(1, (width + tileSize - 1) / tileSize)
        let tilesY = max(1, (height + tileSize - 1) / tileSize)
        let tileCount = tilesX * tilesY
        var counts = [Int](repeating: 0, count: tileCount)
        let ptr = gaussianBuffer.contents().bindMemory(to: GaussianParamGPU.self, capacity: count)
        @inline(__always) func project(_ p: SIMD3<Float>) -> SIMD3<Float> {
            let v = SIMD4<Float>(p.x, p.y, p.z, 1)
            let cam = worldToCam * v
            return SIMD3<Float>(cam.x, cam.y, cam.z)
        }
        // Precompute projection data once; reuse for fill and sorting
        struct ProjInfo { var u: Float; var v: Float; var zf: Float; var r: Float; var valid: Bool }
        var proj = [ProjInfo](repeating: ProjInfo(u: 0, v: 0, zf: 0, r: 0, valid: false), count: count)
        for i in 0..<count {
            let g = ptr[i]
            let cam = project(g.position)
            let zf: Float = (zSign < 0) ? -cam.z : cam.z
            if zf <= 1e-5 { continue }
            let u: Float = intrinsics.fx * (cam.x / zf) + intrinsics.cx
            let v: Float = intrinsics.fy * (cam.y / zf) + intrinsics.cy
            if enableFrustumCull && (u < -Float(tileSize) || u > Float(width + tileSize) || v < -Float(tileSize) || v > Float(height + tileSize)) { continue }
            let sx = max(1e-5, g.scale.x)
            let sy = max(1e-5, g.scale.y)
            let rX = 3.0 * abs(intrinsics.fx * sx / zf)
            let rY = 3.0 * abs(intrinsics.fy * sy / zf)
            let r = max(rX, rY)
            proj[i] = ProjInfo(u: u, v: v, zf: zf, r: r, valid: true)
            let minTX = Int(floor((u - r) / Float(tileSize)))
            let maxTX = Int(floor((u + r) / Float(tileSize)))
            let minTY = Int(floor((v - r) / Float(tileSize)))
            let maxTY = Int(floor((v + r) / Float(tileSize)))
            let a = max(0, minTX)
            let b = min(tilesX - 1, maxTX)
            let c = max(0, minTY)
            let d = min(tilesY - 1, maxTY)
            if a > b || c > d { continue }
            for ty in c...d {
                let row = ty * tilesX
                for tx in a...b { counts[row + tx] += 1 }
            }
        }
        var offsets = [UInt32](repeating: 0, count: tileCount + 1)
        var total = 0
        for i in 0..<tileCount { total += counts[i]; offsets[i+1] = UInt32(total) }
        var list = [UInt32](repeating: 0, count: total)
        var cursor = [Int](repeating: 0, count: tileCount)
        for i in 0..<tileCount { cursor[i] = Int(offsets[i]) }
        for i in 0..<count where proj[i].valid {
            let p = proj[i]
            let minTX = Int(floor((p.u - p.r) / Float(tileSize)))
            let maxTX = Int(floor((p.u + p.r) / Float(tileSize)))
            let minTY = Int(floor((p.v - p.r) / Float(tileSize)))
            let maxTY = Int(floor((p.v + p.r) / Float(tileSize)))
            let a = max(0, minTX)
            let b = min(tilesX - 1, maxTX)
            let c = max(0, minTY)
            let d = min(tilesY - 1, maxTY)
            if a > b || c > d { continue }
            for ty in c...d {
                let row = ty * tilesX
                for tx in a...b {
                    let tid = row + tx
                    let dst = cursor[tid]
                    if dst < total { list[dst] = UInt32(i); cursor[tid] = dst + 1 }
                }
            }
        }
        // Depth ordering per tile. Hybrid strategy:
        // 1. For small tiles or when disabled: full sort (near->far)
        // 2. For large tiles and if enabledDepthBucketOrdering: bucket + local insertion preserving near->far approx
        lastTilesBucketed = 0; lastTilesSorted = 0
        if enableDepthBucketOrdering {
            // Compute min/max depth of visible gaussians once to normalize bucket indices
            var minZ: Float = Float.greatestFiniteMagnitude
            var maxZ: Float = -Float.greatestFiniteMagnitude
            for i in 0..<count where proj[i].valid {
                let z = proj[i].zf
                if z < minZ { minZ = z }
                if z > maxZ { maxZ = z }
            }
            if minZ == Float.greatestFiniteMagnitude { minZ = 0; maxZ = 1 }
            let span = max(1e-6, maxZ - minZ)
            let bins = max(4, depthBucketBinCount)
            let invSpan = 1.0 / span
            // Temporary buffer reused per tile (allocate max once)
            var bucketStorage = [UInt32]()
            bucketStorage.reserveCapacity(256)
            // Iterate tiles
            for tileId in 0..<tileCount {
                let start = Int(offsets[tileId])
                let end = Int(offsets[tileId + 1])
                let n = end - start
                if n <= 1 { continue }
                // Heuristic: use bucketed approach if tile has many gaussians
                if n >= 32 { // threshold tunable
                    // Clear per-tile buckets
                    var bucketCounts = [Int](repeating: 0, count: bins)
                    // First pass: count
                    for k in start..<end {
                        let gi = Int(list[k])
                        let z = proj[gi].zf
                        var b = Int((z - minZ) * invSpan * Float(bins))
                        if b < 0 { b = 0 } else if b >= bins { b = bins - 1 }
                        bucketCounts[b] += 1
                    }
                    // Exclusive prefix
                    var prefix = [Int](repeating: 0, count: bins + 1)
                    for b in 0..<bins { prefix[b+1] = prefix[b] + bucketCounts[b] }
                    let totalB = prefix[bins]
                    if bucketStorage.count < totalB { bucketStorage = [UInt32](repeating: 0, count: totalB) }
                    // Second pass: fill
                    var cursorB = prefix
                    for k in start..<end {
                        let gi = Int(list[k])
                        let z = proj[gi].zf
                        var b = Int((z - minZ) * invSpan * Float(bins))
                        if b < 0 { b = 0 } else if b >= bins { b = bins - 1 }
                        let dst = cursorB[b]
                        if dst < totalB { bucketStorage[dst] = list[k]; cursorB[b] = dst + 1 }
                    }
                    // Optional: within each bucket keep approximate ordering by insertion (already sequential). We now copy back
                    for k in 0..<totalB { list[start + k] = bucketStorage[k] }
                    lastTilesBucketed += 1
                } else {
                    list[start..<end].sort { (ia, ib) -> Bool in
                        let da = proj[Int(ia)].zf
                        let db = proj[Int(ib)].zf
                        return da < db
                    }
                    lastTilesSorted += 1
                }
            }
        } else {
            for tileId in 0..<tileCount {
                let start = Int(offsets[tileId])
                let end = Int(offsets[tileId + 1])
                if end - start <= 1 { continue }
                list[start..<end].sort { (ia, ib) -> Bool in
                    let da = proj[Int(ia)].zf
                    let db = proj[Int(ib)].zf
                    return da < db
                }
                lastTilesSorted += 1
            }
        }
        guard let offsetsBuf = device.makeBuffer(bytes: offsets, length: MemoryLayout<UInt32>.stride * offsets.count, options: .storageModeShared),
              let listBuf = device.makeBuffer(bytes: list, length: MemoryLayout<UInt32>.stride * max(1, list.count), options: .storageModeShared) else { return nil }
        let built = (offsetsBuf, listBuf, tilesX, tilesY)
        if enableTileReuse { tileCache[key] = built }
        lastTileBuildMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        if debugLogging {
            // Diagnostics: how many valid projections, total indices, fraction in frustum
            var validProj = 0
            var totalIndices = list.count
            for i in 0..<count { if proj[i].valid { validProj += 1 } }
            print("[TileDiag] size=\(width)x\(height) tiles=\(tilesX)x\(tilesY) gauss=\(count) valid=\(validProj) (\(String(format: "%.1f", 100.0 * (validProj == 0 ? 0 : Float(validProj)/Float(count))))%) indices=\(totalIndices)")
        }
        return built
    }

    func makeTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2D
        desc.pixelFormat = .rgba16Float
        desc.width = width
        desc.height = height
        desc.usage = [.shaderWrite, .shaderRead]
    // Use .shared to allow CPU readback via getBytes
        desc.storageMode = .shared
        return device.makeTexture(descriptor: desc)
    }

    func makeBuffer<T>(array: [T], options: MTLResourceOptions = [.storageModeShared]) -> MTLBuffer? {
        // Metal disallows zero-length buffers; allocate a minimal 1-stride padding when array empty
        let count = array.count
        let stride = MemoryLayout<T>.stride
        let length = stride * max(count, 1)
        guard let buf = device.makeBuffer(length: length, options: options) else {
            if debugLogging { print("[MetalRenderer] Failed to allocate buffer length=\(length) (array type \(T.self))") }
            return nil
        }
        if count > 0 {
            array.withUnsafeBytes { rawBuf in
                if let base = rawBuf.baseAddress {
                    memcpy(buf.contents(), base, stride * count)
                }
            }
        } else if debugLogging {
            print("[MetalRenderer] Allocated padded buffer (1 element) for empty array of type \(T.self)")
        }
        return buf
    }

    @inline(__always)
    private func safeMakeBuffer(length: Int, options: MTLResourceOptions = .storageModeShared, tag: String) -> MTLBuffer? {
        let safeLen = max(length, 1)
        if length == 0 && debugLogging {
            print("[MetalRenderer] Padding zero-length allocation for tag=\(tag)")
        }
        return device.makeBuffer(length: safeLen, options: options)
    }

    func setUseTiledKernel(_ v: Bool) { useTiledKernel = v }

    struct SHModeUniform { var enableL2: UInt32 }

    var enableForwardCache: Bool = true // host flag (will expose via Trainer later)
    private var forwardCacheBuffer: MTLBuffer? = nil
    private var forwardCacheGaussiansVersion: Int = -1
    private var forwardCacheCameraKey: UInt64 = 0
    private var gaussiansVersionCounter: Int = 0 // increment when gaussians updated externally
    var lastForwardCacheBuildMS: Double = 0.0
    // Forward cache reuse instrumentation
    private var forwardCacheHits: Int = 0
    private var forwardCacheMisses: Int = 0
    var debugLogging: Bool = true

    func forwardCacheStats() -> (hits: Int, misses: Int, hitRate: Double) {
        let total = forwardCacheHits + forwardCacheMisses
        let rate = total > 0 ? Double(forwardCacheHits) / Double(total) : 0.0
        return (forwardCacheHits, forwardCacheMisses, rate)
    }

    // Call when gaussian parameters (positions/scales) changed so we rebuild cache next forward
    func markGaussiansDirty() { gaussiansVersionCounter &+= 1 }

    private func makeCameraKey(worldToCam: simd_float4x4, intr: CameraIntrinsics, zSign: Float) -> UInt64 {
        // Hash a subset of matrix/intrinsics for quick reuse detection
        var hasher = Hasher()
        hasher.combine(worldToCam.columns.0.x)
        hasher.combine(worldToCam.columns.0.y)
        hasher.combine(worldToCam.columns.1.x)
        hasher.combine(worldToCam.columns.1.y)
        hasher.combine(worldToCam.columns.2.z)
        hasher.combine(worldToCam.columns.3.x)
        hasher.combine(intr.fx); hasher.combine(intr.fy); hasher.combine(intr.cx); hasher.combine(intr.cy)
        hasher.combine(zSign)
        return UInt64(bitPattern: Int64(hasher.finalize()))
    }

    func forward(gaussiansBuffer: MTLBuffer, gaussianCount: Int, worldToCam: simd_float4x4, intrinsics: CameraIntrinsics, target: MTLTexture, pointScale: Float = 800.0, zSign: Float = -1.0, enableSHL2: Bool = false) {
        let tStart = CFAbsoluteTimeGetCurrent()
        var intrinsics = intrinsics
        // Heuristic auto-fix: if cx,cy far from center (>25% of dimension) recenter & proportionally scale fx,fy if they look like they were defined for a tiny (e.g. 16x16) space.
        let cxExpected = Float(target.width) * 0.5
        let cyExpected = Float(target.height) * 0.5
        let dx = abs(intrinsics.cx - cxExpected)
        let dy = abs(intrinsics.cy - cyExpected)
        if dx > Float(target.width) * 0.25 || dy > Float(target.height) * 0.25 {
            // Detect scale mismatch: assume original (bad) intrinsics referenced a nominal base size near their (cx,cy) magnitude *2
            let approxBaseW = max(16.0, intrinsics.cx * 2.0)
            let approxBaseH = max(16.0, intrinsics.cy * 2.0)
            let sx = Float(target.width) / approxBaseW
            let sy = Float(target.height) / approxBaseH
            // If fx,fy are small compared to target width/height, scale them.
            if (intrinsics.fx < Float(target.width) * 0.1) || (intrinsics.fy < Float(target.height) * 0.1) {
                intrinsics.fx *= sx
                intrinsics.fy *= sy
            }
            if debugLogging {
                print(String(format: "[IntrinsicsFix] Recenter (cx,cy): (%.2f, %.2f)->(%.2f, %.2f) scale fx,fy->(%.2f, %.2f)",
                             intrinsics.cx, intrinsics.cy, cxExpected, cyExpected, intrinsics.fx, intrinsics.fy))
            }
            intrinsics.cx = cxExpected; intrinsics.cy = cyExpected
        }
        // Use legacy per-pixel kernel (renderGaussians) for now; tiled kernel placeholder hook
        guard let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return }
        if useTiledKernel, let tiled = tiledPipeline {
            // Tiled dispatch: threadgroups = tilesX x tilesY, threadsPerThreadgroup ~ tileSize x tileSize
            enc.setComputePipelineState(tiled)
            var uniforms = RenderUniforms(worldToCam: worldToCam, fx: intrinsics.fx, fy: intrinsics.fy, cx: intrinsics.cx, cy: intrinsics.cy, zSign: zSign, imageWidth: UInt32(target.width), imageHeight: UInt32(target.height), gaussianCount: UInt32(gaussianCount), pointScale: pointScale, tilesX: 1, tilesY: 1, tileSize: 0, useCache: enableForwardCache ? 1 : 0)
            // We still need tiling meta from CPU
            let tileSz = 16
            let tiles = buildTiles(gaussianBuffer: gaussiansBuffer, count: gaussianCount, worldToCam: worldToCam, intrinsics: intrinsics, width: target.width, height: target.height, zSign: zSign, tileSize: tileSz)
            if let t = tiles {
                uniforms.tilesX = UInt32(t.tilesX); uniforms.tilesY = UInt32(t.tilesY); uniforms.tileSize = UInt32(tileSz)
                enc.setBuffer(gaussiansBuffer, offset: 0, index: 0)
                enc.setBytes(&uniforms, length: MemoryLayout<RenderUniforms>.stride, index: 1)
                enc.setBuffer(t.offsets, offset: 0, index: 2)
                enc.setBuffer(t.list, offset: 0, index: 3)
                var shMode = SHModeUniform(enableL2: enableSHL2 ? 1 : 0)
                // Build or reuse screen cache (always rebuild for now; later add reuse detection)
                var localCacheBuf: MTLBuffer? = nil
                if enableForwardCache {
                    // Attempt reuse; only build if changed
                    let camKey = makeCameraKey(worldToCam: worldToCam, intr: intrinsics, zSign: zSign)
                    if let existing = forwardCacheBuffer, camKey == forwardCacheCameraKey, forwardCacheGaussiansVersion == gaussiansVersionCounter, existing.length >= gaussianCount * MemoryLayout<GaussianScreenCache>.stride {
                        forwardCacheHits &+= 1
                        localCacheBuf = existing
                    } else {
                        forwardCacheMisses &+= 1
                        localCacheBuf = buildForwardScreenCache(gaussiansBuffer: gaussiansBuffer, gaussianCount: gaussianCount, worldToCam: worldToCam, intrinsics: intrinsics, zSign: zSign)
                        forwardCacheBuffer = localCacheBuf
                        forwardCacheCameraKey = camKey
                        forwardCacheGaussiansVersion = gaussiansVersionCounter
                    }
                }
                enc.setTexture(target, index: 0)
                enc.setBytes(&shMode, length: MemoryLayout<SHModeUniform>.stride, index: 4)
                if let cb = localCacheBuf { enc.setBuffer(cb, offset: 0, index: 5) }
                let tgSize = MTLSize(width: tileSz, height: tileSz, depth: 1)
                let grid = MTLSize(width: t.tilesX, height: t.tilesY, depth: 1)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tgSize)
                enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted(); return
            } else { enc.setComputePipelineState(pipeline) } // fallback
        } else {
            enc.setComputePipelineState(pipeline)
        }
    let tileSz = 16
        let tiles = buildTiles(gaussianBuffer: gaussiansBuffer, count: gaussianCount, worldToCam: worldToCam, intrinsics: intrinsics, width: target.width, height: target.height, zSign: zSign, tileSize: tileSz)
        let tilesX: UInt32
        let tilesY: UInt32
        let tileSizeVal: UInt32
        let tileOffsets: MTLBuffer
        let tileList: MTLBuffer
        if let t = tiles {
            tilesX = UInt32(t.tilesX)
            tilesY = UInt32(t.tilesY)
            tileSizeVal = UInt32(tileSz)
            tileOffsets = t.offsets
            tileList = t.list
            if debugLogging {
                // Quick CPU stats: number of indices in list buffer
                let listCount = t.list.length / MemoryLayout<UInt32>.stride
                if gaussianCount > 0 && listCount == 0 {
                    print("[ForwardDiag] Tile build produced 0 indices for gaussianCount=\(gaussianCount)")
                }
            }
        } else {
            tileSizeVal = UInt32(max(target.width, target.height))
            tilesX = 1; tilesY = 1
            let offsetsData: [UInt32] = [0, UInt32(gaussianCount)]
            // SAFE padding: allocate at least 1 element
            let listCount = max(gaussianCount, 1)
            var listData = [UInt32](repeating: 0, count: listCount)
            if gaussianCount > 0 { for i in 0..<gaussianCount { listData[i] = UInt32(i) } }
            if debugLogging {
                print("[Forward] Fallback tile path gaussians=\(gaussianCount) padded listCount=\(listCount)")
            }
            guard let to = device.makeBuffer(bytes: offsetsData, length: MemoryLayout<UInt32>.stride * offsetsData.count, options: .storageModeShared),
                  let tl = device.makeBuffer(bytes: listData, length: MemoryLayout<UInt32>.stride * listData.count, options: .storageModeShared) else { return }
            tileOffsets = to; tileList = tl
        }
        var uniforms = RenderUniforms(worldToCam: worldToCam, fx: intrinsics.fx, fy: intrinsics.fy, cx: intrinsics.cx, cy: intrinsics.cy, zSign: zSign, imageWidth: UInt32(target.width), imageHeight: UInt32(target.height), gaussianCount: UInt32(gaussianCount), pointScale: pointScale, tilesX: tilesX, tilesY: tilesY, tileSize: tileSizeVal, useCache: enableForwardCache ? 1 : 0)
        enc.setBuffer(gaussiansBuffer, offset: 0, index: 0)
        enc.setBytes(&uniforms, length: MemoryLayout<RenderUniforms>.stride, index: 1)
    enc.setBuffer(tileOffsets, offset: 0, index: 2)
    enc.setBuffer(tileList, offset: 0, index: 3)
    var shMode = SHModeUniform(enableL2: enableSHL2 ? 1 : 0)
    enc.setBytes(&shMode, length: MemoryLayout<SHModeUniform>.stride, index: 4)
    if enableForwardCache {
        let camKey = makeCameraKey(worldToCam: worldToCam, intr: intrinsics, zSign: zSign)
        if let existing = forwardCacheBuffer, camKey == forwardCacheCameraKey, forwardCacheGaussiansVersion == gaussiansVersionCounter, existing.length >= gaussianCount * MemoryLayout<GaussianScreenCache>.stride {
            forwardCacheHits &+= 1
            enc.setBuffer(existing, offset: 0, index: 5)
        } else {
            forwardCacheMisses &+= 1
            if let cb = buildForwardScreenCache(gaussiansBuffer: gaussiansBuffer, gaussianCount: gaussianCount, worldToCam: worldToCam, intrinsics: intrinsics, zSign: zSign) {
                forwardCacheBuffer = cb
                forwardCacheCameraKey = camKey
                forwardCacheGaussiansVersion = gaussiansVersionCounter
                enc.setBuffer(cb, offset: 0, index: 5)
            } else if debugLogging {
                print("[ForwardCache] Failed to build cache buffer; proceeding with dummy.")
            }
        }
    } else {
        // Provide a 1-element dummy buffer to satisfy pipeline binding at index 5
        if dummyOneFloatBuffer == nil {
            dummyOneFloatBuffer = device.makeBuffer(length: max(16, MemoryLayout<GaussianScreenCache>.stride), options: .storageModeShared)
        }
        if let d = dummyOneFloatBuffer { enc.setBuffer(d, offset: 0, index: 5) }
    }
        if debugLogging {
            // Visibility diagnostic: sample first 5 gaussians (position.z, opacity, scale)
            let gptr = gaussiansBuffer.contents().bindMemory(to: GaussianParamCPU.self, capacity: gaussianCount)
            var minOpacity: Float = 1e9, maxOpacity: Float = -1e9
            var minScale: Float = 1e9, maxScale: Float = -1e9
            let sampleN = min(gaussianCount, 5)
            for i in 0..<gaussianCount {
                let gp = gptr[i]
                minOpacity = min(minOpacity, gp.opacity); maxOpacity = max(maxOpacity, gp.opacity)
                let sMin = min(gp.scale.x, min(gp.scale.y, gp.scale.z))
                let sMax = max(gp.scale.x, max(gp.scale.y, gp.scale.z))
                minScale = min(minScale, sMin); maxScale = max(maxScale, sMax)
            }
            var samples: [String] = []
            for i in 0..<sampleN {
                let gp = gptr[i]
                samples.append(String(format: "[%d z=%.3f op=%.3f s=(%.3f %.3f %.3f)]", i, gp.position.z, gp.opacity, gp.scale.x, gp.scale.y, gp.scale.z))
            }
            let listCountEst = (tileList.length / MemoryLayout<UInt32>.stride)
            print("[ForwardDiag] N=\(gaussianCount) tiles=\(tilesX)x\(tilesY) list=\(listCountEst) op=[\(String(format: "%.3g", minOpacity)), \(String(format: "%.3g", maxOpacity))] scale=[\(String(format: "%.3g", minScale)), \(String(format: "%.3g", maxScale))] samples=\(samples.joined(separator: ", "))")
            // Intrinsics sanity
            if intrinsics.cx < 0 || intrinsics.cy < 0 { print("[ForwardDiag] WARNING cx/cy negative: cx=\(intrinsics.cx) cy=\(intrinsics.cy)") }
            if abs(Float(target.width)/2 - intrinsics.cx) > Float(target.width) * 0.25 || abs(Float(target.height)/2 - intrinsics.cy) > Float(target.height) * 0.25 {
                print("[ForwardDiag] POSSIBLE MIS-CENTER: cx=\(intrinsics.cx) (~\(Float(target.width)/2)), cy=\(intrinsics.cy) (~\(Float(target.height)/2))")
            }
            if maxOpacity <= 0.0001 { print("[ForwardDiag] All opacities ~0 ⇒ invisible") }
            if maxScale < 1e-4 { print("[ForwardDiag] Extremely small scales ⇒ splats may be subpixel invisible") }
        }
    enc.setTexture(target, index: 0)
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let tg = MTLSize(width: w, height: max(1, h), depth: 1)
        let grid = MTLSize(width: target.width, height: target.height, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        lastForwardMS = (CFAbsoluteTimeGetCurrent() - tStart) * 1000.0
    }

    // Readback utility for preview
    func cgImage(from texture: MTLTexture) -> CGImage? {
        let width = texture.width
        let height = texture.height

    // Ideally we'd blit/convert to RGBA8 via GPU; for simplicity we do a CPU half-float -> 8-bit conversion.

        // CPU conversion path
    let pixelCount = width * height

    // Read back RGBA16F bytes and convert to 8-bit on CPU.

        let rowBytes = width * MemoryLayout<UInt16>.stride * 4
        let raw = UnsafeMutableRawPointer.allocate(byteCount: rowBytes * height, alignment: 0x1000)
        defer { raw.deallocate() }
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.getBytes(raw, bytesPerRow: rowBytes, from: region, mipmapLevel: 0)

        // Convert half->float and then float->8-bit
        let src = raw.bindMemory(to: UInt16.self, capacity: pixelCount * 4)
        var rgba8 = [UInt8](repeating: 0, count: pixelCount * 4)
        @inline(__always) func linearToSRGB(_ x: Float) -> Float {
            if x <= 0.0031308 { return 12.92 * x }
            return 1.055 * powf(x, 1.0/2.4) - 0.055
        }
        for i in 0..<(pixelCount) {
            let r16 = src[i*4 + 0]
            let g16 = src[i*4 + 1]
            let b16 = src[i*4 + 2]
            // let a16 = src[i*4 + 3]
            func halfToFloat(_ h: UInt16) -> Float {
                // Minimal half->float conversion
                let s = (h & 0x8000) >> 15
                let e = (h & 0x7C00) >> 10
                let f = (h & 0x03FF)
                var out: Float
                if e == 0 {
                    if f == 0 { out = 0 }
                    else {
                        let mant = Float(f) / 1024.0
                        let val = ldexpf(mant, Int32(-14))
                        out = s == 1 ? -val : val
                    }
                } else if e == 31 {
                    out = Float.nan
                } else {
                    let mant = 1.0 + Float(f) / 1024.0
                    let val = ldexpf(mant, Int32(Int(e) - 15))
                    out = s == 1 ? -val : val
                }
                return out
            }
            let rLin = max(0.0, min(1.0, halfToFloat(r16)))
            let gLin = max(0.0, min(1.0, halfToFloat(g16)))
            let bLin = max(0.0, min(1.0, halfToFloat(b16)))
            // Apply gamma encode for preview (linear -> sRGB) to restore perceived saturation
            let r = max(0.0, min(1.0, linearToSRGB(rLin)))
            let g = max(0.0, min(1.0, linearToSRGB(gLin)))
            let b = max(0.0, min(1.0, linearToSRGB(bLin)))
            rgba8[i*4+0] = UInt8(r * 255 + 0.5)
            rgba8[i*4+1] = UInt8(g * 255 + 0.5)
            rgba8[i*4+2] = UInt8(b * 255 + 0.5)
            rgba8[i*4+3] = 255
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: CGBitmapInfo = [
            .byteOrder32Big,
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        ]
        guard let provider = CGDataProvider(data: Data(rgba8) as CFData),
              let cg = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent)
        else { return nil }
        return cg
    }

    struct GradBuffers {
        let c0R: MTLBuffer
        let c0G: MTLBuffer
        let c0B: MTLBuffer
        let r1x: MTLBuffer
        let r1y: MTLBuffer
        let r1z: MTLBuffer
        let g1x: MTLBuffer
        let g1y: MTLBuffer
        let g1z: MTLBuffer
        let b1x: MTLBuffer
        let b1y: MTLBuffer
        let b1z: MTLBuffer
        // SH L2 gradients packed per channel: layout = 6 consecutive segments of length N (coeff-major)
        let r2: MTLBuffer
        let g2: MTLBuffer
        let b2: MTLBuffer
        let opacity: MTLBuffer
        let sigma: MTLBuffer
                let posX: MTLBuffer
                let posY: MTLBuffer
                let posZ: MTLBuffer
                let sX: MTLBuffer
                let sY: MTLBuffer
                let sZ: MTLBuffer
    }

    func makeGradBuffers(count: Int) -> GradBuffers? {
        let safeCount = max(count, 1)
        let len = MemoryLayout<UInt32>.stride * safeCount
        let lenL2 = MemoryLayout<UInt32>.stride * safeCount * 6 // six coeffs per channel
        if count == 0 && debugLogging {
            print("[MetalRenderer] makeGradBuffers invoked with count=0 (allocating padded buffers of 1 element)")
        }
                        guard let r = safeMakeBuffer(length: len, tag: "grad.c0R"),
                    let g = safeMakeBuffer(length: len, tag: "grad.c0G"),
                    let b = safeMakeBuffer(length: len, tag: "grad.c0B"),
                    let r1x = safeMakeBuffer(length: len, tag: "grad.r1x"),
                    let r1y = safeMakeBuffer(length: len, tag: "grad.r1y"),
                    let r1z = safeMakeBuffer(length: len, tag: "grad.r1z"),
                    let g1x = safeMakeBuffer(length: len, tag: "grad.g1x"),
                    let g1y = safeMakeBuffer(length: len, tag: "grad.g1y"),
                    let g1z = safeMakeBuffer(length: len, tag: "grad.g1z"),
                    let b1x = safeMakeBuffer(length: len, tag: "grad.b1x"),
                    let b1y = safeMakeBuffer(length: len, tag: "grad.b1y"),
                    let b1z = safeMakeBuffer(length: len, tag: "grad.b1z"),
                                        let r2 = safeMakeBuffer(length: lenL2, tag: "grad.r2"),
                                        let g2 = safeMakeBuffer(length: lenL2, tag: "grad.g2"),
                                        let b2 = safeMakeBuffer(length: lenL2, tag: "grad.b2"),
                    let o = safeMakeBuffer(length: len, tag: "grad.opacity"),
                                        let s = safeMakeBuffer(length: len, tag: "grad.sigma"),
                                        let px = safeMakeBuffer(length: len, tag: "grad.posX"),
                                        let py = safeMakeBuffer(length: len, tag: "grad.posY"),
                                        let pz = safeMakeBuffer(length: len, tag: "grad.posZ"),
                                        let sx = safeMakeBuffer(length: len, tag: "grad.sX"),
                                        let sy = safeMakeBuffer(length: len, tag: "grad.sY"),
                                        let sz = safeMakeBuffer(length: len, tag: "grad.sZ") else { return nil }
                        for buf in [r,g,b,r1x,r1y,r1z,g1x,g1y,g1z,b1x,b1y,b1z,r2,g2,b2,o,s,px,py,pz,sx,sy,sz] { memset(buf.contents(), 0, buf.length) }
                        return GradBuffers(c0R: r, c0G: g, c0B: b, r1x: r1x, r1y: r1y, r1z: r1z, g1x: g1x, g1y: g1y, g1z: g1z, b1x: b1x, b1y: b1y, b1z: b1z, r2: r2, g2: g2, b2: b2, opacity: o, sigma: s, posX: px, posY: py, posZ: pz, sX: sx, sY: sy, sZ: sz)
    }

    func backward(gaussiansBuffer: MTLBuffer, gaussianCount: Int, worldToCam: simd_float4x4, intrinsics: CameraIntrinsics, residual: MTLTexture, grads: GradBuffers, zSign: Float = -1.0, enableSHL2: Bool = false, enableCache: Bool = true) {
        if gaussianCount == 0 {
            if debugLogging {
                print("[Backward] Skip: gaussianCount == 0 (no gradients dispatched)")
            }
            return
        }
        let tStart = CFAbsoluteTimeGetCurrent()
        guard let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return }
    // Select backward pipeline
    let useTiledBwd = useTiledBackward && tiledBackwardPipeline != nil
    let pipelineBwd = useTiledBwd ? tiledBackwardPipeline! : backwardPipeline
    enc.setComputePipelineState(pipelineBwd)
        // Build tiles (fallback to single tile if binning fails)
        let tileSz = 16
        let tiles = buildTiles(gaussianBuffer: gaussiansBuffer, count: gaussianCount, worldToCam: worldToCam, intrinsics: intrinsics, width: residual.width, height: residual.height, zSign: zSign, tileSize: tileSz)
        let tilesX: UInt32
        let tilesY: UInt32
        let tileSizeVal: UInt32
        let tileOffsets: MTLBuffer
        let tileList: MTLBuffer
        if let t = tiles {
            tilesX = UInt32(t.tilesX)
            tilesY = UInt32(t.tilesY)
            tileSizeVal = UInt32(tileSz)
            tileOffsets = t.offsets
            tileList = t.list
        } else {
            tileSizeVal = UInt32(max(residual.width, residual.height))
            tilesX = 1; tilesY = 1
            let offsetsData: [UInt32] = [0, UInt32(gaussianCount)]
            // SAFE: allocate at least 1 element to avoid zero-length Metal buffer assertion
            let listCount = max(gaussianCount, 1)
            var listData = [UInt32](repeating: 0, count: listCount)
            if gaussianCount > 0 {
                for i in 0..<gaussianCount { listData[i] = UInt32(i) }
            }
            if debugLogging {
                print("[Backward] Fallback tile path. gaussianCount=\(gaussianCount) allocated listCount=\(listCount)")
            }
            guard let to = device.makeBuffer(bytes: offsetsData, length: MemoryLayout<UInt32>.stride * offsetsData.count, options: .storageModeShared),
                  let tl = device.makeBuffer(bytes: listData, length: MemoryLayout<UInt32>.stride * listData.count, options: .storageModeShared) else { return }
            tileOffsets = to; tileList = tl
        }
        // Optional per-gaussian screen-space cache (CPU). Currently recomputed every backward; could be reused across frames if camera unchanged.
        var cacheBuffer: MTLBuffer? = nil
        if enableCache {
            // Bind with full stride to avoid misaligned reads (previous short struct caused zero scales => square splats)
            let gptr = gaussiansBuffer.contents().bindMemory(to: GaussianParamCPU.self, capacity: gaussianCount)
            var cache = [GaussianScreenCache](repeating: GaussianScreenCache(uv: .zero, invz: 0, zf: 0, cam4: .zero, j00: 0, j01: 0, j02: 0, pad0: 0, j10: 0, j11: 0, j12: 0, pad1: 0, sxx: 0, sxy: 0, syy: 0, pad2: 0, a00: 0, a01: 0, a11: 0, pad3: 0, viewDir4: .zero), count: gaussianCount)
            for i in 0..<gaussianCount {
                let gp = gptr[i]
                let p4 = SIMD4<Float>(gp.position, 1)
                let cam4 = worldToCam * p4
                let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
                let zf = (zSign < 0) ? (-cam.z) : cam.z
                if zf <= 0 {
                    continue
                }
                let invz = 1.0 / zf
                let uvx = intrinsics.fx * (cam.x * invz) + intrinsics.cx
                let uvy = intrinsics.fy * (cam.y * invz) + intrinsics.cy
                let invz2 = invz * invz
                let j00 = intrinsics.fx * invz; let j01: Float = 0; let j02 = -intrinsics.fx * cam.x * invz2
                let j10: Float = 0; let j11 = intrinsics.fy * invz; let j12 = -intrinsics.fy * cam.y * invz2
                let sx2 = gp.scale.x * gp.scale.x
                let sy2 = gp.scale.y * gp.scale.y
                let sz2 = gp.scale.z * gp.scale.z
                let sxx = j00*j00*sx2 + j01*j01*sy2 + j02*j02*sz2
                let sxy = j00*j10*sx2 + j01*j11*sy2 + j02*j12*sz2
                let syy = j10*j10*sx2 + j11*j11*sy2 + j12*j12*sz2
                let det = max(1e-8, (sxx*syy - sxy*sxy))
                let a00 = syy / det
                let a01 = -sxy / det
                let a11 = sxx / det
                let vdir = simd_normalize(-cam)
                cache[i] = GaussianScreenCache(uv: SIMD2<Float>(uvx, uvy), invz: invz, zf: zf, cam4: cam4, j00: j00, j01: j01, j02: j02, pad0: 0, j10: j10, j11: j11, j12: j12, pad1: 0, sxx: sxx, sxy: sxy, syy: syy, pad2: 0, a00: a00, a01: a01, a11: a11, pad3: 0, viewDir4: SIMD4<Float>(vdir.x, vdir.y, vdir.z, 0))
            }
            let bytes = cache.count * MemoryLayout<GaussianScreenCache>.stride
            cacheBuffer = device.makeBuffer(bytes: cache, length: bytes, options: .storageModeShared)
        }
        var u = GradUniforms(
            worldToCam: worldToCam,
            fx: intrinsics.fx, fy: intrinsics.fy, cx: intrinsics.cx, cy: intrinsics.cy,
            zSign: zSign,
            imageWidth: UInt32(residual.width), imageHeight: UInt32(residual.height),
            gaussianCount: UInt32(gaussianCount),
            tilesX: tilesX,
            tilesY: tilesY,
            tileSize: tileSizeVal,
            useCache: (enableCache && cacheBuffer != nil) ? 1 : 0,
            lossMode: enableCharbonnierLoss ? 1 : 0,
            charbEps: charbEps,
            _pad0: 0,
            _pad1: 0
        )
        enc.setBuffer(gaussiansBuffer, offset: 0, index: 0)
        enc.setBytes(&u, length: MemoryLayout<GradUniforms>.stride, index: 1)
    enc.setTexture(residual, index: 0)
        // Bind gradient buffers in the exact order expected by accumulateGradients kernel signature
        enc.setBuffer(grads.c0R, offset: 0, index: 2)
        enc.setBuffer(grads.c0G, offset: 0, index: 3)
        enc.setBuffer(grads.c0B, offset: 0, index: 4)
        enc.setBuffer(grads.r1x, offset: 0, index: 5)
        enc.setBuffer(grads.r1y, offset: 0, index: 6)
        enc.setBuffer(grads.r1z, offset: 0, index: 7)
        enc.setBuffer(grads.g1x, offset: 0, index: 8)
        enc.setBuffer(grads.g1y, offset: 0, index: 9)
        enc.setBuffer(grads.g1z, offset: 0, index: 10)
        enc.setBuffer(grads.b1x, offset: 0, index: 11)
        enc.setBuffer(grads.b1y, offset: 0, index: 12)
        enc.setBuffer(grads.b1z, offset: 0, index: 13)
    enc.setBuffer(grads.r2, offset: 0, index: 14)
    enc.setBuffer(grads.g2, offset: 0, index: 15)
    enc.setBuffer(grads.b2, offset: 0, index: 16)
    enc.setBuffer(grads.opacity, offset: 0, index: 17)
    enc.setBuffer(grads.sigma, offset: 0, index: 18)
    enc.setBuffer(grads.posX, offset: 0, index: 19)
    enc.setBuffer(grads.posY, offset: 0, index: 20)
    enc.setBuffer(grads.posZ, offset: 0, index: 21)
    enc.setBuffer(grads.sX, offset: 0, index: 22)
    enc.setBuffer(grads.sY, offset: 0, index: 23)
    enc.setBuffer(grads.sZ, offset: 0, index: 24)
    enc.setBuffer(tileOffsets, offset: 0, index: 25)
    enc.setBuffer(tileList, offset: 0, index: 26)
    if let cb = cacheBuffer {
        enc.setBuffer(cb, offset: 0, index: 27)
    } else {
        if dummyOneFloatBuffer == nil {
            dummyOneFloatBuffer = device.makeBuffer(length: max(16, MemoryLayout<GaussianScreenCache>.stride), options: .storageModeShared)
        }
        if let d = dummyOneFloatBuffer { enc.setBuffer(d, offset: 0, index: 27) }
    }

        if useTiledBwd {
            // Launch one threadgroup per tile with tileSz x tileSz threads
            let tileSzLaunch = Int(tileSizeVal)
            let gridTG = MTLSize(width: Int(tilesX), height: Int(tilesY), depth: 1)
            let tgSize = MTLSize(width: tileSzLaunch, height: tileSzLaunch, depth: 1)
            // Scratch memory length: one int per thread (sequential component reduction)
            enc.setThreadgroupMemoryLength(MemoryLayout<Int32>.stride * tileSzLaunch * tileSzLaunch, index: 0)
            enc.dispatchThreadgroups(gridTG, threadsPerThreadgroup: tgSize)
        } else {
            let w = backwardPipeline.threadExecutionWidth
            let h = backwardPipeline.maxTotalThreadsPerThreadgroup / w
            let tg = MTLSize(width: w, height: max(1, h), depth: 1)
            let grid = MTLSize(width: residual.width, height: residual.height, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        lastBackwardMS = (CFAbsoluteTimeGetCurrent() - tStart) * 1000.0
        if debugLogging {
            print(String(format: "[Backward] Done: %.3f ms tiles(%d x %d) tileSize=%d gaussians=%d", lastBackwardMS, tilesX, tilesY, tileSizeVal, gaussianCount))
        }
    }

    // Forward screen-space cache builder (CPU). Reuses GaussianScreenCache layout.
    private func buildForwardScreenCache(gaussiansBuffer: MTLBuffer, gaussianCount: Int, worldToCam: simd_float4x4, intrinsics: CameraIntrinsics, zSign: Float) -> MTLBuffer? {
        if gaussianCount == 0 {
            if debugLogging { print("[ForwardCache] Skip build: gaussianCount == 0") }
            // Allocate a minimal padded buffer (1 element) to keep binding logic simple
            let dummy = [GaussianScreenCache(uv: .zero, invz: 0, zf: 0, cam4: .zero, j00: 0, j01: 0, j02: 0, pad0: 0, j10: 0, j11: 0, j12: 0, pad1: 0, sxx: 0, sxy: 0, syy: 0, pad2: 0, a00: 0, a01: 0, a11: 0, pad3: 0, viewDir4: .zero)]
            return device.makeBuffer(bytes: dummy, length: MemoryLayout<GaussianScreenCache>.stride, options: .storageModeShared)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let gptr = gaussiansBuffer.contents().bindMemory(to: GaussianParamCPU.self, capacity: gaussianCount)
        var cache = [GaussianScreenCache](repeating: GaussianScreenCache(uv: .zero, invz: 0, zf: 0, cam4: .zero, j00: 0, j01: 0, j02: 0, pad0: 0, j10: 0, j11: 0, j12: 0, pad1: 0, sxx: 0, sxy: 0, syy: 0, pad2: 0, a00: 0, a01: 0, a11: 0, pad3: 0, viewDir4: .zero), count: gaussianCount)
        for i in 0..<gaussianCount {
            let gp = gptr[i]
            let p4 = SIMD4<Float>(gp.position, 1)
            let cam4 = worldToCam * p4
            let cam = SIMD3<Float>(cam4.x, cam4.y, cam4.z)
            let zf = (zSign < 0) ? (-cam.z) : cam.z
            if zf <= 0 { continue }
            let invz = 1.0 / zf
            let uvx = intrinsics.fx * (cam.x * invz) + intrinsics.cx
            let uvy = intrinsics.fy * (cam.y * invz) + intrinsics.cy
            let invz2 = invz * invz
            let j00 = intrinsics.fx * invz; let j01: Float = 0; let j02 = -intrinsics.fx * cam.x * invz2
            let j10: Float = 0; let j11 = intrinsics.fy * invz; let j12 = -intrinsics.fy * cam.y * invz2
            let sx2 = gp.scale.x * gp.scale.x
            let sy2 = gp.scale.y * gp.scale.y
            let sz2 = gp.scale.z * gp.scale.z
            let sxx = j00*j00*sx2 + j01*j01*sy2 + j02*j02*sz2
            let sxy = j00*j10*sx2 + j01*j11*sy2 + j02*j12*sz2
            let syy = j10*j10*sx2 + j11*j11*sy2 + j12*j12*sz2
            let det = max(1e-8, (sxx*syy - sxy*sxy))
            let a00 = syy / det
            let a01 = -sxy / det
            let a11 = sxx / det
            let vdir = simd_normalize(-cam)
            cache[i] = GaussianScreenCache(uv: SIMD2<Float>(uvx, uvy), invz: invz, zf: zf, cam4: cam4, j00: j00, j01: j01, j02: j02, pad0: 0, j10: j10, j11: j11, j12: j12, pad1: 0, sxx: sxx, sxy: sxy, syy: syy, pad2: 0, a00: a00, a01: a01, a11: a11, pad3: 0, viewDir4: SIMD4<Float>(vdir.x, vdir.y, vdir.z, 0))
        }
        let buf = device.makeBuffer(bytes: cache, length: cache.count * MemoryLayout<GaussianScreenCache>.stride, options: .storageModeShared)
        lastForwardCacheBuildMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        return buf
    }
}
