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
}

final class MetalRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let backwardPipeline: MTLComputePipelineState
    private let tiledPipeline: MTLComputePipelineState?

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
    var enableTileReuse: Bool = true
    var enableFrustumCull: Bool = true
    private var useTiledKernel: Bool = false

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
        } catch {
            print("Failed to create pipeline: \(error)")
            return nil
        }
    }

    // Build screen-space tiling (CPU) for current camera and gaussians
    private func buildTiles(gaussianBuffer: MTLBuffer, count: Int, worldToCam: simd_float4x4, intrinsics: CameraIntrinsics, width: Int, height: Int, zSign: Float, tileSize: Int) -> (offsets: MTLBuffer, list: MTLBuffer, tilesX: Int, tilesY: Int)? {
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
        for i in 0..<count {
            let g = ptr[i]
            let cam = project(g.position)
            let zf: Float = (zSign < 0) ? -cam.z : cam.z
            if zf <= 1e-5 { continue }
            // Basic frustum cull in NDC (assuming pinhole): x/z within +- (width/ (2*fx)) * z maybe not needed; skip for now beyond z>0
            let u: Float = intrinsics.fx * (cam.x / zf) + intrinsics.cx
            let v: Float = intrinsics.fy * (cam.y / zf) + intrinsics.cy
            if enableFrustumCull && (u < -Float(tileSize) || u > Float(width + tileSize) || v < -Float(tileSize) || v > Float(height + tileSize)) { continue }
            let sx = max(1e-5, g.scale.x)
            let sy = max(1e-5, g.scale.y)
            let rX = 3.0 * abs(intrinsics.fx * sx / zf)
            let rY = 3.0 * abs(intrinsics.fy * sy / zf)
            let r = max(rX, rY)
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
                for tx in a...b {
                    let tid = row + tx
                    let dst = cursor[tid]
                    if dst < total { list[dst] = UInt32(i); cursor[tid] = dst + 1 }
                }
            }
        }
        guard let offsetsBuf = device.makeBuffer(bytes: offsets, length: MemoryLayout<UInt32>.stride * offsets.count, options: .storageModeShared),
              let listBuf = device.makeBuffer(bytes: list, length: MemoryLayout<UInt32>.stride * max(1, list.count), options: .storageModeShared) else { return nil }
        let built = (offsetsBuf, listBuf, tilesX, tilesY)
        if enableTileReuse { tileCache[key] = built }
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
        let length = MemoryLayout<T>.stride * array.count
        guard let buf = device.makeBuffer(length: length, options: options) else { return nil }
        if length > 0 {
            array.withUnsafeBytes { rawBuf in
                if let base = rawBuf.baseAddress {
                    memcpy(buf.contents(), base, length)
                }
            }
        }
        return buf
    }

    func setUseTiledKernel(_ v: Bool) { useTiledKernel = v }

    struct SHModeUniform { var enableL2: UInt32 }

    func forward(gaussiansBuffer: MTLBuffer, gaussianCount: Int, worldToCam: simd_float4x4, intrinsics: CameraIntrinsics, target: MTLTexture, pointScale: Float = 800.0, zSign: Float = -1.0, enableSHL2: Bool = false) {
        // Use legacy per-pixel kernel (renderGaussians) for now; tiled kernel placeholder hook
        guard let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return }
        if useTiledKernel, let tiled = tiledPipeline {
            // Tiled dispatch: threadgroups = tilesX x tilesY, threadsPerThreadgroup ~ tileSize x tileSize
            enc.setComputePipelineState(tiled)
            var uniforms = RenderUniforms(worldToCam: worldToCam, fx: intrinsics.fx, fy: intrinsics.fy, cx: intrinsics.cx, cy: intrinsics.cy, zSign: zSign, imageWidth: UInt32(target.width), imageHeight: UInt32(target.height), gaussianCount: UInt32(gaussianCount), pointScale: pointScale, tilesX: 1, tilesY: 1, tileSize: 0)
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
                enc.setTexture(target, index: 0)
                enc.setBytes(&shMode, length: MemoryLayout<SHModeUniform>.stride, index: 4)
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
        } else {
            tileSizeVal = UInt32(max(target.width, target.height))
            tilesX = 1; tilesY = 1
            let offsetsData: [UInt32] = [0, UInt32(gaussianCount)]
            let listData: [UInt32] = gaussianCount > 0 ? Array(0..<UInt32(gaussianCount)) : []
            guard let to = device.makeBuffer(bytes: offsetsData, length: MemoryLayout<UInt32>.stride * offsetsData.count, options: .storageModeShared),
                  let tl = device.makeBuffer(bytes: listData, length: MemoryLayout<UInt32>.stride * listData.count, options: .storageModeShared) else { return }
            tileOffsets = to; tileList = tl
        }
        var uniforms = RenderUniforms(worldToCam: worldToCam, fx: intrinsics.fx, fy: intrinsics.fy, cx: intrinsics.cx, cy: intrinsics.cy, zSign: zSign, imageWidth: UInt32(target.width), imageHeight: UInt32(target.height), gaussianCount: UInt32(gaussianCount), pointScale: pointScale, tilesX: tilesX, tilesY: tilesY, tileSize: tileSizeVal)
        enc.setBuffer(gaussiansBuffer, offset: 0, index: 0)
        enc.setBytes(&uniforms, length: MemoryLayout<RenderUniforms>.stride, index: 1)
    enc.setBuffer(tileOffsets, offset: 0, index: 2)
    enc.setBuffer(tileList, offset: 0, index: 3)
    var shMode = SHModeUniform(enableL2: enableSHL2 ? 1 : 0)
    enc.setBytes(&shMode, length: MemoryLayout<SHModeUniform>.stride, index: 4)
    enc.setTexture(target, index: 0)
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let tg = MTLSize(width: w, height: max(1, h), depth: 1)
        let grid = MTLSize(width: target.width, height: target.height, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
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
        let len = MemoryLayout<UInt32>.stride * count
        let lenL2 = MemoryLayout<UInt32>.stride * count * 6 // six coeffs per channel
            guard let r = device.makeBuffer(length: len, options: .storageModeShared),
          let g = device.makeBuffer(length: len, options: .storageModeShared),
          let b = device.makeBuffer(length: len, options: .storageModeShared),
          let r1x = device.makeBuffer(length: len, options: .storageModeShared),
          let r1y = device.makeBuffer(length: len, options: .storageModeShared),
          let r1z = device.makeBuffer(length: len, options: .storageModeShared),
          let g1x = device.makeBuffer(length: len, options: .storageModeShared),
          let g1y = device.makeBuffer(length: len, options: .storageModeShared),
          let g1z = device.makeBuffer(length: len, options: .storageModeShared),
          let b1x = device.makeBuffer(length: len, options: .storageModeShared),
          let b1y = device.makeBuffer(length: len, options: .storageModeShared),
          let b1z = device.makeBuffer(length: len, options: .storageModeShared),
                    let r2 = device.makeBuffer(length: lenL2, options: .storageModeShared),
                    let g2 = device.makeBuffer(length: lenL2, options: .storageModeShared),
                    let b2 = device.makeBuffer(length: lenL2, options: .storageModeShared),
          let o = device.makeBuffer(length: len, options: .storageModeShared),
                    let s = device.makeBuffer(length: len, options: .storageModeShared),
                    let px = device.makeBuffer(length: len, options: .storageModeShared),
                    let py = device.makeBuffer(length: len, options: .storageModeShared),
                    let pz = device.makeBuffer(length: len, options: .storageModeShared),
                    let sx = device.makeBuffer(length: len, options: .storageModeShared),
                    let sy = device.makeBuffer(length: len, options: .storageModeShared),
                    let sz = device.makeBuffer(length: len, options: .storageModeShared) else { return nil }
                        for buf in [r,g,b,r1x,r1y,r1z,g1x,g1y,g1z,b1x,b1y,b1z,r2,g2,b2,o,s,px,py,pz,sx,sy,sz] { memset(buf.contents(), 0, buf.length) }
                        return GradBuffers(c0R: r, c0G: g, c0B: b, r1x: r1x, r1y: r1y, r1z: r1z, g1x: g1x, g1y: g1y, g1z: g1z, b1x: b1x, b1y: b1y, b1z: b1z, r2: r2, g2: g2, b2: b2, opacity: o, sigma: s, posX: px, posY: py, posZ: pz, sX: sx, sY: sy, sZ: sz)
    }

    func backward(gaussiansBuffer: MTLBuffer, gaussianCount: Int, worldToCam: simd_float4x4, intrinsics: CameraIntrinsics, residual: MTLTexture, grads: GradBuffers, zSign: Float = -1.0, enableSHL2: Bool = false) {
        guard let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(backwardPipeline)
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
            let listData: [UInt32] = gaussianCount > 0 ? Array(0..<UInt32(gaussianCount)) : []
            guard let to = device.makeBuffer(bytes: offsetsData, length: MemoryLayout<UInt32>.stride * offsetsData.count, options: .storageModeShared),
                  let tl = device.makeBuffer(bytes: listData, length: MemoryLayout<UInt32>.stride * listData.count, options: .storageModeShared) else { return }
            tileOffsets = to; tileList = tl
        }
        var u = GradUniforms(
            worldToCam: worldToCam,
            fx: intrinsics.fx, fy: intrinsics.fy, cx: intrinsics.cx, cy: intrinsics.cy,
            zSign: zSign,
            imageWidth: UInt32(residual.width), imageHeight: UInt32(residual.height),
            gaussianCount: UInt32(gaussianCount),
            tilesX: tilesX,
            tilesY: tilesY,
            tileSize: tileSizeVal
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

        let w = backwardPipeline.threadExecutionWidth
        let h = backwardPipeline.maxTotalThreadsPerThreadgroup / w
        let tg = MTLSize(width: w, height: max(1, h), depth: 1)
        let grid = MTLSize(width: residual.width, height: residual.height, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
}
