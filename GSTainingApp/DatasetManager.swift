//
//  DatasetManager.swift
//  GSTainingApp
//
//  Loads local dataset: images and camera params. Provides synthetic fallback.
//

import Foundation
import UIKit
import simd

final class DatasetManager {
    struct Sample {
        let image: UIImage
        let intr: CameraIntrinsics
        let extr: CameraExtrinsics
    }

    let width: Int
    let height: Int
    let samples: [Sample]

    // targetSize: if provided, images are resized with letterboxing and intrinsics are scaled/offset accordingly
    init(width: Int = 256, height: Int = 256, targetSize: CGSize? = nil) {
        // Try to load ARKit/Nerfstudio-style transforms.json first, then our dataset.json
        if let spec = DatasetManager.loadTransformsSpec(),
           let loaded = DatasetManager.loadImages(spec: spec, targetSize: targetSize) {
            if let ts = targetSize, !loaded.isEmpty {
                self.width = Int(ts.width)
                self.height = Int(ts.height)
            } else {
                self.width = spec.width
                self.height = spec.height
            }
            self.samples = loaded
        } else if let spec = DatasetManager.loadDatasetSpec(),
                  let loaded = DatasetManager.loadImages(spec: spec, targetSize: targetSize) {
            if let ts = targetSize, !loaded.isEmpty {
                self.width = Int(ts.width)
                self.height = Int(ts.height)
            } else {
                self.width = spec.width
                self.height = spec.height
            }
            self.samples = loaded
        } else {
            // Synthetic fallback: a few camera poses around origin with a simple image
            self.width = width
            self.height = height
            self.samples = DatasetManager.syntheticDataset(width: width, height: height)
        }
    }

    static func documentsURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static func loadDatasetSpec() -> DatasetSpec? {
        let url = documentsURL().appendingPathComponent("dataset.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(DatasetSpec.self, from: data)
    }

    // MARK: - transforms.json loader (ARKit/nerfstudio style)

    struct TransformsTop: Codable {
        var fl_x: Float?
        var fl_y: Float?
        var cx: Float?
        var cy: Float?
        var w: Int?
        var h: Int?
        var camera_angle_x: Float?
        var camera_angle_y: Float?
        var frames: [TransformsFrame]
    }

    struct TransformsFrame: Codable {
        var file_path: String
        var transform_matrix: [[Float]]?
    // some formats may use an alternative key
        var transform: [[Float]]?
        var fl_x: Float?
        var fl_y: Float?
        var cx: Float?
        var cy: Float?
    }

    static func loadTransformsSpec() -> DatasetSpec? {
        let url = documentsURL().appendingPathComponent("transforms.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let top = try? JSONDecoder().decode(TransformsTop.self, from: data) else { return nil }

    // Determine image dimensions (from JSON or by opening the first file)
        var width = top.w ?? 0
        var height = top.h ?? 0
        if width == 0 || height == 0 {
            // fallback: open first image to get dimensions
            let base = documentsURL()
            if let first = top.frames.first {
                let imgURL = base.appendingPathComponent(first.file_path)
                if let img = UIImage(contentsOfFile: imgURL.path)?.cgImage {
                    width = img.width
                    height = img.height
                }
            }
            if width == 0 || height == 0 { return nil }
        }

    // Global intrinsics
        var fxGlobal: Float? = top.fl_x
        var fyGlobal: Float? = top.fl_y
        var cxGlobal: Float? = top.cx
        var cyGlobal: Float? = top.cy
    // If focal length is missing, use camera_angle_x/y
        if fxGlobal == nil {
            if let cax = top.camera_angle_x { fxGlobal = Float((Double(width)/2.0) / tan(Double(cax)/2.0)) }
        }
        if fyGlobal == nil {
            if let cay = top.camera_angle_y { fyGlobal = Float((Double(height)/2.0) / tan(Double(cay)/2.0)) } else { fyGlobal = fxGlobal }
        }
        if cxGlobal == nil { cxGlobal = Float(width)/2 }
        if cyGlobal == nil { cyGlobal = Float(height)/2 }

        var frames: [FrameData] = []
        frames.reserveCapacity(top.frames.count)
    for fr in top.frames {
        // Per-frame intrinsics (may override globals)
            let fx = fr.fl_x ?? fxGlobal ?? Float(width)/2
            let fy = fr.fl_y ?? fyGlobal ?? fx
            let cx = fr.cx ?? cxGlobal ?? Float(width)/2
            let cy = fr.cy ?? cyGlobal ?? Float(height)/2
            let intr = CameraIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy)

            // 4x4 camera-to-world matrix
            let mat2D = fr.transform_matrix ?? fr.transform
            guard var m2d = mat2D, m2d.count == 4, m2d.allSatisfy({ $0.count == 4 }) else {
                // if matrix missing — skip frame
                continue
            }
            // Some datasets put translation in the last row instead of last column.
            // Detect pattern: last column ~ [0,0,0,1], and elements [3][0..2] hold translation.
            let eps: Float = 1e-5
            let col3isHom = abs(m2d[0][3]) < eps && abs(m2d[1][3]) < eps && abs(m2d[2][3]) < eps && abs(m2d[3][3] - 1) < 1e-3
            let row3hasTrans = (abs(m2d[3][0]) + abs(m2d[3][1]) + abs(m2d[3][2])) > eps
            if col3isHom && row3hasTrans {
                // Transpose so that translation ends up in the last column (row-major flattening)
                var transposed = Array(repeating: Array(repeating: Float(0), count: 4), count: 4)
                for r in 0..<4 { for c in 0..<4 { transposed[r][c] = m2d[c][r] } }
                m2d = transposed
            }

            // Flatten to row-major 16 elements (after possible orientation fix)
            let flat: [Float] = m2d.flatMap { $0 }
            let extr = CameraExtrinsics(m: flat)

            frames.append(FrameData(imageFilename: fr.file_path, intrinsics: intr, extrinsics: extr))
        }

        guard !frames.isEmpty else { return nil }
        return DatasetSpec(frames: frames, width: width, height: height)
    }

    static func loadImages(spec: DatasetSpec, targetSize: CGSize? = nil) -> [Sample]? {
        let base = documentsURL()
        var out: [Sample] = []
        out.reserveCapacity(spec.frames.count)
        for f in spec.frames {
            let imgURL = base.appendingPathComponent(f.imageFilename)
            guard let ui = UIImage(contentsOfFile: imgURL.path), let cg = ui.cgImage else {
                return nil
            }
            var intr = f.intrinsics
            // Work in raw CGImage orientation (ignore EXIF). Use pixel dims for scaling.
            let srcW = CGFloat(cg.width)
            let srcH = CGFloat(cg.height)
            if let ts = targetSize, Int(ts.width) > 0, Int(ts.height) > 0 {
                let scale = min(ts.width / srcW, ts.height / srcH)
                let scaledW = srcW * scale
                let scaledH = srcH * scale
                let padX = (ts.width - scaledW) * 0.5
                let padY = (ts.height - scaledH) * 0.5
                let format = UIGraphicsImageRendererFormat.default()
                format.scale = 1
                let renderer = UIGraphicsImageRenderer(size: ts, format: format)
                let resized = renderer.image { ctx in
                    let drawRect = CGRect(x: padX, y: padY, width: scaledW, height: scaledH)
                    ctx.cgContext.interpolationQuality = .high
                    ctx.cgContext.draw(cg, in: drawRect)
                }
                // Adjust intrinsics: scale, then add padding offset to principal point (letterbox)
                let s = Float(scale)
                intr.fx *= s; intr.fy *= s
                intr.cx = intr.cx * s + Float(padX)
                intr.cy = intr.cy * s + Float(padY)
                out.append(Sample(image: resized, intr: intr, extr: f.extrinsics))
            } else {
                // No resize: normalize to .up by re-wrapping CGImage (removes EXIF orientation)
                let normalized = UIImage(cgImage: cg)
                out.append(Sample(image: normalized, intr: intr, extr: f.extrinsics))
            }
        }
        return out
    }

    static func syntheticDataset(width: Int, height: Int) -> [Sample] {
        // Create a simple gradient image
        func makeImage() -> UIImage {
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            var data = [UInt8](repeating: 0, count: width * height * 4)
            for y in 0..<height {
                for x in 0..<width {
                    let i = (y * width + x) * 4
                    data[i+0] = UInt8((Float(x)/Float(width)) * 255)
                    data[i+1] = UInt8((Float(y)/Float(height)) * 255)
                    data[i+2] = 128
                    data[i+3] = 255
                }
            }
            let provider = CGDataProvider(data: Data(data) as CFData)!
            let bitmapInfo: CGBitmapInfo = [
                .byteOrder32Big,
                CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
            ]
            let cg = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            return UIImage(cgImage: cg)
        }

        let image = makeImage()
        let fx: Float = 300, fy: Float = 300
        let cx: Float = Float(width)/2, cy: Float = Float(height)/2
        let intr = CameraIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy)

        func pose(translation: SIMD3<Float>) -> CameraExtrinsics {
            var m = matrix_identity_float4x4
            m.columns.3 = SIMD4<Float>(translation, 1)
            return CameraExtrinsics(m: [
                m[0,0], m[0,1], m[0,2], m[0,3],
                m[1,0], m[1,1], m[1,2], m[1,3],
                m[2,0], m[2,1], m[2,2], m[2,3],
                m[3,0], m[3,1], m[3,2], m[3,3]
            ])
        }

        let poses = [
            pose(translation: SIMD3<Float>(0, 0, -3)),
            pose(translation: SIMD3<Float>(1, 0.5, -3)),
            pose(translation: SIMD3<Float>(-1, -0.5, -3))
        ]

        return poses.map { Sample(image: image, intr: intr, extr: $0) }
    }
}
