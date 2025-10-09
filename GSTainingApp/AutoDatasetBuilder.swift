//
//  AutoDatasetBuilder.swift
//  GSTainingApp
//
//  Auto-generate dataset.json from a set of photos in Documents/images.
//

import Foundation
import UIKit
import ImageIO
import simd

enum AutoDatasetBuilder {
    struct BuildResult {
        let frames: Int
        let width: Int
        let height: Int
        let usedEXIF: Bool
        let outputURL: URL
    }

    static func build(imagesSubfolder: String = "images", defaultFOVDegrees: Float = 60, radiusMeters: Float = 3.0) throws -> BuildResult {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let imagesURL = docs.appendingPathComponent(imagesSubfolder, isDirectory: true)
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: imagesURL, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]), !items.isEmpty else {
            throw NSError(domain: "AutoDatasetBuilder", code: 1, userInfo: [NSLocalizedDescriptionKey: "No images found in Documents/\(imagesSubfolder)."])
        }

    // Sort by name and keep only image files
        let exts = Set(["png","jpg","jpeg","heic"]) 
        let imageURLs: [URL] = items.compactMap { url in
            let ext = url.pathExtension.lowercased()
            guard exts.contains(ext) else { return nil }
            do {
                let vals = try url.resourceValues(forKeys: [.isRegularFileKey])
                if vals.isRegularFile == true { return url }
            } catch {}
            return nil
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        // Find the first image that actually opens
        guard let firstURL = imageURLs.first(where: { UIImage(contentsOfFile: $0.path) != nil }) else {
            throw NSError(domain: "AutoDatasetBuilder", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to read any image from folder."])
        }
        guard let firstImg = UIImage(contentsOfFile: firstURL.path)?.cgImage else {
            throw NSError(domain: "AutoDatasetBuilder", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to get CGImage of the first image."])
        }
        let width = firstImg.width
        let height = firstImg.height

    // Try to extract EXIF: FocalLength or FocalLenIn35mmFilm
        var usedEXIF = false
        var fx: Float = 0, fy: Float = 0
    if let src = CGImageSourceCreateWithURL(firstURL as CFURL, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
           let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] {
            if let fl35 = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? Double {
                // Estimate horizontal FOV via 35mm equivalent: FOVx = 2*atan(36mm/(2*fl35))
                let fovx = 2.0 * atan(36.0 / (2.0 * fl35))
                let fxPx = (Double(width) / 2.0) / tan(fovx / 2.0)
                fx = Float(fxPx)
                // fy by aspect ratio from vertical FOV (assume 24mm for 35mm film height)
                let fovy = 2.0 * atan(24.0 / (2.0 * fl35))
                let fyPx = (Double(height) / 2.0) / tan(fovy / 2.0)
                fy = Float(fyPx)
                usedEXIF = true
            }
        }
        if fx == 0 || fy == 0 {
            // Default FOV
            let fovx = Double(defaultFOVDegrees) * .pi / 180.0
            fx = Float((Double(width) / 2.0) / tan(fovx / 2.0))
            // Keep fx/fy equal by default
            fy = fx
        }
        let cx = Float(width) / 2
        let cy = Float(height) / 2

    // Extrinsics: simple circle around the scene at y=0 with angle step
        let n = imageURLs.count
        var frames: [FrameData] = []
        frames.reserveCapacity(n)
        for (idx, url) in imageURLs.enumerated() {
            let theta = 2.0 * Float.pi * Float(idx) / Float(max(1, n))
            // Camera position: around Y-axis, facing -Z, similar to synthetic dataset
            let x = radiusMeters * sin(theta)
            let z = -radiusMeters * cos(theta)
            var m = matrix_identity_float4x4
            m.columns.3 = SIMD4<Float>(x, 0, z, 1)
            let extr = CameraExtrinsics(m: [
                m[0,0], m[0,1], m[0,2], m[0,3],
                m[1,0], m[1,1], m[1,2], m[1,3],
                m[2,0], m[2,1], m[2,2], m[2,3],
                m[3,0], m[3,1], m[3,2], m[3,3]
            ])
            let intr = CameraIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy)
            // Path in dataset.json should be relative to Documents, hence imagesSubfolder/filename
            let relPath = imagesSubfolder + "/" + url.lastPathComponent
            frames.append(FrameData(imageFilename: relPath, intrinsics: intr, extrinsics: extr))
        }

        let spec = DatasetSpec(frames: frames, width: width, height: height)
        let outURL = docs.appendingPathComponent("dataset.json")
        let data = try JSONEncoder().encode(spec)
        try data.write(to: outURL, options: .atomic)
        return BuildResult(frames: n, width: width, height: height, usedEXIF: usedEXIF, outputURL: outURL)
    }
}
