//
//  PointsPLYParser.swift
//  GSTainingApp
//
//  Simple PLY parser (ASCII and binary_little_endian) to read vertices x,y,z and optional r,g,b and 3DGS fields.
//

import Foundation

struct PLYPoint {
    var position: SIMD3<Float>
    var color: SIMD3<Float>?
    var opacity: Float? // if present (3DGS), in [0,1]
    var sigma: Float?   // approximate isotropic scale if present
}

enum PointsPLYParser {
    static func loadFromDocuments(filename: String = "points.ply", limit: Int? = nil) -> [PLYPoint]? {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)
        return load(url: url, limit: limit)
    }

    static func load(url: URL, limit: Int? = nil) -> [PLYPoint]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
    // Detect ASCII vs binary_little_endian
        guard let headerEndRange = data.range(of: Data("end_header\n".utf8)) else { return nil }
        let headerData = data.subdata(in: 0..<headerEndRange.upperBound)
        guard let headerStr = String(data: headerData, encoding: .utf8) else { return nil }

        var isBinary = false
        var vertexCount = 0
    var hasRGB = false
    var has3DGS = false
        var propertyOrder: [String] = []

        for line in headerStr.split(separator: "\n") {
            let s = line.trimmingCharacters(in: .whitespaces)
            if s.hasPrefix("format") {
                if s.contains("binary_little_endian") { isBinary = true }
            } else if s.hasPrefix("element vertex") {
                let comps = s.split(separator: " ")
                if comps.count >= 3 { vertexCount = Int(comps[2]) ?? 0 }
            } else if s.hasPrefix("property") {
                let comps = s.split(separator: " ")
                if comps.count >= 3 {
                    propertyOrder.append(String(comps.last!))
                }
            }
        }

    hasRGB = propertyOrder.contains("red") && propertyOrder.contains("green") && propertyOrder.contains("blue")
    // 3D Gaussian Splat common fields
    let hasFDC = propertyOrder.contains("f_dc_0") && propertyOrder.contains("f_dc_1") && propertyOrder.contains("f_dc_2")
    let hasOpacity = propertyOrder.contains("opacity")
    let hasScale = propertyOrder.contains("scale_0") && propertyOrder.contains("scale_1") && propertyOrder.contains("scale_2")
    has3DGS = hasFDC && hasOpacity && hasScale

        let maxCount = limit ?? vertexCount
        var points: [PLYPoint] = []
        points.reserveCapacity(min(maxCount, vertexCount))

        if !isBinary {
            // ASCII
            guard let bodyStr = String(data: data.subdata(in: headerEndRange.upperBound..<data.endIndex), encoding: .utf8) else { return nil }
            var parsed = 0
            for line in bodyStr.split(separator: "\n") {
                if parsed >= maxCount { break }
                let comps = line.split(separator: " ").map { String($0) }
                if comps.count < 3 { continue }
                guard let x = Float(comps[safe: propertyOrder.firstIndex(of: "x") ?? 0] ?? comps[0]),
                      let y = Float(comps[safe: propertyOrder.firstIndex(of: "y") ?? 1] ?? comps[1]),
                      let z = Float(comps[safe: propertyOrder.firstIndex(of: "z") ?? 2] ?? comps[2]) else { continue }
                var color: SIMD3<Float>? = nil
                var opacity: Float? = nil
                var sigma: Float? = nil
                if has3DGS {
                    let SH_C0: Float = 0.28209479177387814
                    if let i0 = propertyOrder.firstIndex(of: "f_dc_0"),
                       let i1 = propertyOrder.firstIndex(of: "f_dc_1"),
                       let i2 = propertyOrder.firstIndex(of: "f_dc_2"),
                       i0 < comps.count, i1 < comps.count, i2 < comps.count {
                        let r = (Float(comps[i0]) ?? 0) * SH_C0
                        let g = (Float(comps[i1]) ?? 0) * SH_C0
                        let b = (Float(comps[i2]) ?? 0) * SH_C0
                        color = SIMD3<Float>(max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
                    }
                    if let io = propertyOrder.firstIndex(of: "opacity"), io < comps.count {
                        let logit = Float(comps[io]) ?? 0
                        opacity = 1.0 / (1.0 + exp(-logit))
                    }
                    if let is0 = propertyOrder.firstIndex(of: "scale_0"),
                       let is1 = propertyOrder.firstIndex(of: "scale_1"),
                       let is2 = propertyOrder.firstIndex(of: "scale_2"),
                       is0 < comps.count, is1 < comps.count, is2 < comps.count {
                        let s0 = Float(comps[is0]) ?? 0.01
                        let s1 = Float(comps[is1]) ?? 0.01
                        let s2 = Float(comps[is2]) ?? 0.01
                        sigma = max(0.001, min(0.2, (s0 + s1 + s2) / 3.0))
                    }
                } else if hasRGB {
                    if let rIdx = propertyOrder.firstIndex(of: "red"),
                       let gIdx = propertyOrder.firstIndex(of: "green"),
                       let bIdx = propertyOrder.firstIndex(of: "blue"),
                       rIdx < comps.count, gIdx < comps.count, bIdx < comps.count,
                       let r = Float(comps[rIdx]), let g = Float(comps[gIdx]), let b = Float(comps[bIdx]) {
                        color = SIMD3<Float>(r/255.0, g/255.0, b/255.0)
                    }
                }
                points.append(PLYPoint(position: SIMD3<Float>(x,y,z), color: color, opacity: opacity, sigma: sigma))
                parsed += 1
            }
        } else {
            // binary little endian: read per propertyOrder (support float and uchar for rgb)
            var offset = headerEndRange.upperBound
            func readFloat() -> Float? {
                guard offset + 4 <= data.count else { return nil }
                let val: Float = data.withUnsafeBytes { raw -> Float in
                    let p = raw.baseAddress!.advanced(by: offset).assumingMemoryBound(to: UInt32.self)
                    let bitsLE = UInt32(littleEndian: p.pointee)
                    let f32 = Float32(bitPattern: bitsLE)
                    return Float(f32)
                }
                offset += 4
                return val
            }
            func readUChar() -> UInt8? {
                guard offset + 1 <= data.count else { return nil }
                let val = data[offset]
                offset += 1
                return val
            }
            let n = min(vertexCount, maxCount)
            for _ in 0..<n {
                var x: Float = 0, y: Float = 0, z: Float = 0
                var col: SIMD3<Float>? = nil
                var op: Float? = nil
                var sig: Float? = nil
                for prop in propertyOrder {
                    switch prop {
                    case "x": if let v = readFloat() { x = v }
                    case "y": if let v = readFloat() { y = v }
                    case "z": if let v = readFloat() { z = v }
                    case "red": if let r = readUChar() { if col == nil { col = SIMD3<Float>(0,0,0) }; col!.x = Float(r)/255.0 }
                    case "green": if let g = readUChar() { if col == nil { col = SIMD3<Float>(0,0,0) }; col!.y = Float(g)/255.0 }
                    case "blue": if let b = readUChar() { if col == nil { col = SIMD3<Float>(0,0,0) }; col!.z = Float(b)/255.0 }
                    case "f_dc_0":
                        if let r = readFloat() {
                            let SH_C0: Float = 0.28209479177387814
                            if col == nil { col = SIMD3<Float>(0,0,0) }
                            col!.x = max(0, min(1, r * SH_C0))
                        }
                    case "f_dc_1":
                        if let g = readFloat() {
                            let SH_C0: Float = 0.28209479177387814
                            if col == nil { col = SIMD3<Float>(0,0,0) }
                            col!.y = max(0, min(1, g * SH_C0))
                        }
                    case "f_dc_2":
                        if let b = readFloat() {
                            let SH_C0: Float = 0.28209479177387814
                            if col == nil { col = SIMD3<Float>(0,0,0) }
                            col!.z = max(0, min(1, b * SH_C0))
                        }
                    case "opacity":
                        if let v = readFloat() { op = 1.0 / (1.0 + exp(-v)) }
                    case "scale_0": if let v = readFloat() { sig = (sig ?? 0) + max(0.001, min(0.2, v)) }
                    case "scale_1": if let v = readFloat() { sig = (sig ?? 0) + max(0.001, min(0.2, v)) }
                    case "scale_2": if let v = readFloat() { sig = (sig ?? 0) + max(0.001, min(0.2, v)) }
                    default:
                        _ = readFloat()
                    }
                }
                if let s = sig { sig = max(0.001, min(0.2, s / 3.0)) }
                points.append(PLYPoint(position: SIMD3<Float>(x,y,z), color: col, opacity: op, sigma: sig))
            }
        }

        return points
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? { indices.contains(index) ? self[index] : nil }
}
