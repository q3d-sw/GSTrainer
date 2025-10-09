//
//  CheckpointManager.swift
//  GSTainingApp
//
//  Save/load Gaussian parameters and training state to Documents.
//

import Foundation

struct Checkpoint: Codable {
    var iteration: Int
    var gaussians: [GaussianParam]
    var learningRate: Float
    // Persist SH L1 coefficients per gaussian (RGB, each has x,y,z)
    var shR1: [SIMD3<Float>]?
    var shG1: [SIMD3<Float>]?
    var shB1: [SIMD3<Float>]?
    // Persist SH L2 coefficients (6 per channel packed as two float4 each on GPU). We store them as two SIMD4 blocks per channel for layout parity.
    var shR2_ex1: [SIMD4<Float>]? // (c2_0..c2_3)
    var shR2_ex2: [SIMD4<Float>]? // (c2_4, c2_5, 0, 0)
    var shG2_ex1: [SIMD4<Float>]? // (c2_0..c2_3)
    var shG2_ex2: [SIMD4<Float>]? // (c2_4, c2_5, 0, 0)
    var shB2_ex1: [SIMD4<Float>]? // (c2_0..c2_3)
    var shB2_ex2: [SIMD4<Float>]? // (c2_4, c2_5, 0, 0)
    // Persist anisotropic scales per gaussian (meters)
    var scaleX: [Float]?
    var scaleY: [Float]?
    var scaleZ: [Float]?
}

enum CheckpointManager {
    static func url(filename: String = "checkpoint.json") -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(filename)
    }

    static func save(_ ckpt: Checkpoint, filename: String = "checkpoint.json") {
        let url = url(filename: filename)
        do {
            let data = try JSONEncoder().encode(ckpt)
            try data.write(to: url, options: .atomic)
        } catch {
            print("Failed to save checkpoint: \(error)")
        }
    }

    static func load(filename: String = "checkpoint.json") -> Checkpoint? {
        let url = url(filename: filename)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Checkpoint.self, from: data)
    }
}
