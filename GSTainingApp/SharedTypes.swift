//
//  SharedTypes.swift
//  GSTainingApp
//
//  Basic shared data structures for Gaussian splats and camera parameters.
//

import Foundation
import simd

// MARK: - Gaussian Parameters (CPU & GPU memory layout compatible)

public struct GaussianParam: Codable {
    public var position: SIMD3<Float>   // world-space position (meters)
    public var sigma: Float             // isotropic stddev (meters)
    public var color: SIMD3<Float>      // RGB in [0, 1]
    public var opacity: Float           // alpha in [0, 1]

    public init(position: SIMD3<Float>, sigma: Float, color: SIMD3<Float>, opacity: Float) {
        self.position = position
        self.sigma = sigma
        self.color = color
        self.opacity = opacity
    }
}

// Matches Metal struct memory layout (float3 pads to 16B). Keep fields order.
public struct GaussianParamGPU {
    public var position: SIMD3<Float>
    public var _pad0: Float // alignment padding
    public var scale: SIMD3<Float> // anisotropic stddevs (meters)
    public var opacity: Float
    // SH L1 (existing)
    public var shR: SIMD4<Float>
    public var shG: SIMD4<Float>
    public var shB: SIMD4<Float>
    // SH L2 extra blocks (6 coeffs per channel packed into two float4 each; unused pads = 0)
    public var shR_ex1: SIMD4<Float> // (c2_0, c2_1, c2_2, c2_3)
    public var shR_ex2: SIMD4<Float> // (c2_4, c2_5, pad, pad)
    public var shG_ex1: SIMD4<Float>
    public var shG_ex2: SIMD4<Float>
    public var shB_ex1: SIMD4<Float>
    public var shB_ex2: SIMD4<Float>
}

public extension GaussianParamGPU {
    init(_ g: GaussianParam) {
        // Map isotropic sigma to anisotropic scale, and RGB color to SH DC (c0) with zero linear terms
        self.position = g.position
        self._pad0 = 0
        let s = g.sigma
        self.scale = SIMD3<Float>(repeating: s)
        self.opacity = g.opacity
        self.shR = SIMD4<Float>(g.color.x, 0, 0, 0)
        self.shG = SIMD4<Float>(g.color.y, 0, 0, 0)
        self.shB = SIMD4<Float>(g.color.z, 0, 0, 0)
        self.shR_ex1 = .zero; self.shR_ex2 = .zero
        self.shG_ex1 = .zero; self.shG_ex2 = .zero
        self.shB_ex1 = .zero; self.shB_ex2 = .zero
    }
}

// MARK: - Camera Parameters

public struct CameraIntrinsics: Codable {
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float
}

public struct CameraExtrinsics: Codable { // 4x4 pose matrix (camera-to-world)
    // Row-major 4x4
    public var m: [Float]

    public init(m: [Float]) {
        precondition(m.count == 16)
        self.m = m
    }

    public var matrix4x4: simd_float4x4 {
        simd_float4x4(rows: [
            SIMD4<Float>(m[0], m[1], m[2], m[3]),
            SIMD4<Float>(m[4], m[5], m[6], m[7]),
            SIMD4<Float>(m[8], m[9], m[10], m[11]),
            SIMD4<Float>(m[12], m[13], m[14], m[15])
        ])
    }
}

public struct FrameData: Codable {
    public var imageFilename: String
    public var intrinsics: CameraIntrinsics
    public var extrinsics: CameraExtrinsics
}

public struct DatasetSpec: Codable {
    public var frames: [FrameData]
    public var width: Int
    public var height: Int
}

// MARK: - Math Utilities

public extension simd_float4x4 {
    var inverseSafe: simd_float4x4 {
        return simd_inverse(self)
    }
}
