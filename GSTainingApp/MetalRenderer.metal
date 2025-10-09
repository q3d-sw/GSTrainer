#include <metal_stdlib>
using namespace metal;

struct GaussianParamGPU {
    float3 position; float _pad0;
    float3 scale;    float opacity;
    float4 shR;  // L1
    float4 shG;
    float4 shB;
    float4 shR_ex1; // L2 extra (c2_0..c2_3)
    float4 shR_ex2; // L2 extra (c2_4, c2_5, pad, pad)
    float4 shG_ex1;
    float4 shG_ex2;
    float4 shB_ex1;
    float4 shB_ex2;
};

struct SHModeUniform { uint enableL2; };

struct RenderUniforms {
    float4x4 worldToCam; // column-major (Metal default)
    float fx, fy, cx, cy;
    float zSign; // +1 => +Z forward, -1 => -Z forward
    uint imageWidth;
    uint imageHeight;
    uint gaussianCount;
    float pointScale;    // pixels per meter for sprite size (approx)
    uint tilesX;
    uint tilesY;
    uint tileSize;
};

// Simple splat renderer: for each pixel, accumulate contributions of all gaussians.
// Note: This is O(W*H*N) and meant as a baseline. For real-time, tile/bin or splat per-gaussian.
// FP16 write; math in fp32.

kernel void renderGaussians(
    constant GaussianParamGPU *gaussians [[buffer(0)]],
    constant RenderUniforms &uniforms [[buffer(1)]],
    constant uint *tileOffsets [[buffer(2)]],
    constant uint *tileList [[buffer(3)]],
    constant SHModeUniform &shMode [[buffer(4)]],
    texture2d<half, access::write> outTex [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uniforms.imageWidth || gid.y >= uniforms.imageHeight) return;

    float2 pixel = float2(gid.x + 0.5, gid.y + 0.5);

    float3 accumRGB = float3(0.0);
    float accumA = 0.0;

    uint tx = gid.x / uniforms.tileSize;
    uint ty = gid.y / uniforms.tileSize;
    uint tileId = ty * uniforms.tilesX + tx;
    uint start = tileOffsets[tileId];
    uint end = tileOffsets[tileId + 1];
    for (uint idx = start; idx < end; ++idx) {
        uint i = tileList[idx];
        GaussianParamGPU g = gaussians[i];
        float4 p = float4(g.position, 1.0);
        float4 cam4 = uniforms.worldToCam * p; // column-major multiply
        float3 cam = cam4.xyz;
        // Camera forward along zSign*Z: zf = zSign*(-cam.z) when zSign=-1 (OpenGL), or zf=cam.z when zSign=+1
        float zf = (uniforms.zSign < 0.0) ? (-cam.z) : (cam.z);
        if (zf <= 0.0) continue;
        // Project using intrinsics
        float2 uv = float2(
            uniforms.fx * (cam.x / zf) + uniforms.cx,
            uniforms.fy * (cam.y / zf) + uniforms.cy
        );

        // Evaluate view-dependent color
        float3 viewDir = normalize(-cam);
        float x = viewDir.x, y = viewDir.y, z = viewDir.z;
        float4 basis = float4(1.0, x, y, z); // L1
        float3 color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
        if (shMode.enableL2 != 0) {
            float4 ex1 = float4(1.092548f * x * y, 1.092548f * y * z, 0.315392f * (3.0f * z * z - 1.0f), 1.092548f * x * z);
            float4 ex2 = float4(0.546274f * (x * x - y * y), 0.0, 0.0, 0.0);
            color += float3(dot(g.shR_ex1, ex1) + dot(g.shR_ex2, ex2),
                            dot(g.shG_ex1, ex1) + dot(g.shG_ex2, ex2),
                            dot(g.shB_ex1, ex1) + dot(g.shB_ex2, ex2));
        }
        color = clamp(color, 0.0, 1.0);

        // Anisotropic footprint: project 3D diag covariance to 2D via Jacobian J (2x3)
        // Σ3D = diag(sx^2, sy^2, sz^2)
        float sx2 = g.scale.x * g.scale.x;
        float sy2 = g.scale.y * g.scale.y;
        float sz2 = g.scale.z * g.scale.z;

        // Projection Jacobian at (x,y,z):
        // u = fx * x/z + cx; v = fy * y/z + cy
        // du/dx = fx/z; du/dy = 0; du/dz = -fx*x/z^2
        // dv/dx = 0; dv/dy = fy/z; dv/dz = -fy*y/z^2
        float invz = 1.0 / zf;
        float invz2 = invz * invz;
        float j00 = uniforms.fx * invz;       float j01 = 0.0;                  float j02 = -uniforms.fx * cam.x * invz2;
        float j10 = 0.0;                       float j11 = uniforms.fy * invz;   float j12 = -uniforms.fy * cam.y * invz2;

        // Σ2D = J Σ3D J^T for diagonal Σ3D: expand explicitly
        float sxx = j00*j00*sx2 + j01*j01*sy2 + j02*j02*sz2;
        float sxy = j00*j10*sx2 + j01*j11*sy2 + j02*j12*sz2;
        float syy = j10*j10*sx2 + j11*j11*sy2 + j12*j12*sz2;

        // Inverse of 2x2
        float det = max(1e-8, (sxx*syy - sxy*sxy));
        float inv00 =  syy / det;
        float inv01 = -sxy / det;
        float inv11 =  sxx / det;

        // Evaluate anisotropic Gaussian weight w = exp(-0.5 * d^T Σ2D^{-1} d)
        float2 d = pixel - uv;
        float qx = inv00*d.x + inv01*d.y;
        float qy = inv01*d.x + inv11*d.y;
        float mahal = d.x*qx + d.y*qy;
        float w = exp(-0.5 * mahal);
        float a = clamp(g.opacity * w, 0.0, 1.0);

        // Pre-multiplied alpha blending front-to-back (no depth sort in this baseline)
        float oneMinusA = (1.0 - accumA);
        accumRGB += oneMinusA * a * color;
        accumA += oneMinusA * a;

        if (accumA > 0.995) break; // early exit if almost opaque
    }
    half4 outColor = half4((half3)accumRGB, (half)accumA);
    outTex.write(outColor, gid);
}

// Tiled kernel: one threadgroup per tile, cooperative compositing with shared alpha early-out.
// Threadgroup size should match tileSize x tileSize (or a divisor) launched over (tilesX, tilesY).
kernel void tiledRenderGaussians(
    constant GaussianParamGPU *gaussians [[buffer(0)]],
    constant RenderUniforms &uniforms [[buffer(1)]],
    constant uint *tileOffsets [[buffer(2)]],
    constant uint *tileList [[buffer(3)]],
    constant SHModeUniform &shMode [[buffer(4)]],
    texture2d<half, access::write> outTex [[texture(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint  flatTid [[thread_index_in_threadgroup]],
    threadgroup float *tgAlpha) {
    if (tgid.x >= uniforms.tilesX || tgid.y >= uniforms.tilesY) return;
    // Compute pixel within image
    uint baseX = tgid.x * uniforms.tileSize;
    uint baseY = tgid.y * uniforms.tileSize;
    uint px = baseX + tid2.x;
    uint py = baseY + tid2.y;
    if (px >= uniforms.imageWidth || py >= uniforms.imageHeight) {
        return;
    }
    const uint tileId = tgid.y * uniforms.tilesX + tgid.x;
    uint start = tileOffsets[tileId];
    uint end = tileOffsets[tileId + 1];
    float2 pixel = float2(float(px) + 0.5, float(py) + 0.5);
    float3 accumRGB = float3(0.0);
    float accumA = 0.0;
    // First thread initializes shared alpha
    if (flatTid == 0) *tgAlpha = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = start; idx < end; ++idx) {
        // Tile-wide early out flag
        if (*tgAlpha > 0.995f) break;
        uint gi = tileList[idx];
        GaussianParamGPU g = gaussians[gi];
        float4 cam4 = uniforms.worldToCam * float4(g.position, 1.0);
        float3 cam = cam4.xyz;
        float zf = (uniforms.zSign < 0.0) ? (-cam.z) : (cam.z);
        if (zf <= 0.0) continue;
        float2 uv = float2(uniforms.fx * (cam.x / zf) + uniforms.cx,
                           uniforms.fy * (cam.y / zf) + uniforms.cy);
        // SH color (L1 + optional L2)
        float3 viewDir = normalize(-cam);
        float x = viewDir.x, y = viewDir.y, z = viewDir.z;
        float4 basis = float4(1.0, x, y, z);
        float3 color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
        if (shMode.enableL2 != 0) {
            float4 ex1 = float4(1.092548f * x * y, 1.092548f * y * z, 0.315392f * (3.0f * z * z - 1.0f), 1.092548f * x * z);
            float4 ex2 = float4(0.546274f * (x * x - y * y), 0.0, 0.0, 0.0);
            color += float3(dot(g.shR_ex1, ex1) + dot(g.shR_ex2, ex2),
                            dot(g.shG_ex1, ex1) + dot(g.shG_ex2, ex2),
                            dot(g.shB_ex1, ex1) + dot(g.shB_ex2, ex2));
        }
        color = clamp(color, 0.0, 1.0);
        // Anisotropic projection
        float sx2 = g.scale.x * g.scale.x;
        float sy2 = g.scale.y * g.scale.y;
        float sz2 = g.scale.z * g.scale.z;
        float invz = 1.0 / zf;
        float invz2 = invz * invz;
        float j00 = uniforms.fx * invz;       float j01 = 0.0;                  float j02 = -uniforms.fx * cam.x * invz2;
        float j10 = 0.0;                      float j11 = uniforms.fy * invz;   float j12 = -uniforms.fy * cam.y * invz2;
        float sxx = j00*j00*sx2 + j01*j01*sy2 + j02*j02*sz2;
        float sxy = j00*j10*sx2 + j01*j11*sy2 + j02*j12*sz2;
        float syy = j10*j10*sx2 + j11*j11*sy2 + j12*j12*sz2;
        float det = max(1e-8, (sxx*syy - sxy*sxy));
        float inv00 =  syy / det;
        float inv01 = -sxy / det;
        float inv11 =  sxx / det;
        float2 d = pixel - uv;
        float qx = inv00*d.x + inv01*d.y;
        float qy = inv01*d.x + inv11*d.y;
        float mahal = d.x*qx + d.y*qy;
        float w = exp(-0.5 * mahal);
        float a = clamp(g.opacity * w, 0.0, 1.0);
        float oneMinusA = (1.0 - accumA);
        accumRGB += oneMinusA * a * color;
        accumA += oneMinusA * a;
        if (accumA > 0.995) { *tgAlpha = 1.0; break; }
    }
    outTex.write(half4((half3)accumRGB, (half)accumA), uint2(px, py));
}

// Backward pass: accumulate dL/d(c0_R,G,B) and dL/d(opacity) using residual image (pred - gt) in linear RGB.
// Note: This is a simplified example that ignores blending derivatives and uses current weight w as contribution proxy.
struct GradUniforms {
    float4x4 worldToCam;
    float fx, fy, cx, cy;
    float zSign;
    uint imageWidth;
    uint imageHeight;
    uint gaussianCount;
    uint tilesX;
    uint tilesY;
    uint tileSize;
};

kernel void accumulateGradients(
    constant GaussianParamGPU *gaussians [[buffer(0)]],
    constant GradUniforms &uni [[buffer(1)]],
    texture2d<half, access::sample> residualTex [[texture(0)]],
    device atomic_int *gradC0R [[buffer(2)]],
    device atomic_int *gradC0G [[buffer(3)]],
    device atomic_int *gradC0B [[buffer(4)]],
    device atomic_int *gradR1x [[buffer(5)]],
    device atomic_int *gradR1y [[buffer(6)]],
    device atomic_int *gradR1z [[buffer(7)]],
    device atomic_int *gradG1x [[buffer(8)]],
    device atomic_int *gradG1y [[buffer(9)]],
    device atomic_int *gradG1z [[buffer(10)]],
    device atomic_int *gradB1x [[buffer(11)]],
    device atomic_int *gradB1y [[buffer(12)]],
    device atomic_int *gradB1z [[buffer(13)]],
    device atomic_int *gradR2 [[buffer(14)]], // packed: coeff blocks of size N (0..5)
    device atomic_int *gradG2 [[buffer(15)]],
    device atomic_int *gradB2 [[buffer(16)]],
    device atomic_int *gradOpacity [[buffer(17)]],
    device atomic_int *gradSigma [[buffer(18)]],
    device atomic_int *gradPosX [[buffer(19)]],
    device atomic_int *gradPosY [[buffer(20)]],
    device atomic_int *gradPosZ [[buffer(21)]],
    device atomic_int *gradScaleX [[buffer(22)]],
    device atomic_int *gradScaleY [[buffer(23)]],
    device atomic_int *gradScaleZ [[buffer(24)]],
    constant uint *tileOffsets [[buffer(25)]],
    constant uint *tileList [[buffer(26)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uni.imageWidth || gid.y >= uni.imageHeight) return;
    constexpr sampler s(coord::pixel, filter::nearest);
    float3 r = float3(residualTex.sample(s, (float2(gid) + 0.5) / float2(uni.imageWidth, uni.imageHeight)).rgb);

    float2 pixel = float2(gid) + 0.5;

    // Pass 1: compute final accumulated RGB using simplified path with SH color.
    float3 finalRGB = float3(0.0);
    float accumA_final = 0.0;
    uint tx = gid.x / uni.tileSize;
    uint ty = gid.y / uni.tileSize;
    uint tileId = ty * uni.tilesX + tx;
    uint start = tileOffsets[tileId];
    uint end = tileOffsets[tileId + 1];
    for (uint it = start; it < end; ++it) {
        uint i = tileList[it];
        GaussianParamGPU g = gaussians[i];
        float4 cam4 = uni.worldToCam * float4(g.position, 1.0);
        float3 cam = cam4.xyz;
        float zf = (uni.zSign < 0.0) ? (-cam.z) : (cam.z);
        if (zf <= 0.0) continue;
        float2 uv = float2(uni.fx * (cam.x / zf) + uni.cx,
                           uni.fy * (cam.y / zf) + uni.cy);

        // Approximate anisotropic weight using isotropic proxy (fast path)
        float invz = 1.0 / zf;

        float s_world = (g.scale.x + g.scale.y + g.scale.z) / 3.0;
        float K = max(uni.fx, uni.fy) * invz * 0.5; // screen sigma per world sigma
        float screenSigma = max(1.0, s_world) * K;
        float2 d = pixel - uv;
        float r2 = dot(d,d);
        // culling: skip very far pixels beyond 3 sigma (avoid exp)
        float sig2 = screenSigma * screenSigma;
        if (r2 > 9.0 * sig2) {
            continue;
        }
        float w = exp(-0.5 * r2 / sig2);
        float a = clamp(g.opacity * w, 0.0, 1.0);
        // SH color
        float3 vdir = normalize(-cam);
        float4 basis = float4(1.0, vdir.x, vdir.y, vdir.z);
        float3 color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
        color = clamp(color, 0.0, 1.0);
        float T = (1.0 - accumA_final);
        finalRGB += T * a * color;
        accumA_final += T * a;
        if (accumA_final > 0.995) break;
    }

    // Pass 2: accumulate gradients using downstream contribution approximation.
    float accumA = 0.0;
    float3 accumRGB_partial = float3(0.0);
    // Pass 2 tiled loop
    tx = gid.x / uni.tileSize;
    ty = gid.y / uni.tileSize;
    tileId = ty * uni.tilesX + tx;
    start = tileOffsets[tileId];
    end = tileOffsets[tileId + 1];
    for (uint it2 = start; it2 < end; ++it2) {
        uint i = tileList[it2];
        GaussianParamGPU g = gaussians[i];
        float4 cam4 = uni.worldToCam * float4(g.position, 1.0);
        float3 cam = cam4.xyz;
        float zf = (uni.zSign < 0.0) ? (-cam.z) : (cam.z);
        if (zf <= 0.0) continue;
        float2 uv = float2(uni.fx * (cam.x / zf) + uni.cx,
                           uni.fy * (cam.y / zf) + uni.cy);

        float invz = 1.0 / zf;
        float2 d = pixel - uv;
        // Isotropic proxy for compositing path (used to form T,a and downstream)
        float s_world_iso = (g.scale.x + g.scale.y + g.scale.z) / 3.0;
        float Kiso = max(uni.fx, uni.fy) * invz * 0.5;
        float screenSigma = max(1.0, s_world_iso) * Kiso;
        float r2 = dot(d,d);
        float sig2 = screenSigma * screenSigma;
        if (r2 > 9.0 * sig2) {
            continue;
        }
        float w_iso = exp(-0.5 * r2 / sig2);
        float a = clamp(g.opacity * w_iso, 0.0, 1.0);
        float T = (1.0 - accumA);
        // SH basis and color (L1 only for gradient path initially; we treat L2 similarly below)
        float3 vdir = normalize(-cam);
        float bx = vdir.x, by = vdir.y, bz = vdir.z;
        float4 basis = float4(1.0, bx, by, bz);
        float3 color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
        // Optional L2: evaluate for color & gradient contributions
    float l2vals[6];
        // Precompute L2 real SH basis (same ordering as packing) if any coeffs are non-zero (heuristic)
        bool anyL2 = (g.shR_ex1.x != 0.0 || g.shR_ex1.y != 0.0 || g.shR_ex1.z != 0.0 || g.shR_ex1.w != 0.0 || g.shR_ex2.x != 0.0 || g.shR_ex2.y != 0.0 ||
                      g.shG_ex1.x != 0.0 || g.shG_ex1.y != 0.0 || g.shG_ex1.z != 0.0 || g.shG_ex1.w != 0.0 || g.shG_ex2.x != 0.0 || g.shG_ex2.y != 0.0 ||
                      g.shB_ex1.x != 0.0 || g.shB_ex1.y != 0.0 || g.shB_ex1.z != 0.0 || g.shB_ex1.w != 0.0 || g.shB_ex2.x != 0.0 || g.shB_ex2.y != 0.0);
        if (anyL2) {
            // Basis order used in forward: (xy, yz, 3z^2-1, xz, x^2-y^2, <pad>)
            float xy = bx * by;
            float yz = by * bz;
            float xz = bx * bz;
            float z2 = bz * bz;
            l2vals[0] = 1.092548f * xy;           // m=-2? (xy)
            l2vals[1] = 1.092548f * yz;           // (yz)
            l2vals[2] = 0.315392f * (3.0f * z2 - 1.0f); // (3z^2-1)
            l2vals[3] = 1.092548f * xz;           // (xz)
            l2vals[4] = 0.546274f * (bx * bx - by * by); // (x^2 - y^2)
            l2vals[5] = 0.0f; // padding / unused
            color += float3(
                g.shR_ex1.x * l2vals[0] + g.shR_ex1.y * l2vals[1] + g.shR_ex1.z * l2vals[2] + g.shR_ex1.w * l2vals[3] + g.shR_ex2.x * l2vals[4] + g.shR_ex2.y * l2vals[5],
                g.shG_ex1.x * l2vals[0] + g.shG_ex1.y * l2vals[1] + g.shG_ex1.z * l2vals[2] + g.shG_ex1.w * l2vals[3] + g.shG_ex2.x * l2vals[4] + g.shG_ex2.y * l2vals[5],
                g.shB_ex1.x * l2vals[0] + g.shB_ex1.y * l2vals[1] + g.shB_ex1.z * l2vals[2] + g.shB_ex1.w * l2vals[3] + g.shB_ex2.x * l2vals[4] + g.shB_ex2.y * l2vals[5]
            );
        }
        color = clamp(color, 0.0, 1.0);
        float3 cur = T * a * color;
        float3 downstream = finalRGB - (accumRGB_partial + cur);

        // Fixed-point scale for atomics
        const float scale = 1024.0f;

    // Color DC grads: r * T * a
        int gc0r = (int)clamp(r.x * (T * a) * scale, -1e9f, 1e9f);
        int gc0g = (int)clamp(r.y * (T * a) * scale, -1e9f, 1e9f);
        int gc0b = (int)clamp(r.z * (T * a) * scale, -1e9f, 1e9f);
        // SH L1 grads: multiply by basis (bx,by,bz)
        int gr1x = (int)clamp(r.x * (T * a) * bx * scale, -1e9f, 1e9f);
        int gr1y = (int)clamp(r.x * (T * a) * by * scale, -1e9f, 1e9f);
        int gr1z = (int)clamp(r.x * (T * a) * bz * scale, -1e9f, 1e9f);
        int gg1x = (int)clamp(r.y * (T * a) * bx * scale, -1e9f, 1e9f);
        int gg1y = (int)clamp(r.y * (T * a) * by * scale, -1e9f, 1e9f);
        int gg1z = (int)clamp(r.y * (T * a) * bz * scale, -1e9f, 1e9f);
        int gb1x = (int)clamp(r.z * (T * a) * bx * scale, -1e9f, 1e9f);
        int gb1y = (int)clamp(r.z * (T * a) * by * scale, -1e9f, 1e9f);
        int gb1z = (int)clamp(r.z * (T * a) * bz * scale, -1e9f, 1e9f);

        // Opacity gradient via full chain: dL/da = r · (T*color - downstream); da/d(opacity)=w
    float dL_da = dot(r, (T * color - downstream));
    int gop = (int)clamp((dL_da * w_iso) * scale, -1e9f, 1e9f);

    // Sigma gradient via chain: dL/dw = dL/da * d a/d w = dL/da * opacity
    // dw/ds_screen = w * (r^2 / s_screen^3), ds_screen/ds_world = K
    float dLdw = dL_da * g.opacity;
    // Exact anisotropic scale derivatives via Σ2D = J Σ3D J^T and A = Σ2D^{-1}
    float sx2 = g.scale.x * g.scale.x;
    float sy2 = g.scale.y * g.scale.y;
    float sz2 = g.scale.z * g.scale.z;
    // Jacobian terms
    float invz2 = invz * invz;
    float j00 = uni.fx * invz;       float j01 = 0.0;                  float j02 = -uni.fx * cam.x * invz2;
    float j10 = 0.0;                  float j11 = uni.fy * invz;        float j12 = -uni.fy * cam.y * invz2;
    // Σ2D entries
    float sxx = j00*j00*sx2 + j01*j01*sy2 + j02*j02*sz2;
    float sxy = j00*j10*sx2 + j01*j11*sy2 + j02*j12*sz2;
    float syy = j10*j10*sx2 + j11*j11*sy2 + j12*j12*sz2;
    float det = max(1e-8, (sxx*syy - sxy*sxy));
    float a00 =  syy / det;
    float a01 = -sxy / det;
    float a11 =  sxx / det;
    // u = A d
    float2 u = float2(a00*d.x + a01*d.y, a01*d.x + a11*d.y);
    // t_k = j_k^T u
    float t0 = j00 * u.x + j10 * u.y;
    float t1 = j01 * u.x + j11 * u.y;
    float t2 = j02 * u.x + j12 * u.y;
    // w_aniso for chain
    float mahalA = d.x * u.x + d.y * u.y;
    float w_aniso = exp(-0.5 * mahalA);
    float dsx = dLdw * (w_aniso * g.scale.x * (t0 * t0));
    float dsy = dLdw * (w_aniso * g.scale.y * (t1 * t1));
    float dsz = dLdw * (w_aniso * g.scale.z * (t2 * t2));
    // Also accumulate isotropic sigma gradient using proxy for legacy sigma
    float dLds_world = dLdw * (w_iso * (r2 / (screenSigma*screenSigma*screenSigma)) * Kiso);
    int gs = (int)clamp(dLds_world * scale, -1e9f, 1e9f);

    // Position gradient (world), isotropic approx: ∂w/∂cam = J^T * (w * d / sig2)
    float2 Pd = d / max(1e-8, sig2);
    float3 dwdcam = float3(j00*Pd.x + j10*Pd.y,
                   j01*Pd.x + j11*Pd.y,
                   j02*Pd.x + j12*Pd.y) * w_iso;
    float3 rc0 = float3(uni.worldToCam[0][0], uni.worldToCam[0][1], uni.worldToCam[0][2]);
    float3 rc1 = float3(uni.worldToCam[1][0], uni.worldToCam[1][1], uni.worldToCam[1][2]);
    float3 rc2 = float3(uni.worldToCam[2][0], uni.worldToCam[2][1], uni.worldToCam[2][2]);
    float3 gpos = float3(dot(rc0, dwdcam), dot(rc1, dwdcam), dot(rc2, dwdcam)) * dLdw;

    // dsx,dsy,dsz computed exactly above

        atomic_fetch_add_explicit(&gradC0R[i], gc0r, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradC0G[i], gc0g, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradC0B[i], gc0b, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradR1x[i], gr1x, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradR1y[i], gr1y, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradR1z[i], gr1z, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradG1x[i], gg1x, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradG1y[i], gg1y, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradG1z[i], gg1z, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradB1x[i], gb1x, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradB1y[i], gb1y, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradB1z[i], gb1z, memory_order_relaxed);
        // SH L2 gradients if any (per coefficient buffers)
        if (anyL2) {
            int gr2_0 = (int)clamp(r.x * (T * a) * l2vals[0] * scale, -1e9f, 1e9f);
            int gr2_1 = (int)clamp(r.x * (T * a) * l2vals[1] * scale, -1e9f, 1e9f);
            int gr2_2 = (int)clamp(r.x * (T * a) * l2vals[2] * scale, -1e9f, 1e9f);
            int gr2_3 = (int)clamp(r.x * (T * a) * l2vals[3] * scale, -1e9f, 1e9f);
            int gr2_4 = (int)clamp(r.x * (T * a) * l2vals[4] * scale, -1e9f, 1e9f);
            int gr2_5 = (int)clamp(r.x * (T * a) * l2vals[5] * scale, -1e9f, 1e9f);
            int gg2_0 = (int)clamp(r.y * (T * a) * l2vals[0] * scale, -1e9f, 1e9f);
            int gg2_1 = (int)clamp(r.y * (T * a) * l2vals[1] * scale, -1e9f, 1e9f);
            int gg2_2 = (int)clamp(r.y * (T * a) * l2vals[2] * scale, -1e9f, 1e9f);
            int gg2_3 = (int)clamp(r.y * (T * a) * l2vals[3] * scale, -1e9f, 1e9f);
            int gg2_4 = (int)clamp(r.y * (T * a) * l2vals[4] * scale, -1e9f, 1e9f);
            int gg2_5 = (int)clamp(r.y * (T * a) * l2vals[5] * scale, -1e9f, 1e9f);
            int gb2_0 = (int)clamp(r.z * (T * a) * l2vals[0] * scale, -1e9f, 1e9f);
            int gb2_1 = (int)clamp(r.z * (T * a) * l2vals[1] * scale, -1e9f, 1e9f);
            int gb2_2 = (int)clamp(r.z * (T * a) * l2vals[2] * scale, -1e9f, 1e9f);
            int gb2_3 = (int)clamp(r.z * (T * a) * l2vals[3] * scale, -1e9f, 1e9f);
            int gb2_4 = (int)clamp(r.z * (T * a) * l2vals[4] * scale, -1e9f, 1e9f);
            int gb2_5 = (int)clamp(r.z * (T * a) * l2vals[5] * scale, -1e9f, 1e9f);
            uint base = i; // stride = gaussianCount
            uint stride = uni.gaussianCount;
            atomic_fetch_add_explicit(&gradR2[base + 0*stride], gr2_0, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradR2[base + 1*stride], gr2_1, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradR2[base + 2*stride], gr2_2, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradR2[base + 3*stride], gr2_3, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradR2[base + 4*stride], gr2_4, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradR2[base + 5*stride], gr2_5, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradG2[base + 0*stride], gg2_0, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradG2[base + 1*stride], gg2_1, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradG2[base + 2*stride], gg2_2, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradG2[base + 3*stride], gg2_3, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradG2[base + 4*stride], gg2_4, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradG2[base + 5*stride], gg2_5, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradB2[base + 0*stride], gb2_0, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradB2[base + 1*stride], gb2_1, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradB2[base + 2*stride], gb2_2, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradB2[base + 3*stride], gb2_3, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradB2[base + 4*stride], gb2_4, memory_order_relaxed);
            atomic_fetch_add_explicit(&gradB2[base + 5*stride], gb2_5, memory_order_relaxed);
        }
        atomic_fetch_add_explicit(&gradOpacity[i], gop, memory_order_relaxed);
    atomic_fetch_add_explicit(&gradSigma[i], gs, memory_order_relaxed);
    atomic_fetch_add_explicit(&gradPosX[i], (int)clamp(gpos.x * scale, -1e9f, 1e9f), memory_order_relaxed);
    atomic_fetch_add_explicit(&gradPosY[i], (int)clamp(gpos.y * scale, -1e9f, 1e9f), memory_order_relaxed);
    atomic_fetch_add_explicit(&gradPosZ[i], (int)clamp(gpos.z * scale, -1e9f, 1e9f), memory_order_relaxed);
    atomic_fetch_add_explicit(&gradScaleX[i], (int)clamp(dsx * scale, -1e9f, 1e9f), memory_order_relaxed);
    atomic_fetch_add_explicit(&gradScaleY[i], (int)clamp(dsy * scale, -1e9f, 1e9f), memory_order_relaxed);
    atomic_fetch_add_explicit(&gradScaleZ[i], (int)clamp(dsz * scale, -1e9f, 1e9f), memory_order_relaxed);

        // Update partials like forward
        accumRGB_partial += cur;
        accumA += T * a;
        if (accumA > 0.995) break;
    }
}
