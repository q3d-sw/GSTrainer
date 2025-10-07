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
    uint useCache; // 0=no cache (recompute), 1=use screen-space cache buffer
};

// Screen-space per-gaussian cache reused by forward (optional) and backward passes.
// Contains projection results and inverse covariance for quick weight evaluation.
struct GaussianScreenCache {
    float2 uv; float invz; float zf; // uv + inverse depth + forward depth
    float4 cam4;                     // camera-space position (xyz, w pad)
    float j00,j01,j02,pad0;          // projection Jacobian row 0
    float j10,j11,j12,pad1;          // projection Jacobian row 1
    float sxx,sxy,syy,pad2;          // 2x2 covariance components
    float a00,a01,a11,pad3;          // inverse covariance (symmetric)
    float4 viewDir4;                 // normalized view dir (-cam) (xyz, pad)
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
    constant GaussianScreenCache *cache [[buffer(5)]],
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
        float3 color; float2 uv; float a00,a01,a11; float3 cam; float zf;
        if (uniforms.useCache != 0) {
            GaussianScreenCache c = cache[i];
            if (c.zf <= 0.0) continue;
            uv = c.uv;
            cam = c.cam4.xyz;
            float3 viewDir = normalize(-cam);
            float x = viewDir.x, y = viewDir.y, z = viewDir.z;
            float4 basis = float4(1.0, x, y, z);
            color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
            if (shMode.enableL2 != 0) {
                float4 ex1 = float4(1.092548f * x * y, 1.092548f * y * z, 0.315392f * (3.0f * z * z - 1.0f), 1.092548f * x * z);
                float4 ex2 = float4(0.546274f * (x * x - y * y), 0.0, 0.0, 0.0);
                color += float3(dot(g.shR_ex1, ex1) + dot(g.shR_ex2, ex2),
                                dot(g.shG_ex1, ex1) + dot(g.shG_ex2, ex2),
                                dot(g.shB_ex1, ex1) + dot(g.shB_ex2, ex2));
            }
            color = clamp(color, 0.0, 1.0);
            a00 = c.a00; a01 = c.a01; a11 = c.a11;
            float2 d = float2(pixel.x, pixel.y) - uv;
            float qx = a00*d.x + a01*d.y;
            float qy = a01*d.x + a11*d.y;
            float mahal = d.x*qx + d.y*qy;
            float w = exp(-0.5 * mahal);
            float a = clamp(g.opacity * w, 0.0, 1.0);
            float oneMinusA = (1.0 - accumA);
            accumRGB += oneMinusA * a * color;
            accumA += oneMinusA * a;
            if (accumA > 0.995) break;
            continue;
        } else {
            float4 p = float4(g.position, 1.0);
            float4 cam4 = uniforms.worldToCam * p;
            cam = cam4.xyz;
            zf = (uniforms.zSign < 0.0) ? (-cam.z) : (cam.z);
            if (zf <= 0.0) continue;
            uv = float2(uniforms.fx * (cam.x / zf) + uniforms.cx,
                        uniforms.fy * (cam.y / zf) + uniforms.cy);
            float3 viewDir = normalize(-cam);
            float x = viewDir.x, y = viewDir.y, z = viewDir.z;
            float4 basis = float4(1.0, x, y, z);
            color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
            if (shMode.enableL2 != 0) {
                float4 ex1 = float4(1.092548f * x * y, 1.092548f * y * z, 0.315392f * (3.0f * z * z - 1.0f), 1.092548f * x * z);
                float4 ex2 = float4(0.546274f * (x * x - y * y), 0.0, 0.0, 0.0);
                color += float3(dot(g.shR_ex1, ex1) + dot(g.shR_ex2, ex2),
                                dot(g.shG_ex1, ex1) + dot(g.shG_ex2, ex2),
                                dot(g.shB_ex1, ex1) + dot(g.shB_ex2, ex2));
            }
            color = clamp(color, 0.0, 1.0);
            float sx2 = g.scale.x * g.scale.x;
            float sy2 = g.scale.y * g.scale.y;
            float sz2 = g.scale.z * g.scale.z;
            float invz = 1.0 / zf;
            float invz2 = invz * invz;
            float j00 = uniforms.fx * invz;       float j01 = 0.0;                  float j02 = -uniforms.fx * cam.x * invz2;
            float j10 = 0.0;                       float j11 = uniforms.fy * invz;   float j12 = -uniforms.fy * cam.y * invz2;
            float sxx = j00*j00*sx2 + j01*j01*sy2 + j02*j02*sz2;
            float sxy = j00*j10*sx2 + j01*j11*sy2 + j02*j12*sz2;
            float syy = j10*j10*sx2 + j11*j11*sy2 + j12*j12*sz2;
            float det = max(1e-8, (sxx*syy - sxy*sxy));
            float inv00 =  syy / det;
            float inv01 = -sxy / det;
            float inv11 =  sxx / det;
            float2 d = float2(pixel.x, pixel.y) - uv;
            float qx = inv00*d.x + inv01*d.y;
            float qy = inv01*d.x + inv11*d.y;
            float mahal = d.x*qx + d.y*qy;
            float w = exp(-0.5 * mahal);
            float a = clamp(g.opacity * w, 0.0, 1.0);
            float oneMinusA = (1.0 - accumA);
            accumRGB += oneMinusA * a * color;
            accumA += oneMinusA * a;
            if (accumA > 0.995) break;
        }
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
    constant GaussianScreenCache *cache [[buffer(5)]],
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
        if (*tgAlpha > 0.995f) break;
        uint gi = tileList[idx];
        GaussianParamGPU g = gaussians[gi];
        float2 uv; float3 color; float3 cam;
        if (uniforms.useCache != 0) {
            GaussianScreenCache c = cache[gi];
            if (c.zf <= 0.0) continue;
            uv = c.uv; cam = c.cam4.xyz;
            float3 viewDir = normalize(-cam);
            float x = viewDir.x, y = viewDir.y, z = viewDir.z;
            float4 basis = float4(1.0, x, y, z);
            color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
            if (shMode.enableL2 != 0) {
                float4 ex1 = float4(1.092548f * x * y, 1.092548f * y * z, 0.315392f * (3.0f * z * z - 1.0f), 1.092548f * x * z);
                float4 ex2 = float4(0.546274f * (x * x - y * y), 0.0, 0.0, 0.0);
                color += float3(dot(g.shR_ex1, ex1) + dot(g.shR_ex2, ex2),
                                dot(g.shG_ex1, ex1) + dot(g.shG_ex2, ex2),
                                dot(g.shB_ex1, ex1) + dot(g.shB_ex2, ex2));
            }
            color = clamp(color, 0.0, 1.0);
            float2 d = pixel - uv;
            float qx = c.a00*d.x + c.a01*d.y;
            float qy = c.a01*d.x + c.a11*d.y;
            float mahal = d.x*qx + d.y*qy;
            float w = exp(-0.5 * mahal);
            float a = clamp(g.opacity * w, 0.0, 1.0);
            float oneMinusA = (1.0 - accumA);
            accumRGB += oneMinusA * a * color;
            accumA += oneMinusA * a;
            if (accumA > 0.995) { *tgAlpha = 1.0; break; }
            continue;
        } else {
            float4 cam4 = uniforms.worldToCam * float4(g.position, 1.0);
            cam = cam4.xyz;
            float zf = (uniforms.zSign < 0.0) ? (-cam.z) : (cam.z);
            if (zf <= 0.0) continue;
            uv = float2(uniforms.fx * (cam.x / zf) + uniforms.cx,
                        uniforms.fy * (cam.y / zf) + uniforms.cy);
            float3 viewDir = normalize(-cam);
            float x = viewDir.x, y = viewDir.y, z = viewDir.z;
            float4 basis = float4(1.0, x, y, z);
            color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
            if (shMode.enableL2 != 0) {
                float4 ex1 = float4(1.092548f * x * y, 1.092548f * y * z, 0.315392f * (3.0f * z * z - 1.0f), 1.092548f * x * z);
                float4 ex2 = float4(0.546274f * (x * x - y * y), 0.0, 0.0, 0.0);
                color += float3(dot(g.shR_ex1, ex1) + dot(g.shR_ex2, ex2),
                                dot(g.shG_ex1, ex1) + dot(g.shG_ex2, ex2),
                                dot(g.shB_ex1, ex1) + dot(g.shB_ex2, ex2));
            }
            color = clamp(color, 0.0, 1.0);
            float sx2 = g.scale.x * g.scale.x;
            float sy2 = g.scale.y * g.scale.y;
            float sz2 = g.scale.z * g.scale.z;
            float invz = 1.0 / ((uniforms.zSign < 0.0) ? (-cam.z) : (cam.z));
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
    uint useCache; // 0=no cache, 1=use screen cache buffer
    uint lossMode; // 0=L2,1=Charb
    float charbEps;
    float _pad0, _pad1; // align to 16-byte
};

// (Definition moved earlier)

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
    constant GaussianScreenCache *cache [[buffer(27)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uni.imageWidth || gid.y >= uni.imageHeight) return;
    constexpr sampler s(coord::pixel, filter::nearest);
    float3 r = float3(residualTex.sample(s, (float2(gid) + 0.5) / float2(uni.imageWidth, uni.imageHeight)).rgb);
    if (uni.lossMode == 1) {
        // Charbonnier scaling: r' = r / sqrt(r^2 + eps^2)
        float eps2 = uni.charbEps * uni.charbEps;
        r = r / sqrt(r * r + float3(eps2));
    }

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
        float3 cam;
        float zf;
        float2 uv;
        float invz; // may come from cache
        if (uni.useCache != 0) {
            GaussianScreenCache c = cache[i];
            uv = c.uv;
            invz = c.invz;
            zf = c.zf;
            cam = c.cam4.xyz;
            if (zf <= 0.0) continue;
        } else {
            float4 cam4 = uni.worldToCam * float4(g.position, 1.0);
            cam = cam4.xyz;
            zf = (uni.zSign < 0.0) ? (-cam.z) : (cam.z);
            if (zf <= 0.0) continue;
            uv = float2(uni.fx * (cam.x / zf) + uni.cx,
                        uni.fy * (cam.y / zf) + uni.cy);
            invz = 1.0 / zf;
        }

        // Approximate anisotropic weight using isotropic proxy (fast path)
        // (invz already set above if cached)

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
        float3 cam;
        float zf;
        float2 uv;
        float invz;
        if (uni.useCache != 0) {
            GaussianScreenCache c = cache[i];
            uv = c.uv;
            invz = c.invz;
            zf = c.zf;
            cam = c.cam4.xyz;
            if (zf <= 0.0) continue;
        } else {
            float4 cam4 = uni.worldToCam * float4(g.position, 1.0);
            cam = cam4.xyz;
            zf = (uni.zSign < 0.0) ? (-cam.z) : (cam.z);
            if (zf <= 0.0) continue;
            uv = float2(uni.fx * (cam.x / zf) + uni.cx,
                        uni.fy * (cam.y / zf) + uni.cy);
            invz = 1.0 / zf;
        }

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
    float invz2 = invz * invz;
    float j00,j01,j02,j10,j11,j12;
    float sxx,sxy,syy;
    float a00,a01,a11;
    if (uni.useCache != 0) {
        GaussianScreenCache c = cache[i];
        j00 = c.j00; j01 = c.j01; j02 = c.j02; 
        j10 = c.j10; j11 = c.j11; j12 = c.j12;
        sxx = c.sxx; sxy = c.sxy; syy = c.syy;
        a00 = c.a00; a01 = c.a01; a11 = c.a11;
        // (invz2 already computed above; det not needed explicitly when cached)
    } else {
        j00 = uni.fx * invz;       j01 = 0.0;                  j02 = -uni.fx * cam.x * invz2;
        j10 = 0.0;                 j11 = uni.fy * invz;        j12 = -uni.fy * cam.y * invz2;
        sxx = j00*j00*sx2 + j01*j01*sy2 + j02*j02*sz2;
        sxy = j00*j10*sx2 + j01*j11*sy2 + j02*j12*sz2;
        syy = j10*j10*sx2 + j11*j11*sy2 + j12*j12*sz2;
        float det = max(1e-8, (sxx*syy - sxy*sxy));
        a00 =  syy / det;
        a01 = -sxy / det;
        a11 =  sxx / det;
    }
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

    // Position gradient (world) with improved analytic handling of dA/dcam:
    // w = exp(-0.5 d^T A d) where A = Σ^{-1}, Σ = J Σ3D J^T.
    // Base term (ignoring A dependence): ∂w/∂cam |_base = w * u^T * J_cam with u = A d.
    // Include first order from A' via A' = -A Σ' A. For Σ' we differentiate only entries affected by j02,j12 (depend on cam.x, cam.y, cam.z) and j00,j11 via invz.
    // Let q = A d ( = u ). Correction term: ∂w/∂cam |_A = -0.5 w * (d^T A' d) * (ignored higher order) + w * ( - (A' d) )^T * (pixel - note: exact expansion). We approximate dominant contribution through (A' d).
    // We compute t = (A' d) using Σ' pieces then form correction vector added to base.
    float d_invz_dz = -invz * invz; // invz = 1/zf (zf ~ +/- cam.z)
    float signZ = (uni.zSign < 0.0f) ? -1.0f : 1.0f; // zf = sign * cam.z or -cam.z when zSign<0
    // Partial derivatives of j00,j11 wrt cam.z via invz
    float fx = uni.fx; float fy = uni.fy;
    float dj00_dz = fx * d_invz_dz * signZ;
    float dj11_dz = fy * d_invz_dz * signZ;
    // j02 = -fx * cam.x * invz^2 => derivative wrt cam.x and cam.z
    float dj02_dx = -fx * invz * invz;
    float dj12_dy = -fy * invz * invz;
    float dj02_dz = -fx * cam.x * 2.0f * invz * d_invz_dz * signZ;
    float dj12_dz = -fy * cam.y * 2.0f * invz * d_invz_dz * signZ;
    // Reconstruct column vectors jx,jy,jz for clarity
    float2 jx = float2(j00, j10);
    float2 jy = float2(j01, j11);
    float2 jzv = float2(j02, j12);
    // Σ = sx2 * jx jx^T + sy2 * jy jy^T + sz2 * jzv jzv^T
    // Only j02,j12 include cam.x, cam.y explicitly; all depend on cam.z via invz. Compute partial Σ' contributions.
    float2 djx_dz = float2(dj00_dz, 0.0f); // j10 constant wrt z except through invz? j10=0 => derivative 0
    float2 djy_dz = float2(0.0f, dj11_dz);
    float2 djz_dx = float2(dj02_dx, 0.0f);
    float2 djz_dy = float2(0.0f, dj12_dy);
    float2 djz_dz = float2(dj02_dz, dj12_dz);
    // Helper to compute A' d ≈ -A Σ' (A d) = -A Σ' u2d, where u2d = d (since A d already computed). We'll use u (A d) inside.
    // First compute Σ' * u2d for each axis component partially.
    float2 u2d = d; // screen-space d
    // Σ'_x contribution via jz wrt cam.x: Σ'_x = sz2*(djz_dx jz^T + jz djz_dx^T)
    float2 s_z_dx = sz2 * (djz_dx * (dot(jzv, u2d)) + jzv * (dot(djz_dx, u2d)));
    // Σ'_y contribution
    float2 s_z_dy = sz2 * (djz_dy * (dot(jzv, u2d)) + jzv * (dot(djz_dy, u2d)));
    // Σ'_z contribution aggregates derivatives of jx,jy,jz wrt z
    float2 s_x_dz = sx2 * (djx_dz * (dot(jx, u2d)) + jx * (dot(djx_dz, u2d)));
    float2 s_y_dz = sy2 * (djy_dz * (dot(jy, u2d)) + jy * (dot(djy_dz, u2d)));
    float2 s_z_dz = sz2 * (djz_dz * (dot(jzv, u2d)) + jzv * (dot(djz_dz, u2d)));
    float2 sigmaPrime_x = s_z_dx; // only jz depends on x
    float2 sigmaPrime_y = s_z_dy; // only jz depends on y
    float2 sigmaPrime_z = s_x_dz + s_y_dz + s_z_dz;
    // Compute A Σ' u  using A (symmetric 2x2: [a00 a01; a01 a11])
    // Apply A (symmetric 2x2) without using lambdas/blocks (for wider deployment target compatibility)
    float2 A_sigmaPrime_u_x = float2(a00*sigmaPrime_x.x + a01*sigmaPrime_x.y, a01*sigmaPrime_x.x + a11*sigmaPrime_x.y);
    float2 A_sigmaPrime_u_y = float2(a00*sigmaPrime_y.x + a01*sigmaPrime_y.y, a01*sigmaPrime_y.x + a11*sigmaPrime_y.y);
    float2 A_sigmaPrime_u_z = float2(a00*sigmaPrime_z.x + a01*sigmaPrime_z.y, a01*sigmaPrime_z.x + a11*sigmaPrime_z.y);
    // A' d ≈ -A Σ' A d = -A Σ' u  => correction components
    float2 corr2d_x = -A_sigmaPrime_u_x;
    float2 corr2d_y = -A_sigmaPrime_u_y;
    float2 corr2d_z = -A_sigmaPrime_u_z;
    // Base term (ignoring A')
    float3 dwdcam_base = float3(
        u.x * j00 + u.y * j10,
        u.x * j01 + u.y * j11,
        u.x * j02 + u.y * j12
    ) * w_aniso;
    // Project 2D corrections through J_cam columns approximated: x,y use their respective columns; z uses average sensitivity of uv to z (via dj00_dz,dj11_dz)
    float3 dwdcam_corr = float3(
        (corr2d_x.x * j00 + corr2d_x.y * j10),
        (corr2d_y.x * j01 + corr2d_y.y * j11),
        (corr2d_z.x * (j02) + corr2d_z.y * (j12))
    ) * w_aniso;
    float3 dwdcam = dwdcam_base + dwdcam_corr;
    float3 rc0 = float3(uni.worldToCam[0][0], uni.worldToCam[0][1], uni.worldToCam[0][2]);
    float3 rc1 = float3(uni.worldToCam[1][0], uni.worldToCam[1][1], uni.worldToCam[1][2]);
    float3 rc2 = float3(uni.worldToCam[2][0], uni.worldToCam[2][1], uni.worldToCam[2][2]);
    float3 gpos = float3(dot(rc0, dwdcam), dot(rc1, dwdcam), dot(rc2, dwdcam)) * dLdw;

    // dsx,dsy,dsz computed exactly above

    // Direct atomic accumulation (TODO: introduce per-tile shared-memory reduction in a dedicated tiled backward kernel)
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

// Optimized (experimental) tiled backward with per-gaussian threadgroup reduction to reduce global atomic pressure.
// Strategy:
// 1. Precompute finalRGB per pixel (same as pass 1 of accumulateGradients) to obtain downstream color later.
// 2. Iterate gaussians in tile order near-to-far. For each gaussian:
//    - Each thread computes its pixel's contribution grad components (fixed-point scaled ints).
//    - Sequentially reduce each component across the threadgroup using a scratch array and atomic add once per gaussian.
// This dramatically cuts atomics from (#pixels * components) to (#gaussians * components) per tile at the cost of extra barriers.
// NOTE: Maintains semantic parity with accumulateGradients (same formulas) for DC, SH(L1/L2), opacity, sigma (isotropic proxy), position, anisotropic scales.
// Threadgroup reduction helper (file-scope). Each invocation:
// 1) Writes value to scratch[flatTid]
// 2) Barrier
// 3) Thread 0 linear-reduces and does a single atomic add
// 4) Barrier (to ensure scratch reuse safety for next component)
inline void tgReduceAndAtomicAdd(int value,
                                 device atomic_int *dstBase,
                                 uint index,
                                 threadgroup int *scratch,
                                 uint flatTid,
                                 uint tgElems) {
    scratch[flatTid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (flatTid == 0) {
        int sum = 0;
        for (uint k = 0; k < tgElems; ++k) sum += scratch[k];
        atomic_fetch_add_explicit(dstBase + index, sum, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void tiledAccumulateGradients(
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
    device atomic_int *gradR2 [[buffer(14)]],
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
    constant GaussianScreenCache *cache [[buffer(27)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint flatTid [[thread_index_in_threadgroup]],
    threadgroup int *scratch)
{
    constexpr sampler s(coord::pixel, filter::nearest);
    // Tile bounds
    if (tgid.x >= uni.tilesX || tgid.y >= uni.tilesY) return;
    uint baseX = tgid.x * uni.tileSize;
    uint baseY = tgid.y * uni.tileSize;
    uint px = baseX + tid2.x;
    uint py = baseY + tid2.y;
    if (px >= uni.imageWidth || py >= uni.imageHeight) return;
    uint tileId = tgid.y * uni.tilesX + tgid.x;
    uint start = tileOffsets[tileId];
    uint end = tileOffsets[tileId + 1];
    float2 pixel = float2(float(px) + 0.5, float(py) + 0.5);
    // Sample residual (pred - gt) already exposure-normalized
    float3 r = float3(residualTex.sample(s, (float2(px, py) + 0.5) / float2(uni.imageWidth, uni.imageHeight)).rgb);
    if (uni.lossMode == 1) {
        float eps2 = uni.charbEps * uni.charbEps;
        r = r / sqrt(r * r + float3(eps2));
    }
    // Pass 1: compute finalRGB for downstream logic.
    float3 finalRGB = float3(0.0);
    float accumA_final = 0.0;
    for (uint it = start; it < end; ++it) {
        uint gi = tileList[it];
        GaussianParamGPU g = gaussians[gi];
        float3 cam; float zf; float2 uv; float invz;
        if (uni.useCache != 0) {
            GaussianScreenCache c = cache[gi];
            uv = c.uv; invz = c.invz; zf = c.zf; cam = c.cam4.xyz;
            if (zf <= 0.0) continue;
        } else {
            float4 cam4 = uni.worldToCam * float4(g.position, 1.0);
            cam = cam4.xyz;
            zf = (uni.zSign < 0.0) ? (-cam.z) : (cam.z);
            if (zf <= 0.0) continue;
            uv = float2(uni.fx * (cam.x / zf) + uni.cx,
                        uni.fy * (cam.y / zf) + uni.cy);
            invz = 1.0 / zf;
        }
        float s_world = (g.scale.x + g.scale.y + g.scale.z) / 3.0;
        float K = max(uni.fx, uni.fy) * invz * 0.5;
        float screenSigma = max(1.0, s_world) * K;
        float2 d = pixel - uv;
        float r2 = dot(d,d);
        float sig2 = screenSigma * screenSigma;
        if (r2 > 9.0 * sig2) continue;
        float w_iso = exp(-0.5 * r2 / sig2);
        float a = clamp(g.opacity * w_iso, 0.0, 1.0);
        float T = (1.0 - accumA_final);
        // SH color (L1 + optional L2)
        float3 vdir = normalize(-cam);
        float bx = vdir.x, by = vdir.y, bz = vdir.z;
        float4 basis = float4(1.0, bx, by, bz);
        float3 color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
        bool anyL2 = (g.shR_ex1.x != 0.0 || g.shR_ex1.y != 0.0 || g.shR_ex1.z != 0.0 || g.shR_ex1.w != 0.0 || g.shR_ex2.x != 0.0 || g.shR_ex2.y != 0.0 ||
                      g.shG_ex1.x != 0.0 || g.shG_ex1.y != 0.0 || g.shG_ex1.z != 0.0 || g.shG_ex1.w != 0.0 || g.shG_ex2.x != 0.0 || g.shG_ex2.y != 0.0 ||
                      g.shB_ex1.x != 0.0 || g.shB_ex1.y != 0.0 || g.shB_ex1.z != 0.0 || g.shB_ex1.w != 0.0 || g.shB_ex2.x != 0.0 || g.shB_ex2.y != 0.0);
        if (anyL2) {
            float xy = bx * by; float yz = by * bz; float xz = bx * bz; float z2 = bz * bz;
            float l2_0 = 1.092548f * xy; float l2_1 = 1.092548f * yz; float l2_2 = 0.315392f * (3.0f * z2 - 1.0f);
            float l2_3 = 1.092548f * xz; float l2_4 = 0.546274f * (bx * bx - by * by); float l2_5 = 0.0f;
            color += float3(
                g.shR_ex1.x * l2_0 + g.shR_ex1.y * l2_1 + g.shR_ex1.z * l2_2 + g.shR_ex1.w * l2_3 + g.shR_ex2.x * l2_4 + g.shR_ex2.y * l2_5,
                g.shG_ex1.x * l2_0 + g.shG_ex1.y * l2_1 + g.shG_ex1.z * l2_2 + g.shG_ex1.w * l2_3 + g.shG_ex2.x * l2_4 + g.shG_ex2.y * l2_5,
                g.shB_ex1.x * l2_0 + g.shB_ex1.y * l2_1 + g.shB_ex1.z * l2_2 + g.shB_ex1.w * l2_3 + g.shB_ex2.x * l2_4 + g.shB_ex2.y * l2_5);
        }
        color = clamp(color, 0.0, 1.0);
        finalRGB += T * a * color;
        accumA_final += T * a;
        if (accumA_final > 0.995) break;
    }
    // Pass 2: accumulate gradients per gaussian with threadgroup reduction.
    float3 accumRGB_partial = float3(0.0);
    float accumA = 0.0;
    for (uint it = start; it < end; ++it) {
        uint i = tileList[it];
        GaussianParamGPU g = gaussians[i];
        float3 cam; float zf; float2 uv; float invz; float invz2;
        if (uni.useCache != 0) {
            GaussianScreenCache c = cache[i];
            uv = c.uv; invz = c.invz; zf = c.zf; cam = c.cam4.xyz; invz2 = invz * invz;
            if (zf <= 0.0) continue;
        } else {
            float4 cam4 = uni.worldToCam * float4(g.position, 1.0);
            cam = cam4.xyz; zf = (uni.zSign < 0.0) ? (-cam.z) : (cam.z);
            if (zf <= 0.0) continue;
            uv = float2(uni.fx * (cam.x / zf) + uni.cx, uni.fy * (cam.y / zf) + uni.cy);
            invz = 1.0 / zf; invz2 = invz * invz;
        }
        float2 d = pixel - uv;
        float s_world_iso = (g.scale.x + g.scale.y + g.scale.z) / 3.0;
        float Kiso = max(uni.fx, uni.fy) * invz * 0.5;
        float screenSigma = max(1.0, s_world_iso) * Kiso;
        float r2 = dot(d,d); float sig2 = screenSigma * screenSigma;
        if (r2 > 9.0 * sig2) continue;
        float w_iso = exp(-0.5 * r2 / sig2);
        float a = clamp(g.opacity * w_iso, 0.0, 1.0);
        float T = (1.0 - accumA);
        float3 vdir = normalize(-cam);
        float bx = vdir.x, by = vdir.y, bz = vdir.z;
        float4 basis = float4(1.0, bx, by, bz);
        float3 color = float3(dot(g.shR, basis), dot(g.shG, basis), dot(g.shB, basis));
        // L2
        bool anyL2 = (g.shR_ex1.x != 0.0 || g.shR_ex1.y != 0.0 || g.shR_ex1.z != 0.0 || g.shR_ex1.w != 0.0 || g.shR_ex2.x != 0.0 || g.shR_ex2.y != 0.0 ||
                      g.shG_ex1.x != 0.0 || g.shG_ex1.y != 0.0 || g.shG_ex1.z != 0.0 || g.shG_ex1.w != 0.0 || g.shG_ex2.x != 0.0 || g.shG_ex2.y != 0.0 ||
                      g.shB_ex1.x != 0.0 || g.shB_ex1.y != 0.0 || g.shB_ex1.z != 0.0 || g.shB_ex1.w != 0.0 || g.shB_ex2.x != 0.0 || g.shB_ex2.y != 0.0);
        float l2vals[6];
        if (anyL2) {
            float xy = bx * by; float yz = by * bz; float xz = bx * bz; float z2 = bz * bz;
            l2vals[0] = 1.092548f * xy;
            l2vals[1] = 1.092548f * yz;
            l2vals[2] = 0.315392f * (3.0f * z2 - 1.0f);
            l2vals[3] = 1.092548f * xz;
            l2vals[4] = 0.546274f * (bx * bx - by * by);
            l2vals[5] = 0.0f;
            color += float3(
                g.shR_ex1.x * l2vals[0] + g.shR_ex1.y * l2vals[1] + g.shR_ex1.z * l2vals[2] + g.shR_ex1.w * l2vals[3] + g.shR_ex2.x * l2vals[4] + g.shR_ex2.y * l2vals[5],
                g.shG_ex1.x * l2vals[0] + g.shG_ex1.y * l2vals[1] + g.shG_ex1.z * l2vals[2] + g.shG_ex1.w * l2vals[3] + g.shG_ex2.x * l2vals[4] + g.shG_ex2.y * l2vals[5],
                g.shB_ex1.x * l2vals[0] + g.shB_ex1.y * l2vals[1] + g.shB_ex1.z * l2vals[2] + g.shB_ex1.w * l2vals[3] + g.shB_ex2.x * l2vals[4] + g.shB_ex2.y * l2vals[5]
            );
        }
        color = clamp(color, 0.0, 1.0);
        float3 cur = T * a * color;
        float3 downstream = finalRGB - (accumRGB_partial + cur);
        const float scaleFix = 1024.0f;
        int gc0r = (int)clamp(r.x * (T * a) * scaleFix, -1e9f, 1e9f);
        int gc0g = (int)clamp(r.y * (T * a) * scaleFix, -1e9f, 1e9f);
        int gc0b = (int)clamp(r.z * (T * a) * scaleFix, -1e9f, 1e9f);
        int gr1x = (int)clamp(r.x * (T * a) * bx * scaleFix, -1e9f, 1e9f);
        int gr1y = (int)clamp(r.x * (T * a) * by * scaleFix, -1e9f, 1e9f);
        int gr1z = (int)clamp(r.x * (T * a) * bz * scaleFix, -1e9f, 1e9f);
        int gg1x = (int)clamp(r.y * (T * a) * bx * scaleFix, -1e9f, 1e9f);
        int gg1y = (int)clamp(r.y * (T * a) * by * scaleFix, -1e9f, 1e9f);
        int gg1z = (int)clamp(r.y * (T * a) * bz * scaleFix, -1e9f, 1e9f);
        int gb1x = (int)clamp(r.z * (T * a) * bx * scaleFix, -1e9f, 1e9f);
        int gb1y = (int)clamp(r.z * (T * a) * by * scaleFix, -1e9f, 1e9f);
        int gb1z = (int)clamp(r.z * (T * a) * bz * scaleFix, -1e9f, 1e9f);
        float dL_da = dot(r, (T * color - downstream));
        int gop = (int)clamp((dL_da * w_iso) * scaleFix, -1e9f, 1e9f);
        float dLdw = dL_da * g.opacity;
        // Exact anisotropic scale derivatives like accumulateGradients
        float sx2 = g.scale.x * g.scale.x; float sy2 = g.scale.y * g.scale.y; float sz2 = g.scale.z * g.scale.z;
        float j00,j01,j02,j10,j11,j12; float sxx,sxy,syy; float a00,a01,a11;
        if (uni.useCache != 0) {
            GaussianScreenCache c = cache[i];
            j00 = c.j00; j01 = c.j01; j02 = c.j02;
            j10 = c.j10; j11 = c.j11; j12 = c.j12;
            sxx = c.sxx; sxy = c.sxy; syy = c.syy;
            a00 = c.a00; a01 = c.a01; a11 = c.a11;
        } else {
            j00 = uni.fx * invz; j01 = 0.0; j02 = -uni.fx * cam.x * invz2;
            j10 = 0.0;          j11 = uni.fy * invz; j12 = -uni.fy * cam.y * invz2;
            sxx = j00*j00*sx2 + j01*j01*sy2 + j02*j02*sz2;
            sxy = j00*j10*sx2 + j01*j11*sy2 + j02*j12*sz2;
            syy = j10*j10*sx2 + j11*j11*sy2 + j12*j12*sz2;
            float det = max(1e-8, (sxx*syy - sxy*sxy));
            a00 =  syy / det; a01 = -sxy / det; a11 =  sxx / det;
        }
        float2 u = float2(a00*d.x + a01*d.y, a01*d.x + a11*d.y);
        float t0 = j00 * u.x + j10 * u.y;
        float t1 = j01 * u.x + j11 * u.y;
        float t2 = j02 * u.x + j12 * u.y;
        float mahalA = d.x * u.x + d.y * u.y;
        float w_aniso = exp(-0.5 * mahalA);
        float dsx = dLdw * (w_aniso * g.scale.x * (t0 * t0));
        float dsy = dLdw * (w_aniso * g.scale.y * (t1 * t1));
        float dsz = dLdw * (w_aniso * g.scale.z * (t2 * t2));
        float dLds_world = dLdw * (w_iso * (r2 / (screenSigma*screenSigma*screenSigma)) * Kiso);
        int gs = (int)clamp(dLds_world * scaleFix, -1e9f, 1e9f);
        // Anisotropic position gradient as in accumulateGradients (see explanation there)
        float3 dwdcam = float3(
            u.x * j00 + u.y * j10,
            u.x * j01 + u.y * j11,
            u.x * j02 + u.y * j12
        ) * w_aniso;
        float3 rc0 = float3(uni.worldToCam[0][0], uni.worldToCam[0][1], uni.worldToCam[0][2]);
        float3 rc1 = float3(uni.worldToCam[1][0], uni.worldToCam[1][1], uni.worldToCam[1][2]);
        float3 rc2 = float3(uni.worldToCam[2][0], uni.worldToCam[2][1], uni.worldToCam[2][2]);
        float3 gpos3 = float3(dot(rc0, dwdcam), dot(rc1, dwdcam), dot(rc2, dwdcam)) * dLdw;
        int gposX = (int)clamp(gpos3.x * scaleFix, -1e9f, 1e9f);
        int gposY = (int)clamp(gpos3.y * scaleFix, -1e9f, 1e9f);
        int gposZ = (int)clamp(gpos3.z * scaleFix, -1e9f, 1e9f);
        int gsx = (int)clamp(dsx * scaleFix, -1e9f, 1e9f);
        int gsy = (int)clamp(dsy * scaleFix, -1e9f, 1e9f);
        int gsz = (int)clamp(dsz * scaleFix, -1e9f, 1e9f);
        uint tgElems = uni.tileSize * uni.tileSize;
        tgReduceAndAtomicAdd(gc0r, gradC0R, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gc0g, gradC0G, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gc0b, gradC0B, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gr1x, gradR1x, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gr1y, gradR1y, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gr1z, gradR1z, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gg1x, gradG1x, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gg1y, gradG1y, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gg1z, gradG1z, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gb1x, gradB1x, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gb1y, gradB1y, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gb1z, gradB1z, i, scratch, flatTid, tgElems);
        if (anyL2) {
            int gr2_0 = (int)clamp(r.x * (T * a) * l2vals[0] * scaleFix, -1e9f, 1e9f);
            int gr2_1 = (int)clamp(r.x * (T * a) * l2vals[1] * scaleFix, -1e9f, 1e9f);
            int gr2_2 = (int)clamp(r.x * (T * a) * l2vals[2] * scaleFix, -1e9f, 1e9f);
            int gr2_3 = (int)clamp(r.x * (T * a) * l2vals[3] * scaleFix, -1e9f, 1e9f);
            int gr2_4 = (int)clamp(r.x * (T * a) * l2vals[4] * scaleFix, -1e9f, 1e9f);
            int gr2_5 = (int)clamp(r.x * (T * a) * l2vals[5] * scaleFix, -1e9f, 1e9f);
            int gg2_0 = (int)clamp(r.y * (T * a) * l2vals[0] * scaleFix, -1e9f, 1e9f);
            int gg2_1 = (int)clamp(r.y * (T * a) * l2vals[1] * scaleFix, -1e9f, 1e9f);
            int gg2_2 = (int)clamp(r.y * (T * a) * l2vals[2] * scaleFix, -1e9f, 1e9f);
            int gg2_3 = (int)clamp(r.y * (T * a) * l2vals[3] * scaleFix, -1e9f, 1e9f);
            int gg2_4 = (int)clamp(r.y * (T * a) * l2vals[4] * scaleFix, -1e9f, 1e9f);
            int gg2_5 = (int)clamp(r.y * (T * a) * l2vals[5] * scaleFix, -1e9f, 1e9f);
            int gb2_0 = (int)clamp(r.z * (T * a) * l2vals[0] * scaleFix, -1e9f, 1e9f);
            int gb2_1 = (int)clamp(r.z * (T * a) * l2vals[1] * scaleFix, -1e9f, 1e9f);
            int gb2_2 = (int)clamp(r.z * (T * a) * l2vals[2] * scaleFix, -1e9f, 1e9f);
            int gb2_3 = (int)clamp(r.z * (T * a) * l2vals[3] * scaleFix, -1e9f, 1e9f);
            int gb2_4 = (int)clamp(r.z * (T * a) * l2vals[4] * scaleFix, -1e9f, 1e9f);
            int gb2_5 = (int)clamp(r.z * (T * a) * l2vals[5] * scaleFix, -1e9f, 1e9f);
            uint stride = uni.gaussianCount;
            tgReduceAndAtomicAdd(gr2_0, gradR2, i + 0*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gr2_1, gradR2, i + 1*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gr2_2, gradR2, i + 2*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gr2_3, gradR2, i + 3*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gr2_4, gradR2, i + 4*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gr2_5, gradR2, i + 5*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gg2_0, gradG2, i + 0*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gg2_1, gradG2, i + 1*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gg2_2, gradG2, i + 2*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gg2_3, gradG2, i + 3*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gg2_4, gradG2, i + 4*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gg2_5, gradG2, i + 5*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gb2_0, gradB2, i + 0*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gb2_1, gradB2, i + 1*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gb2_2, gradB2, i + 2*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gb2_3, gradB2, i + 3*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gb2_4, gradB2, i + 4*stride, scratch, flatTid, tgElems);
            tgReduceAndAtomicAdd(gb2_5, gradB2, i + 5*stride, scratch, flatTid, tgElems);
        }
        tgReduceAndAtomicAdd(gop, gradOpacity, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gs, gradSigma, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gposX, gradPosX, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gposY, gradPosY, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gposZ, gradPosZ, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gsx, gradScaleX, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gsy, gradScaleY, i, scratch, flatTid, tgElems);
        tgReduceAndAtomicAdd(gsz, gradScaleZ, i, scratch, flatTid, tgElems);
        // Advance compositing prefix
        accumRGB_partial += cur;
        accumA += T * a;
        if (accumA > 0.995) break;
    }
}

// NOTE: Gradient atomics currently performed per pixel per gaussian. Next optimization planned:
// Implement a tiled backward kernel mirroring 'tiledRenderGaussians' that first accumulates per-tile
// partial gradients for the gaussians referenced by that tile into threadgroup memory, then flushes
// a single atomic per gaussian per gradient component, drastically reducing contention.

// ------------------------------------------------------------
// Fused GPU loss & exposure kernels
// 1) computeExposureSums: accumulate numerator and denominator for optimal scalar exposure gain
// 2) computeResidualAndLoss: apply gain, build residual texture (half), accumulate L2 loss
// Loss = mean over pixels of ||gain*pred - gt||^2 (linear RGB)
// We operate in linear domain already because renderer writes linear color.
// ------------------------------------------------------------
constant float3 kLumW = float3(0.2126, 0.7152, 0.0722);
constant float  kExposureScale = 4096.0; // scale for exposure accum (num,den)
constant float  kLossScale = 1024.0;     // scale for loss accumulation

kernel void computeExposureSums(
    texture2d<half, access::read> pred [[texture(0)]],
    texture2d<half, access::read> gt   [[texture(1)]],
    device atomic_uint *accum          [[buffer(0)]], // 0:num,1:den
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= pred.get_width() || gid.y >= pred.get_height()) return;
    half4 p4 = pred.read(gid);
    half4 g4 = gt.read(gid);
    float3 p = float3(p4.xyz);
    float3 g = float3(g4.xyz);
    float pl = dot(p, kLumW);
    float gl = dot(g, kLumW);
    float num = pl * gl;
    float den = pl * pl;
    uint addNum = (uint)clamp(num * kExposureScale, 0.0, 4290000000.0);
    uint addDen = (uint)clamp(den * kExposureScale, 0.0, 4290000000.0);
    atomic_fetch_add_explicit(&accum[0], addNum, memory_order_relaxed);
    atomic_fetch_add_explicit(&accum[1], addDen, memory_order_relaxed);
}

struct ResidualUniforms { float gain; uint width; uint height; uint lossMode; float charbEps; };

kernel void computeResidualAndLoss(
    texture2d<half, access::read> pred [[texture(0)]],
    texture2d<half, access::read> gt   [[texture(1)]],
    texture2d<half, access::write> residual [[texture(2)]],
    constant ResidualUniforms &u [[buffer(0)]],
    device atomic_uint *lossAccum [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.width || gid.y >= u.height) return;
    half4 p4 = pred.read(gid);
    half4 g4 = gt.read(gid);
    float3 p = float3(p4.xyz) * u.gain;
    float3 g = float3(g4.xyz);
    float3 r = p - g;
    if (u.lossMode == 0) {
        // L2
        residual.write(half4(half(r.x), half(r.y), half(r.z), half(0.0)), gid);
        float se = dot(r, r);
        uint addLoss = (uint)clamp(se * kLossScale, 0.0, 4290000000.0);
        atomic_fetch_add_explicit(&lossAccum[0], addLoss, memory_order_relaxed);
    } else {
        // Charbonnier: sqrt(r^2 + eps^2); store original r (for backward scaling)
        float eps2 = u.charbEps * u.charbEps;
        float3 v = sqrt(r * r + float3(eps2));
        float charb = v.x + v.y + v.z; // sum per-pixel (will average later)
        residual.write(half4(half(r.x), half(r.y), half(r.z), half(0.0)), gid);
        uint addLoss = (uint)clamp(charb * kLossScale, 0.0, 4290000000.0);
        atomic_fetch_add_explicit(&lossAccum[0], addLoss, memory_order_relaxed);
    }
}

// ------------------------------------------------------------
// Residual Heatmap Downsampling
// Produces a coarse heatmap (RGBA16F, value in .r) by averaging a magnitude
// over blocks (block x block) of the residual texture. Two modes:
//  - L2:      magnitude = length(r)
//  - Charb:   magnitude = sum_c ( sqrt(r_c^2 + eps^2) - eps )  (less bias for small r)
// The result can drive densification (split/prune) decisions.
// ------------------------------------------------------------
struct HeatmapDownsampleUniforms {
    uint inWidth;
    uint inHeight;
    uint outWidth;
    uint outHeight;
    uint block;      // downsample factor
    uint lossMode;   // 0=L2,1=Charbonnier (match renderer flags)
    float charbEps;
};

kernel void downsampleResidualHeatmap(
    texture2d<half, access::read> residual [[texture(0)]],
    texture2d<half, access::write> heatmap [[texture(1)]],
    constant HeatmapDownsampleUniforms &u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.outWidth || gid.y >= u.outHeight) return;
    uint x0 = gid.x * u.block;
    uint y0 = gid.y * u.block;
    uint x1 = min(u.inWidth,  x0 + u.block);
    uint y1 = min(u.inHeight, y0 + u.block);
    float accum = 0.0f;
    uint count = 0;
    for (uint y = y0; y < y1; ++y) {
        for (uint x = x0; x < x1; ++x) {
            float3 r = float3(residual.read(uint2(x,y)).xyz);
            float mag;
            if (u.lossMode == 0) {
                mag = length(r);
            } else {
                float eps2 = u.charbEps * u.charbEps;
                float3 v = sqrt(r*r + float3(eps2)) - u.charbEps; // per-channel adjusted
                mag = v.x + v.y + v.z;
            }
            accum += mag;
            count++;
        }
    }
    float avg = (count > 0) ? (accum / (float)count) : 0.0f;
    heatmap.write(half4(half(avg), half(0.0), half(0.0), half(0.0)), gid);
}
