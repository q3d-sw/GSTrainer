# Gaussian Splat Trainer (iOS, Swift + Metal)

This app is a fully offline Gaussian Splatting trainer that runs on iOS using Swift/SwiftUI and Metal. It loads camera datasets (`transforms.json` or `dataset.json`) and point clouds (`points.ply`), renders via a custom compute pipeline, and optimizes appearance (and optionally geometry) with Adam to minimize L2 loss against ground-truth images. Checkpoints and a simple SwiftUI UI are included.

What’s new (Recent updates):
- Forward now uses SH(L1) view-dependent color and anisotropic 3D scales projected to 2D via Jacobian.
- Backward runs on GPU with two-pass transmittance-aware accumulation; gradients for DC, SH(L1), opacity, isotropic `sigma`, positions (XYZ), and anisotropic scales (proxy) are computed.
- Geometry optimization: positions and per-axis scales are updated with Adam on-device.
- Checkpoints persist SH(L1) and anisotropic scales (scaleX/Y/Z) in addition to base parameters.
- Screen-space tiling/binning: per-pixel loops iterate only gaussians overlapping the pixel’s tile (tile size 16px), significantly reducing O(W·H·N) cost.

- Platforms: iOS 17+ (tested with iOS 17/18 simulators/devices)
- Tech stack: Swift 5.9+, SwiftUI, Metal compute (RGBA16F), simd; optional MPSGraph scaffolding
- Formats: nerfstudio-like `transforms.json`, custom `dataset.json`, 3DGS-aware `points.ply`

## 1. How it works (High-level)

- Loads the dataset (intrinsics + extrinsics + images). Handles letterbox resizing and `cx, cy, fx, fy` scaling.
- Loads `points.ply` (colors/opacity/scale if present) and uploads Gaussian parameters to GPU buffers.
- Renders predicted frames with a Metal compute kernel (front-to-back compositing, RGBA16F).
- Computes L2 loss in linear color space, normalizes per-frame exposure with a 1D gain, applies Adam to update Gaussian appearance (and optionally positions).
- Iterates training; shows preview, overlay, and logs. Checkpoints allow saving/restoring state.

## 2. Architecture

- SwiftUI UI (`TrainingView`) drives the training loop (`Trainer`).
- `DatasetManager` loads and scales intrinsics, normalizes image orientation, and supports both `transforms.json` and `dataset.json`.
- `PointsPLYParser` parses ASCII/binary little-endian PLY, recognizing 3DGS fields (f_dc_*, opacity, scale_*).
- `MetalRenderer` sets up a compute pipeline, creates RGBA16F textures and shared buffers, and runs the renderer per frame.
- `Trainer` orchestrates: rendering, loss/gradients, Adam updates, EMA, diagnostics, Sim3 refinement, optional geometry updates, and logging.
- `CheckpointManager` saves/loads model state.
- `AutoDatasetBuilder` can synthesize `dataset.json` from photos placed in Documents/images.

## 3. Data formats

### 3.1 transforms.json (nerfstudio-like)
- Contains frames with `intrinsics` (or computable from FOV) and `transform_matrix` per camera.
- This app supports matrices with translation in row or column and transposes when necessary.

### 3.2 dataset.json (custom)
```
{
  "width": <int>,
  "height": <int>,
  "frames": [
    {
      "imageFilename": "images/xxx.png",
      "intrinsics": {"fx":..., "fy":..., "cx":..., "cy":...},
      "extrinsics": {"m": [16 floats row-major]}
    }, ...
  ]
}
```
- Intrinsics are scaled with letterbox resize.
- Extrinsics are 4x4 row-major.

### 3.3 points.ply
- XYZ required. Optionally color (3DGS spherical harmonics DC: `f_dc_0/1/2`), `opacity`, and anisotropic scales (`scale_0/1/2`).
- Binary little-endian or ASCII supported.

## 4. Rendering (Metal)

- Forward compute kernel rasterizes Gaussians front-to-back with pre-multiplied alpha compositing into an RGBA16F target.
- View-dependent color via SH(L1): each channel stores `[c0, c1x, c1y, c1z]` and is evaluated against the view direction.
- Anisotropic splats: 3D diagonal covariance `diag(scale.x^2, scale.y^2, scale.z^2)` is projected with the 2×3 Jacobian to screen-space `Σ2D = J Σ3D J^T`; weight uses the Mahalanobis distance.
- Screen-space tiling/binning: per-pixel iteration is limited to gaussians overlapping that pixel’s tile; tile buffers are rebuilt per frame on CPU.
- A CPU path converts RGBA16F to RGBA8 for UI preview.

## 5. Optimization

- Loss: L2 in linear RGB, with per-frame scalar exposure gain (closed-form least squares) for robustness to auto-exposure.
- Optimizer: Adam (EMA, decoupled weight decay) for color, opacity, `sigma`, and geometry (positions, anisotropic scales).
- Backward on GPU: two-pass accumulation that accounts for transmittance; gradients computed for DC, SH(L1), opacity, isotropic `sigma`, positions (XYZ), and anisotropic scales (proxy via isotropic world sigma split).
- EMA of loss for smoothed logging.
- Geometry fine-tuning: positions are updated using GPU position gradients; anisotropic scales are optimized per-axis and synced to GPU; current scale gradients use a stable proxy and will be refined.

## 6. Sim(3) alignment

- Auto-align utility: heuristic search over flips, scale, and camera forward sign to avoid black frames and increase in-bounds coverage.
- Sim3 coordinate descent: adjusts global scale/rotation/translation to improve reprojection.
- Sim3 Gauss–Newton: precise 7-parameter refinement with numeric Jacobian, JTJ/JTr solve (7×7), backtracking, adaptive damping (Levenberg–Marquardt-like), log-scale parameterization, and residual normalization.

## 7. Auto dataset builder

- Place photos into Documents/images on the device/simulator.
- Builder reads EXIF (`FocalLenIn35mmFilm`) when available to estimate FOV; otherwise uses a default FOV.
- Generates a circular camera path and writes `dataset.json` to Documents.

## 8. UI and usage

- Start/Pause: controls the training loop.
- Gen dataset.json: builds dataset.json from Documents/images and reloads the dataset.
- Auto-align: tries to fix flips/scale/forward and prints diagnostics.
- Sim3 Refine / Sim3 Refine (GN): global alignment refinement.
- Reseed colors: initializes Gaussian colors by sampling GT in linear space.
- Resolution: set downscaled long-side for faster training.
- Gaussians: select number of Gaussians.
- Toggles: Overlay projections; Geometry fine-tuning.
- Preview and Overlay panels visualize current prediction and projected centers.
- Logs: prints step/loss stats, exposure, opacity, coverage, and diagnostics.

Tips:
- Begin with fewer Gaussians (1k–4k) and a smaller resolution for speed, then scale up.
- Use Auto-align if you see black frames or very low in-bounds percentage.
- GN refine after a few hundred steps to stabilize pose.

## 9. Checkpoints

- Saves current Gaussian parameters and trainer state.
- Persists SH(L1) coefficients and anisotropic scales (scaleX/Y/Z) for each gaussian.
- Can restore from the latest checkpoint after relaunch.

## 10. Troubleshooting

- Previews fail to install on Simulator (Client not entitled): ensure your Scheme’s Run/Previews build configuration is Debug, clean DerivedData, or try a different simulator.
- Black frames: toggle camera forward (zSign) via Auto-align or wait for auto toggle; check intrinsics scaling.
- Exploding loss: reduce learning rate; ensure linear color conversion and exposure gain are active.
- PLY parsing issues: verify binary little-endian vs ASCII and 3DGS field names.

## 11. Performance notes

- Screen-space tiling (16×16) reduces per-pixel gaussian loops to only nearby splats and improves frame time significantly for large `N`.
- Use smaller downscale values for faster iterations on device.
- RGBA16F compute + shared storage avoids excessive CPU-GPU copies.
- Avoid too many Gaussians early on to keep render time manageable.

## 12. Recent changes (Changelog)

- Added SH(L1) view-dependent color in forward and full DC/SH(L1) grads in backward.
- Implemented anisotropic splats via projected covariance and added grads for isotropic `sigma` and proxy grads for per-axis scales.
- Moved opacity and compositing gradients to a two-pass, transmittance-aware formulation; improved stability and quality.
- Implemented GPU gradients for positions (XYZ) and wired Adam updates on-device.
- Persisted SH(L1) and per-axis scales in checkpoints; loader upgraded for backward compatibility.
- Introduced screen-space culling (3σ) and added screen-space tiling with per-tile gaussian lists (rebuilt on CPU each frame).
- Fixed assorted bugs and improved logs/UI text; ensured portrait UI and Previews reliability.
- Added optional SH(L2) (9 coeffs / channel) path with packed gradient accumulation, Adam updates for 6 extra coeffs per channel, global gradient clipping including L2, and periodic diagnostics (RMS grads, active coeff counts).

---
