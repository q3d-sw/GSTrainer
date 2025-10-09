# USER GUIDE (EN)

Practical, step-by-step instructions for preparing a dataset, placing files on device/simulator, launching and supervising training, and understanding every control in the UI. Conceptual architecture lives in `README.md`.

---
## 1. Supported Dataset Inputs
You can supply ONE of:
1. `transforms.json` (nerfstudio-like)
2. `dataset.json` (custom format, see below)
3. Raw photos in `Documents/images/` (the app can synthesize `dataset.json` via the "Gen dataset.json" button)
4. Optional `points.ply` (initial Gaussian setup)

### 1.1 Folder Placement
App sandbox Documents directory (device or simulator):
```
Documents/
  transforms.json          (optional alternative to dataset.json)
  dataset.json             (or generated)
  points.ply               (optional)
  images/                  (if auto-building dataset.json)
    frame_000.png
    frame_001.png
    ...
```
Simulator path pattern (macOS): `~/Library/Developer/CoreSimulator/Devices/<UDID>/data/Containers/Data/Application/<APP_UUID>/Documents/`

### 1.2 dataset.json Format
```
{
  "width": <int>,
  "height": <int>,
  "frames": [
    {
      "imageFilename": "images/frame_000.png",
      "intrinsics": {"fx":..., "fy":..., "cx":..., "cy":...},
      "extrinsics": {"m": [16 floats row-major]}
    }
  ]
}
```
Intrinsics are pre-downscale; the app rescales if you choose a smaller resolution. Extrinsics are camera-to-world (row-major) and internally inverted.

### 1.3 transforms.json
Standard nerfstudio style. Priority order if multiple inputs exist:
1. `transforms.json`
2. `dataset.json`
3. Auto-generated dataset from images

### 1.4 points.ply
Fields (ASCII or binary little-endian): `x y z` (required) and optional `f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2`. Missing fields get defaults.

---
## 2. Launching the App
1. Open project in Xcode, pick device/simulator (iOS 17+).
2. Build & Run. Logs will show dataset detection.
3. If no dataset, add images then click "Gen dataset.json".

---
## 3. UI Controls Reference
| Control | Description |
|---------|-------------|
| Start | Begin training loop |
| Pause | Stop training & save checkpoint |
| Gen dataset.json | Build dataset from `images/` (circular path + EXIF focal) |
| Auto-align | Search flips/scale/forward sign to maximize in-bounds coverage |
| Sim3 Refine | Coordinate-descent global Sim(3) refinement |
| Sim3 Refine (GN) | Gauss–Newton 7-param refinement (damping/backtracking) |
| Reseed colors | Reset Gaussian colors by sampling GT at projected centers |
| Geometry fine-tuning | Enable position & anisotropic scale optimization |
| Overlay | Show projected centers (red dots) over GT |
| Resolution | Downscale long side for speed |
| Gaussians | Set number of Gaussians (recreates buffers) |
| SH L2 | Enable higher-order spherical harmonics (extra 6 coeffs/channel) |
| (Auto) Reseed multi | Internal logged multi-frame reseed triggers |

Metrics & state appear in the log panel.

---
## 4. Typical Workflow
1. Place dataset files (or just images/) in Documents.
2. (Optional) Add `points.ply`.
3. Press Start; monitor loss ↓ and PSNR ↑.
4. Use Auto-align if black frames / low coverage.
5. After ~200–500 iters run Sim3 Refine (and GN) for global pose fine-tune.
6. Reseed colors if saturation collapses (low variance logs) or for early acceleration.
7. Upscale Gaussian count or resolution once PSNR plateaus (<0.1 dB gains over hundreds iters).
8. Pause to persist checkpoint; restart auto-loads the latest.

---
## 5. Metrics
| Metric | Meaning |
|--------|---------|
| loss | Mean per-pixel (dR^2 + dG^2 + dB^2) (linear) |
| PSNR | 10*log10(3 / loss) (channel MSE = loss/3) |
| EMA loss/PSNR | Exponential smoothing (alpha 0.05 new) |
| avgOpacity | Mean opacity over Gaussians |
| var / gVar | Image luminance variance / Gaussian color variance |
| sat | Mean Gaussian color saturation |
| reproj front/inBounds | Coverage diagnostics |

Plateau: PSNR increase <0.1 dB across few hundred iterations.

---
## 6. Recommendations
- Start small (1024–4096 Gaussians, 256–384px long side).
- Early Auto-align if black frames.
- Reseed colors once orientation stable (10–50 iters).
- Scale resolution & count after initial convergence.
- Reduce base LR (×0.5) only if oscillations appear.

---
## 7. Troubleshooting
| Symptom | Fix |
|---------|-----|
| Black frames / loss static | Auto-align (forward sign), verify intrinsics scaling |
| Early PSNR stagnation | Reseed colors, Sim3 refine, lower LR |
| Gray / desaturated colors | Reseed, allow auto saturation boost |
| Slow per-step | Downscale resolution, reduce Gaussians initially |
| Checkpoint ignored | Ensure JSON present & readable |
| PLY fields skipped | Use 3DGS names (`f_dc_*`, `opacity`, `scale_*`) |

---
## 8. Roadmap Highlights
Exact anisotropic scale gradients; SSIM logging; adaptive gradient scaling; SH L2/L3 energy monitoring.

---
## 9. Version & Contact
Feature set: see Changelog in `README.md`.
For issues: inspect logs (first 20 iterations surface config errors).

---
End of USER_GUIDE_EN.md
