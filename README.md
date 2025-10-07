# Gaussian Splat Trainer (iOS, Swift + Metal)

This app is a fully offline Gaussian Splatting trainer that runs on iOS using Swift/SwiftUI and Metal. It loads camera datasets (`transforms.json` or `dataset.json`) and point clouds (`points.ply`), renders via a custom compute pipeline, and optimizes appearance (and optionally geometry) with Adam to minimize L2 loss against ground-truth images. Checkpoints and a simple SwiftUI UI are included.

What’s new (Recent updates):
- Forward now uses SH(L1) view-dependent color and anisotropic 3D scales projected to 2D via Jacobian.
- Backward runs on GPU with two-pass transmittance-aware accumulation; gradients for DC, SH(L1), opacity, isotropic `sigma`, positions (XYZ), and anisotropic scales (proxy) are computed.
 - Optional screen-space per-gaussian cache (backward) reuses projected uv / 1/z / Jacobian / Σ2D / Σ2D^{-1} to reduce redundant math; toggle in UI ("Screen cache (backward)").
- Geometry optimization: positions and per-axis scales are updated with Adam on-device.
- Checkpoints persist SH(L1) and anisotropic scales (scaleX/Y/Z) in addition to base parameters.
- Screen-space tiling/binning: per-pixel loops iterate only gaussians overlapping the pixel’s tile (tile size 16px), significantly reducing O(W·H·N) cost.
 - Hybrid depth ordering inside tiles: automatic switch between full sort (small tiles) and bucketed depth ordering (large occupancy) with instrumentation (bucketed vs sorted tile counts).

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

- Loss: L2 in linear RGB, with per-frame scalar exposure gain (closed-form least squares) for robustness to auto-exposure. Optional robust Charbonnier loss (sqrt(r^2 + ε^2)) can be enabled via the UI toggle ("Robust loss (Charbonnier)"). This smoothly transitions from L2 near zero to L1 for large residuals, reducing the influence of outliers, specular highlights, or mis-modeled pixels. Typical ε range: 5e-4 – 2e-3 (default 1e-3). Gradients are scaled by r / sqrt(r^2 + ε^2) in the backward pass.
- Optimizer: Adam (EMA, decoupled weight decay) for color, opacity, `sigma`, and geometry (positions, anisotropic scales).
- Backward on GPU: two-pass accumulation that accounts for transmittance; gradients computed for DC, SH(L1), opacity, isotropic `sigma`, positions (XYZ), and anisotropic scales (proxy via isotropic world sigma split).
- EMA of loss for smoothed logging.
- Mini-batch gradient accumulation: set "Batch frames" > 1 to accumulate gradients over K frames before an Adam step (buffers sum, then averaged); stabilizes training with fewer gaussians or noisy frames.
  - Implementation note: buffers are zeroed only at the start of a batch (when the accumulator is empty). Changing the value of "Batch frames" mid-batch resets the accumulation counters to avoid mixing scales. Valid range is 1–32; UI clamps out-of-range values.
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
 - Fused GPU exposure+residual+loss path (removes CPU per-pixel loops) feeding residual texture directly to backward.
 - Per-tile depth ordering (CPU heuristic near-to-far) improving compositing correctness and enabling future early-outs.
 - Hybrid bucket-based depth ordering in tile builder (large tiles use O(n) bucket pass, small tiles keep full sort); metrics exposed via MetalRenderer.depthOrderingStats().
 - Robust Charbonnier loss (GPU) with UI toggle and epsilon slider; backward scales residuals for stable robust gradients.
 - Multi-frame mini-batch gradient accumulation (Batch frames control) averaging per-frame gradients before Adam update.
 - Residual heatmap downsampling kernel (GPU) and initial densification/pruning (split/prune) pipeline with UI controls (experimental).
 - Screen-space cache (uv, invz, zf, J rows, Σ2D, Σ2D^{-1}, viewDir) now implemented for BOTH backward and forward passes; forward reuse skips rebuild when camera + gaussian parameters unchanged (camera hash + gaussians version). Metrics: fCacheBuild=XXms (last miss), cacheHits, cacheMisses, cacheHitRate.
 - Analytic position gradient refined: includes first-order dA/dcam via Σ' contributions (partial A' = -A Σ' A) improving anisotropic accuracy.
 - Tiled backward kernel now performs threadgroup reductions for color, opacity, sigma, position, scale gradients cutting global atomic traffic.

---

## 13. Roadmap / Performance Metrics (Preview)

Roadmap (phased):
- Phase 1: (DONE) GPU fused loss (removed CPU residual build), (DONE) per-tile depth ordering heuristic, (DONE) shared-memory tiled gradient reduction (experimental toggle).
 - Phase 1.1: (DONE) Hybrid bucket depth ordering (CPU) to cut large-tile sort cost; instrumentation of bucket vs full sort counts.
- Phase 2: Exact anisotropic position gradients, robust loss (Charbonnier), layerwise LR schedules.
- Phase 3: Densification (split high-error splats) + pruning, multi-frame mini-batch, log-scale parameterization of anisotropic scales.
- Phase 4: LOD / clustering for large N, foveated update, compressed FP16 checkpoints.
- Phase 5: Gradient finite-diff self-test, residual heatmap-driven adaptive sampling.

Runtime metrics now logged every N steps (perf line):
- forward=X ms, backward=Y ms, tile=Z ms (CPU tile build), fCacheBuild=F ms (screen cache CPU build on last miss), cacheHits=H cacheMisses=M cacheHitRate=R%.
- Preview PNG generation throttled to reduce UI overhead.

---

### (Updated) Exact Anisotropic Scale (partial) & Position Gradients

Scale per-axis gradients were upgraded earlier via explicit Σ2D = J Σ3D J^T differentiation (dsx, dsy, dsz block in the Metal backward). Position gradients have now been switched from an isotropic proxy (treating a single screen sigma) to an anisotropic formulation that uses the inverse projected covariance A = Σ2D^{-1} applied to the pixel offset.

Implemented now:
1. Per-axis scale grads: dL/dσ_i = dL/dw * w * σ_i * (t_i^2) with t_i = j_i^T (A d) (already present). This comes from chain: Σ2D = Σ_k σ_k^2 j_k j_k^T, u = A d, w = exp(-1/2 d^T A d).
2. Position grads: ∂w/∂d = -w * (A d) = -w * u. Because d = pixel - uv(X), and uv depends on camera differential via perspective Jacobian J_cam (du/dcam rows j0, j1), we obtain (ignoring ∂A/∂X second-order terms):
  ∂w/∂cam ≈ w * u^T * J_cam. This replaces the earlier isotropic screen-sigma approximation.

Remaining approximations / future work:
- We ignore derivative of A (i.e., of Σ2D^{-1}) w.r.t. camera-space position in ∂w/∂X. This term is second-order and typically smaller but can matter for extreme anisotropy.
- Forward caching is still not performed: per-pixel recomputation of J and Σ2D inverse continues; we will introduce a per-gaussian cache (u,v, 1/z, J rows, Σ2D, Σ2D^{-1}) to cut redundant math and enable full exact gradients.
- Anisotropy regularization (ratio clamp/penalty) not yet active.

Original derivation reference kept below for completeness.

Exact formulation proceeds via the 2D projected covariance:

1. 3D diagonal covariance: Σ3D = diag(σx^2, σy^2, σz^2).
2. Projection Jacobian J (2×3) maps local 3D differentials to screen (depends on intrinsics and perspective division). Screen-space covariance: Σ2D = J Σ3D J^T.
3. Per-pixel Gaussian density (ignoring color/alpha): G = exp(-0.5 * d^T Σ2D^{-1} d), where d is screen-space offset.
4. Composited color contribution C ∝ α * G (with transmittance T). Loss L = Σ_p ℓ( (C_pred - C_gt)_p ).

Derivatives needed:
- dL/dσi via chain: dL/dG * dG/dΣ2D * dΣ2D/dσi.  For Σ2D = Σ over axes k: (dΣ2D/dσi) = J E_i J^T * 2σi where E_i selects axis i.
- dG/dΣ2D uses identity: ∂/∂Σ ( -0.5 d^T Σ^{-1} d ) = 0.5 Σ^{-T} (d d^T) Σ^{-T} - 0.5 Σ^{-T}.
  Thus dG/dΣ2D = G * (0.5 Σ^{-1} (d d^T) Σ^{-1} - 0.5 Σ^{-1}).
- Positions: screen position u depends on world position X through projection; dG/dX = (∂G/∂d) (∂d/∂u) (∂u/∂X). We have ∂G/∂d = -G * Σ2D^{-1} d. Perspective: u = (K [R|t] X)_xy / z_cam; differentiate quotient (store per-splat inverse depth and Jacobian during forward to reuse).

Planned phases:
Phase A (DONE partial): Scale grads exact; position grads upgraded to anisotropic (A d) chain (without dA/dX) — no cache yet.
Phase B (NEXT): Introduce forward cache (u,v, 1/z, J rows, Σ2D, Σ2D^{-1}) to avoid per-pixel recomputation and enable reuse in backward.
Phase C: Incorporate dA/dX term for fully exact position derivatives; add finite-difference validation harness.
Phase D: Regularization schedule: penalize extreme anisotropy ratio (max(σ)/min(σ)); optional log-parameterization of σ_i.

Numerical stability notes:
- Clamp eigenvalues of Σ2D to ≥ 1e-6 before inversion.
- Use symmetric 2×2 inverse closed-form: for [[a,b],[b,c]] inv = (1/Δ) [[c,-b],[-b,a]].
- Accumulate gradients in float32; optional Kahan summation if precision issues appear in large tiles.

RU (кратко): точные градиенты по σx/σy/σz и позициям через явную производную Σ2D = J Σ3D J^T. Кешируем J, Σ2D^{-1}, центр и 1/z во форварде; в бэкварде применяем формулу ∂G/∂Σ и цепочку. Фазы A–D (кэш → dσ → dX → регуляризация).

---

### (Planned) Densification & Pruning

Goal: adaptively increase model capacity where residual error persists while pruning ineffective splats.

Heatmap generation:
1. From residual texture (already linear half), compute per-pixel magnitude m = ||r||_2 (optionally robust-adjusted if Charbonnier active: use sqrt(r^2+ε^2)-ε approximation).
2. Downsample to e.g. 1/4 resolution via 2×2 or 4×4 average pooling on GPU (new compute kernel) to obtain a coarse heatmap H.
3. For each Gaussian, sample H at its projected center (cache screen coords from forward or recompute lightweight projection) to get an importance score.

Split criterion:
- If importance > splitThreshold AND opacity in [minOpacityForSplit, maxOpacityForSplit] AND scale within allowed range, duplicate Gaussian into 2–4 children.
- Children positions: jitter inside an ellipsoid proportional to current anisotropic scales (e.g. ±0.3 σ along principal axes or axis-aligned for now).
- Color & SH: inherit parent coefficients; opacity divided by childCount * attenuationFactor to preserve transmittance.

Prune criterion:
- If opacity < pruneOpacityThreshold OR (importance < lowResidualThreshold AND age > minAge), mark for removal.
- Age tracked as iterations since creation (store alongside GaussianParam if extended struct is added later; interim can use parallel Int array).

Execution schedule:
- Every densifyInterval iterations (e.g. 200):
  1. Build heatmap.
  2. Score gaussians; collect splits & prunes (cap total new gaussians per pass to maxNewPerDensify).
  3. Apply prunes (compact arrays), then apply splits (append children), reallocate GPU buffers if capacity exceeded.
  4. Reinitialize optimizer moments for new gaussians (zeros) and copy moments for split children proportionally if desired.

Data structure adjustments:
- Maintain parallel arrays (gaussians, scaleX/Y/Z, SH, optimizer state). Splits append; pruning compacts with swap-remove + consistent reordering of all arrays.
- Consider capacity reservation to reduce reallocs (e.g. reserve 1.5× initial count when starting densification phase).

Numerical safeguards:
- Enforce minimum scale (σ > 0.002) and maximum scale.
- Prevent opacity explosion: cap parent opacity before split; ensure children opacity <= parentOpacity * 0.6 / childCount.

RU кратко: каждые N шагов строим тепловую карту остатка, сплитаем сплаты с высоким остатком (деля opacity) и удаляем слабые/прозрачные. Ограничиваем количество новых за проход, обновляем буферы и моменты Adam.

---

# Дополнение: Roadmap / Метрики (RU)

Фазы:
1. (СДЕЛАНО) Fused GPU loss, (СДЕЛАНО) глубинная сортировка внутри тайла, (СДЕЛАНО) редукция градиентов в shared memory (экспериментальная tiled backward).
2. Точные градиенты по позициям (анизотропия), робастный лосс, layerwise LR / расписания.
3. Деснификация (split) + pruning, мини-батч по нескольким кадрам, лог-параметризация масштабов.
4. LOD/кластеризация, фовеированное обновление, сжатые FP16 чекпоинты.
5. Finite-diff тест градиентов, тепловая карта остатка для адаптивного семплинга.

Логируются метрики: forward/backward/tile (мс) каждые perfLogInterval итераций; превью PNG реже (previewInterval) для снижения задержек.

---

# Gaussian Splat Trainer — Документация (RU)

Приложение — офлайновый тренер Gaussian Splatting для iOS на Swift/SwiftUI и Metal. Загружает датасеты камер (`transforms.json`/`dataset.json`) и облака точек (`points.ply`), рендерит предсказания на GPU и обучает параметры гауссиан (и опционально геометрию) с помощью Adam, минимизируя L2-лосс по GT-кадрам. Есть чекпоинты и простой интерфейс.

- Платформы: iOS 17+ (проверено на iOS 17/18)
- Стек: Swift 5.9+, SwiftUI, Metal (RGBA16F), simd; опционально MPSGraph
- Форматы: `transforms.json` (nerfstudio), `dataset.json` (кастом), `points.ply` (с поддержкой 3DGS полей)

## 1. Как это работает

- Загружается датасет (интринсики, экстрансики, изображения). Интринсики корректируются при letterbox-ресайзе.
- Загружается `points.ply` (цвет/прозрачность/скейлы, если есть) и отправляется в GPU-буферы.
- Рендеринг на Metal compute (front-to-back, RGBA16F).
- Считается L2-лосс в линейном цветовом пространстве, нормализуется экспозиция per-frame скалярным коэффициентом, параметры обновляются Adam.
- Итеративное обучение; предпросмотр, оверлей и логи в UI. Чекпоинты — для сохранения/восстановления.

## 2. Архитектура

- SwiftUI (`TrainingView`) управляет циклом обучения (`Trainer`).
- `DatasetManager` грузит и масштабирует интринсики, нормализует ориентацию изображений; поддерживает `transforms.json` и `dataset.json`.
- `PointsPLYParser` парсит PLY (ASCII/бинарный LE), распознаёт 3DGS-поля (`f_dc_*`, `opacity`, `scale_*`).
- `MetalRenderer` создаёт compute pipeline, RGBA16F-цели и shared-буферы, запускает рендер каждого кадра.
- `Trainer` выполняет рендер, лосс/градиенты, Adam, EMA, диагностику, Sim3-рефайн, опциональные обновления геометрии и логи.
- `CheckpointManager` — сохранение/загрузка состояния.
- `AutoDatasetBuilder` — автогенерация `dataset.json` из фотографий в Documents/images.

## 3. Форматы данных

### 3.1 transforms.json (как в nerfstudio)
- Кадры содержат интринсики (или выводимые из FOV) и `transform_matrix`.
- Поддерживаются варианты, где перенос лежит в строке или столбце; при необходимости делается транспонирование.

### 3.2 dataset.json (кастом)
```
{
  "width": <int>,
  "height": <int>,
  "frames": [
    {
      "imageFilename": "images/xxx.png",
      "intrinsics": {"fx":..., "fy":..., "cx":..., "cy":...},
      "extrinsics": {"m": [16 float, row-major]}
    }, ...
  ]
}
```
- Интринсики масштабируются с учётом letterbox.
- Экстрансики — 4x4 row-major.

### 3.3 points.ply
- Обязательны XYZ. Опционально: цвет (3DGS DC: `f_dc_0/1/2`), `opacity`, `scale_0/1/2`.
- Поддерживаются LE-бинарный и ASCII.

## 4. Рендеринг (Metal)

- Форвард растеризует гауссианы «спереди-назад» с pre-multiplied alpha в RGBA16F-цель.
- Зависимый от вида цвет через SH(L1): на канал хранится `[c0, c1x, c1y, c1z]`, оценивается по направлению взгляда.
- Анизотропные следы: 3D-диагональная ковариация `diag(scale.x^2, scale.y^2, scale.z^2)` проектируется Якобианом в экран `Σ2D = J Σ3D J^T`; вес по расстоянию Махаланобиса.
- Экранный тайлинг/биннинг: для каждого пикселя перебираются только гауссианы, перекрывающие его тайл; списки тайлов строятся на CPU каждый кадр.
- CPU-конвертация RGBA16F→RGBA8 для предпросмотра в UI.

## 5. Оптимизация

- Лосс: L2 в линейном RGB + per-frame экспозиция (замкнутая форма МНК). Опционально робастный Charbonnier (sqrt(r^2 + ε^2)) — переключатель в UI. При малых r ≈ L2, при больших r плавно ведёт себя как L1, подавляя выбросы / пересветы. Типичный диапазон ε: 5e-4 – 2e-3 (по умолчанию 1e-3). В бэкварде остаток масштабируется r / sqrt(r^2 + ε^2).
- Оптимизатор: Adam (EMA, decoupled weight decay) для цвета, opacity, `sigma` и геометрии (позиции, анизотропные масштабы).
- Бэквард на GPU: двухпроходный с учётом трансмиттанса; градиенты: DC, SH(L1), opacity, изотропный `sigma`, позиции (XYZ), анизотропные масштабы (прокси).
- EMA лосса для сглаженного логирования.
- Мини-батч: параметр "Batch frames" > 1 накапливает градиенты по K кадрам и делает один Adam шаг (усреднение), повышая стабильность.
  - Реализация: буферы обнуляются только в начале батча. Если значение "Batch frames" меняется во время накопления, счётчики сбрасываются для корректности. Допустимый диапазон 1–32; UI делает кламп.
- Геометрия: позиции обновляются GPU-градами; масштабы по осям оптимизируются отдельно; точные производные по анизотропии в плане.

## 6. Sim(3)

- Auto-align: перебор флип/скейл/знак forward, чтобы избежать чёрных кадров и улучшить покрытие.
- Coordinate Descent: глобальный Sim3-рефайн.
- Gauss–Newton (точный): 7 параметров, численный Якобиан, JTJ/JTr (7×7), бэктрекинг, адаптивное демпфирование, лог-шкала для скейла, нормализация остатков.

## 7. Автогенерация датасета

- Сложите фото в Documents/images на устройстве/симуляторе.
- Читает EXIF (`FocalLenIn35mmFilm`) и оценивает FOV; иначе берёт FOV по умолчанию.
- Генерирует круговую траекторию и пишет `dataset.json` в Documents.

## 8. UI и использование

- Start/Pause — управление обучением.
- Gen dataset.json — создать датасет из фото в Documents/images и подгрузить его.
- Auto-align — подбирает флипы/скейл/forward; печатает диагностику.
- Sim3 Refine / Sim3 Refine (GN) — глобальная подстройка позы.
- Reseed colors — инициализация цветов гауссиан по GT в линейном пространстве.
- Resolution — изменение масштаба (long side) для скорости.
- Gaussians — количество гауссиан.
- Переключатели: Overlay projections; Geometry fine-tuning.
- Preview/Overlay — визуализация предсказания и центров.
- Logs — шаг/лосс/экспозиция/opacity/покрытие/диагностика.

Подсказки:
- Начните с 1k–4k гауссиан и уменьшенного разрешения; затем увеличивайте.
- Используйте Auto-align при чёрных кадрах или низком in-bounds.
- Делайте GN-рефайн после нескольких сотен шагов для стабилизации позы.

## 9. Чекпоинты

- Сохраняет параметры гауссиан и состояние тренера.
- Теперь сохраняются SH(L1) и анизотропные масштабы (scaleX/Y/Z) по каждой гауссиане.
- Можно восстановиться после перезапуска.

## 10. Решение проблем

- Previews не ставятся в симулятор (Client not entitled): для схемы Run/Preview выберите Debug, очистите DerivedData, попробуйте другой симулятор.
- Чёрные кадры: смените знак forward (zSign) через Auto-align или дождитесь авто-переключения; проверьте масштаб интринсик.
- Взрыв лосса: уменьшите learning rate; проверьте линейную конверсию и экспозицию.
- PLY не читается: проверьте LE/ASCII и имена полей 3DGS.

## 11. Производительность

- Экранный тайлинг (16×16) сокращает перебор гауссиан на пиксель до близлежащих сплатов и существенно ускоряет кадр при больших `N`.
- Для скорости уменьшайте downscale.
- RGBA16F + shared storage минимизируют копирования CPU↔GPU.
- Не начинайте с слишком большого числа гауссиан.

## 12. Недавние изменения

- Добавлены SH(L1) для цвета во форварде и градиенты DC/SH(L1) в бэкварде.
- Реализованы анизотропные следы через проекцию ковариации; добавлены градиенты по изотропному `sigma` и прокси-грады для осевых масштабов.
- Улучшён бэквард композитинга: двухпроходный, учитывает трансмиттанс — стабильнее и точнее.
- Добавлены GPU-градиенты по позициям (XYZ) и Adam-обновления на устройстве.
- Чекпоинты теперь сохраняют SH(L1) и осевые масштабы; добавлена обратная совместимость при загрузке.
- Экранный отсев по 3σ и тайлинг экрана со списками гауссиан на тайл (строятся на CPU каждый кадр).
- Исправлены баги, улучшены логи/тексты UI; портретный UI и стабильные Previews.
 - Добавлена опциональная поддержка SH(L2) (9 гармоник/канал) с пакетным накоплением градиентов, Adam-обновлениями для 6 доп. коэффициентов на канал, глобальным клиппингом градиентов и периодической диагностикой (RMS градиентов, счёт активных коэффициентов).
 - Робастный Charbonnier лосс (GPU) с переключателем и слайдером ε; устойчивее к выбросам.
 - Мини-батч накопление градиентов (Batch frames) с усреднением перед Adam.
 - Тепловая карта остатка (даунсемплинг на GPU) и первичная деснификация (split/prune) с UI контролами (экспериментально).

---

## PPPs (Progress, Plans, Problems)

Progress
- Exposure pipeline hardened: added warmup freeze (stable early color) + master exposure enable toggle + per-iteration gain logging (with override flag indicator).
- Gradient diagnostics framework: periodic RMS / max / zero-fraction stats for DC, SH(L1), opacity, sigma, XYZ position, per-axis scales, and combined SH(L2) (if enabled). Helps detect vanishing / exploding grads early.
- Intrinsics auto-recentering & scale sanity heuristic prevents black frames when dataset intrinsics are off-center or mis-scaled; logs correction action.
- Texture allocation fix: removed legacy 16×16 placeholder; runtime reallocation safeguards guarantee training at dataset resolution (eliminated gray/flat frames issue).
- Dummy buffer bindings for forward/backward when screen cache disabled (prevents Metal validation errors on required indices).
- Screen-space forward cache toggle implemented (unified with backward cache approach); dummy fallbacks ensure stable kernel dispatch when disabled.
- Retained and integrated prior milestones: SH(L1) forward color, anisotropic splats, two-pass transmittance-aware backward, Adam on geometry, checkpoint persistence for SH(L1)+scales, 3σ culling & 16×16 tiling, hybrid depth ordering, SH(L2) optional path, Charbonnier robust loss, mini-batch accumulation, residual heatmap & early densification scaffolding.

Plans
1. Full anisotropic position gradients: incorporate missing dA/dX (Σ2D^{-1} derivative) term for exact ∂L/∂X.
2. Exact per-axis scale gradients (replace proxy) via explicit dΣ2D/dσ_i and Mahalanobis chain; add anisotropy ratio regularizer.
3. Finite-difference harness (select M splats, ε ~1e-3) validating position & scale grads (target relative error <1e-2); gate new analytic terms.
4. Tile profiling & heuristic tuning: collect occupancy histogram → adjust bucket vs full sort threshold adaptively.
5. UI toggles: runtime switches for gradient diagnostics verbosity, tiled backward, exposure warmup length, forward cache.
6. Extended checkpoints: persist exposure gain state, warmup params, charb ε, cache hit stats, SH L2 enable flag.
7. Residual distribution logging → adaptive Charbonnier ε scheduler (percentile-based) + optional residual clipping.
8. Memory / allocation optimization: buffer pool for tile lists & screen cache; reuse reduction buffers; optional FP16 parameter storage.
9. (Stretch) Early-out in tiles using accumulated alpha; frustum plane generation & rejection in CPU binning to cut large-N cost.

Problems / Risks
- Position gradients still approximate: missing dΣ^{-1}/dX path can bias optimization for highly elongated splats until Phase 1 complete.
- Scale gradients proxy may mis-shape anisotropy (elongated artifacts) on structured scenes; priority to replace.
- Gradient diagnostics O(N) CPU scan adds overhead at high N; mitigation: sampling or reduced frequency (already gated after 200 steps & every 500 thereafter).
- CPU tile build cost scales with N and view changes; need cache/reuse or partial incremental rebuild.
- Potential numerical instability for extreme anisotropy (ill-conditioned Σ2D) — currently clamped via min eigenvalue heuristic only; may require log-parameterization.
- Exposure warmup can lock suboptimal brightness if dataset frames vary strongly in luminance ordering (mitigate by shorter warmup or per-scene heuristic).
- Additional SH(L2) increases gradient variance; may need layerwise LR or separate clipping in larger scenes.

---

## PPPs (Прогресс, Планы, Проблемы)

Прогресс
- Экспозиция: добавлен warmup (заморозка gain первых N итераций), глобальный переключатель, логирование коэффициента на каждом шаге.
- Градиентная диагностика: периодический расчёт RMS / max / доли нулевых значений по группам (DC, SH(L1), opacity, sigma, XYZ, масштабы по осям, SH(L2) при включении).
- Автоисправление интринсик: центрирование и масштабная проверка предотвращают «чёрные кадры» при смещённых `cx, cy`.
- Исправление 16×16 плейсхолдера: гарантируется выделение текстур под фактическое разрешение датасета + проверка и переаллокация в рантайме.
- Dummy-буферы при отключённом screen cache (forward/backward) устраняют ошибки привязки Metal.
- Опциональный forward cache с безопасным фоллбеком — единообразная логика с backward cache.
- Сохранены предыдущие достижения: SH(L1/L2), анизотропный рендер, двухпроходный бэквард, Adam по геометрии, расширенные чекпоинты, гибридная глубинная сортировка, Charbonnier, мини-батчи, heatmap + начальная деснификация.

Планы
1. Полные позиционные градиенты: добавить dA/dX член (точная производная Σ2D^{-1}) для ∂L/∂X при сильной анизотропии.
2. Точные градиенты масштабов: dΣ2D/dσ_i + регуляризация отношения max/min σ (предотвращение вырожденности).
3. Finite-difference harness (M≈8, ε≈1e-3) для проверки позиций и масштабов (ошибка <1e-2) до включения новых аналитических терминов по умолчанию.
4. Профилировка тайлинга и автоподбор порога bucket vs sort на основе гистограммы заполненности.
5. UI-переключатели: частота диагностики, включение tiled backward, длина warmup, forward cache.
6. Расширенные чекпоинты: сохранять состояние экспозиции, warmup, charb ε, статистику кэшей, флаг SH L2.
7. Лог распределения резидуалов → адаптивный Charbonnier ε (перцентили) + опциональный клип.
8. Оптимизация памяти: пул для tile/caches, переиспользование редукционных буферов, опция FP16 хранения параметров.
9. (Stretch) Ранний выход по альфе на тайле; фрустум-плоскости в биннинге.

Проблемы / Риски
- Позиционные градиенты пока без dΣ^{-1}/dX: возможен лёгкий bias при вытянутых сплатах.
- Прокси-градиенты масштабов могут искажать анизотропию — приоритет на замену.
- O(N) пересчёт статистики градиентов — накладные расходы при больших N; требуется сэмплирование или реже логировать.
- CPU биннинг масштабируется по N; нужен частичный reuse либо инкрементальные обновления.
- Возможная численная нестабильность при экстремальной анизотропии (малые собственные значения Σ2D) — пока только минимальный кламп.
- Warmup экспозиции может «застрять» на под- или переоценённом уровне при неоднородной яркости кадров.
- SH(L2) повышает вариативность градиентов; возможна потребность в отдельном клиппинге / LR.
