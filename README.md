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

- Лосс: L2 в линейном RGB; per-frame коэффициент экспозиции (замкнутая форма МНК).
- Оптимизатор: Adam (EMA, decoupled weight decay) для цвета, opacity, `sigma` и геометрии (позиции, анизотропные масштабы).
- Бэквард на GPU: двухпроходное накопление с учётом трансмиттанса; считаются градиенты для DC, SH(L1), opacity, изотропного `sigma`, позиций (XYZ) и анизотропных масштабов (пока прокси через распределение изотропного града по осям).
- EMA лосса для сглаженного логирования.
- Геометрическое дообучение: позиции обновляются по GPU-градиентам; анизотропные масштабы оптимизируются по осям и синхронизируются в GPU; точные производные для масштабов запланированы.

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

---

## PPPs (Progress, Plans, Problems)

Progress
- Implemented SH(L1) view-dependent color and anisotropic splats in forward; improved visual fidelity.
- Moved backward to the GPU with two-pass transmittance-aware accumulation; added grads for DC, SH(L1), opacity, `sigma`, positions (XYZ), and proxy grads for anisotropic scales.
- Hooked up Adam updates for geometry (positions and per-axis scales) and synced parameters to GPU every step.
- Extended checkpointing to persist SH(L1) and scaleX/Y/Z; added robust load path.
- Added 3σ screen-space culling and new screen-space tiling (16×16) with per-tile gaussian lists; noticeable speedups at scale.
- Fixed build issues, preview reliability, and localized UI/logs.
 - Dataset ingestion: supports nerfstudio-like `transforms.json` and custom `dataset.json` with proper intrinsics scaling (letterbox) and orientation handling.
 - PLY parser: ASCII/binary LE with 3DGS fields (`f_dc_*`, `opacity`, `scale_*`); initializes GPU buffers from `points.ply`.
 - Per-frame exposure gain estimation in linear color space (closed-form least squares) for robust photometric training.
 - Auto-align utility (flips/scale/forward sweep) and Sim(3) refinement via coordinate descent; precise Gauss–Newton Sim(3) with damping, backtracking, log-scale parameterization, and residual normalization.
 - Auto dataset builder: generates `dataset.json` from photos in `Documents/images`, uses EXIF (`FocalLenIn35mmFilm`) when available.
 - UI/UX: Start/Pause, build dataset.json, overlay projections, reseed colors, resolution and gaussian-count controls, logs and diagnostics.
 - Rendering/preview pipeline: RGBA16F compute target and CPU RGBA16F→RGBA8 conversion path for SwiftUI previews.
 - Bilingual documentation (EN/RU) and troubleshooting notes.

Plans
- Replace proxy scale gradients with exact derivatives through `Σ2D = J Σ3D J^T` and Mahalanobis chain; add regularization schedules.
- Further speedups: tile/frustum culling, per-tile early-out by alpha, and buffer reuse across frames.
- Quality improvements: exposure/opacity priors tuning, gradient clipping policies, optional SH L2/L3.
- Add minimal tests for checkpoint round-trip and binning correctness.

Problems / Risks
- Anisotropic scale gradients are currently a proxy; can bias shape if relied upon heavily — scheduled for exact derivation.
- CPU binning per frame has overhead for very large `N` and high FPS; mitigated by small tiles and potential reuse.
- Mobile GPUs have tight threadgroup limits; careful tuning of tile size and kernel occupancy is needed.

---

## PPPs (Прогресс, Планы, Проблемы)

Прогресс
- Реализованы SH(L1) и анизотропные сплаты во форварде; выросло качество.
- Бэквард перенесён на GPU, двухпроходный с учётом трансмиттанса; добавлены градиенты: DC, SH(L1), opacity, `sigma`, позиции (XYZ) и прокси-грады для анизотропных масштабов.
- Подключены Adam-обновления геометрии (позиции, масштабы по осям) и синхронизация параметров в GPU на каждом шаге.
- Чекпоинты расширены: сохраняются SH(L1) и scaleX/Y/Z; загрузка обратносовместима.
- Добавлены экранный отсев по 3σ и тайлинг (16×16) со списками гауссиан на тайл; заметное ускорение при больших сценах.
- Исправлены баги сборки и стабильность Previews; локализованы UI/логи.
 - Загрузка датасета: поддерживаются `transforms.json` (как в nerfstudio) и кастомный `dataset.json` с корректным масштабированием интринсик (letterbox) и нормализацией ориентации.
 - Парсер PLY: ASCII/бинарный LE с поддержкой 3DGS-полей (`f_dc_*`, `opacity`, `scale_*`); инициализация GPU-буферов из `points.ply`.
 - Оценка экспозиции per-frame в линейном пространстве (замкнутая форма МНК) — устойчивее к автоэкспозиции.
 - Auto-align (перебор флип/скейл/forward) и Sim(3)-рефайн методом координатного спуска; точный Gauss–Newton по 7 параметрам с демпфированием, бэктрекингом, лог-шкалой и нормировкой остатков.
 - Автогенерация датасета: строит `dataset.json` из фото в `Documents/images`, использует EXIF (`FocalLenIn35mmFilm`) при наличии.
 - UI/UX: Start/Pause, генерация dataset.json, overlay projections, reseed colors, управление разрешением и числом гауссиан, логи и диагностика.
 - Рендер/превью: RGBA16F-цель и CPU-конвертация RGBA16F→RGBA8 для SwiftUI-предпросмотра.
 - Документация на русском и английском, раздел с решением проблем.

Планы
- Заменить прокси-градиенты масштабов на точные производные через `Σ2D` и цепочку с расстоянием Махаланобиса; добавить регуляры.
- Ещё ускорение: фрустум- и per-tile отсев, ранний выход по альфе на тайле, переиспользование буферов между кадрами.
- Качество: настройка экспозиции/opacity-приоров, клиппинг градиентов, опция SH L2/L3.
- Мини-тесты на целостность чекпоинтов и корректность биннинга.

Проблемы / Риски
- Текущие градиенты по анизотропным масштабам — прокси; возможна предвзятость формы — планируется точная формула.
- Биннинг на CPU даёт накладные расходы при очень больших `N` и высоком FPS; смягчается малыми тайлами и будущим переиспользованием.
- Аппаратные ограничения мобильных GPU требуют подбора размера тайла и загрузки потоков.
