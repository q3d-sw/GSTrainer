#!/usr/bin/env python3
"""
Quick auto-align helper for Gaussian Splatting datasets.
- Reads ASCII PLY (x y z) and dataset.json (Nerfstudio-like)
- Tries all flips (X/Y/Z) × camera forward sign (−Z / +Z)
- Computes front% and inBounds% across a sample of frames
- Recommends the best combination

Usage:
  python3 tools/gs_auto_align.py --ply path/to/points.ply \
      --dataset path/to/dataset.json \
      --sample-frames 12 --w 1920 --h 1080 --fx 1437.1254 --fy 1437.1254 --cx 963.4962 --cy 537.2931

If fx/fy/cx/cy are omitted, script tries reading them from dataset.json.
"""
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Camera:
    w2c: List[List[float]]  # 4x4 world-to-camera (row-major)
    fx: float
    fy: float
    cx: float
    cy: float
    w: int
    h: int


def _normalize_c2w(m: List[List[float]]) -> List[List[float]]:
    """
    Many Nerfstudio/ARKit dumps store 4x4 as column-major (translation in last ROW).
    Detect this case and transpose so translation ends up in last COLUMN.
    """
    if len(m) == 4 and all(len(r) == 4 for r in m):
        eps = 1e-7
        last_col_zero = abs(m[0][3]) < eps and abs(m[1][3]) < eps and abs(m[2][3]) < eps
        last_row_has_t = (abs(m[3][0]) > eps) or (abs(m[3][1]) > eps) or (abs(m[3][2]) > eps)
        last_row_one = abs(m[3][3] - 1.0) < 1e-5
        if last_col_zero and last_row_has_t and last_row_one:
            # looks like translation is in last ROW -> transpose
            mt = [list(row) for row in zip(*m)]
            return mt
    return m


def load_dataset(dataset_path: str) -> Tuple[List[List[List[float]]], dict]:
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    frames = data["frames"] if isinstance(data.get("frames"), list) else []
    intr = {
        "w": data.get("w"),
        "h": data.get("h"),
        "fl_x": data.get("fl_x") or data.get("fx"),
        "fl_y": data.get("fl_y") or data.get("fy"),
        "cx": data.get("cx"),
        "cy": data.get("cy"),
    }
    # dataset typically stores camera_to_world; convert to world_to_camera
    c2w_list = []
    for f in frames:
        m = f.get("transform_matrix")
        if not m:
            continue
        # Some datasets store as 4x4 with last row as translation; normalize
        if len(m) == 4 and len(m[0]) == 4:
            c2w_list.append(_normalize_c2w(m))
    return c2w_list, intr


def mat4_inverse(m: List[List[float]]) -> List[List[float]]:
    # simple 4x4 inverse for rigid/affine matrices
    import numpy as np
    M = np.array(m, dtype=float)
    Minv = np.linalg.inv(M)
    return Minv.tolist()


def load_ply_xyz(ply_path: str) -> List[Tuple[float, float, float]]:
    pts = []
    with open(ply_path, 'r') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                break
            header.append(line.strip())
            if line.strip() == 'end_header':
                break
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[:3])
            except:
                continue
            pts.append((x, y, z))
    return pts


def project_point(cam: Camera, w2c, p: Tuple[float, float, float], z_sign: int) -> Tuple[bool, bool]:
    # w2c: 4x4 row-major
    x, y, z = p
    X = [x, y, z, 1.0]
    # multiply row-major: v' = M @ v
    Xc = [
        w2c[0][0]*X[0] + w2c[0][1]*X[1] + w2c[0][2]*X[2] + w2c[0][3]*X[3],
        w2c[1][0]*X[0] + w2c[1][1]*X[1] + w2c[1][2]*X[2] + w2c[1][3]*X[3],
        w2c[2][0]*X[0] + w2c[2][1]*X[1] + w2c[2][2]*X[2] + w2c[2][3]*X[3],
        w2c[3][0]*X[0] + w2c[3][1]*X[1] + w2c[3][2]*X[2] + w2c[3][3]*X[3],
    ]
    if Xc[3] != 0:
        Xc = [Xc[0]/Xc[3], Xc[1]/Xc[3], Xc[2]/Xc[3], 1.0]
    # camera forward sign: if +Z, point is in front when z>0; if -Z, when z<0
    zc = Xc[2]
    in_front = (zc * z_sign) > 0
    if not in_front:
        return False, False
    u = cam.fx * (Xc[0]/zc) + cam.cx
    v = cam.fy * (Xc[1]/zc) + cam.cy
    in_bounds = (u >= 0) and (u < cam.w) and (v >= 0) and (v < cam.h)
    return True, in_bounds


def evaluate(points: List[Tuple[float,float,float]], c2w_list: List[List[List[float]]], intr: dict,
             flips: Tuple[int,int,int], z_sign: int, sample_frames: int) -> Tuple[float,float]:
    import numpy as np
    w = intr.get("w")
    h = intr.get("h")
    fx = intr.get("fl_x")
    fy = intr.get("fl_y")
    cx = intr.get("cx")
    cy = intr.get("cy")
    if None in (w,h,fx,fy,cx,cy):
        raise ValueError("Missing intrinsics (w,h,fl_x,fl_y,cx,cy). Pass via CLI or ensure dataset.json contains them.")
    # make cameras
    indices = list(range(len(c2w_list)))
    if sample_frames and sample_frames < len(indices):
        random.seed(0)
        indices = random.sample(indices, sample_frames)
    cams: List[Camera] = []
    w2cs: List[List[List[float]]] = []
    for i in indices:
        c2w = c2w_list[i]
        w2c = mat4_inverse(c2w)
        cams.append(Camera(w2c=w2c, fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h))
        w2cs.append(w2c)
    # apply flips to points
    fx_, fy_, fz_ = flips
    pts = [(fx_*x, fy_*y, fz_*z) for (x,y,z) in points]
    # sample subset of points for speed
    N = min(20000, len(pts))
    if len(pts) > N:
        random.seed(0)
        pts = random.sample(pts, N)
    total = 0
    front = 0
    inb = 0
    for cam, w2c in zip(cams, w2cs):
        for p in pts:
            total += 1
            f, ib = project_point(cam, w2c, p, z_sign)
            if f:
                front += 1
            if ib:
                inb += 1
    front_pct = 100.0 * front / max(1,total)
    inb_pct = 100.0 * inb / max(1,total)
    return front_pct, inb_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--sample-frames', type=int, default=12)
    ap.add_argument('--w', type=int)
    ap.add_argument('--h', type=int)
    ap.add_argument('--fx', type=float)
    ap.add_argument('--fy', type=float)
    ap.add_argument('--cx', type=float)
    ap.add_argument('--cy', type=float)
    args = ap.parse_args()

    c2w_list, intr = load_dataset(args.dataset)
    # override intrinsics if provided
    for k in ['w','h','fl_x','fl_y','cx','cy']:
        v = getattr(args, k if k in ['w','h','cx','cy'] else ('fx' if k=='fl_x' else 'fy' if k=='fl_y' else k), None)
        if v is not None:
            intr[k] = v
    # map fx/fy from args
    if args.fx is not None:
        intr['fl_x'] = args.fx
    if args.fy is not None:
        intr['fl_y'] = args.fy

    pts = load_ply_xyz(args.ply)

    combos = []
    signs = [-1, 1]  # flips per axis
    for fx_ in signs:
        for fy_ in signs:
            for fz_ in signs:
                for zsign in [-1, 1]:  # -Z or +Z forward
                    front_pct, inb_pct = evaluate(pts, c2w_list, intr, (fx_,fy_,fz_), zsign, args.sample_frames)
                    combos.append({
                        'flipX': fx_==-1,
                        'flipY': fy_==-1,
                        'flipZ': fz_==-1,
                        'zSignPlusZ': zsign==1,
                        'front%': front_pct,
                        'inBounds%': inb_pct,
                    })
    combos.sort(key=lambda d: (round(d['inBounds%'],3), round(d['front%'],3)), reverse=True)
    best = combos[0]
    print("Best configuration:")
    print(json.dumps(best, indent=2))
    print("\nTop 5:")
    for c in combos[:5]:
        print(json.dumps(c, indent=2))

if __name__ == '__main__':
    main()
