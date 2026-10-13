
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from field import m0util
_PACK = 1 << 21
_OFF = _PACK // 2

def _voxel_keys(pts: np.ndarray, voxel: float) -> np.ndarray:
    idx = np.floor(pts / voxel).astype(np.int64) + _OFF
    if idx.min() < 0 or idx.max() >= _PACK:
        raise ValueError('koordinatlar voksel paketleme araligini asti')
    return (idx[:, 0] * _PACK + idx[:, 1]) * _PACK + idx[:, 2]

class _Accum:

    def __init__(self) -> None:
        self.k = np.empty(0, dtype=np.int64)
        self.sxyz = np.empty((0, 3))
        self.sconf = np.empty(0)
        self.mconf = np.empty(0)
        self.sd = np.empty(0)
        self.hits = np.empty(0, dtype=np.int64)
        self.views = np.empty(0, dtype=np.int64)

    def add(self, keys, pts, conf, dist, frame_ids) -> None:
        keys = np.asarray(keys, dtype=np.int64)
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        if len(keys) != len(frame_ids):
            raise ValueError('keys ve frame_ids uzunlugu esit olmali')
        chunk_k, inv = np.unique(keys, return_inverse=True)
        chunk_m = len(chunk_k)
        chunk_sxyz = np.zeros((chunk_m, 3))
        np.add.at(chunk_sxyz, inv, pts)
        chunk_sconf = np.zeros(chunk_m)
        np.add.at(chunk_sconf, inv, conf)
        chunk_sd = np.zeros(chunk_m)
        np.add.at(chunk_sd, inv, dist)
        chunk_hits = np.zeros(chunk_m, dtype=np.int64)
        np.add.at(chunk_hits, inv, 1)
        chunk_mconf = np.full(chunk_m, np.inf)
        np.minimum.at(chunk_mconf, inv, conf)
        pairs = np.stack([keys, frame_ids], axis=1)
        view_keys = np.unique(pairs, axis=0)[:, 0]
        view_pos = np.searchsorted(chunk_k, view_keys)
        chunk_views = np.zeros(chunk_m, dtype=np.int64)
        np.add.at(chunk_views, view_pos, 1)
        k = np.concatenate([self.k, chunk_k])
        sxyz = np.concatenate([self.sxyz, chunk_sxyz])
        sconf = np.concatenate([self.sconf, chunk_sconf])
        mconf = np.concatenate([self.mconf, chunk_mconf])
        sd = np.concatenate([self.sd, chunk_sd])
        hits = np.concatenate([self.hits, chunk_hits])
        views = np.concatenate([self.views, chunk_views])
        uk, inv = np.unique(k, return_inverse=True)
        m = len(uk)
        self.k = uk
        self.sxyz = np.zeros((m, 3))
        np.add.at(self.sxyz, inv, sxyz)
        self.sconf = np.zeros(m)
        np.add.at(self.sconf, inv, sconf)
        self.sd = np.zeros(m)
        np.add.at(self.sd, inv, sd)
        self.hits = np.zeros(m, dtype=np.int64)
        np.add.at(self.hits, inv, hits)
        self.views = np.zeros(m, dtype=np.int64)
        np.add.at(self.views, inv, views)
        self.mconf = np.full(m, np.inf)
        np.minimum.at(self.mconf, inv, mconf)

def write_ply(path: Path, xyz: np.ndarray, fields: dict[str, np.ndarray]) -> None:
    n = len(xyz)
    names = list(fields)
    head = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}', 'property float x', 'property float y', 'property float z']
    head += [f'property float {nm}' for nm in names]
    head += ['end_header', '']
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2]] + [fields[nm] for nm in names]
    data = np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)
    with path.open('wb') as f:
        f.write('\n'.join(head).encode('ascii'))
        f.write(np.ascontiguousarray(data, dtype='<f4').tobytes())

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--stride', type=int, default=4, help='her N. kare (varsayilan 4)')
    ap.add_argument('--voxel', type=float, default=0.01, help='voksel boyu m (varsayilan 0.01)')
    ap.add_argument('--chunk', type=int, default=120, help='kac karede bir kaynastir')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--pixel-convention', choices=list(m0util.PIXEL_CONVENTIONS), default=m0util.DEFAULT_PIXEL_CONVENTION, help="RGB->derinlik ana nokta olceklemesi; 'center' yarim-piksel duzeltmesini uygular (bkz m0util.scale_K)")
    ap.add_argument('--depth-correction', type=Path, default=None, help="kosu basina menzil duzeltmesi JSON'u (fit_depth_correction.py; schema cca.depth-correction.v1). Verilmezse KAPALI: d' = d - f(d), backprojection'dan once; esik degil, nokta silmez (kural 1/2)")
    a = ap.parse_args()
    if a.stride < 1 or a.voxel <= 0.0 or a.chunk < 1:
        ap.error('stride/chunk at least one and voxel positive olmali')
    depth_model = m0util.load_depth_correction(a.depth_correction) if a.depth_correction else None
    frames = m0util.list_frames(a.run_dir)[::a.stride]
    out = a.out or Path('data/cc') / f'{Path(a.run_dir).name}_conf.ply'
    print(f'{len(frames)} kare (stride {a.stride}), voksel {a.voxel * 100:.1f} cm, piksel konvansiyonu {a.pixel_convention}')
    if depth_model is not None:
        print(f'menzil duzeltmesi ACIK: {a.depth_correction} {m0util.depth_correction_describe(depth_model)}')
    acc = _Accum()
    bk, bp, bc, bd, bf = ([], [], [], [], [])
    used = skipped = 0
    import cv2
    for i, fr in enumerate(frames, 1):
        if fr.depth_path is None or fr.conf_path is None:
            skipped += 1
            continue
        try:
            T, K_rgb, rgb_wh = m0util.load_pose_intrinsics_rgb_wh(fr.json_path)
        except (ValueError, KeyError):
            skipped += 1
            continue
        if m0util.pose_ortho_error(T) > 0.001:
            skipped += 1
            continue
        d = m0util.read_depth(fr.depth_path)
        c = m0util.read_cv_image(fr.conf_path, cv2.IMREAD_UNCHANGED)
        if c is None or c.shape != d.shape:
            skipped += 1
            continue
        m = m0util.valid_mask(d)
        if not m.any():
            skipped += 1
            continue
        vs, us = np.nonzero(m)
        ds = d[m].astype(np.float64)
        if depth_model is not None:
            m0util.validate_depth_correction_grid(depth_model, d.shape)
            ds = m0util.apply_depth_correction(ds, depth_model, us=us, vs=vs)
        K = m0util.scale_K(K_rgb, rgb_wh, (d.shape[1], d.shape[0]), pixel_convention=a.pixel_convention)
        pts = m0util.backproject_world(us, vs, ds, K, T)
        bk.append(_voxel_keys(pts, a.voxel))
        bp.append(pts)
        bc.append(c[m].astype(np.float64))
        bd.append(ds)
        bf.append(np.full(len(ds), fr.idx, dtype=np.int64))
        used += 1
        if len(bk) >= a.chunk:
            acc.add(np.concatenate(bk), np.concatenate(bp), np.concatenate(bc), np.concatenate(bd), np.concatenate(bf))
            bk, bp, bc, bd, bf = ([], [], [], [], [])
            print(f'  {i}/{len(frames)} kare -> {len(acc.k):,} voksel', flush=True)
    if bk:
        acc.add(np.concatenate(bk), np.concatenate(bp), np.concatenate(bc), np.concatenate(bd), np.concatenate(bf))
    if not len(acc.k):
        print('kullanilabilir depth-confidence karesi yok', file=sys.stderr)
        return 2
    xyz = acc.sxyz / acc.hits[:, None]
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, xyz, {'conf_mean': acc.sconf / acc.hits, 'conf_min': acc.mconf, 'range_m': acc.sd / acc.hits, 'n_views': acc.views.astype(np.float64), 'n_hits': acc.hits.astype(np.float64)})
    print(f'\nkullanilan {used} kare, atlanan {skipped}')
    print(f'{len(xyz):,} nokta -> {out}  ({out.stat().st_size / 1000000.0:.0f} MB)')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
