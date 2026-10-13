
from __future__ import annotations
import argparse
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from field import m0util
from field.make_conf_cloud import _Accum, _voxel_keys
_STOPS = np.array([[220, 40, 40], [235, 185, 20], [40, 190, 70]], dtype=np.float64)
MAX_KEYFRAME_CANDIDATES = 6
VISIBILITY_TOLERANCE_FLOOR_M = 0.05
VISIBILITY_TOLERANCE_QUADRATIC_M_PER_M2 = 0.01
UNASSIGNED_RGB = (255, 0, 255)
UNASSIGNED_KEYFRAME_DISTANCE = -1.0
DEPTH_FILL_PRIOR_M = 16.384
DEPTH_FILL_CONF = 2
DEPTH_FILL_PROPERTY = 'fill_share'

class ColorCloudProcessingError(RuntimeError):
    pass

@dataclass(frozen=True)
class ColorCloudResult:
    rgb_path: Path
    confidence_path: Path
    point_count: int
    indexed_frame_count: int
    eligible_frame_count: int
    used_frame_count: int
    skipped_frame_count: int
    voxel_m: float
    keyframe_count: int = 0
    rgb_assigned_point_count: int = 0
    rgb_unassigned_point_count: int = 0
    max_keyframe_candidates: int = MAX_KEYFRAME_CANDIDATES
    visibility_tolerance_floor_m: float = VISIBILITY_TOLERANCE_FLOOR_M
    visibility_tolerance_quadratic_m_per_m2: float = VISIBILITY_TOLERANCE_QUADRATIC_M_PER_M2
    unassigned_rgb: tuple[int, int, int] = UNASSIGNED_RGB
    unassigned_keyframe_distance: float = UNASSIGNED_KEYFRAME_DISTANCE
    depth_fill_prior_m: float = DEPTH_FILL_PRIOR_M
    depth_fill_conf: int = DEPTH_FILL_CONF
    depth_fill_property: str = DEPTH_FILL_PROPERTY
    depth_fill_labelled_pixel_count: int = 0
    depth_fill_valid_pixel_count: int = 0
    depth_fill_voxel_count_share_ge_0_5: int = 0
    depth_correction: dict | None = None
    ray_recovery: dict | None = None
ProgressCallback = Callable[[str], None]

def conf_to_rgb(conf: np.ndarray) -> np.ndarray:
    c = np.clip(conf, 0.0, 2.0)
    lo = np.floor(c).astype(int).clip(0, 1)
    t = (c - lo)[:, None]
    return (_STOPS[lo] * (1 - t) + _STOPS[lo + 1] * t).astype(np.uint8)

def write_ply_rgb(path: Path, xyz: np.ndarray, rgb: np.ndarray, extras: dict[str, np.ndarray]) -> None:
    n = len(xyz)
    names = list(extras)
    head = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}', 'property float x', 'property float y', 'property float z', 'property uchar red', 'property uchar green', 'property uchar blue']
    head += [f'property float {nm}' for nm in names] + ['end_header', '']
    dt = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    dt += [(nm, '<f4') for nm in names]
    arr = np.empty(n, dtype=np.dtype(dt))
    arr['x'], arr['y'], arr['z'] = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    arr['red'], arr['green'], arr['blue'] = (rgb[:, 0], rgb[:, 1], rgb[:, 2])
    for nm in names:
        arr[nm] = extras[nm]
    with path.open('wb') as f:
        f.write('\n'.join(head).encode('ascii'))
        f.write(arr.tobytes())

def depth_fill_mask(depth_m: np.ndarray, conf: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    confidence = np.asarray(conf)
    return (confidence == DEPTH_FILL_CONF) & (depth >= DEPTH_FILL_PRIOR_M)

def visibility_tolerance_m(reference_depth_m: np.ndarray) -> np.ndarray:
    d = np.asarray(reference_depth_m, dtype=np.float64)
    return np.maximum(VISIBILITY_TOLERANCE_FLOOR_M, VISIBILITY_TOLERANCE_QUADRATIC_M_PER_M2 * d * d)

def keyframe_candidates(keyframe_indices: np.ndarray, source_idx: int, limit: int) -> list[int]:
    keys = np.asarray(keyframe_indices, dtype=np.int64)
    if limit < 1 or not len(keys):
        return []
    right = int(np.searchsorted(keys, source_idx, side='left'))
    left = right - 1
    out: list[int] = []
    while len(out) < limit and (left >= 0 or right < len(keys)):
        take_left = right >= len(keys) or (left >= 0 and source_idx - int(keys[left]) <= int(keys[right]) - source_idx)
        if take_left:
            out.append(int(keys[left]))
            left -= 1
        else:
            out.append(int(keys[right]))
            right += 1
    return out

@dataclass(frozen=True)
class _Keyframe:
    idx: int
    world_to_cv: np.ndarray
    intrinsics_rgb: np.ndarray
    depth: np.ndarray
    image: np.ndarray

    def sample(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(points_world)
        visible = np.zeros(n, dtype=bool)
        bgr = np.zeros((n, 3), dtype=np.float64)
        if n == 0:
            return (visible, bgr)
        homogeneous = np.concatenate([points_world, np.ones((n, 1))], axis=1)
        cam = homogeneous @ self.world_to_cv.T
        z = cam[:, 2]
        front = z > m0util.MIN_VALID_DEPTH_M
        if not front.any():
            return (visible, bgr)
        fx, fy = (self.intrinsics_rgb[0, 0], self.intrinsics_rgb[1, 1])
        cx, cy = (self.intrinsics_rgb[0, 2], self.intrinsics_rgb[1, 2])
        with np.errstate(divide='ignore', invalid='ignore'):
            u_rgb = np.where(front, fx * cam[:, 0] / z + cx, -1.0)
            v_rgb = np.where(front, fy * cam[:, 1] / z + cy, -1.0)
        height_d, width_d = self.depth.shape
        height_r, width_r = self.image.shape[:2]
        u_d = np.rint(u_rgb * (width_d / width_r)).astype(np.int64)
        v_d = np.rint(v_rgb * (height_d / height_r)).astype(np.int64)
        inside = front & (u_d >= 0) & (u_d < width_d) & (v_d >= 0) & (v_d < height_d)
        if not inside.any():
            return (visible, bgr)
        sel = np.nonzero(inside)[0]
        reference = self.depth[v_d[sel], u_d[sel]].astype(np.float64)
        has_return = m0util.valid_mask(reference)
        agree = np.zeros(len(sel), dtype=bool)
        agree[has_return] = np.abs(z[sel][has_return] - reference[has_return]) <= visibility_tolerance_m(reference[has_return])
        ok = sel[agree]
        if not len(ok):
            return (visible, bgr)
        u_r = np.rint(u_rgb[ok]).astype(np.int64).clip(0, width_r - 1)
        v_r = np.rint(v_rgb[ok]).astype(np.int64).clip(0, height_r - 1)
        visible[ok] = True
        bgr[ok] = self.image[v_r, u_r].astype(np.float64)
        return (visible, bgr)

class _KeyframeCache:

    def __init__(self, frames_by_idx: dict[int, m0util.RawFrame], capacity: int, depth_model: dict | None=None, ray_recovery: dict | None=None) -> None:
        self._frames = frames_by_idx
        self._capacity = max(1, capacity)
        self._depth_model = depth_model
        self._ray_recovery = ray_recovery
        self._cache: OrderedDict[int, _Keyframe | None] = OrderedDict()

    def get(self, idx: int) -> _Keyframe | None:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        keyframe = self._load(self._frames[idx], self._depth_model, self._ray_recovery)
        self._cache[idx] = keyframe
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return keyframe

    @staticmethod
    def _load(frame: m0util.RawFrame, depth_model: dict | None=None, ray_recovery: dict | None=None) -> _Keyframe | None:
        try:
            transform, intrinsics_rgb, rgb_wh = m0util.load_pose_intrinsics_rgb_wh(frame.json_path)
        except (ValueError, KeyError):
            return None
        if m0util.pose_ortho_error(transform) > 0.001:
            return None
        try:
            image = m0util.read_sensor_image(frame.rgb_path, cv2.IMREAD_COLOR, expected_wh=rgb_wh)
        except (OSError, ValueError):
            return None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        depth = m0util.read_depth(frame.depth_path)
        raw_depth = depth
        if depth_model is not None:
            m0util.validate_depth_correction_grid(depth_model, depth.shape)
            rows, columns = np.indices(depth.shape)
            depth = m0util.apply_depth_correction(depth, depth_model, us=columns, vs=rows)
        if ray_recovery and frame.idx in ray_recovery['by_frame']:
            from geometry.ray_recovery import apply_recovery_to_depth
            recovered = apply_recovery_to_depth(raw_depth, ray_recovery['by_frame'][frame.idx])
            moved = np.isfinite(recovered) & (recovered != raw_depth)
            depth = depth.copy()
            depth[moved] = recovered[moved]
        world_to_cv = np.linalg.inv(m0util.camera_cv_to_world_matrix(transform))
        return _Keyframe(idx=frame.idx, world_to_cv=world_to_cv, intrinsics_rgb=intrinsics_rgb, depth=depth, image=image)

def _assign_keyframe_rgb(points_world: np.ndarray, source_idx: int, keyframe_indices: np.ndarray, cache: _KeyframeCache, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(points_world)
    assigned = np.zeros(n, dtype=bool)
    rgb = np.zeros((n, 3), dtype=np.float64)
    distance = np.full(n, UNASSIGNED_KEYFRAME_DISTANCE, dtype=np.float64)
    for candidate in keyframe_candidates(keyframe_indices, source_idx, limit):
        if assigned.all():
            break
        keyframe = cache.get(candidate)
        if keyframe is None:
            continue
        pending = np.nonzero(~assigned)[0]
        visible, bgr = keyframe.sample(points_world[pending])
        hit = pending[visible]
        if not len(hit):
            continue
        assigned[hit] = True
        rgb[hit] = bgr[visible][:, ::-1]
        distance[hit] = abs(candidate - source_idx)
    return (assigned, rgb, distance)

def process_color_clouds(run_dir: str | Path, *, voxel_m: float=0.015, chunk_frames: int=80, output_dir: str | Path=Path('data/cc'), output_stem: str | None=None, progress: ProgressCallback | None=None, max_keyframe_candidates: int=MAX_KEYFRAME_CANDIDATES, depth_correction: str | Path | dict | None=None, ray_recovery: dict | None=None) -> ColorCloudResult:
    source = Path(run_dir)
    target_dir = Path(output_dir)
    if voxel_m <= 0.0:
        raise ValueError('voxel_m must be positive')
    if chunk_frames < 1:
        raise ValueError('chunk_frames must be at least one')
    if max_keyframe_candidates < 1:
        raise ValueError('max_keyframe_candidates must be at least one')
    if output_stem is not None and (not output_stem or not output_stem.isascii() or Path(output_stem).name != output_stem):
        raise ValueError('output_stem must be a non-empty ASCII filename stem')

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)
    depth_model: dict | None = None
    if depth_correction is not None:
        if isinstance(depth_correction, dict):
            if '_coef' not in depth_correction:
                raise ValueError("depth_correction dict must come from m0util.load_depth_correction (missing '_coef'); pass the JSON path instead")
            depth_model = depth_correction
        else:
            depth_model = m0util.load_depth_correction(depth_correction)
        emit(f"depth correction ON: {depth_model.get('_source', '<dict>')} {m0util.depth_correction_describe(depth_model)} (held constant beyond; no cutoff, no point dropped)")
    indexed_frames = m0util.list_frames(source)
    frames = [frame for frame in indexed_frames if frame.depth_path and frame.conf_path]
    keyframes = {frame.idx: frame for frame in indexed_frames if frame.rgb_path and frame.depth_path}
    keyframe_indices = np.array(sorted(keyframes), dtype=np.int64)
    emit(f'{len(frames)} depth+conf frames, {len(keyframe_indices)} RGB keyframes, voxel {voxel_m * 100:.1f} cm, <= {max_keyframe_candidates} keyframes per point')
    cache = _KeyframeCache(keyframes, capacity=2 * max_keyframe_candidates + 2, depth_model=depth_model, ray_recovery=ray_recovery)
    acc = _Accum()
    col = _Accum()
    fill = _Accum()
    bk: list[np.ndarray] = []
    bp: list[np.ndarray] = []
    bc: list[np.ndarray] = []
    bd: list[np.ndarray] = []
    bf: list[np.ndarray] = []
    ck: list[np.ndarray] = []
    ccol: list[np.ndarray] = []
    cdist: list[np.ndarray] = []
    cf: list[np.ndarray] = []
    fk: list[np.ndarray] = []
    ff: list[np.ndarray] = []
    used = 0
    fill_labelled_pixels = 0
    fill_valid_pixels = 0
    recovered_keys = []
    recovered_fill_keys = []
    recovered_pixel_count = 0

    def flush() -> None:
        nonlocal bk, bp, bc, bd, bf, ck, ccol, cdist, cf, fk, ff
        if not bk:
            return
        acc.add(np.concatenate(bk), np.concatenate(bp), np.concatenate(bc), np.concatenate(bd), np.concatenate(bf))
        if ck:
            keys = np.concatenate(ck)
            col.add(keys, np.concatenate(ccol), np.concatenate(cdist), np.zeros(len(keys)), np.concatenate(cf))
        if fk:
            keys = np.concatenate(fk)
            fill.add(keys, np.zeros((len(keys), 3)), np.zeros(len(keys)), np.zeros(len(keys)), np.concatenate(ff))
        bk, bp, bc, bd, bf, ck, ccol, cdist, cf = ([], [], [], [], [], [], [], [], [])
        fk, ff = ([], [])
    for index, frame in enumerate(frames, 1):
        try:
            transform, intrinsics_rgb, rgb_wh = m0util.load_pose_intrinsics_rgb_wh(frame.json_path)
        except (ValueError, KeyError):
            continue
        if m0util.pose_ortho_error(transform) > 0.001:
            continue
        depth = m0util.read_depth(frame.depth_path)
        confidence = m0util.read_cv_image(frame.conf_path, cv2.IMREAD_UNCHANGED)
        if confidence is None or confidence.shape != depth.shape:
            continue
        valid = m0util.valid_mask(depth)
        if not valid.any():
            continue
        rows, columns = np.nonzero(valid)
        distances = depth[valid].astype(np.float64)
        geometry_distances = distances
        if depth_model is not None:
            m0util.validate_depth_correction_grid(depth_model, depth.shape)
            geometry_distances = m0util.apply_depth_correction(distances, depth_model, us=columns, vs=rows)
        relocated = np.zeros(len(distances), dtype=bool)
        if ray_recovery and frame.idx in ray_recovery['by_frame']:
            from geometry.ray_recovery import apply_recovery_to_depth
            recovered = apply_recovery_to_depth(depth, ray_recovery['by_frame'][frame.idx])[valid]
            relocated = np.isfinite(recovered) & (recovered != distances)
            geometry_distances = geometry_distances.copy()
            geometry_distances[relocated] = recovered[relocated]
            recovered_pixel_count += int(relocated.sum())
        intrinsics_depth = m0util.scale_K(intrinsics_rgb, rgb_wh, (depth.shape[1], depth.shape[0]))
        points = m0util.backproject_world(columns, rows, geometry_distances, intrinsics_depth, transform)
        keys = _voxel_keys(points, voxel_m)
        if relocated.any():
            recovered_keys.append(keys[relocated])
        frame_ids = np.full(len(keys), frame.idx, dtype=np.int64)
        conf_valid = confidence[valid]
        bk.append(keys)
        bp.append(points)
        bc.append(conf_valid.astype(np.float64))
        bd.append(geometry_distances)
        bf.append(frame_ids)
        labelled = depth_fill_mask(distances, conf_valid)
        if np.any(labelled & relocated):
            recovered_fill_keys.append(keys[labelled & relocated])
        fill_valid_pixels += int(len(distances))
        labelled_count = int(np.count_nonzero(labelled))
        fill_labelled_pixels += labelled_count
        if labelled_count:
            fk.append(keys[labelled])
            ff.append(frame_ids[labelled])
        assigned, rgb, kf_distance = _assign_keyframe_rgb(points, frame.idx, keyframe_indices, cache, max_keyframe_candidates)
        if assigned.any():
            ck.append(keys[assigned])
            ccol.append(rgb[assigned])
            cdist.append(kf_distance[assigned])
            cf.append(frame_ids[assigned])
        used += 1
        if len(bk) >= chunk_frames:
            flush()
            emit(f'{index}/{len(frames)} frames -> {len(acc.k):,} voxels')
    flush()
    if not len(acc.k):
        raise ColorCloudProcessingError('All Data folder has no usable depth/confidence frame')
    xyz = acc.sxyz / acc.hits[:, None]
    conf_mean = acc.sconf / acc.hits
    n = len(acc.k)
    rgb_real = np.empty((n, 3), dtype=np.uint8)
    rgb_real[:] = UNASSIGNED_RGB
    rgb_kf_dist = np.full(n, UNASSIGNED_KEYFRAME_DISTANCE)
    rgb_hits = np.zeros(n)
    if len(col.k):
        position = np.searchsorted(acc.k, col.k)
        if position.max() >= n or not np.array_equal(acc.k[position], col.k):
            raise ColorCloudProcessingError('colour voxel keys are not a subset of geometry')
        rgb_real[position] = np.clip(col.sxyz / col.hits[:, None], 0, 255).astype(np.uint8)
        rgb_kf_dist[position] = col.mconf
        rgb_hits[position] = col.hits.astype(np.float64)
    assigned_count = int(np.count_nonzero(rgb_hits > 0))
    fill_share = np.zeros(n, dtype=np.float32)
    if len(fill.k):
        position = np.searchsorted(acc.k, fill.k)
        if position.max() >= n or not np.array_equal(acc.k[position], fill.k):
            raise ColorCloudProcessingError('depth-fill voxel keys are not a subset of geometry')
        share = fill.hits.astype(np.float64) / acc.hits[position].astype(np.float64)
        fill_share[position] = np.clip(share, 0.0, 1.0).astype(np.float32)
    fill_voxels_ge_half = int(np.count_nonzero(fill_share >= 0.5))
    recovery_share = np.zeros(n, dtype=np.float32)
    fill_recovery_share = np.zeros(n, dtype=np.float32)
    if recovered_keys:
        recovered_unique, recovered_counts = np.unique(np.concatenate(recovered_keys), return_counts=True)
        positions = np.searchsorted(acc.k, recovered_unique)
        recovery_share[positions] = recovered_counts / acc.hits[positions]
    if recovered_fill_keys:
        recovered_unique, recovered_counts = np.unique(np.concatenate(recovered_fill_keys), return_counts=True)
        positions = np.searchsorted(acc.k, recovered_unique)
        fill_recovery_share[positions] = recovered_counts / acc.hits[positions]
    extras = {'conf_mean': conf_mean, 'conf_min': acc.mconf, 'range_m': acc.sd / acc.hits, 'n_views': acc.views.astype(np.float64), 'n_hits': acc.hits.astype(np.float64), 'rgb_kf_dist': rgb_kf_dist, 'rgb_hits': rgb_hits, DEPTH_FILL_PROPERTY: fill_share}
    if ray_recovery is not None:
        extras['recovery_share'] = recovery_share
        extras['fill_recovery_share'] = fill_recovery_share
    stem = output_stem or source.name
    target_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = target_dir / f'{stem}_rgb.ply'
    confidence_path = target_dir / f'{stem}_confcolor.ply'
    write_ply_rgb(rgb_path, xyz, rgb_real, extras)
    write_ply_rgb(confidence_path, xyz, conf_to_rgb(conf_mean), extras)
    emit(f'{assigned_count:,} of {n:,} voxels took keyframe RGB, {n - assigned_count:,} unassigned (rgb_kf_dist = -1)')
    emit(f'depth-fill label (conf=={DEPTH_FILL_CONF} & depth>={DEPTH_FILL_PRIOR_M} m): {fill_labelled_pixels:,} of {fill_valid_pixels:,} pixels, {fill_voxels_ge_half:,} of {n:,} voxels with fill_share >= 0.5; nothing removed')
    for path in (rgb_path, confidence_path):
        emit(f'{len(xyz):,} points -> {path} ({path.stat().st_size / 1000000.0:.0f} MB)')
    return ColorCloudResult(rgb_path=rgb_path, confidence_path=confidence_path, point_count=len(xyz), indexed_frame_count=len(indexed_frames), eligible_frame_count=len(frames), used_frame_count=used, skipped_frame_count=len(frames) - used, voxel_m=float(voxel_m), keyframe_count=int(len(keyframe_indices)), rgb_assigned_point_count=assigned_count, rgb_unassigned_point_count=int(n - assigned_count), max_keyframe_candidates=int(max_keyframe_candidates), visibility_tolerance_floor_m=float(VISIBILITY_TOLERANCE_FLOOR_M), visibility_tolerance_quadratic_m_per_m2=float(VISIBILITY_TOLERANCE_QUADRATIC_M_PER_M2), unassigned_rgb=(int(UNASSIGNED_RGB[0]), int(UNASSIGNED_RGB[1]), int(UNASSIGNED_RGB[2])), unassigned_keyframe_distance=float(UNASSIGNED_KEYFRAME_DISTANCE), depth_fill_prior_m=float(DEPTH_FILL_PRIOR_M), depth_fill_conf=int(DEPTH_FILL_CONF), depth_fill_property=str(DEPTH_FILL_PROPERTY), depth_fill_labelled_pixel_count=int(fill_labelled_pixels), depth_fill_valid_pixel_count=int(fill_valid_pixels), depth_fill_voxel_count_share_ge_0_5=fill_voxels_ge_half, depth_correction=m0util.depth_correction_provenance(depth_model), ray_recovery={**ray_recovery['provenance'], 'relocated_pixels': recovered_pixel_count} if ray_recovery is not None else None)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--voxel', type=float, default=0.015)
    ap.add_argument('--chunk', type=int, default=80)
    ap.add_argument('--outdir', type=Path, default=Path('data/cc'))
    ap.add_argument('--kf-candidates', type=int, default=MAX_KEYFRAME_CANDIDATES, help=f'nokta basina denenen en fazla keyframe (varsayilan {MAX_KEYFRAME_CANDIDATES})')
    ap.add_argument('--depth-correction', type=Path, default=None, help="kosu basina menzil duzeltmesi JSON'u (field/fit_depth_correction.py, schema cca.depth-correction.v1). Verilmezse KAPALI. d' = d - f(d) backprojection'dan once; dolgu etiketi HAM derinlikten; esik degil, nokta silmez (kural 1/2)")
    a = ap.parse_args()
    if a.voxel <= 0.0 or a.chunk < 1 or a.kf_candidates < 1:
        ap.error('voxel positive, chunk and kf-candidates at least one olmali')
    try:
        process_color_clouds(a.run_dir, voxel_m=a.voxel, chunk_frames=a.chunk, output_dir=a.outdir, progress=lambda message: print(message, flush=True), max_keyframe_candidates=a.kf_candidates, depth_correction=a.depth_correction)
    except ColorCloudProcessingError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
