
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
import cv2
import Imath
import numpy as np
import OpenEXR
MIN_VALID_DEPTH_M = 0.05
LEGACY_RGB_WH = (1920, 1440)
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0])
CV_TO_ARKIT_4 = np.diag([1.0, -1.0, -1.0, 1.0])
ARKIT_WORLD_TO_ZUP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
_CHANNEL_PREFERENCE = ('R', 'Y', 'Z', 'G', 'B', 'A')
_PIXEL_DTYPE = {Imath.PixelType(Imath.PixelType.FLOAT).v: np.float32, Imath.PixelType(Imath.PixelType.HALF).v: np.float16, Imath.PixelType(Imath.PixelType.UINT).v: np.uint32}

@dataclass
class RawFrame:
    idx: int
    json_path: Path
    depth_path: Path | None
    conf_path: Path | None
    rgb_path: Path | None
_IDX_RE = re.compile('(\\d+)')

def _idx_of(p: Path) -> int:
    m = _IDX_RE.search(p.stem)
    if not m:
        raise ValueError(f'indeks cikarilamadi: {p.name}')
    return int(m.group(1))

def list_frames(run_dir: Path) -> list[RawFrame]:
    run_dir = Path(run_dir)
    by_idx: dict[int, dict] = {}
    for pattern, key in (('frame_*.json', 'json'), ('depth_*.exr', 'depth'), ('depth_*.png', 'depth'), ('conf_*.png', 'conf'), ('frame_*.jpg', 'rgb')):
        for p in sorted(run_dir.glob(pattern)):
            by_idx.setdefault(_idx_of(p), {})[key] = p
    frames = [RawFrame(idx=i, json_path=d['json'], depth_path=d.get('depth'), conf_path=d.get('conf'), rgb_path=d.get('rgb')) for i, d in sorted(by_idx.items()) if 'json' in d]
    if not frames:
        raise FileNotFoundError(f"{run_dir} icinde frame_*.json yok - 3D Scanner App 'All Data' export mu?")
    return frames
DEPTH_PNG_SCALE_M = 1.0 / 1000.0

def read_depth(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == '.exr':
        return read_depth_exr(path)
    if path.suffix.lower() == '.png':
        return read_depth_png(path)
    raise ValueError(f'bilinmeyen derinlik uzantisi: {path.name}')

def read_depth_png(path: Path) -> np.ndarray:
    arr = read_cv_image(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise OSError(f'derinlik PNG okunamadi: {path}')
    if arr.ndim != 2:
        raise ValueError(f'{path.name}: tek kanal bekleniyordu, shape={arr.shape}')
    if arr.dtype != np.uint16:
        raise ValueError(f'{path.name}: uint16 bekleniyordu, dtype={arr.dtype}')
    out = arr.astype(np.float32) * DEPTH_PNG_SCALE_M
    out[arr == 0] = np.nan
    return out

def read_cv_image(path: Path, flags: int) -> np.ndarray | None:
    try:
        encoded = np.fromfile(Path(path), dtype=np.uint8)
    except OSError:
        return None
    if not len(encoded):
        return None
    return cv2.imdecode(encoded, flags)

def read_sensor_image(path: Path, flags: int, *, expected_wh: tuple[int, int] | None=None) -> np.ndarray:
    image = read_cv_image(Path(path), flags | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.ndim not in {2, 3}:
        raise OSError(f'sensor RGB cannot be read: {path}')
    actual_wh = (int(image.shape[1]), int(image.shape[0]))
    if expected_wh is not None and actual_wh != tuple(expected_wh):
        raise ValueError(f'{Path(path).name}: stored RGB {actual_wh} != intrinsics basis {tuple(expected_wh)}')
    return image

def read_depth_exr(path: Path) -> np.ndarray:
    f = OpenEXR.InputFile(str(path))
    try:
        header = f.header()
        dw = header['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        channels = header['channels']
        name = next((c for c in _CHANNEL_PREFERENCE if c in channels), next(iter(channels)))
        dtype = _PIXEL_DTYPE.get(channels[name].type.v, np.float32)
        arr = np.frombuffer(f.channel(name), dtype=dtype).reshape(h, w)
        return arr.astype(np.float32)
    finally:
        f.close()

def write_depth_exr(path: Path, depth: np.ndarray) -> None:
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    h, w = depth.shape
    header = OpenEXR.Header(w, h)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    out = OpenEXR.OutputFile(str(path), header)
    try:
        out.writePixels({'R': depth.tobytes()})
    finally:
        out.close()

def valid_mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > MIN_VALID_DEPTH_M)

def load_pose_intrinsics_rgb_wh(json_path: Path, *, legacy_rgb_wh: tuple[int, int] | None=None) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    d = json.loads(Path(json_path).read_text(encoding='utf-8'))
    pose_raw = d.get('cameraPoseARFrame') or d.get('pose')
    K_raw = d.get('intrinsics') or d.get('calibrationMatrix')
    if pose_raw is None or K_raw is None:
        raise KeyError(f'{json_path.name}: cameraPoseARFrame/intrinsics alanlari yok')
    T = np.asarray(pose_raw, dtype=np.float64).reshape(4, 4)
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-05):
        if np.allclose(T[:, 3], [0, 0, 0, 1], atol=1e-05):
            T = T.T
        else:
            raise ValueError(f'{json_path.name}: poz matrisi 4x4 homojen degil')
    K = np.asarray(K_raw, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f'{json_path.name}: intrinsics gecersiz')
    rgb_wh_raw = d.get('rgb_wh')
    if rgb_wh_raw is None:
        rgb_wh = LEGACY_RGB_WH if legacy_rgb_wh is None else legacy_rgb_wh
        if len(rgb_wh) != 2 or any((not isinstance(value, int) or isinstance(value, bool) for value in rgb_wh)) or any((value < 1 or value > 16384 for value in rgb_wh)):
            raise ValueError(f'{json_path.name}: legacy_rgb_wh gecersiz')
    else:
        if not isinstance(rgb_wh_raw, list) or len(rgb_wh_raw) != 2 or any((not isinstance(value, int) or isinstance(value, bool) for value in rgb_wh_raw)) or any((value < 1 or value > 16384 for value in rgb_wh_raw)):
            raise ValueError(f'{json_path.name}: rgb_wh gecersiz')
        rgb_wh = (rgb_wh_raw[0], rgb_wh_raw[1])
    return (T, K, rgb_wh)

def load_pose_intrinsics(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    T, K, _ = load_pose_intrinsics_rgb_wh(json_path)
    return (T, K)

def pose_ortho_error(T: np.ndarray) -> float:
    R = T[:3, :3]
    return float(np.abs(R @ R.T - np.eye(3)).max())
PIXEL_CONVENTIONS = ('corner', 'center')
DEFAULT_PIXEL_CONVENTION = 'corner'

def scale_K(K_rgb: np.ndarray, rgb_wh: tuple[int, int], depth_wh: tuple[int, int], *, pixel_convention: str=DEFAULT_PIXEL_CONVENTION) -> np.ndarray:
    if pixel_convention not in PIXEL_CONVENTIONS:
        raise ValueError(f'bilinmeyen pixel_convention: {pixel_convention!r} (gecerli: {PIXEL_CONVENTIONS})')
    sx = depth_wh[0] / rgb_wh[0]
    sy = depth_wh[1] / rgb_wh[1]
    K = K_rgb.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    if pixel_convention == 'corner':
        K[0, 2] *= sx
        K[1, 2] *= sy
    else:
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K

def projection_matrix_principal_point(projection_matrix, rgb_wh: tuple[int, int]) -> dict:
    M = np.asarray(projection_matrix, dtype=np.float64)
    if M.size != 16:
        raise ValueError(f'projectionMatrix 16 sayi olmali, {M.size} geldi')
    M = M.reshape(4, 4)
    if np.isclose(M[3, 2], -1.0, atol=1e-06):
        layout = 'row-major'
    elif np.isclose(M[2, 3], -1.0, atol=1e-06):
        layout = 'column-major'
        M = M.T
    else:
        raise ValueError(f'projectionMatrix ARKit/GL perspektif matrisi degil (M[3,2]={M[3, 2]:.6g}, M[2,3]={M[2, 3]:.6g})')
    w, h = (float(rgb_wh[0]), float(rgb_wh[1]))
    if M[0, 0] <= 0 or M[1, 1] <= 0:
        raise ValueError('projectionMatrix odak terimleri pozitif degil')
    return dict(fx=float(M[0, 0] * w / 2.0), fy=float(M[1, 1] * h / 2.0), cx=float((1.0 - M[0, 2]) * w / 2.0), cy=float((1.0 + M[1, 2]) * h / 2.0), layout=layout)

def principal_point_offset_px(json_path: Path, *, legacy_rgb_wh: tuple[int, int] | None=None) -> dict:
    json_path = Path(json_path)
    d = json.loads(json_path.read_text(encoding='utf-8'))
    pm = d.get('projectionMatrix')
    if pm is None:
        raise KeyError(f'{json_path.name}: projectionMatrix alani yok')
    _T, K, rgb_wh = load_pose_intrinsics_rgb_wh(json_path, legacy_rgb_wh=legacy_rgb_wh)
    implied = projection_matrix_principal_point(pm, rgb_wh)
    return dict(frame=json_path.name, rgb_wh=(int(rgb_wh[0]), int(rgb_wh[1])), layout=implied['layout'], cx_intrinsics=float(K[0, 2]), cy_intrinsics=float(K[1, 2]), cx_projection=implied['cx'], cy_projection=implied['cy'], dx_px=implied['cx'] - float(K[0, 2]), dy_px=implied['cy'] - float(K[1, 2]), fx_intrinsics=float(K[0, 0]), fy_intrinsics=float(K[1, 1]), fx_projection=implied['fx'], fy_projection=implied['fy'], dfx_px=implied['fx'] - float(K[0, 0]), dfy_px=implied['fy'] - float(K[1, 1]))

def backproject_world(us: np.ndarray, vs: np.ndarray, ds: np.ndarray, K_depth: np.ndarray, T_wc: np.ndarray, pose_conv: str='arkit-gl') -> np.ndarray:
    fx, fy, cx, cy = (K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2])
    x = (np.asarray(us, dtype=np.float64) - cx) / fx * ds
    y = (np.asarray(vs, dtype=np.float64) - cy) / fy * ds
    pts_cv = np.stack([x, y, np.asarray(ds, dtype=np.float64)], axis=-1)
    if pose_conv == 'arkit-gl':
        pts_cam = pts_cv @ CV_TO_ARKIT.T
    elif pose_conv == 'cv':
        pts_cam = pts_cv
    else:
        raise ValueError(f'bilinmeyen pose_conv: {pose_conv}')
    h = np.concatenate([pts_cam, np.ones((*pts_cam.shape[:-1], 1))], axis=-1)
    return (h @ T_wc.T)[..., :3]
DEPTH_CORRECTION_SCHEMA = 'cca.depth-correction.v1'
DEPTH_CORRECTION_KINDS = ('poly_no_intercept', 'cell_range_table')

def _table_from_json(values, shape: tuple[int, ...], what: str) -> np.ndarray:
    try:
        arr = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f'{what}: sayi/null listesi olmali ({e})') from e
    if arr.shape != tuple(shape):
        raise ValueError(f'{what}: beklenen sekil {tuple(shape)}, bulunan {arr.shape}')
    if np.isinf(arr).any():
        raise ValueError(f'{what}: sonsuz deger')
    return arr

def _fill_nearest_along_d(tab: np.ndarray) -> np.ndarray:
    a = np.asarray(tab, dtype=np.float64)
    n = a.shape[-1]
    idx = np.maximum.accumulate(np.where(np.isfinite(a), np.arange(n), -1), axis=-1)
    out = np.where(idx >= 0, np.take_along_axis(a, np.maximum(idx, 0), axis=-1), np.nan)
    idx2 = np.where(np.isfinite(out), np.arange(n), n)
    idx2 = np.minimum.accumulate(idx2[..., ::-1], axis=-1)[..., ::-1]
    out = np.where(idx2 < n, np.take_along_axis(out, np.minimum(idx2, n - 1), axis=-1), np.nan)
    return np.where(np.isfinite(out), out, 0.0)

def _lerp_index(x, n: int):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, n - 1.0)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    return (i0, i1, x - i0)

def _positive(dct: dict, key: str, what: str, integer: bool=False):
    val = dct.get(key)
    try:
        fv = float(val)
    except (TypeError, ValueError):
        raise ValueError(f'{what}.{key} pozitif sayi olmali, bulunan {val!r}') from None
    if not np.isfinite(fv) or fv <= 0.0 or (integer and fv != int(fv)):
        raise ValueError(f'{what}.{key} pozitif sayi olmali, bulunan {val!r}')
    return int(fv) if integer else fv

def _prepare_table_model(spec: dict, model: dict, where: str) -> np.ndarray:
    if spec.get('interp', 'trilinear') != 'trilinear':
        raise ValueError(f"{where}: model.interp yalnizca 'trilinear' desteklenir")
    rt = spec.get('range_table') or {}
    ct = spec.get('cell_table') or {}
    n_r = _positive(rt, 'n_bins', f'{where}: range_table', integer=True)
    rb = _positive(rt, 'bin_m', f'{where}: range_table')
    cu = _positive(ct, 'cells_u', f'{where}: cell_table', integer=True)
    cv = _positive(ct, 'cells_v', f'{where}: cell_table', integer=True)
    w = _positive(ct, 'width', f'{where}: cell_table', integer=True)
    h = _positive(ct, 'height', f'{where}: cell_table', integer=True)
    nb = _positive(ct, 'n_bins', f'{where}: cell_table', integer=True)
    cb = _positive(ct, 'bin_m', f'{where}: cell_table')
    if cu > w or cv > h:
        raise ValueError(f'{where}: cell_table hucre sayisi ({cu}x{cv}) izgaradan ({w}x{h}) buyuk')
    range_tab = _table_from_json(rt.get('values'), (n_r,), f'{where}: range_table.values')
    cell_tab = _table_from_json(ct.get('values'), (cu, cv, nb), f'{where}: cell_table.values')
    if not np.isfinite(range_tab).any():
        raise ValueError(f'{where}: range_table tamamen bos (hic dolu kutu yok)')
    model['_range_tab'] = _fill_nearest_along_d(range_tab)
    model['_range_bin'] = rb
    model['_cell_tab'] = _fill_nearest_along_d(cell_tab)
    model['_cell_bin'] = cb
    model['_wh'] = (w, h)
    ref = spec.get('reference_poly') or {}
    coef = np.asarray(ref.get('coef', []), dtype=np.float64)
    if coef.ndim != 1 or not np.all(np.isfinite(coef)):
        raise ValueError(f'{where}: reference_poly.coef sonlu bir liste olmali')
    return coef

def validate_depth_correction(model: dict, source: str='<dict>') -> dict:
    if model.get('schema') != DEPTH_CORRECTION_SCHEMA:
        raise ValueError(f"{source}: beklenen schema {DEPTH_CORRECTION_SCHEMA!r}, bulunan {model.get('schema')!r}")
    spec = model.get('model') or {}
    kind = spec.get('kind')
    if kind not in DEPTH_CORRECTION_KINDS:
        raise ValueError(f'{source}: desteklenmeyen model kind {kind!r} (desteklenen: {DEPTH_CORRECTION_KINDS})')
    if spec.get('unit', 'm') != 'm':
        raise ValueError(f"{source}: model unit 'm' olmali (kural 8), bulunan {spec.get('unit')!r}")
    fit_range = model.get('fit_range_m')
    if not isinstance(fit_range, (list, tuple)) or len(fit_range) != 2 or (not np.isfinite(fit_range[1])) or (float(fit_range[1]) <= 0.0):
        raise ValueError(f'{source}: fit_range_m [lo, hi] pozitif sonlu bir ust sinirla verilmeli')
    if kind == 'poly_no_intercept':
        coef = np.asarray(spec.get('coef', []), dtype=np.float64)
        if coef.ndim != 1 or coef.size == 0 or (not np.all(np.isfinite(coef))):
            raise ValueError(f'{source}: model.coef bos olmayan sonlu bir liste olmali')
    else:
        coef = _prepare_table_model(spec, model, source)
    model['_coef'] = coef
    model['_d_max'] = float(fit_range[1])
    model['_source'] = source
    return model

def load_depth_correction(path: str | Path) -> dict:
    path = Path(path)
    model = json.loads(path.read_text(encoding='utf-8'))
    return validate_depth_correction(model, source=str(path))

def _table_offset(d: np.ndarray, u: np.ndarray, v: np.ndarray, model: dict) -> np.ndarray:
    rt, ct = (model['_range_tab'], model['_cell_tab'])
    cu, cv, nb = ct.shape
    w, h = model['_wh']
    i0, i1, f = _lerp_index(d / model['_range_bin'] - 0.5, rt.shape[0])
    out = rt[i0] * (1.0 - f) + rt[i1] * f
    x0, x1, fx = _lerp_index((u + 0.5) * cu / w - 0.5, cu)
    y0, y1, fy = _lerp_index((v + 0.5) * cv / h - 0.5, cv)
    z0, z1, fz = _lerp_index(d / model['_cell_bin'] - 0.5, nb)
    for xi, wx in ((x0, 1.0 - fx), (x1, fx)):
        for yi, wy in ((y0, 1.0 - fy), (y1, fy)):
            for zi, wz in ((z0, 1.0 - fz), (z1, fz)):
                out = out + wx * wy * wz * ct[xi, yi, zi]
    return out

def depth_correction_offset(ds, model: dict, us=None, vs=None) -> np.ndarray:
    d = np.asarray(ds, dtype=np.float64)
    if model['model']['kind'] == 'poly_no_intercept':
        x = np.minimum(d, model['_d_max'])
        out = np.zeros_like(d)
        for power, c in enumerate(model['_coef'], start=1):
            out += c * x ** power
        return out
    if us is None or vs is None:
        raise ValueError('cell_range_table modeli piksel indeksi ister: us (sutun) ve vs (satir) verilmeli')
    u = np.asarray(us)
    v = np.asarray(vs)
    if u.shape != d.shape or v.shape != d.shape:
        raise ValueError(f'us/vs sekli derinlikle ayni olmali: {u.shape}/{v.shape} vs {d.shape}')
    w, h = model['_wh']
    if d.size and (u.min() < 0 or u.max() >= w or v.min() < 0 or (v.max() >= h)):
        raise ValueError(f'piksel indeksi tablonun {w}x{h} derinlik izgarasi disinda')
    out = np.full(d.shape, np.nan)
    ok = np.isfinite(d)
    if ok.any():
        out[ok] = _table_offset(d[ok], u[ok].astype(np.float64), v[ok].astype(np.float64), model)
    return out

def apply_depth_correction(ds, model: dict, us=None, vs=None) -> np.ndarray:
    d = np.asarray(ds, dtype=np.float64)
    return d - depth_correction_offset(d, model, us=us, vs=vs)

def validate_depth_correction_grid(model: dict, shape: tuple[int, int]) -> None:
    if model['model']['kind'] == 'cell_range_table':
        height, width = shape
        if model['_wh'] != (width, height):
            raise ValueError(f"depth model raster {model['_wh']} != sensor raster {(width, height)}")

def depth_correction_describe(model: dict) -> str:
    if model['model']['kind'] == 'poly_no_intercept':
        return f"poly coef={model['_coef'].tolist()} fit_range={model['fit_range_m']} m (otesinde sabit)"
    cu, cv, nb = model['_cell_tab'].shape
    return f"cell_range_table {cu}x{cv} hucre x {nb} kutu ({model['_cell_bin']} m) + {model['_range_tab'].shape[0]} menzil kutusu ({model['_range_bin']} m), trilinear, fit_range={model['fit_range_m']} m (otesinde sabit)"

def depth_correction_provenance(model: dict | None) -> dict | None:
    if model is None:
        return None
    spec = model['model']
    prov = {'schema': model.get('schema'), 'source': model.get('_source'), 'run': model.get('run'), 'kind': spec.get('kind'), 'fit_range_m': [float(v) for v in model['fit_range_m']]}
    if spec.get('kind') == 'poly_no_intercept':
        prov['coef'] = [float(c) for c in model['_coef']]
    else:
        cu, cv, nb = model['_cell_tab'].shape
        prov['interp'] = 'trilinear'
        prov['range_table'] = {'bin_m': model['_range_bin'], 'n_bins': int(model['_range_tab'].shape[0])}
        prov['cell_table'] = {'cells_u': cu, 'cells_v': cv, 'width': model['_wh'][0], 'height': model['_wh'][1], 'bin_m': model['_cell_bin'], 'n_bins': nb}
        prov['reference_poly_coef'] = [float(c) for c in model['_coef']]
    return prov

def camera_cv_to_world_matrix(T_wc: np.ndarray) -> np.ndarray:
    T_wc = np.asarray(T_wc, dtype=np.float64)
    if T_wc.shape != (4, 4):
        raise ValueError('T_wc must be 4x4')
    return T_wc @ CV_TO_ARKIT_4

def depth_ray_to_world_matrix(K_depth: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    K_depth = np.asarray(K_depth, dtype=np.float64)
    if K_depth.shape != (3, 3):
        raise ValueError('K_depth must be 3x3')
    K4_inv = np.eye(4, dtype=np.float64)
    K4_inv[:3, :3] = np.linalg.inv(K_depth)
    return camera_cv_to_world_matrix(T_wc) @ K4_inv

def detect_apriltags(gray: np.ndarray) -> dict[int, np.ndarray]:
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.flatten(), corners)}
