
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from field import m0util

@dataclass(frozen=True)
class RayRecoveryConfig:
    min_angle_deg: float = 2.0
    max_reprojection_px: float = 3.0
    agreement_m: float = 0.08
    minimum_change_m: float = 0.05
    neighbours: int = 8
    min_support: int = 3

def triangulate_rays(xy_a, xy_b, pose_a, pose_b, k_a, k_b, config=None):
    c = config or RayRecoveryConfig()
    a, b = (np.asarray(xy_a, float), np.asarray(xy_b, float))
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('matching pixel arrays must be Nx2')
    rays = []
    cameras = []
    for pixels, pose, k in ((a, pose_a, k_a), (b, pose_b, k_b)):
        camera = m0util.camera_cv_to_world_matrix(pose)
        q = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
        rays.append(q @ camera[:3, :3].T)
        cameras.append(camera)
    qa, qb = rays
    delta = cameras[1][:3, 3] - cameras[0][:3, 3]
    aa, bb, ab = (np.sum(qa * qa, axis=1), np.sum(qb * qb, axis=1), np.sum(qa * qb, axis=1))
    det = aa * bb - ab * ab
    with np.errstate(invalid='ignore', divide='ignore'):
        da = (bb * (qa @ delta) - ab * (qb @ delta)) / det
        db = (ab * (qa @ delta) - aa * (qb @ delta)) / det
        sin2 = det / (aa * bb)
    pa = cameras[0][:3, 3] + da[:, None] * qa
    pb = cameras[1][:3, 3] + db[:, None] * qb
    point = (pa + pb) * 0.5
    errors = []
    for camera, k, xy in ((cameras[0], k_a, a), (cameras[1], k_b, b)):
        local = (point - camera[:3, 3]) @ camera[:3, :3]
        image = local @ np.asarray(k).T
        with np.errstate(divide='ignore', invalid='ignore'):
            errors.append(np.linalg.norm(image[:, :2] / image[:, 2:] - xy, axis=1))
    valid = np.isfinite(da) & np.isfinite(db) & (da > m0util.MIN_VALID_DEPTH_M) & (db > m0util.MIN_VALID_DEPTH_M) & (sin2 >= np.sin(np.deg2rad(c.min_angle_deg)) ** 2) & (np.maximum(*errors) <= c.max_reprojection_px)
    return (da, valid)

def recover_feature_depths(features, bridge_config, *, config=None, progress=None):
    from geometry.visual_bridge import _mutual_ratio_matches, _sample_depth, _read_depth_unicode
    c = config or RayRecoveryConfig()
    prepared = []
    for feature in features:
        pose, k, wh = m0util.load_pose_intrinsics_rgb_wh(feature.frame.json_path, legacy_rgb_wh=feature.image_wh)
        prepared.append((pose, k, wh))
    audit = []
    for index, feature in enumerate(features):
        pose, k, wh = prepared[index]
        depth = _read_depth_unicode(feature.frame.depth_path)
        measured = _sample_depth(depth, feature.xy_full, wh, 0)
        tracks = [[] for _ in feature.xy_full]
        candidates = sorted((j for j in range(len(features)) if j != index and features[j].frame.run_id == feature.frame.run_id), key=lambda j: abs(features[j].frame.frame_index - feature.frame.frame_index))
        chosen = []
        for j in candidates:
            baseline = np.linalg.norm(prepared[j][0][:3, 3] - pose[:3, 3])
            if baseline >= 0.12:
                chosen.append(j)
            if len(chosen) >= c.neighbours:
                break
        for j in chosen:
            other = features[j]
            matches = _mutual_ratio_matches(feature, other, bridge_config)
            if not matches:
                continue
            ia, ib = np.asarray([(x, y) for x, y, _ in matches]).T
            values, valid = triangulate_rays(feature.xy_full[ia], other.xy_full[ib], pose, prepared[j][0], k, prepared[j][1], c)
            for point_index, value in zip(ia[valid], values[valid]):
                tracks[point_index].append((j, float(value)))
        proposal = np.full(len(measured), np.nan)
        support = np.zeros(len(measured), dtype=np.int16)
        for i, observations in enumerate(tracks):
            if len(observations) < c.min_support or not np.isfinite(measured[i]):
                continue
            values = np.array([v for _, v in observations])
            estimate = float(np.median(values[:-1]))
            if np.max(np.abs(values - estimate)) > c.agreement_m:
                continue
            if abs(estimate - measured[i]) < c.minimum_change_m:
                continue
            proposal[i] = estimate
            support[i] = len(values)
        feature.recovered_depths = proposal
        feature.recovery_support = support
        v, u = depth.shape
        for i in np.flatnonzero(np.isfinite(proposal)):
            audit.append({'frame_index': feature.frame.frame_index, 'u': int(np.clip(np.rint(feature.xy_full[i, 0] * u / wh[0]), 0, u - 1)), 'v': int(np.clip(np.rint(feature.xy_full[i, 1] * v / wh[1]), 0, v - 1)), 'raw_depth_m': float(measured[i]), 'proposed_depth_m': float(proposal[i]), 'support_views': int(support[i]), 'held_out_view': True})
        if progress is not None and (index % 10 == 0 or index + 1 == len(features)):
            progress(f'RGB isin aramasi {index + 1}/{len(features)} / {len(audit)} nokta adayi')
    return audit

def apply_recovery_to_depth(depth, observations):
    output = np.asarray(depth).copy()
    groups = {}
    for item in observations:
        if item.get('held_out_view') is not True or item.get('support_views', 0) < 3:
            continue
        u, v = (int(item['u']), int(item['v']))
        if not (0 <= v < output.shape[0] and 0 <= u < output.shape[1]):
            raise ValueError('recovery pixel outside depth grid')
        value = float(item['proposed_depth_m'])
        if not np.isfinite(value) or value <= m0util.MIN_VALID_DEPTH_M:
            raise ValueError('invalid recovered depth')
        if not np.isclose(float(depth[v, u]), item['raw_depth_m'], atol=0.0011):
            raise ValueError('recovery does not match its raw observation')
        groups.setdefault((v, u), []).append(value)
    for pixel, values in groups.items():
        if np.ptp(values) <= 0.08:
            output[pixel] = np.median(values)
    return output

def attach_recovery_to_features(features, model):
    if model is None:
        return 0
    total = 0
    for feature in features:
        rows = model['by_frame'].get(feature.frame.frame_index, ())
        if not rows:
            continue
        raw = m0util.read_depth(feature.frame.depth_path)
        after = apply_recovery_to_depth(raw, rows)
        height, width = raw.shape
        us = np.clip(np.rint(feature.xy_full[:, 0] * width / feature.image_wh[0]).astype(int), 0, width - 1)
        vs = np.clip(np.rint(feature.xy_full[:, 1] * height / feature.image_wh[1]).astype(int), 0, height - 1)
        moved = np.isfinite(after[vs, us]) & (after[vs, us] != raw[vs, us])
        feature.recovered_depths = np.where(moved, after[vs, us], np.nan)
        total += int(moved.sum())
    return total
