
import time
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from field import m0util
from field.make_color_clouds import depth_fill_mask
from geometry.ray_structure import PatchCatalog, StructureConfig, search_catalog

def prepare_structure_inputs(run_dir, *, config=None, progress=None, frame_indices=None, query_frames=None, source_snapshot=None, reference_step=None, query_pixels=None):
    from viewer.raw_processing_service import snapshot_source
    c = config or StructureConfig()
    if c.patch_radius_px != 2 or c.pixel_step < 5 or c.frame_budget < 4 or (c.query_stride < 1):
        raise ValueError('structure search uses 5x5 patches and at least four frames')
    source = Path(run_dir).resolve()
    before = source_snapshot if source_snapshot is not None else snapshot_source(source)
    all_frames = [f for f in m0util.list_frames(source) if f.depth_path is not None]
    chosen = np.unique(np.linspace(0, len(all_frames) - 1, min(c.frame_budget, len(all_frames))).astype(int)) if frame_indices is None else np.asarray(frame_indices, int)
    points, origins, frames, pixels, depths, views, queries = ([], [], [], [], [], [], [])
    fill_centres, fill_patches, confidence_views = ([], [], {})
    du, dv = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    sampled = incomplete = offset = 0
    for index in chosen:
        frame = all_frames[index]
        raw = m0util.read_depth(frame.depth_path)
        conf = m0util.read_cv_image(frame.conf_path, cv2.IMREAD_UNCHANGED) if frame.conf_path else np.zeros(raw.shape, np.uint8)
        if conf is None or conf.shape != raw.shape:
            raise ValueError(f'confidence/depth shape mismatch: {frame.idx}')
        fill = depth_fill_mask(raw, conf)
        h, w = raw.shape
        pose, k_rgb, wh = m0util.load_pose_intrinsics_rgb_wh(frame.json_path)
        k = m0util.scale_K(k_rgb, wh, (w, h), pixel_convention=c.pixel_convention)
        camera = m0util.camera_cv_to_world_matrix(pose)
        is_query = query_frames is None or frame.idx in query_frames
        step = c.pixel_step if is_query or reference_step is None else reference_step
        exact_query = is_query and query_pixels is not None and (frame.idx in query_pixels)
        if exact_query:
            uv = np.asarray(query_pixels[frame.idx])
            if uv.ndim != 2 or uv.shape[1] != 2 or (not np.issubdtype(uv.dtype, np.integer)) or np.any(uv < 2) or np.any(uv[:, 0] >= w - 2) or np.any(uv[:, 1] >= h - 2):
                raise ValueError('query pixels must be integer centres inside the 5x5 raster border')
            u, v = uv.T
        else:
            u, v = np.meshgrid(np.arange(2, w - 2, step), np.arange(2, h - 2, step))
            u, v = (u.ravel(), v.ravel())
        us, vs = (u[:, None] + du.ravel(), v[:, None] + dv.ravel())
        d = raw[vs, us]
        valid = np.isfinite(d).all(axis=1) & (d > 0).all(axis=1)
        sampled += len(u)
        incomplete += int((~valid).sum())
        us, vs, d = (us[valid], vs[valid], d[valid])
        p = m0util.backproject_world(us, vs, d, k, pose)
        points.append(p)
        take = np.ones(valid.sum(), bool) if exact_query else is_query & ((u[valid] - 2) // step % c.query_stride == 0) & ((v[valid] - 2) // step % c.query_stride == 0)
        queries.extend((offset + np.flatnonzero(take)).tolist())
        offset += len(p)
        origins.append(np.tile(camera[:3, 3], (len(p), 1)))
        frames.append(np.full(len(p), frame.idx, dtype=int))
        pixels.append(np.c_[u[valid], v[valid]])
        depths.append(d[:, 12])
        fill_centres.append(fill[v[valid], u[valid]])
        fill_patches.append(fill[vs, us].any(axis=1))
        views.append((frame.idx, camera, k, raw, fill))
        confidence_views[frame.idx] = conf
    if not points:
        raise ValueError('no depth frames for structure search')
    catalog = PatchCatalog(*(np.concatenate(x) for x in (points, origins, frames, pixels, depths)), sensor_fill=np.concatenate(fill_centres), patch_fill=np.concatenate(fill_patches))
    if progress:
        progress(f'{len(chosen)} kare / {len(catalog.points)} yerel yapi / ham kaynak korunuyor')
    return {'source': source, 'snapshot': before, 'catalog': catalog, 'queries': queries, 'views': views, 'confidence_views': confidence_views, 'frames_total': len(all_frames), 'frames_sampled': len(chosen), 'sampled_patch_centres': sampled, 'incomplete_patches_retained_in_raw': incomplete, 'raw_frames': {all_frames[i].idx: all_frames[i] for i in chosen}}

def search_prepared_structure(prepared, *, config=None, progress=None):
    c = config or StructureConfig()
    result = search_catalog(prepared['catalog'], config=c, progress=progress, query_indices=prepared['queries'])
    for row in result['candidates'] + result.get('alignment_anchors', []):
        is_anchor = row['status'] == 'ALIGNMENT_ANCHOR_CANDIDATE'
        tolerance = row['anchor_tolerance_m'] if is_anchor else c.agreement_m
        point = np.asarray(row['candidate_point'])
        votes, contradictions, occlusions, uncertain, observations = ([], [], [], [], [])
        independent = []
        for frame_id, camera, k, raw, fill in sorted(prepared['views'], key=lambda x: abs(x[0] - row['frame_index'])):
            if frame_id == row['frame_index'] or frame_id in row['support_frames']:
                continue
            if any((np.linalg.norm(camera[:3, 3] - origin) < c.min_baseline_m for origin in independent)):
                continue
            p = (point - camera[:3, 3]) @ camera[:3, :3]
            if p[2] <= 0:
                continue
            uvh = p @ k.T
            u, v = np.rint(uvh[:2] / uvh[2]).astype(int)
            if not (0 <= v < raw.shape[0] and 0 <= u < raw.shape[1]) or not np.isfinite(raw[v, u]) or fill[v, u]:
                continue
            independent.append(camera[:3, 3])
            residual = float(raw[v, u] - p[2])
            confidence = prepared.get('confidence_views', {}).get(frame_id)
            confidence = int(confidence[v, u]) if confidence is not None else None
            if residual > tolerance:
                (contradictions if confidence == 2 else uncertain).append(frame_id)
                observation = 'FREE_SPACE_CONFLICT' if confidence == 2 else 'UNVERIFIED_DISAGREEMENT'
            elif residual < -tolerance:
                occlusions.append(frame_id)
                observation = 'OCCLUDED'
            else:
                votes.append(frame_id)
                observation = 'DEPTH_AGREES'
            observations.append({'frame': int(frame_id), 'u': int(u), 'v': int(v), 'raw_confidence': confidence, 'raw_depth_m': float(raw[v, u]), 'candidate_depth_m': float(p[2]), 'residual_m': residual, 'observation': observation})
            if len(independent) >= 8:
                break
        row.update(additional_support_frames=votes, free_space_conflict_frames=contradictions, unverified_visibility_frames=uncertain, visibility_observations=observations, occluded_in_frames=occlusions)
        if contradictions:
            row['status'] = 'ANCHOR_FREE_SPACE_CONFLICT' if is_anchor else 'FREE_SPACE_CONFLICT'
            if is_anchor:
                row['eligible_for_correspondence_search'] = False
        elif uncertain and (is_anchor or row['status'] == 'REVIEWABLE_STRUCTURE_CANDIDATE'):
            row['status'] = 'ANCHOR_VISIBILITY_UNVERIFIED' if is_anchor else 'VISIBILITY_UNVERIFIED'
            if is_anchor:
                row['eligible_for_correspondence_search'] = False
    decisions = {(r['frame_index'], r['u'], r['v']): r['status'] for r in result['candidates'] + result.get('alignment_anchors', [])}
    for row in result.get('query_ledger', []):
        row['role'] = decisions.get((row['frame_index'], row['u'], row['v']), row['role'])
    result['anchor_summary'] = {'eligible': sum((r['eligible_for_correspondence_search'] for r in result.get('alignment_anchors', []))), 'free_space_rejected': sum((r['status'] == 'ANCHOR_FREE_SPACE_CONFLICT' for r in result.get('alignment_anchors', []))), 'visibility_unverified': sum((r['status'] == 'ANCHOR_VISIBILITY_UNVERIFIED' for r in result.get('alignment_anchors', []))), 'position_certified': False, 'photo_model_correspondences_verified': False, 'policy': 'LIDAR_SUPPORT_ONLY; DO_NOT_ALIGN_FROM_RESTORATION_OR_UNKNOWN_REGIONS'}
    for name, status in (('reviewable', 'REVIEWABLE_STRUCTURE_CANDIDATE'), ('foreground_warning', 'FOREGROUND_UNRESOLVED'), ('free_space_conflicts', 'FREE_SPACE_CONFLICT'), ('visibility_unverified', 'VISIBILITY_UNVERIFIED')):
        result['counts'][name] = sum((row['status'] == status for row in result['candidates']))
    _attach_same_frame_rgb(prepared, result)
    return result

def _attach_same_frame_rgb(prepared, result):
    images = {}
    views = {v[0]: v for v in prepared['views']}
    frames = prepared.get('raw_frames', {})
    du, dv = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))

    def sample(frame_id, u, v):
        frame = frames.get(frame_id)
        if frame is None or frame.rgb_path is None:
            return None
        if frame_id not in images:
            images[frame_id] = m0util.read_sensor_image(frame.rgb_path, cv2.IMREAD_COLOR)
        image = images[frame_id]
        if image is None:
            return None
        h, w = views[frame_id][3].shape
        uu = np.rint((u + du.ravel()) * image.shape[1] / w).astype(int)
        vv = np.rint((v + dv.ravel()) * image.shape[0] / h).astype(int)
        return image[vv.clip(0, image.shape[0] - 1), uu.clip(0, image.shape[1] - 1), ::-1].tolist()
    for row in result['candidates'] + result.get('alignment_anchors', []):
        row['measured_patch_rgb'] = sample(row['frame_index'], row['u'], row['v'])
        for ref in row['support_patches']:
            ref['rgb'] = sample(ref['frame'], *ref['pixel'])
    result['rgb_color_source'] = 'SAME_FRAME_SENSOR_RGB_ONLY; missing RGB remains grey'

def build_structure_search(run_dir, output, *, config=None, progress=None, temporal=True):
    if temporal:
        return build_temporal_structure_search(run_dir, output, config=config, progress=progress)
    from viewer.raw_processing_service import _write_ascii_json, snapshot_source
    c = config or StructureConfig()
    source, output = (Path(run_dir).resolve(), Path(output).resolve())
    if source == output or source in output.parents:
        raise ValueError('structure evidence must be outside raw capture')
    start = time.monotonic()
    prepared = prepare_structure_inputs(source, config=c, progress=progress)
    result = search_prepared_structure(prepared, config=c, progress=progress)
    if prepared['snapshot'] != snapshot_source(source):
        raise ValueError('raw source changed during structure search')
    payload = {'schema': 'cca.structure-search.v1', 'run': source.name, 'unit': 'm', 'axis': 'ARKit_Y_up', 'pixel_convention': c.pixel_convention, 'source_snapshot': asdict(prepared['snapshot']), 'raw_preserved': True, 'config': asdict(c), **{key: prepared[key] for key in ('frames_total', 'frames_sampled', 'sampled_patch_centres', 'incomplete_patches_retained_in_raw')}, 'scope': 'SAMPLED_5X5_PATCH_HYPOTHESES_NOT_ALL_PIXELS', 'applied_observations': 0, 'position_certified': False, 'seconds': round(time.monotonic() - start, 3), **result}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_ascii_json(output, payload)
    return payload

def temporal_reference_indices(origins, anchor, *, baseline_m=0.12, neighbours=8):
    xyz = np.asarray(origins, float)
    selected = [int(anchor)]
    for index in sorted(range(len(xyz)), key=lambda i: (abs(i - anchor), i)):
        if all((np.linalg.norm(xyz[index] - xyz[j]) >= baseline_m for j in selected)):
            selected.append(index)
        if len(selected) >= neighbours + 1:
            break
    return selected

def build_temporal_structure_search(run_dir, output, *, config=None, progress=None, reference_step=4, negative_control_frames=0, rgb_queries=False):
    from dataclasses import replace
    from viewer.raw_processing_service import _write_ascii_json, snapshot_source
    c = config or StructureConfig()
    source, output = (Path(run_dir).resolve(), Path(output).resolve())
    if source == output or source in output.parents:
        raise ValueError('structure evidence must be outside raw capture')
    if reference_step not in (4, 8):
        raise ValueError('reference_step must be 4 or 8 pixels')
    start = time.monotonic()
    before = snapshot_source(source)
    frames = [f for f in m0util.list_frames(source) if f.depth_path is not None]
    if not frames:
        raise ValueError('no depth frames for structure search')
    origins = np.asarray([m0util.load_pose_intrinsics_rgb_wh(f.json_path)[0][:3, 3] for f in frames])
    query_pool = np.asarray([i for i, frame in enumerate(frames) if not rgb_queries or frame.rgb_path is not None])
    if not len(query_pool):
        raise ValueError('no RGB/depth query frames for anchor search')
    anchors = query_pool[np.unique(np.linspace(0, len(query_pool) - 1, min(c.frame_budget, len(query_pool))).astype(int))]
    counts, candidates, windows, used_frames = ({}, [], [], set())
    alignment_anchors, query_ledger = ([], [])
    control_queries = control_reviewable = 0
    control_anchors = 0
    rng = np.random.default_rng(381)
    for number, anchor in enumerate(anchors):
        chosen = temporal_reference_indices(origins, int(anchor), baseline_m=c.min_baseline_m)
        prepared = prepare_structure_inputs(source, config=c, frame_indices=chosen, query_frames={frames[anchor].idx}, source_snapshot=before, reference_step=reference_step)
        result = search_prepared_structure(prepared, config=c)
        for key, value in result['counts'].items():
            counts[key] = counts.get(key, 0) + value
        candidates.extend(result['candidates'])
        alignment_anchors.extend(result.get('alignment_anchors', []))
        query_ledger.extend(result.get('query_ledger', []))
        used_frames.update(chosen)
        windows.append({'query_frame': frames[anchor].idx, 'reference_frames': [frames[i].idx for i in chosen[1:]], 'patches': len(prepared['catalog'].points), 'counts': result['counts']})
        if number < negative_control_frames:
            catalog = prepared['catalog']
            shifted = catalog.points.copy()
            for frame_id in np.unique(catalog.frames):
                if frame_id != frames[anchor].idx:
                    shifted[catalog.frames == frame_id] += rng.uniform(-3.0, 3.0, 3)
            control = search_prepared_structure({**prepared, 'catalog': replace(catalog, points=shifted)}, config=c)
            control_queries += control['counts']['queries']
            control_reviewable += control['counts']['reviewable']
            control_anchors += control['anchor_summary']['eligible']
        if progress and (number % 8 == 0 or number == len(anchors) - 1):
            progress(f"Yakin zaman {number + 1}/{len(anchors)} / {counts.get('reviewable', 0)} aday")
    if before != snapshot_source(source):
        raise ValueError('raw source changed during structure search')
    payload = {'schema': 'cca.structure-search.v1', 'run': source.name, 'unit': 'm', 'axis': 'ARKit_Y_up', 'pixel_convention': c.pixel_convention, 'source_snapshot': asdict(before), 'raw_preserved': True, 'config': asdict(c), 'search_strategy': 'TEMPORAL_CAMERA_BASELINE_WINDOWS', 'reference_step_px': reference_step, 'frames_total': len(frames), 'frames_sampled': len(used_frames), 'query_frames': len(anchors), 'scope': 'SAMPLED_QUERY_PIXELS_WITH_NEAR_TIME_REFERENCES_NOT_ALL_PIXELS', 'applied_observations': 0, 'position_certified': False, 'counts': counts, 'candidates': candidates, 'windows': windows, 'alignment_anchors': alignment_anchors, 'query_ledger': query_ledger, 'query_frames_have_rgb': bool(rgb_queries), 'anchor_summary': {'eligible': sum((r['eligible_for_correspondence_search'] for r in alignment_anchors)), 'free_space_rejected': sum((r['status'] == 'ANCHOR_FREE_SPACE_CONFLICT' for r in alignment_anchors)), 'visibility_unverified': sum((r['status'] == 'ANCHOR_VISIBILITY_UNVERIFIED' for r in alignment_anchors)), 'position_certified': False, 'photo_model_correspondences_verified': False}, 'negative_control_queries': control_queries, 'negative_control_reviewable': control_reviewable, 'negative_control_anchors': control_anchors, 'negative_control': 'Other frames displaced independently +/-3m; not a calibrated field false-positive rate.', 'seconds': round(time.monotonic() - start, 3)}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_ascii_json(output, payload)
    return payload
