
from __future__ import annotations
from dataclasses import replace
from itertools import combinations
import cv2
import numpy as np
from field import m0util
from geometry.visual_bridge import BridgeConfig, enumerate_side_frames, extract_features, build_retrieval_index, retrieve_candidates, _mutual_ratio_matches
from geometry.temporal_endpoint_registration import _side, select_endpoint_frames, EndpointWindowConfig

def spread_frames(frames, count=24):
    if len(frames) <= count:
        return tuple(frames)
    return tuple((frames[i] for i in np.linspace(0, len(frames) - 1, count, dtype=int)))

def color_signature(path):
    image = m0util.read_sensor_image(path, cv2.IMREAD_COLOR)
    small = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 12, 12], [0, 256] * 3).reshape(-1)
    return hist / max(float(hist.sum()), 1.0)

def compare_rgb_groups(source, target, config, signatures, *, max_checks=4):
    if not source or not target:
        return {'score': 0.0, 'rgb_similarity': 0.0, 'supports': [], 'matching_details': 0}
    for feature in (*source, *target):
        path = feature.frame.rgb_path
        if path not in signatures:
            signatures[path] = color_signature(path)
    color_matches = [(float(np.minimum(signatures[a.frame.rgb_path], signatures[b.frame.rgb_path]).sum()), i, j) for i, a in enumerate(source) for j, b in enumerate(target)]
    color_matches.sort(reverse=True)
    candidates = retrieve_candidates(source, target, config, retrieval_index=build_retrieval_index(target))
    seen = {(i, j) for i, j, _, _ in candidates[:max_checks]}
    candidates = candidates[:max_checks] + [(i, j, 0.0, 0) for _, i, j in color_matches[:2] if (i, j) not in seen]
    supports = []
    for i, j, vote, _ in candidates:
        a, b = (source[i], target[j])
        matches = _mutual_ratio_matches(a, b, config)
        if len(matches) < 8:
            continue
        xa = np.array([a.xy_full[x] for x, _, _ in matches], np.float32)
        xb = np.array([b.xy_full[y] for _, y, _ in matches], np.float32)
        _, mask = cv2.findHomography(xa, xb, cv2.RANSAC, config.homography_threshold_px)
        inliers = int(mask.sum()) if mask is not None else 0
        for f in (a, b):
            if f.frame.rgb_path not in signatures:
                signatures[f.frame.rgb_path] = color_signature(f.frame.rgb_path)
        color = float(np.minimum(signatures[a.frame.rgb_path], signatures[b.frame.rgb_path]).sum())
        if inliers >= 8:
            supports.append({'source_frame': a.frame.frame_index, 'target_frame': b.frame.frame_index, 'rgb_similarity': color, 'matched_details': inliers, 'match_ratio': inliers / len(matches), 'orb_vote': float(vote)})
    supports.sort(key=lambda row: row['matched_details'] * row['match_ratio'], reverse=True)
    details = sum((row['matched_details'] for row in supports))
    color = float(np.mean([r['rgb_similarity'] for r in supports])) if supports else color_matches[0][0]
    score = color * (sum((np.log1p(r['matched_details']) * r['match_ratio'] for r in supports)) if supports else 0.1)
    return {'score': float(score), 'rgb_similarity': color, 'supports': supports, 'matching_details': details, 'evidence_kind': 'RGB_DETAILS' if supports else 'COLOR_ONLY'}

def discover_rgb_candidates(raw_root, sequence, exclusions=None, progress=None):
    excluded = set(exclusions or {})
    config = replace(BridgeConfig(), orb_features=900, max_candidates=12, retrieval_top_k=3, max_image_width=640)
    groups, signatures = ({}, {})
    for index, run in enumerate(sequence.runs):
        if run.scan_id in excluded:
            continue
        frames = enumerate_side_frames(_side(run.scan_id, None), raw_root)
        head = spread_frames(select_endpoint_frames(frames, 'HEAD', EndpointWindowConfig()), 12)
        tail = spread_frames(select_endpoint_frames(frames, 'TAIL', EndpointWindowConfig()), 12)
        whole = spread_frames(frames, 24)
        unique = {f.frame_index: f for f in (*head, *tail, *whole)}
        feats = {f.frame.frame_index: f for f in extract_features(tuple(unique.values()), config)}
        groups[run.scan_id] = {key: [feats[f.frame_index] for f in chosen if f.frame_index in feats] for key, chosen in (('HEAD', head), ('TAIL', tail), ('ALL', whole))}
        if progress:
            progress(f'RGB aday indeksi {index + 1}/{len(sequence.runs)} / {run.scan_id}')
    jobs = [(p.source_scan_id, p.target_scan_id, 'ENDPOINT', p.gap_s) for p in sequence.pairs if p.source_scan_id in groups and p.target_scan_id in groups]
    jobs += [(a, b, 'OVERLAP', None) for a, b in combinations(groups, 2)]
    output = []
    for index, (a, b, kind, gap) in enumerate(jobs):
        first, second = ('TAIL', 'HEAD') if kind == 'ENDPOINT' else ('ALL', 'ALL')
        value = compare_rgb_groups(groups[a][first], groups[b][second], config, signatures)
        output.append({'source_scan_id': a, 'target_scan_id': b, 'kind': kind, 'gap_s': gap, 'status': 'RGB_CANDIDATE_NOT_ALIGNED', **value})
        if progress and (index % 10 == 0 or index + 1 == len(jobs)):
            progress(f'RGB ciftleri {index + 1}/{len(jobs)}')
    return sorted(output, key=lambda row: row['score'], reverse=True)
