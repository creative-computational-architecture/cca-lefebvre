
from pathlib import Path
from dataclasses import asdict
import hashlib
import json

def build_recovery(run_dir, output, progress=None):
    from geometry.visual_bridge import BridgeConfig, enumerate_side_frames, extract_features
    from geometry.temporal_endpoint_registration import _side
    from geometry.ray_recovery import RayRecoveryConfig, recover_feature_depths
    from viewer.raw_processing_service import snapshot_source, _write_ascii_json
    source, output = (Path(run_dir).resolve(), Path(output).resolve())
    if source in output.parents:
        raise ValueError('recovery evidence must be outside raw capture')
    before = snapshot_source(source)
    config = BridgeConfig()
    features = extract_features(enumerate_side_frames(_side(source.name, None), source.parent), config, source.name)
    observations = recover_feature_depths(features, config, progress=progress)
    if before != snapshot_source(source):
        raise ValueError('raw source changed during ray recovery')
    payload = {'schema': 'cca.ray-recovery.v1', 'run': source.name, 'source_snapshot': asdict(before), 'unit': 'm', 'raw_preserved': True, 'config': asdict(RayRecoveryConfig()), 'rgb_frames': len(features), 'observations': observations, 'scope': 'SPARSE_RGB_TRACKS_ONLY', 'unresolved': 'KEEP_MEASURED_DEPTH'}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_ascii_json(output, payload)
    return output

def load_recovery(path, run_dir):
    from viewer.raw_processing_service import snapshot_source
    content = Path(path).read_bytes()
    doc = json.loads(content)
    source = Path(run_dir).resolve()
    if not isinstance(doc, dict) or doc.get('schema') != 'cca.ray-recovery.v1' or doc.get('run') != source.name or (doc.get('unit') != 'm') or (doc.get('raw_preserved') is not True):
        raise ValueError('recovery document/run contract mismatch')
    if doc.get('source_snapshot') != asdict(snapshot_source(source)):
        raise ValueError('recovery evidence is stale for the raw source')
    if not isinstance(doc.get('observations'), list):
        raise ValueError('recovery observations must be a list')
    by_frame = {}
    for row in doc['observations']:
        frame = row.get('frame_index')
        if type(frame) is not int or frame < 0:
            raise ValueError('invalid recovery frame index')
        by_frame.setdefault(frame, []).append(row)
    provenance = {'schema': doc['schema'], 'run': doc['run'], 'sha256': hashlib.sha256(content).hexdigest(), 'proposed_observations': len(doc['observations']), 'scope': doc['scope'], 'source_tree_sha256': doc['source_snapshot']['tree_sha256']}
    return {'by_frame': by_frame, 'provenance': provenance}
