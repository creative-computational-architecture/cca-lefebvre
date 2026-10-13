
from __future__ import annotations
import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any
import numpy as np

class PlacementError(ValueError):
    pass

def placement_matrix(*, tx_m: float, ty_m: float, tz_m: float, yaw_deg: float) -> np.ndarray:
    values = (tx_m, ty_m, tz_m, yaw_deg)
    if not all((math.isfinite(float(value)) for value in values)):
        raise PlacementError('placement values must be finite')
    radians = math.radians(float(yaw_deg))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray([[cosine, -sine, 0.0, float(tx_m)], [sine, cosine, 0.0, float(ty_m)], [0.0, 0.0, 1.0, float(tz_m)], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding='ascii'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlacementError(f'cannot read {label}: {source}') from exc
    if not isinstance(value, dict):
        raise PlacementError(f'{label} must be a JSON object')
    return value

def _validated_base_matrix(value: Any, run_id: str) -> list[list[float]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PlacementError(f'run {run_id} has an invalid Z-up base transform')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-09):
        raise PlacementError(f'run {run_id} base transform has an invalid homogeneous row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-06):
        raise PlacementError(f'run {run_id} base transform is not rigid')
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-06):
        raise PlacementError(f'run {run_id} base transform is not proper')
    return matrix.tolist()

def _placement() -> dict[str, float | str]:
    return {'tx_m': 0.0, 'ty_m': 0.0, 'tz_m': 0.0, 'yaw_deg': 0.0, 'scale': 1.0, 'status': 'UNPLACED'}

def build_manual_workspace(fusion_artifact: str | Path, preview_artifact: str | Path) -> dict[str, Any]:
    fusion = _read_json(fusion_artifact, 'fusion artifact')
    preview = _read_json(preview_artifact, 'registration preview')
    policy = fusion.get('coordinate_policy')
    if not isinstance(policy, dict) or not policy.get('provisional_component_runs_kept_independent'):
        raise PlacementError('fusion artifact does not preserve provisional runs')
    rows = preview.get('runs')
    if not isinstance(rows, list):
        raise PlacementError('registration preview runs must be a list')
    preview_by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('run'), str):
            raise PlacementError('registration preview has an invalid run row')
        run_id = row['run']
        if run_id in preview_by_run:
            raise PlacementError(f'duplicate registration preview run: {run_id}')
        preview_by_run[run_id] = row
    components = fusion.get('components')
    if not isinstance(components, list):
        raise PlacementError('fusion components must be a list')
    units: list[dict[str, Any]] = []

    def append_unit(unit_id: str, component_id: str, status: str, run_ids: list[str]) -> None:
        unit_runs = []
        for run_id in run_ids:
            row = preview_by_run.get(run_id)
            if row is None:
                raise PlacementError(f'run {run_id} is missing from registration preview')
            if row.get('component') != component_id:
                raise PlacementError(f'run {run_id} component disagrees across artifacts')
            source_ply = row.get('source_ply')
            if not isinstance(source_ply, str) or not source_ply:
                raise PlacementError(f'run {run_id} has no preview PLY')
            unit_runs.append({'id': run_id, 'source_ply': source_ply.replace('\\', '/'), 'base_T_context_zup_from_run_yup': _validated_base_matrix(row.get('T_global_zup_from_run'), run_id), 'rgb_hex': str(row.get('rgb_hex', '#b0b0b0')), 'point_count': int(row.get('point_count', 0))})
        units.append({'id': unit_id, 'component': component_id, 'source_status': status, 'runs': run_ids, 'run_previews': unit_runs, 'placement': _placement()})
    for component in components:
        if not isinstance(component, dict):
            raise PlacementError('fusion component row must be an object')
        component_id = component.get('id')
        status = component.get('status')
        run_ids = component.get('runs')
        if not isinstance(component_id, str) or not isinstance(status, str):
            raise PlacementError('fusion component id and status must be strings')
        if not isinstance(run_ids, list) or not run_ids or (not all((isinstance(item, str) for item in run_ids))):
            raise PlacementError(f'component {component_id} has invalid runs')
        if status == 'PROVISIONAL':
            for run_id in run_ids:
                append_unit(f'RUN_{run_id}', component_id, status, [run_id])
        else:
            append_unit(component_id, component_id, status, list(run_ids))
    unit_ids = [unit['id'] for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise PlacementError('unit ids must be unique')
    return {'schema': 'cca.manual-plan-workspace.v1', 'status': 'MANUAL_PLAN_CONTEXT_ONLY', 'coordinate_contract': {'input': 'CAD Z-up provisional context', 'placement': 'T_plan_zup_from_context_zup', 'output': 'CAD Z-up plan metres', 'scale': 1.0}, 'workspace_contract': {'unit_ids': unit_ids}, 'plan': None, 'units': units, 'evidence_boundary': {'manual_placement_is_registration_evidence': False, 'source_plys_read_only': True, 'distance_cutoff_m': None, 'points_rejected': 0}}

def _validate_workspace(workspace: dict[str, Any]) -> None:
    if workspace.get('schema') != 'cca.manual-plan-workspace.v1':
        raise PlacementError('unsupported placement workspace schema')
    contract = workspace.get('coordinate_contract')
    if not isinstance(contract, dict) or contract.get('scale') != 1.0:
        raise PlacementError('workspace scale must remain 1.0')
    workspace_contract = workspace.get('workspace_contract')
    expected_ids = workspace_contract.get('unit_ids') if isinstance(workspace_contract, dict) else None
    units = workspace.get('units')
    if not isinstance(expected_ids, list) or not all((isinstance(item, str) for item in expected_ids)):
        raise PlacementError('workspace contract has invalid unit ids')
    if not isinstance(units, list):
        raise PlacementError('workspace units must be a list')
    actual_ids = [unit.get('id') for unit in units if isinstance(unit, dict)]
    if actual_ids != expected_ids:
        raise PlacementError('unit ids do not match workspace contract')
    if len(actual_ids) != len(set(actual_ids)):
        raise PlacementError('unit ids must be unique')
    for unit in units:
        placement = unit.get('placement')
        if not isinstance(placement, dict):
            raise PlacementError(f"unit {unit.get('id')} has no placement")
        if placement.get('scale') != 1.0:
            raise PlacementError(f"unit {unit.get('id')} scale must remain 1.0")
        for name in ('tx_m', 'ty_m', 'tz_m', 'yaw_deg'):
            value = placement.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(float(value))):
                raise PlacementError(f"unit {unit.get('id')} {name} must be finite")
        if placement.get('status') not in {'UNPLACED', 'MANUAL_PLAN_PLACEMENT'}:
            raise PlacementError(f"unit {unit.get('id')} has an invalid placement status")
        placement_matrix(tx_m=float(placement['tx_m']), ty_m=float(placement['ty_m']), tz_m=float(placement['tz_m']), yaw_deg=float(placement['yaw_deg']))

def save_placement_session(workspace: dict[str, Any], destination: str | Path) -> None:
    _validate_workspace(workspace)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(workspace, indent=2, sort_keys=True, ensure_ascii=True) + '\n'
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='ascii', newline='\n', dir=output.parent, prefix=f'.{output.name}.', suffix='.tmp', delete=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = Path(stream.name)
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

def load_placement_session(source: str | Path) -> dict[str, Any]:
    workspace = _read_json(source, 'placement session')
    _validate_workspace(workspace)
    return copy.deepcopy(workspace)
