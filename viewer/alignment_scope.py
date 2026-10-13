
from __future__ import annotations
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
ALIGNMENT_EXCLUSION_REASONS = ('BASEMENT', 'ADMIN', 'OTHER')

class AlignmentScopeError(ValueError):
    pass

def _validate(project: Any, exclusions: Mapping[str, str]) -> dict[str, str]:
    scans = {str(scan.scan_id) for scan in getattr(project, 'scans', ())}
    if not isinstance(exclusions, Mapping):
        raise AlignmentScopeError('alignment_exclusions must be a mapping')
    validated: dict[str, str] = {}
    for scan_id, reason in exclusions.items():
        if not isinstance(scan_id, str) or scan_id not in scans:
            raise AlignmentScopeError(f'alignment exclusion references unknown scan: {scan_id!r}')
        if not isinstance(reason, str) or reason not in ALIGNMENT_EXCLUSION_REASONS:
            raise AlignmentScopeError(f'alignment exclusion reason must be one of {ALIGNMENT_EXCLUSION_REASONS}: {reason!r}')
        try:
            reason.encode('ascii')
        except UnicodeEncodeError as error:
            raise AlignmentScopeError('alignment exclusion reason must be ASCII') from error
        validated[scan_id] = reason
    return validated

def alignment_exclusions(project: Any) -> Mapping[str, str]:
    raw = getattr(project, 'metadata', {}).get('alignment_exclusions', {})
    return MappingProxyType(_validate(project, raw))

def resolve_alignment_exclusion(project: Any, scan_id: str) -> str | None:
    return alignment_exclusions(project).get(scan_id)

def set_alignment_exclusions(project: Any, exclusions: Mapping[str, str]) -> Mapping[str, str]:
    validated = _validate(project, exclusions)
    project.metadata = dict(getattr(project, 'metadata', {}))
    project.metadata['alignment_exclusions'] = dict(validated)
    return MappingProxyType(validated)
__all__ = ['ALIGNMENT_EXCLUSION_REASONS', 'AlignmentScopeError', 'alignment_exclusions', 'resolve_alignment_exclusion', 'set_alignment_exclusions']
