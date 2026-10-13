
from __future__ import annotations
from typing import Any
from mcp.server import MCPServer
from viewer.lefebvre_mcp_bridge import LefebvreBridgeError, enqueue_command, read_state
from viewer.layer_controller import EvidenceProfile
mcp = MCPServer('cca-lefebvre', title='CCA Lefebvre', description='Local control for CCA Lefebvre. Reads active project/view state and queues display-only source, selection and visibility changes. Raw scan files stay read-only.')

def _state_or_error() -> dict[str, Any]:
    try:
        return read_state()
    except LefebvreBridgeError as error:
        return {'ok': False, 'error': str(error), 'next_step': 'Open CCA Lefebvre first.'}

@mcp.tool(name='lefebvre_project_status', description='Read the active Lefebvre project, source profile and visible/missing scan counts.')
def project_status() -> dict[str, Any]:
    return _state_or_error()

@mcp.tool(name='lefebvre_list_scans', description='List active project scans with their visibility, source layer and availability.')
def list_scans(visible_only: bool=False) -> dict[str, Any]:
    state = _state_or_error()
    if state.get('ok') is False:
        return state
    scans = state.get('scans', [])
    if visible_only:
        scans = [scan for scan in scans if scan.get('visible')]
    return {'ok': True, 'project': state.get('project'), 'scans': scans}

@mcp.tool(name='lefebvre_select_scan', description='Select one scan in the running Lefebvre window. Does not alter data or transforms.')
def select_scan(scan_id: str) -> dict[str, Any]:
    state = _state_or_error()
    if state.get('ok') is False:
        return state
    known = {str(scan.get('scan_id')) for scan in state.get('scans', [])}
    if scan_id not in known:
        return {'ok': False, 'error': f'unknown scan id: {scan_id}'}
    command = enqueue_command('select_scan', {'scan_id': scan_id})
    return {'ok': True, 'queued': command['command_id'], 'scan_id': scan_id}

@mcp.tool(name='lefebvre_set_visibility', description='Show or hide named scans in the running Lefebvre view. No source file is changed.')
def set_visibility(scan_ids: list[str], visible: bool) -> dict[str, Any]:
    state = _state_or_error()
    if state.get('ok') is False:
        return state
    known = {str(scan.get('scan_id')) for scan in state.get('scans', [])}
    requested = tuple(dict.fromkeys((str(item) for item in scan_ids)))
    unknown = sorted(set(requested) - known)
    if not requested or unknown:
        return {'ok': False, 'error': 'scan_ids must name known scans', 'unknown': unknown}
    command = enqueue_command('set_visibility', {'scan_ids': list(requested), 'visible': bool(visible)})
    return {'ok': True, 'queued': command['command_id'], 'scan_ids': list(requested), 'visible': bool(visible)}

@mcp.tool(name='lefebvre_set_source', description='Choose a source profile in the running Lefebvre view. KAYNAK shows every native source without selecting corrected geometry.')
def set_source(profile: str) -> dict[str, Any]:
    try:
        selected = EvidenceProfile(profile)
    except ValueError:
        allowed = [item.value for item in EvidenceProfile]
        return {'ok': False, 'error': 'unsupported source profile', 'allowed': allowed}
    state = _state_or_error()
    if state.get('ok') is False:
        return state
    command = enqueue_command('set_source', {'profile': selected.value})
    return {'ok': True, 'queued': command['command_id'], 'profile': selected.value}
if __name__ == '__main__':
    mcp.run(transport='stdio')
