
from __future__ import annotations
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
BRIDGE_SCHEMA = 'cca.lefebvre.mcp-bridge.v1'
_ACTIONS = frozenset({'select_scan', 'set_visibility', 'set_source'})

class LefebvreBridgeError(ValueError):
    pass

def bridge_root(root: str | Path | None=None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve(strict=False)
    override = os.environ.get('CCA_LEFEBVRE_BRIDGE_ROOT', '').strip()
    if override:
        return Path(override).expanduser().resolve(strict=False)
    base = os.environ.get('LOCALAPPDATA', '').strip()
    parent = Path(base) if base else Path.home() / '.local' / 'share'
    return (parent / 'CCA Lefebvre' / 'mcp').resolve(strict=False)

def state_file(root: str | Path | None=None) -> Path:
    return bridge_root(root) / 'state.json'

def commands_dir(root: str | Path | None=None) -> Path:
    return bridge_root(root) / 'commands'

def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    temporary.replace(path)

def publish_state(payload: Mapping[str, Any], *, root: str | Path | None=None) -> Path:
    state = dict(payload)
    state['schema'] = BRIDGE_SCHEMA
    state['published_at_utc'] = datetime.now(UTC).isoformat()
    target = state_file(root)
    _write_json_atomic(target, state)
    return target

def read_state(*, root: str | Path | None=None) -> dict[str, Any]:
    path = state_file(root)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        raise LefebvreBridgeError('Lefebvre is not running or has not published MCP state') from error
    except (OSError, json.JSONDecodeError) as error:
        raise LefebvreBridgeError('Lefebvre MCP state cannot be read') from error
    if not isinstance(payload, dict) or payload.get('schema') != BRIDGE_SCHEMA:
        raise LefebvreBridgeError('Lefebvre MCP state schema is invalid')
    return payload

def enqueue_command(action: str, payload: Mapping[str, Any], *, root: str | Path | None=None) -> dict[str, Any]:
    if action not in _ACTIONS:
        raise LefebvreBridgeError(f'unsupported Lefebvre MCP action: {action}')
    if not isinstance(payload, Mapping):
        raise LefebvreBridgeError('Lefebvre MCP command payload must be an object')
    command_id = uuid.uuid4().hex
    command = {'schema': BRIDGE_SCHEMA, 'command_id': command_id, 'action': action, 'payload': dict(payload), 'created_at_utc': datetime.now(UTC).isoformat()}
    target = commands_dir(root) / f'{command_id}.json'
    _write_json_atomic(target, command)
    return command

def take_commands(*, root: str | Path | None=None) -> tuple[dict[str, Any], ...]:
    directory = commands_dir(root)
    if not directory.is_dir():
        return ()
    claimed: list[dict[str, Any]] = []
    for source in sorted(directory.glob('*.json')):
        processing = source.with_suffix('.processing')
        try:
            source.replace(processing)
            payload = json.loads(processing.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        finally:
            try:
                processing.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(payload, dict) and payload.get('schema') == BRIDGE_SCHEMA and (payload.get('action') in _ACTIONS) and isinstance(payload.get('payload'), dict):
            claimed.append(payload)
    return tuple(claimed)
__all__ = ['BRIDGE_SCHEMA', 'LefebvreBridgeError', 'bridge_root', 'commands_dir', 'enqueue_command', 'publish_state', 'read_state', 'state_file', 'take_commands']
