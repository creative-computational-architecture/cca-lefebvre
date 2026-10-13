
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
PRODUCT_NAME = 'CCA Lefebvre'
PANEL_TITLE = 'Lefebvre'
ORGANIZATION_NAME = 'Caglar Celik Architects'
CREATIVE_NAME = 'Creative Computational Architecture'
BRAND_ACRONYM = 'CCA'
BRAND_RELEASE = '3.0.0'
COMPANY_MARK_SHA256 = 'f2402d68f9853b2fad9f9494c9da35794e3cee922ee3eb263f935919fae5e2e9'
FONT_SANS_CANDIDATES = ('Space Grotesk', 'Segoe UI')
FONT_MONO_CANDIDATES = ('JetBrains Mono', 'Consolas')
SKIN_C_DARK = {'background': '#1A1A1A', 'surface': '#202020', 'sidebar': '#151515', 'ink': '#F5F5F5', 'muted': '#A8A8A3', 'approval': '#3F8F63', 'warning': '#E08430', 'accent': '#C8102E'}
_SCHEMA = 'cca.points-brand-assets.v1'
_SVG_FILENAME = 'cca-mono-dark.svg'
_MANIFEST_FILENAME = 'asset-manifest.json'
_EXPECTED_ASSET_NAMES = frozenset((_SVG_FILENAME,))
_EXPECTED_IDENTITY = {'acronym': BRAND_ACRONYM, 'creative_name': CREATIVE_NAME, 'mark': 'cca-mono', 'organization': ORGANIZATION_NAME, 'product': PRODUCT_NAME, 'skin': 'Skin C', 'usage': 'dark surfaces'}

class BrandError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class BrandAssets:
    manifest: Path
    company_svg: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (self.company_svg,)

def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BrandError(f'duplicate manifest key: {key}')
        result[key] = value
    return result

def _read_manifest(manifest: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest.read_text(encoding='ascii'), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BrandError(f'could not read brand asset manifest: {error}') from error
    if not isinstance(payload, dict):
        raise BrandError('brand asset manifest must be an object')
    return payload

def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise BrandError(f'could not read brand asset: {path.name}') from error

def verify_manifest_hashes(payload: dict[str, Any], assets: BrandAssets) -> None:
    if payload.get('schema') != _SCHEMA:
        raise BrandError('unsupported brand asset manifest')
    if payload.get('registry_release') != BRAND_RELEASE:
        raise BrandError('unexpected brand registry release')
    if payload.get('identity') != _EXPECTED_IDENTITY:
        raise BrandError('unexpected brand identity metadata')
    files = payload.get('files')
    if not isinstance(files, dict) or set(files) != _EXPECTED_ASSET_NAMES:
        raise BrandError('brand asset manifest has undeclared or missing files')
    if any((not isinstance(value, str) or len(value) != 64 for value in files.values())):
        raise BrandError('brand asset manifest has invalid hashes')
    source_hash = payload.get('source_svg_sha256')
    if source_hash != COMPANY_MARK_SHA256:
        raise BrandError('brand source SVG hash mismatch')
    if files[_SVG_FILENAME] != COMPANY_MARK_SHA256:
        raise BrandError('brand SVG manifest hash mismatch')
    actual_files = {path.name for path in assets.manifest.parent.iterdir() if path.is_file()}
    expected_files = _EXPECTED_ASSET_NAMES | {_MANIFEST_FILENAME}
    if actual_files != expected_files:
        raise BrandError('brand asset directory has undeclared product assets')
    for path in assets.all_paths():
        if not path.is_file():
            raise BrandError(f'missing brand asset: {path.name}')
        actual = _sha256(path)
        if actual != files[path.name]:
            raise BrandError(f'brand asset hash mismatch: {path.name}')

def load_brand_assets(asset_dir: Path | None=None) -> BrandAssets:
    root = asset_dir or Path(__file__).resolve().parent / 'assets' / 'brand' / 'v3'
    manifest = root / _MANIFEST_FILENAME
    payload = _read_manifest(manifest)
    if payload.get('schema') != _SCHEMA:
        raise BrandError('unsupported brand asset manifest')
    assets = BrandAssets(manifest=manifest, company_svg=root / _SVG_FILENAME)
    verify_manifest_hashes(payload, assets)
    return assets

def skin_color(name: str) -> tuple[float, float, float]:
    try:
        token = SKIN_C_DARK[name]
    except KeyError as error:
        raise BrandError(f'unknown Skin C color: {name}') from error
    if not isinstance(token, str) or len(token) != 7 or (not token.startswith('#')):
        raise BrandError(f'invalid Skin C color token: {name}')
    try:
        channels = bytes.fromhex(token[1:])
    except ValueError as error:
        raise BrandError(f'invalid Skin C color token: {name}') from error
    if len(channels) != 3:
        raise BrandError(f'invalid Skin C color token: {name}')
    return tuple((channel / 255 for channel in channels))
