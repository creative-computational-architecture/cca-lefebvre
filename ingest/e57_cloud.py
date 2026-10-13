
from __future__ import annotations
import argparse
import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
_HASH_CHUNK_BYTES = 1024 * 1024

class E57ConversionError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class E57ConversionResult:
    source_path: Path
    destination_path: Path
    point_count: int
    scan_count: int
    has_color: bool
    has_intensity: bool
    source_sha256: str
    destination_sha256: str
    reader: str

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (block := stream.read(_HASH_CHUNK_BYTES)):
            digest.update(block)
    return digest.hexdigest()

def _read_with_e57(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    import e57
    cloud = e57.read_points(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    color = None
    if getattr(cloud, 'color', None) is not None and len(cloud.color):
        colors = np.asarray(cloud.color, dtype=np.float64)
        if colors.max() <= 1.0:
            colors = colors * 255.0
        color = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    intensity = None
    if getattr(cloud, 'intensity', None) is not None and len(cloud.intensity):
        intensity = np.asarray(cloud.intensity, dtype=np.float32)
    return (points, color, intensity, 1)

def _read_with_pye57(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    import pye57
    handle = pye57.E57(str(path))
    parts: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    has_color = has_intensity = True
    for index in range(handle.scan_count):
        data = handle.read_scan(index, ignore_missing_fields=True, colors=True, intensity=True, transform=True)
        parts.append(np.column_stack((data['cartesianX'], data['cartesianY'], data['cartesianZ'])).astype(np.float64))
        if all((k in data for k in ('colorRed', 'colorGreen', 'colorBlue'))):
            colors.append(np.column_stack((data['colorRed'], data['colorGreen'], data['colorBlue'])))
        else:
            has_color = False
        if 'intensity' in data:
            intensities.append(np.asarray(data['intensity'], dtype=np.float32))
        else:
            has_intensity = False
    if not parts:
        raise E57ConversionError('E57 file contains no scans')
    points = np.concatenate(parts, axis=0)
    color = None
    if has_color and colors:
        stacked = np.concatenate(colors, axis=0).astype(np.float64)
        if stacked.max() <= 1.0:
            stacked = stacked * 255.0
        color = np.clip(np.rint(stacked), 0, 255).astype(np.uint8)
    intensity = np.concatenate(intensities) if has_intensity and intensities else None
    return (points, color, intensity, handle.scan_count)

def read_e57_points(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, str]:
    source = Path(path)
    errors: list[str] = []
    for name, reader in (('pye57', _read_with_pye57), ('e57', _read_with_e57)):
        try:
            points, color, intensity, scans = reader(source)
        except ImportError as error:
            errors.append(f'{name}: {error}')
            continue
        except E57ConversionError:
            raise
        except Exception as error:
            errors.append(f'{name}: {error}')
            continue
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise E57ConversionError('E57 reader returned no cartesian points')
        return (points, color, intensity, scans, name)
    raise E57ConversionError('no E57 reader could open the file: ' + '; '.join(errors))

def _write_ply(destination: Path, points: np.ndarray, color: np.ndarray | None, intensity: np.ndarray | None) -> None:
    fields: list[tuple[str, Any]] = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if color is not None:
        fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if intensity is not None:
        fields.append(('intensity', '<f4'))
    dtype = np.dtype(fields)
    record = np.empty(points.shape[0], dtype=dtype)
    record['x'] = points[:, 0]
    record['y'] = points[:, 1]
    record['z'] = points[:, 2]
    if color is not None:
        record['red'], record['green'], record['blue'] = (color[:, 0], color[:, 1], color[:, 2])
    if intensity is not None:
        record['intensity'] = intensity
    type_names = {'<f4': 'float', 'u1': 'uchar'}
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {points.shape[0]}']
    header += [f'property {type_names[fmt]} {name}' for name, fmt in fields]
    header.append('end_header')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=destination.parent, prefix=f'.{destination.name}.', suffix='.tmp')
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(('\n'.join(header) + '\n').encode('ascii'))
            record.tofile(stream)
        os.replace(temp_path, destination)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

def convert_e57_to_ply(source: str | Path, destination: str | Path, *, y_up: bool=False) -> E57ConversionResult:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path.suffix.lower() != '.e57' or not source_path.is_file():
        raise E57ConversionError(f'E57 source does not exist: {source_path}')
    if destination_path.suffix.lower() != '.ply':
        raise E57ConversionError('destination must use the .ply extension')
    if destination_path == source_path:
        raise E57ConversionError('destination must differ from the source')
    sha_before = _sha256(source_path)
    points, color, intensity, scans, reader = read_e57_points(source_path)
    if not np.isfinite(points).all():
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        color = None if color is None else color[finite]
        intensity = None if intensity is None else intensity[finite]
    if y_up:
        points = np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))
    _write_ply(destination_path, points, color, intensity)
    sha_after = _sha256(source_path)
    if sha_after != sha_before:
        raise E57ConversionError('E57 source changed while it was being converted')
    return E57ConversionResult(source_path=source_path, destination_path=destination_path, point_count=int(points.shape[0]), scan_count=int(scans), has_color=color is not None, has_intensity=intensity is not None, source_sha256=sha_before, destination_sha256=_sha256(destination_path), reader=reader)

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Convert an E57 point cloud into a Z-up metre PLY.')
    parser.add_argument('source', help='input .e57 file')
    parser.add_argument('destination', help='output .ply file')
    parser.add_argument('--y-up', action='store_true', help='source is Y-up; apply (x, y, z) -> (x, -z, y)')
    args = parser.parse_args(argv)
    try:
        result = convert_e57_to_ply(args.source, args.destination, y_up=args.y_up)
    except E57ConversionError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    print(f'{result.destination_path} | {result.point_count:,} points | scans {result.scan_count} | color {result.has_color} | intensity {result.has_intensity} | reader {result.reader}')
    return 0
__all__ = ['E57ConversionError', 'E57ConversionResult', 'convert_e57_to_ply', 'read_e57_points']
if __name__ == '__main__':
    raise SystemExit(main())
