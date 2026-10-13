
from __future__ import annotations
import math
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import BinaryIO
import numpy as np
_SCALAR_DTYPES = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4', 'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8', 'int64': 'i8', 'uint64': 'u8'}
_FLOAT_TYPES = {'float', 'float32', 'double', 'float64'}
_FORMATS = {'ascii', 'binary_little_endian', 'binary_big_endian'}
_TOKEN_RE = re.compile(b'\\S+')

class PlyFormatError(ValueError):
    pass

@dataclass(frozen=True)
class PlyProperty:
    name: str
    scalar_type: str | None = None
    list_count_type: str | None = None
    list_item_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.scalar_type is None

@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    properties: tuple[PlyProperty, ...]

@dataclass(frozen=True)
class PlySchema:
    path: Path
    format: str
    version: str
    header: bytes
    data_offset: int
    elements: tuple[PlyElement, ...]
    vertex: PlyElement

    @property
    def vertex_signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(((prop.scalar_type or 'list', prop.name) for prop in self.vertex.properties))

def read_ply_schema(path: str | Path) -> PlySchema:
    source = Path(path)
    header_parts: list[bytes] = []
    format_name = ''
    version = ''
    elements: list[PlyElement] = []
    current_name: str | None = None
    current_count = 0
    current_properties: list[PlyProperty] = []

    def close_current() -> None:
        nonlocal current_name, current_count, current_properties
        if current_name is not None:
            elements.append(PlyElement(current_name, current_count, tuple(current_properties)))
            current_name = None
            current_count = 0
            current_properties = []
    with source.open('rb') as stream:
        first = stream.readline()
        if first.strip() != b'ply':
            raise PlyFormatError(f'not a PLY file: {source}')
        header_parts.append(first)
        while True:
            raw = stream.readline()
            if not raw:
                raise PlyFormatError(f'PLY header has no end_header: {source}')
            header_parts.append(raw)
            try:
                line = raw.decode('ascii').strip()
            except UnicodeDecodeError as exc:
                raise PlyFormatError(f'PLY header is not ASCII: {source}') from exc
            parts = line.split()
            if not parts or parts[0] in {'comment', 'obj_info'}:
                continue
            if parts[0] == 'format':
                if len(parts) != 3 or parts[1] not in _FORMATS:
                    raise PlyFormatError(f'unsupported PLY format line: {line}')
                format_name, version = (parts[1], parts[2])
            elif parts[0] == 'element':
                close_current()
                if len(parts) != 3:
                    raise PlyFormatError(f'invalid PLY element line: {line}')
                current_name = parts[1]
                try:
                    current_count = int(parts[2])
                except ValueError as exc:
                    raise PlyFormatError(f'invalid PLY element count: {line}') from exc
                if current_count < 0:
                    raise PlyFormatError(f'negative PLY element count: {line}')
            elif parts[0] == 'property':
                if current_name is None:
                    raise PlyFormatError(f'PLY property outside an element: {line}')
                if len(parts) == 3:
                    scalar_type, name = (parts[1], parts[2])
                    if scalar_type not in _SCALAR_DTYPES:
                        raise PlyFormatError(f'unsupported PLY scalar type: {scalar_type}')
                    current_properties.append(PlyProperty(name=name, scalar_type=scalar_type))
                elif len(parts) == 5 and parts[1] == 'list':
                    count_type, item_type, name = (parts[2], parts[3], parts[4])
                    if count_type not in _SCALAR_DTYPES or item_type not in _SCALAR_DTYPES:
                        raise PlyFormatError(f'unsupported PLY list property: {line}')
                    current_properties.append(PlyProperty(name=name, list_count_type=count_type, list_item_type=item_type))
                else:
                    raise PlyFormatError(f'invalid PLY property line: {line}')
            elif parts[0] == 'end_header':
                close_current()
                data_offset = stream.tell()
                break
    if not format_name:
        raise PlyFormatError(f'PLY format is missing: {source}')
    if version != '1.0':
        raise PlyFormatError(f'unsupported PLY version {version}: {source}')
    vertex_matches = [element for element in elements if element.name == 'vertex']
    if len(vertex_matches) != 1:
        raise PlyFormatError(f'PLY needs exactly one vertex element: {source}')
    vertex = vertex_matches[0]
    if not elements or elements[0].name != 'vertex':
        raise PlyFormatError(f'vertex must be the first PLY element: {source}')
    if any((prop.is_list for prop in vertex.properties)):
        raise PlyFormatError(f'list properties in vertex records are unsupported: {source}')
    names = [prop.name for prop in vertex.properties]
    if len(names) != len(set(names)):
        raise PlyFormatError(f'duplicate vertex property names: {source}')
    for name in ('x', 'y', 'z'):
        if name not in names:
            raise PlyFormatError(f'vertex property {name} is missing: {source}')
        prop = vertex.properties[names.index(name)]
        if prop.scalar_type not in _FLOAT_TYPES:
            raise PlyFormatError(f'vertex property {name} must be floating point: {source}')
    normal_names = {'nx', 'ny', 'nz'}.intersection(names)
    if normal_names and normal_names != {'nx', 'ny', 'nz'}:
        raise PlyFormatError(f'PLY has an incomplete normal triplet: {source}')
    for name in normal_names:
        prop = vertex.properties[names.index(name)]
        if prop.scalar_type not in _FLOAT_TYPES:
            raise PlyFormatError(f'normal property {name} must be floating point: {source}')
    return PlySchema(path=source, format=format_name, version=version, header=b''.join(header_parts), data_offset=data_offset, elements=tuple(elements), vertex=vertex)

def _numpy_vertex_dtype(schema: PlySchema) -> np.dtype:
    if schema.format == 'binary_little_endian':
        endian = '<'
    elif schema.format == 'binary_big_endian':
        endian = '>'
    else:
        endian = '='
    fields = []
    for prop in schema.vertex.properties:
        code = _SCALAR_DTYPES[prop.scalar_type or '']
        fields.append((prop.name, code if code.endswith('1') else endian + code))
    return np.dtype(fields, align=False)

def iter_ply_vertex_chunks(source: str | Path, chunk_vertices: int=250000) -> Iterator[np.ndarray]:
    if chunk_vertices < 1:
        raise ValueError('chunk_vertices must be positive')
    schema = read_ply_schema(source)
    dtype = _numpy_vertex_dtype(schema)
    with schema.path.open('rb') as input_stream:
        input_stream.seek(schema.data_offset)
        remaining = schema.vertex.count
        while remaining:
            count = min(remaining, chunk_vertices)
            if schema.format != 'ascii':
                byte_count = count * dtype.itemsize
                raw = input_stream.read(byte_count)
                if len(raw) != byte_count:
                    raise PlyFormatError(f'unexpected EOF in binary vertices: {schema.path}')
                yield np.frombuffer(raw, dtype=dtype, count=count).copy()
                remaining -= count
                continue
            lines = list(islice(input_stream, count))
            if len(lines) != count:
                raise PlyFormatError(f'unexpected EOF in ASCII vertices: {schema.path}')
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', DeprecationWarning)
                    warnings.filterwarnings('ignore', message='loadtxt: input contained no data', category=UserWarning)
                    chunk = np.loadtxt(lines, dtype=dtype, comments=None, ndmin=1)
                if len(chunk) != count:
                    raise ValueError('blank ASCII vertex row')
            except (OverflowError, ValueError, DeprecationWarning):
                chunk = np.empty(count, dtype=dtype)
                for row_index, line in enumerate(lines):
                    tokens = _TOKEN_RE.findall(line)
                    if len(tokens) != len(schema.vertex.properties):
                        raise PlyFormatError(f'ASCII vertex token count {len(tokens)} != {len(schema.vertex.properties)} in {schema.path}')
                    try:
                        for prop, token in zip(schema.vertex.properties, tokens):
                            if prop.scalar_type in _FLOAT_TYPES:
                                chunk[prop.name][row_index] = float(token)
                            else:
                                chunk[prop.name][row_index] = int(token)
                    except (OverflowError, ValueError) as exc:
                        raise PlyFormatError(f'invalid ASCII scalar value in {schema.path}') from exc
            yield chunk
            remaining -= count

def _validate_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('PLY transform must be a finite 4x4 matrix')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-08):
        raise ValueError('PLY transform has an invalid homogeneous row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-07):
        raise ValueError('PLY transform rotation is not rigid')
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-07):
        raise ValueError('PLY transform rotation is not proper')
    return matrix

def _transform_structured_vertices(array: np.ndarray, transform: np.ndarray) -> None:
    xyz = np.column_stack([array['x'], array['y'], array['z']]).astype(np.float64)
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    array['x'], array['y'], array['z'] = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    if all((name in array.dtype.names for name in ('nx', 'ny', 'nz'))):
        normals = np.column_stack([array['nx'], array['ny'], array['nz']]).astype(np.float64)
        normals = normals @ transform[:3, :3].T
        array['nx'], array['ny'], array['nz'] = (normals[:, 0], normals[:, 1], normals[:, 2])

def _format_ascii_float(value: float, scalar_type: str) -> bytes:
    if scalar_type in {'float', 'float32'}:
        cast = float(np.float32(value))
        return format(cast, '.9g').encode('ascii')
    return format(float(value), '.17g').encode('ascii')

def _transform_ascii_line(line: bytes, schema: PlySchema, transform: np.ndarray) -> bytes:
    matches = list(_TOKEN_RE.finditer(line))
    properties = schema.vertex.properties
    if len(matches) != len(properties):
        raise PlyFormatError(f'ASCII vertex token count {len(matches)} != {len(properties)} in {schema.path}')
    index = {prop.name: i for i, prop in enumerate(properties)}
    try:
        xyz = np.array([float(matches[index[name]].group()) for name in ('x', 'y', 'z')], dtype=np.float64)
    except ValueError as exc:
        raise PlyFormatError(f'invalid ASCII XYZ value in {schema.path}') from exc
    transformed_xyz = transform[:3, :3] @ xyz + transform[:3, 3]
    replacements: dict[int, bytes] = {index[name]: _format_ascii_float(transformed_xyz[axis], properties[index[name]].scalar_type or 'float') for axis, name in enumerate(('x', 'y', 'z'))}
    if all((name in index for name in ('nx', 'ny', 'nz'))):
        try:
            normal = np.array([float(matches[index[name]].group()) for name in ('nx', 'ny', 'nz')], dtype=np.float64)
        except ValueError as exc:
            raise PlyFormatError(f'invalid ASCII normal value in {schema.path}') from exc
        transformed_normal = transform[:3, :3] @ normal
        for axis, name in enumerate(('nx', 'ny', 'nz')):
            replacements[index[name]] = _format_ascii_float(transformed_normal[axis], properties[index[name]].scalar_type or 'float')
    output: list[bytes] = []
    cursor = 0
    for token_index, match in enumerate(matches):
        output.append(line[cursor:match.start()])
        output.append(replacements.get(token_index, match.group()))
        cursor = match.end()
    output.append(line[cursor:])
    return b''.join(output)

def compatible_vertex_schema(first: PlySchema, second: PlySchema) -> bool:
    return first.format == second.format and first.version == second.version and (first.vertex.properties == second.vertex.properties)

def non_vertex_elements(schema: PlySchema) -> tuple[PlyElement, ...]:
    return tuple((element for element in schema.elements if element.name != 'vertex'))

def merged_vertex_header(schema: PlySchema, count: int) -> bytes:
    lines = ['ply', f'format {schema.format} {schema.version}', 'comment geometry.register property-preserving merged preview', f'element vertex {int(count)}']
    for prop in schema.vertex.properties:
        if prop.is_list:
            lines.append(f'property list {prop.list_count_type} {prop.list_item_type} {prop.name}')
        else:
            lines.append(f'property {prop.scalar_type} {prop.name}')
    lines.extend(['end_header', ''])
    return '\n'.join(lines).encode('ascii')

def _temporary_output_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent)
    os.close(descriptor)
    return Path(name)

def _stream_transformed_vertices(input_stream: BinaryIO, schema: PlySchema, transform: np.ndarray, sinks: Sequence[tuple[BinaryIO, bool]], chunk_vertices: int) -> None:
    if schema.format == 'ascii':
        for _ in range(schema.vertex.count):
            line = input_stream.readline()
            if not line:
                raise PlyFormatError(f'unexpected EOF in ASCII vertices: {schema.path}')
            transformed = _transform_ascii_line(line, schema, transform)
            for sink, require_newline in sinks:
                sink.write(transformed + b'\n' if require_newline and (not transformed.endswith((b'\n', b'\r'))) else transformed)
        return
    dtype = _numpy_vertex_dtype(schema)
    remaining = schema.vertex.count
    while remaining:
        count = min(remaining, chunk_vertices)
        byte_count = count * dtype.itemsize
        buffer = bytearray(input_stream.read(byte_count))
        if len(buffer) != byte_count:
            raise PlyFormatError(f'unexpected EOF in binary vertices: {schema.path}')
        array = np.frombuffer(buffer, dtype=dtype, count=count)
        _transform_structured_vertices(array, transform)
        for sink, _ in sinks:
            sink.write(buffer)
        remaining -= count

def transform_ply_preserving_properties(source: str | Path, destination: str | Path, transform: np.ndarray, merged_vertex_stream: BinaryIO | None=None, chunk_vertices: int=250000) -> PlySchema:
    schema = read_ply_schema(source)
    matrix = _validate_transform(transform)
    target = Path(destination)
    if schema.path.resolve() == target.resolve():
        raise ValueError('source and destination PLY paths must differ')
    target.parent.mkdir(parents=True, exist_ok=True)
    if chunk_vertices < 1:
        raise ValueError('chunk_vertices must be positive')
    temporary = _temporary_output_path(target)
    try:
        with schema.path.open('rb') as input_stream, temporary.open('wb') as output_stream:
            input_stream.seek(schema.data_offset)
            output_stream.write(schema.header)
            sinks: list[tuple[BinaryIO, bool]] = [(output_stream, False)]
            if merged_vertex_stream is not None:
                sinks.append((merged_vertex_stream, True))
            _stream_transformed_vertices(input_stream, schema, matrix, sinks, chunk_vertices)
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return schema

def merge_transformed_plys(sources: Sequence[str | Path], destination: str | Path, transforms: Sequence[np.ndarray], chunk_vertices: int=250000) -> int:
    if not sources or len(sources) != len(transforms):
        raise ValueError('sources and transforms need the same non-zero length')
    if chunk_vertices < 1:
        raise ValueError('chunk_vertices must be positive')
    schemas = [read_ply_schema(source) for source in sources]
    first = schemas[0]
    for schema in schemas:
        extras = non_vertex_elements(schema)
        if extras:
            names = ', '.join((element.name for element in extras))
            raise PlyFormatError(f'merged PLY cannot preserve non-vertex elements ({names}): {schema.path}')
    for schema in schemas[1:]:
        if not compatible_vertex_schema(first, schema):
            raise PlyFormatError(f'merged PLY needs identical format/property schema: {schema.path}')
    target = Path(destination)
    source_paths = {schema.path.resolve() for schema in schemas}
    if target.resolve() in source_paths:
        raise ValueError('merged destination must differ from every source PLY')
    target.parent.mkdir(parents=True, exist_ok=True)
    total = sum((schema.vertex.count for schema in schemas))
    temporary = _temporary_output_path(target)
    try:
        with temporary.open('wb') as output_stream:
            output_stream.write(merged_vertex_header(first, total))
            for schema, transform in zip(schemas, transforms):
                matrix = _validate_transform(transform)
                with schema.path.open('rb') as input_stream:
                    input_stream.seek(schema.data_offset)
                    _stream_transformed_vertices(input_stream, schema, matrix, [(output_stream, True)], chunk_vertices)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return total

def merge_transformed_plys_as_rgb(sources: Sequence[str | Path], destination: str | Path, transforms: Sequence[np.ndarray], chunk_vertices: int=250000) -> int:
    if not sources or len(sources) != len(transforms):
        raise ValueError('sources and transforms need the same non-zero length')
    if chunk_vertices < 1:
        raise ValueError('chunk_vertices must be positive')
    schemas = [read_ply_schema(source) for source in sources]
    for schema in schemas:
        names = {prop.name for prop in schema.vertex.properties}
        missing = {'x', 'y', 'z', 'red', 'green', 'blue'}.difference(names)
        if missing:
            joined = ', '.join(sorted(missing))
            raise PlyFormatError(f'RGB merge needs {joined}: {schema.path}')
    matrices = [_validate_transform(transform) for transform in transforms]
    target = Path(destination)
    if target.resolve() in {schema.path.resolve() for schema in schemas}:
        raise ValueError('merged destination must differ from every source PLY')
    target.parent.mkdir(parents=True, exist_ok=True)
    total = sum((schema.vertex.count for schema in schemas))
    header = f'ply\nformat binary_little_endian 1.0\ncomment derived placed RGB display cloud; inputs remain authoritative\nelement vertex {total}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode('ascii')
    output_dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')], align=False)
    temporary = _temporary_output_path(target)
    try:
        with temporary.open('wb') as output_stream:
            output_stream.write(header)
            for source, matrix in zip(sources, matrices, strict=True):
                for vertices in iter_ply_vertex_chunks(source, chunk_vertices=chunk_vertices):
                    xyz = np.column_stack([vertices['x'], vertices['y'], vertices['z']]).astype(np.float64)
                    xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
                    output = np.empty(len(vertices), dtype=output_dtype)
                    output['x'], output['y'], output['z'] = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
                    output['red'] = vertices['red']
                    output['green'] = vertices['green']
                    output['blue'] = vertices['blue']
                    output_stream.write(output.tobytes())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return total
