
from __future__ import annotations
RAW_PROCESSING_SCHEMA_V1 = 'cca.raw-processing.v1'
RAW_PROCESSING_SCHEMA_V2 = 'cca.raw-processing.v2'
RAW_PROCESSING_SCHEMA_V3 = 'cca.raw-processing.v3'
CURRENT_RAW_PROCESSING_SCHEMA = RAW_PROCESSING_SCHEMA_V3
SUPPORTED_RAW_PROCESSING_SCHEMAS = frozenset({RAW_PROCESSING_SCHEMA_V1, RAW_PROCESSING_SCHEMA_V2, RAW_PROCESSING_SCHEMA_V3})

def is_supported_raw_processing_schema(value: object) -> bool:
    return isinstance(value, str) and value in SUPPORTED_RAW_PROCESSING_SCHEMAS
__all__ = ['CURRENT_RAW_PROCESSING_SCHEMA', 'RAW_PROCESSING_SCHEMA_V1', 'RAW_PROCESSING_SCHEMA_V2', 'RAW_PROCESSING_SCHEMA_V3', 'SUPPORTED_RAW_PROCESSING_SCHEMAS', 'is_supported_raw_processing_schema']
