"""Immutable validation vocabulary shared by provider profile values."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from jharness.kernel import ModelCapabilities, freeze_json_value


def required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def immutable_json_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    mapping = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) or not key for key in mapping):
        raise ValueError(f"{label} keys must be non-empty strings")
    typed_mapping = cast(Mapping[str, Any], value)
    frozen = freeze_json_value(
        typed_mapping,
        label=label,
        error_message=f"{label} is immutable",
    )
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], frozen)


def immutable_string_mapping(value: object, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, str] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label} values must be non-empty strings")
        result[key] = item
    return MappingProxyType(result)


def string_set(value: object, label: str) -> frozenset[str]:
    if not isinstance(value, frozenset):
        raise TypeError(f"{label} must be a frozenset")
    result = cast(frozenset[object], value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{label} must contain non-empty strings")
    return cast(frozenset[str], result)


def validate_capabilities(
    value: object,
    *,
    profile: str,
    input_modalities: frozenset[str],
    output_modalities: frozenset[str],
) -> ModelCapabilities:
    """Validate one protocol profile against its actual codec surface."""

    if not isinstance(value, ModelCapabilities):
        raise TypeError("capabilities must be ModelCapabilities")
    unsupported_inputs = value.input_modalities.difference(input_modalities)
    if unsupported_inputs:
        modality = min(unsupported_inputs)
        raise ValueError(f"unsupported {profile} input modality: {modality}")
    unsupported_outputs = value.output_modalities.difference(output_modalities)
    if unsupported_outputs:
        modality = min(unsupported_outputs)
        raise ValueError(f"unsupported {profile} output modality: {modality}")
    return value
