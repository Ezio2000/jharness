"""Canonical metadata and pure numeric rules for ``AskQuestion``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from types import MappingProxyType
from typing import Literal, TypeAlias

QuestionKind: TypeAlias = Literal[
    "confirm",
    "single_choice",
    "multiple_choice",
    "text",
    "number",
    "date",
    "scale",
    "ranking",
]

SUPPORTED_QUESTION_KINDS: tuple[QuestionKind, ...] = (
    "confirm",
    "single_choice",
    "multiple_choice",
    "text",
    "number",
    "date",
    "scale",
    "ranking",
)
SCHEMA_VERSION = 1
DEFAULT_MAX_QUESTIONS = 8
DEFAULT_MAX_OPTIONS = 12
DEFAULT_MAX_PROMPT_CHARS = 2_000
DEFAULT_MAX_ANSWER_CHARS = 16_384
OPTION_VALUE_CHARS = 128

COMMON_QUESTION_FIELDS = frozenset({"id", "kind", "prompt", "description", "required"})
QUESTION_KIND_FIELDS: Mapping[QuestionKind, frozenset[str]] = MappingProxyType(
    {
        "confirm": frozenset({"default"}),
        "single_choice": frozenset({"options", "allow_custom", "default"}),
        "multiple_choice": frozenset(
            {"options", "allow_custom", "min_selections", "max_selections", "default"}
        ),
        "text": frozenset({"multiline", "placeholder", "min_length", "max_length", "default"}),
        "number": frozenset({"minimum", "maximum", "step", "integer_only", "default"}),
        "date": frozenset({"minimum", "maximum", "default"}),
        "scale": frozenset(
            {
                "minimum",
                "maximum",
                "step",
                "minimum_label",
                "maximum_label",
                "default",
            }
        ),
        "ranking": frozenset({"options", "min_ranked", "max_ranked", "default"}),
    }
)
CANONICAL_REQUIRED_QUESTION_FIELDS: Mapping[QuestionKind, frozenset[str]] = MappingProxyType(
    {
        "confirm": frozenset(),
        "single_choice": frozenset({"options", "allow_custom"}),
        "multiple_choice": frozenset(
            {"options", "allow_custom", "min_selections", "max_selections"}
        ),
        "text": frozenset({"multiline", "min_length", "max_length"}),
        "number": frozenset({"integer_only"}),
        "date": frozenset(),
        "scale": frozenset({"minimum", "maximum", "step"}),
        "ranking": frozenset({"options", "min_ranked", "max_ranked"}),
    }
)


def is_step_aligned(
    value: int | float,
    step: int | float,
    origin: int | float,
) -> bool:
    """Return whether a number lies on an exact decimal step."""

    try:
        exact_distance = (number_fraction(value) - number_fraction(origin)) / number_fraction(step)
    except (OverflowError, ValueError, ZeroDivisionError):
        return False
    return exact_distance.denominator == 1


def number_fraction(value: int | float) -> Fraction:
    """Preserve decimal float spelling while constructing an exact fraction."""

    return Fraction(value) if isinstance(value, int) else Fraction(str(value))


def integer_answer_exists(
    minimum: int | float | None,
    maximum: int | float | None,
    step: int | float | None,
) -> bool:
    """Return whether integer-only numeric constraints have at least one answer."""

    lower = None if minimum is None else number_fraction(minimum)
    upper = None if maximum is None else number_fraction(maximum)
    if step is None:
        lower_integer = None if lower is None else _fraction_ceiling(lower)
        upper_integer = None if upper is None else _fraction_floor(upper)
        return lower_integer is None or upper_integer is None or lower_integer <= upper_integer

    origin = lower if lower is not None else Fraction(0)
    stride = number_fraction(step)
    common_denominator = math.lcm(origin.denominator, stride.denominator)
    origin_units = origin.numerator * (common_denominator // origin.denominator)
    stride_units = stride.numerator * (common_denominator // stride.denominator)
    divisor = math.gcd(stride_units, common_denominator)
    if origin_units % divisor != 0:
        return False

    modulus = common_denominator // divisor
    if modulus == 1:
        first_solution = 0
    else:
        inverse = pow(stride_units // divisor, -1, modulus)
        first_solution = (-origin_units // divisor * inverse) % modulus

    lower_step = None if lower is None else _fraction_ceiling((lower - origin) / stride)
    upper_step = None if upper is None else _fraction_floor((upper - origin) / stride)
    if lower_step is None or upper_step is None:
        return True
    if lower_step > upper_step:
        return False
    multiplier = -(-(lower_step - first_solution) // modulus)
    return first_solution + multiplier * modulus <= upper_step


def _fraction_floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _fraction_ceiling(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)
