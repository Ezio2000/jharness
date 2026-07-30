"""Verify the public API of an isolated five-package JHarness installation."""

from __future__ import annotations

from importlib.util import find_spec


def _load_required_types() -> tuple[object, ...]:
    from jharness.kernel import Runtime
    from jharness.models.decorators import FallbackModel, RetryingModel
    from jharness.models.openai import OpenAIChatCompletionsModel
    from jharness.repository import (
        MemoryRunRepository,
        MySQLRunRepository,
        RedisRunRepository,
        SQLiteRunRepository,
    )
    from jharness.toolkit import ToolRegistry
    from jharness.tools import ReadTool

    return (
        Runtime,
        FallbackModel,
        RetryingModel,
        OpenAIChatCompletionsModel,
        MemoryRunRepository,
        MySQLRunRepository,
        RedisRunRepository,
        SQLiteRunRepository,
        ToolRegistry,
        ReadTool,
    )


def main() -> None:
    """Reject leaked optional drivers and require every public smoke type."""

    leaked = [name for name in ("pymysql", "redis") if find_spec(name) is not None]
    if leaked:
        raise RuntimeError(f"base installation contains optional drivers: {leaked}")
    public_types = _load_required_types()
    if not all(isinstance(value, type) for value in public_types):
        raise TypeError("public API smoke targets must all be types")
    print(f"installed API ok: types={len(public_types)}")


if __name__ == "__main__":
    main()
