"""Verify the public API of an isolated five-package JHarness installation."""

from __future__ import annotations

from importlib.util import find_spec
from typing import cast


def _load_required_types() -> tuple[object, ...]:
    from jharness.kernel import Runtime
    from jharness.models.anthropic import AnthropicModel, AnthropicProfile
    from jharness.models.decorators import FallbackModel, RetryingModel
    from jharness.models.openai import (
        OpenAIChatCompletionsModel,
        OpenAIChatCompletionsProfile,
    )
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
        AnthropicModel,
        AnthropicProfile,
        OpenAIChatCompletionsModel,
        OpenAIChatCompletionsProfile,
        MemoryRunRepository,
        MySQLRunRepository,
        RedisRunRepository,
        SQLiteRunRepository,
        ToolRegistry,
        ReadTool,
    )


def _load_deepseek_profiles() -> tuple[object, object]:
    from jharness.models.anthropic import AnthropicProfile
    from jharness.models.deepseek import (
        deepseek_anthropic_profile,
        deepseek_openai_chat_profile,
    )
    from jharness.models.openai import OpenAIChatCompletionsProfile

    openai_profile = deepseek_openai_chat_profile(thinking=False)
    anthropic_profile = deepseek_anthropic_profile(thinking=False)
    if not isinstance(cast(object, openai_profile), OpenAIChatCompletionsProfile):
        raise TypeError("DeepSeek OpenAI profile factory returned the wrong type")
    if not isinstance(cast(object, anthropic_profile), AnthropicProfile):
        raise TypeError("DeepSeek Anthropic profile factory returned the wrong type")
    return openai_profile, anthropic_profile


def main() -> None:
    """Reject leaked optional drivers and require every public smoke type."""

    leaked = [name for name in ("pymysql", "redis") if find_spec(name) is not None]
    if leaked:
        raise RuntimeError(f"base installation contains optional drivers: {leaked}")
    public_types = _load_required_types()
    if not all(isinstance(value, type) for value in public_types):
        raise TypeError("public API smoke targets must all be types")
    profiles = _load_deepseek_profiles()
    print(f"installed API ok: types={len(public_types)} profiles={len(profiles)}")


if __name__ == "__main__":
    main()
