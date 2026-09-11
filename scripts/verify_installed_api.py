"""Verify the public API of an isolated five-package JHarness installation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any, cast


def _load_required_types() -> tuple[object, ...]:
    from jharness.kernel import Runtime
    from jharness.models.anthropic import AnthropicMessagesModel, AnthropicMessagesProfile
    from jharness.models.decorators import FallbackModel, RetryingModel
    from jharness.models.openai import (
        OpenAIChatModel,
        OpenAIChatProfile,
        OpenAIResponsesArtifactStore,
        OpenAIResponsesModel,
        OpenAIResponsesProfile,
    )
    from jharness.repository import (
        MemoryRunRepository,
        MySQLRunRepository,
        RedisRunRepository,
        SQLiteRunRepository,
    )
    from jharness.toolkit import ToolRegistry
    from jharness.tools import LsTool, ReadTool

    return (
        Runtime,
        FallbackModel,
        RetryingModel,
        AnthropicMessagesModel,
        AnthropicMessagesProfile,
        OpenAIChatModel,
        OpenAIChatProfile,
        OpenAIResponsesModel,
        OpenAIResponsesProfile,
        OpenAIResponsesArtifactStore,
        MemoryRunRepository,
        MySQLRunRepository,
        RedisRunRepository,
        SQLiteRunRepository,
        ToolRegistry,
        LsTool,
        ReadTool,
    )


def _require_exports(
    module: object,
    expected: set[str],
) -> None:
    module_name = getattr(module, "__name__", repr(module))
    exports = getattr(module, "__all__", None)
    if not isinstance(exports, list):
        raise TypeError(f"{module_name} exports differ: {exports!r}")
    raw_exports = cast(list[object], exports)
    if not all(isinstance(name, str) for name in raw_exports):
        raise TypeError(f"{module_name} exports contain non-string names: {exports!r}")
    actual = {cast(str, name) for name in raw_exports}
    if actual != expected:
        raise TypeError(f"{module_name} exports differ: {exports!r}")
    if missing := sorted(name for name in expected if not hasattr(module, name)):
        raise TypeError(f"{module_name} is missing exports: {missing}")


def _verify_model_namespaces() -> None:
    import jharness.models.anthropic as anthropic
    import jharness.models.anthropic.messages as anthropic_messages
    import jharness.models.openai as openai
    import jharness.models.openai.chat as openai_chat
    import jharness.models.openai.responses as openai_responses

    _require_exports(
        openai,
        {
            "OPENAI_RESPONSES_IMAGE_GENERATION",
            "OPENAI_RESPONSES_WEB_SEARCH",
            "OpenAIChatCodec",
            "OpenAIChatError",
            "OpenAIChatModel",
            "OpenAIChatProfile",
            "OpenAIResponsesCodec",
            "OpenAIResponsesError",
            "OpenAIResponsesModel",
            "OpenAIResponsesProfile",
            "OpenAIResponsesArtifactStore",
            "openai_responses_image_generation",
            "openai_responses_profile",
            "openai_responses_web_search",
        },
    )
    _require_exports(
        anthropic,
        {
            "ANTHROPIC_MESSAGES_WEB_SEARCH",
            "AnthropicMessagesCodec",
            "AnthropicMessagesError",
            "AnthropicMessagesModel",
            "AnthropicMessagesProfile",
            "anthropic_messages_profile",
            "anthropic_messages_web_search",
        },
    )
    for implementation in (openai_chat, openai_responses, anthropic_messages):
        _require_exports(implementation, set())


def _load_profiles() -> tuple[object, ...]:
    from jharness.models.anthropic import (
        AnthropicMessagesProfile,
        anthropic_messages_profile,
    )
    from jharness.models.openai import (
        OpenAIChatProfile,
        OpenAIResponsesProfile,
        openai_responses_profile,
    )

    profiles = (
        OpenAIChatProfile(),
        OpenAIResponsesProfile(),
        openai_responses_profile(),
        AnthropicMessagesProfile(),
        anthropic_messages_profile(),
    )
    expected_types = (
        OpenAIChatProfile,
        OpenAIResponsesProfile,
        OpenAIResponsesProfile,
        AnthropicMessagesProfile,
        AnthropicMessagesProfile,
    )
    if not all(
        isinstance(cast(object, profile), expected)
        for profile, expected in zip(profiles, expected_types, strict=True)
    ):
        raise TypeError("profile factory returned the wrong adapter type")
    names = tuple(profile.name for profile in profiles)
    expected_names = (
        "openai-chat",
        "openai-responses",
        "openai-responses",
        "anthropic-messages",
        "anthropic-messages",
    )
    if names != expected_names:
        raise TypeError(f"profile names differ: {names!r}")
    return profiles


def _verify_provider_tool_presets() -> tuple[object, ...]:
    from jharness.models.anthropic import (
        ANTHROPIC_MESSAGES_WEB_SEARCH,
        AnthropicMessagesProfile,
        anthropic_messages_profile,
        anthropic_messages_web_search,
    )
    from jharness.models.openai import (
        OPENAI_RESPONSES_IMAGE_GENERATION,
        OPENAI_RESPONSES_WEB_SEARCH,
        OpenAIResponsesProfile,
        openai_responses_image_generation,
        openai_responses_profile,
        openai_responses_web_search,
    )

    openai_profile = openai_responses_profile()
    anthropic_profile = anthropic_messages_profile()
    openai_tools = frozenset({OPENAI_RESPONSES_WEB_SEARCH, OPENAI_RESPONSES_IMAGE_GENERATION})
    if OpenAIResponsesProfile().capabilities.provider_tools:
        raise TypeError("generic OpenAI Responses profile unexpectedly enables provider tools")
    if AnthropicMessagesProfile().capabilities.provider_tools:
        raise TypeError("generic Anthropic Messages profile unexpectedly enables provider tools")
    if openai_profile.capabilities.provider_tools != openai_tools:
        raise TypeError("OpenAI Responses hosted-tool preset identities differ")
    if anthropic_profile.capabilities.provider_tools != frozenset({ANTHROPIC_MESSAGES_WEB_SEARCH}):
        raise TypeError("Anthropic Messages hosted-tool preset identities differ")
    specs = (
        openai_responses_web_search(),
        openai_responses_image_generation(),
        anthropic_messages_web_search(),
    )
    expected = (
        OPENAI_RESPONSES_WEB_SEARCH,
        OPENAI_RESPONSES_IMAGE_GENERATION,
        ANTHROPIC_MESSAGES_WEB_SEARCH,
    )
    if tuple(spec.tool for spec in specs) != expected:
        raise TypeError("hosted-tool preset specs differ")
    return specs


async def _verify_function_adapters() -> None:
    from jharness.kernel import (
        FreeformToolCall,
        RunContext,
        SettledResult,
        StructuredToolCall,
        ToolContext,
        ToolSuccess,
    )
    from jharness.toolkit import ToolRegistry, freeform_tool, function_tool

    async def ignore_progress(value: Mapping[str, Any]) -> None:
        del value

    @function_tool(
        name="greet",
        description="Greet a user",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        output_schema={"type": "string"},
        context_parameter="ctx",
    )
    async def greet(name: str, *, ctx: ToolContext) -> str:
        return f"{ctx.run.run_id}: hello {name}"

    @freeform_tool(name="count_lines", description="Count text lines")
    async def count_lines(text: str) -> dict[str, int]:
        return {"lines": len(text.splitlines())}

    context = ToolContext(RunContext("smoke", 1.0), ignore_progress, lambda: False)
    catalog = await ToolRegistry((greet, count_lines)).open_catalog()
    result = await catalog.bind(StructuredToolCall("greet-1", "greet", {"name": "world"})).invoke(
        context
    )
    if not isinstance(result, SettledResult) or not isinstance(result.outcome, ToolSuccess):
        raise TypeError("function adapter did not return a success result")
    if result.outcome.structured_content != "smoke: hello world":
        raise TypeError("function adapter argument binding or result adaptation differs")
    lines = await catalog.bind(FreeformToolCall("lines-1", "count_lines", "first\nsecond")).invoke(
        context
    )
    if lines.outcome.structured_content != {"lines": 2}:
        raise TypeError("freeform adapter input or result adaptation differs")


def main() -> None:
    """Reject leaked optional drivers and require every public smoke type."""

    leaked = [name for name in ("pymysql", "redis") if find_spec(name) is not None]
    if leaked:
        raise RuntimeError(f"base installation contains optional drivers: {leaked}")
    _verify_model_namespaces()
    public_types = _load_required_types()
    if not all(isinstance(value, type) for value in public_types):
        raise TypeError("public API smoke targets must all be types")
    profiles = _load_profiles()
    presets = _verify_provider_tool_presets()
    asyncio.run(_verify_function_adapters())
    print(
        "installed API ok: "
        f"types={len(public_types)} profiles={len(profiles)} presets={len(presets)}"
    )


if __name__ == "__main__":
    main()
