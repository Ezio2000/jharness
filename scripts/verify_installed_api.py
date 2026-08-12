"""Verify the public API of an isolated five-package JHarness installation."""

from __future__ import annotations

from collections.abc import Set
from importlib.util import find_spec
from typing import cast, get_args


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
    legacy: Set[str] = frozenset(),
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
    if leaked := sorted(name for name in legacy if hasattr(module, name)):
        raise TypeError(f"{module_name} retains legacy exports: {leaked}")


def _verify_model_namespaces() -> None:
    import jharness.models.anthropic as anthropic
    import jharness.models.anthropic.messages as anthropic_messages
    import jharness.models.deepseek as deepseek
    import jharness.models.openai as openai
    import jharness.models.openai.chat as openai_chat
    import jharness.models.openai.responses as openai_responses

    _require_exports(
        openai,
        {
            "OpenAIChatCodec",
            "OpenAIChatError",
            "OpenAIChatModel",
            "OpenAIChatProfile",
            "OpenAIResponsesCodec",
            "OpenAIResponsesError",
            "OpenAIResponsesModel",
            "OpenAIResponsesProfile",
            "OpenAIResponsesProviderToolStreamUpdate",
            "OpenAIResponsesArtifactStore",
            "OpenAIResponsesImageGenerationTool",
            "OpenAIResponsesProviderToolCodec",
            "OpenAIResponsesProviderToolRegistry",
            "OpenAIResponsesWebSearchTool",
        },
        {
            "ProviderStreamUpdate",
            "ResponsesArtifactStore",
            "ResponsesImageGenerationTool",
            "ResponsesProviderToolCodec",
            "ResponsesProviderToolRegistry",
            "ResponsesWebSearchTool",
            "OpenAIChatCompletionsCodec",
            "OpenAIChatCompletionsError",
            "OpenAIChatCompletionsModel",
            "OpenAIChatCompletionsProfile",
        },
    )
    _require_exports(
        anthropic,
        {
            "AnthropicMessagesCodec",
            "AnthropicMessagesError",
            "AnthropicMessagesModel",
            "AnthropicMessagesProfile",
            "AnthropicMessagesServerToolCodec",
            "AnthropicMessagesServerToolRegistry",
            "anthropic_messages_web_search_codec",
        },
        {
            "AnthropicCodec",
            "AnthropicError",
            "AnthropicModel",
            "AnthropicProfile",
            "AnthropicServerToolCodec",
            "AnthropicServerToolRegistry",
            "anthropic_web_search_codec",
        },
    )
    _require_exports(
        deepseek,
        {
            "DEEPSEEK_MESSAGES_WEB_SEARCH",
            "DEEPSEEK_RESPONSES_WEB_SEARCH",
            "DeepSeekResponsesEffort",
            "DeepSeekThinkingEffort",
            "deepseek_chat_profile",
            "deepseek_messages_profile",
            "deepseek_messages_web_search",
            "deepseek_responses_profile",
            "deepseek_responses_web_search",
        },
        {
            "DEEPSEEK_ANTHROPIC_WEB_SEARCH",
            "deepseek_anthropic_profile",
            "deepseek_anthropic_web_search",
            "deepseek_openai_chat_profile",
            "deepseek_openai_responses_profile",
        },
    )
    if frozenset(get_args(deepseek.DeepSeekThinkingEffort)) != frozenset({"high", "max"}):
        raise TypeError("DeepSeek thinking effort values differ")
    if frozenset(get_args(deepseek.DeepSeekResponsesEffort)) != frozenset(
        {"none", "low", "high", "xhigh", "max"}
    ):
        raise TypeError("DeepSeek Responses effort values differ")
    for implementation in (openai_chat, openai_responses, anthropic_messages):
        _require_exports(implementation, set())
    for legacy_module in (
        "jharness.models.anthropic.errors",
        "jharness.models.anthropic.messages_api",
        "jharness.models.anthropic.profiles",
        "jharness.models.openai.chat_completions",
        "jharness.models.openai.errors",
        "jharness.models.openai.profiles",
        "jharness.models.openai.responses_api",
    ):
        if find_spec(legacy_module) is not None:
            raise TypeError(f"legacy model module remains importable: {legacy_module}")


def _load_profiles() -> tuple[object, ...]:
    from jharness.models.anthropic import AnthropicMessagesProfile
    from jharness.models.deepseek import (
        deepseek_chat_profile,
        deepseek_messages_profile,
        deepseek_responses_profile,
    )
    from jharness.models.openai import OpenAIChatProfile, OpenAIResponsesProfile

    profiles = (
        OpenAIChatProfile(),
        OpenAIResponsesProfile(),
        AnthropicMessagesProfile(),
        deepseek_chat_profile(),
        deepseek_chat_profile(thinking=True),
        deepseek_messages_profile(),
        deepseek_messages_profile(thinking=True),
        deepseek_responses_profile(effort="none"),
    )
    expected_types = (
        OpenAIChatProfile,
        OpenAIResponsesProfile,
        AnthropicMessagesProfile,
        OpenAIChatProfile,
        OpenAIChatProfile,
        AnthropicMessagesProfile,
        AnthropicMessagesProfile,
        OpenAIResponsesProfile,
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
        "anthropic-messages",
        "deepseek-chat",
        "deepseek-chat-thinking",
        "deepseek-messages",
        "deepseek-messages-thinking",
        "deepseek-responses",
    )
    if names != expected_names:
        raise TypeError(f"profile names differ: {names!r}")
    return profiles


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
    print(f"installed API ok: types={len(public_types)} profiles={len(profiles)}")


if __name__ == "__main__":
    main()
