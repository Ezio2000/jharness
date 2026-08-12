"""DeepSeek provider profiles."""

from __future__ import annotations

from typing import Literal, cast

from jharness.kernel import ModelCapabilities, RuntimeToolKind
from jharness.models.anthropic import AnthropicProfile
from jharness.models.anthropic.messages_api.server_tools import (
    AnthropicServerToolRegistry,
    anthropic_web_search_codec,
)
from jharness.models.deepseek.tools import (
    DEEPSEEK_ANTHROPIC_WEB_SEARCH,
    DEEPSEEK_RESPONSES_WEB_SEARCH,
)
from jharness.models.openai.profiles import (
    OpenAIChatCompletionsProfile,
    OpenAIResponsesProfile,
)
from jharness.models.openai.responses_api.provider_tools import ResponsesProviderToolRegistry

from . import _responses

DeepSeekThinkingEffort = Literal["high", "max"]
DeepSeekResponsesEffort = Literal[
    "none",
    "low",
    "high",
    "xhigh",
    "max",
]

_THINKING_EFFORTS = frozenset({"high", "max"})
_RESPONSES_EFFORTS = frozenset({"none", "low", "high", "xhigh", "max"})
_RUNTIME_TOOL_CHOICES = frozenset({"auto", "none", "required", "runtime"})
_ALL_TOOL_CHOICES = frozenset({"auto", "none", "required", "runtime", "provider"})


def deepseek_openai_chat_profile(
    *,
    thinking: bool,
    effort: DeepSeekThinkingEffort | None = None,
) -> OpenAIChatCompletionsProfile:
    """Return a DeepSeek profile for the OpenAI Chat Completions wire protocol."""

    thinking = _validate_options(thinking, effort)
    extra_request_body = _thinking_request_body(thinking=thinking, effort=effort)
    return OpenAIChatCompletionsProfile(
        name=_profile_name("deepseek-openai-chat", thinking),
        capabilities=ModelCapabilities(
            streaming=True,
            runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED}),
            tool_choice_types=(frozenset({"auto"}) if thinking else _RUNTIME_TOOL_CHOICES),
            parallel_runtime_tool_calls=True,
            parallel_runtime_tool_call_control=False,
            input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"}),
            structured_output=False,
            json_mode=True,
            seed=False,
            usage_reporting=True,
        ),
        automatic_tool_choice_mode="implicit" if thinking else "explicit",
        assistant_tool_call_content_mode="required" if thinking else "nullable",
        reasoning_content_mode="required_with_tools" if thinking else "live_only",
        stream_usage_mode="include",
        extra_request_body=extra_request_body,
    )


def deepseek_anthropic_profile(
    *,
    thinking: bool,
    effort: DeepSeekThinkingEffort | None = None,
) -> AnthropicProfile:
    """Return a DeepSeek profile for the Anthropic Messages wire protocol."""

    thinking = _validate_options(thinking, effort)
    extra_request_body = _thinking_request_body(thinking=thinking, effort=None)
    web_search = DEEPSEEK_ANTHROPIC_WEB_SEARCH
    return AnthropicProfile(
        name=_profile_name("deepseek-anthropic", thinking),
        capabilities=ModelCapabilities(
            streaming=True,
            runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED}),
            tool_choice_types=_ALL_TOOL_CHOICES,
            parallel_runtime_tool_calls=True,
            parallel_runtime_tool_call_control=False,
            input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"}),
            provider_tools=frozenset({web_search}),
            structured_output=False,
            json_mode=False,
            seed=False,
            usage_reporting=True,
        ),
        server_tools=AnthropicServerToolRegistry((anthropic_web_search_codec(web_search),)),
        redacted_thinking_mode="reject",
        stream_usage_mode="include",
        extra_request_body=extra_request_body,
        extra_output_config={} if effort is None else {"effort": effort},
    )


def deepseek_openai_responses_profile(
    *,
    effort: DeepSeekResponsesEffort | None = None,
) -> OpenAIResponsesProfile:
    """Return the native DeepSeek-V4-Flash Responses API profile."""

    effort_value = cast(object, effort)
    if effort is not None and (
        not isinstance(effort_value, str) or effort not in _RESPONSES_EFFORTS
    ):
        expected = ", ".join(sorted(_RESPONSES_EFFORTS))
        raise ValueError(f"effort must be one of: {expected}")
    extra_request_body: dict[str, object] = {}
    if effort is not None:
        extra_request_body["reasoning"] = {"effort": effort}
    web_search = DEEPSEEK_RESPONSES_WEB_SEARCH
    return OpenAIResponsesProfile(
        name="deepseek-openai-responses",
        capabilities=ModelCapabilities(
            streaming=True,
            runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED, RuntimeToolKind.FREEFORM}),
            tool_choice_types=(
                frozenset({"auto", "none"}) if effort != "none" else _ALL_TOOL_CHOICES
            ),
            parallel_runtime_tool_calls=True,
            parallel_runtime_tool_call_control=False,
            input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"}),
            provider_tools=frozenset({web_search}),
            structured_output=True,
            json_mode=True,
            seed=False,
            usage_reporting=True,
        ),
        reasoning_history_mode="content",
        store=None,
        include=frozenset(),
        provider_tool_registry=ResponsesProviderToolRegistry(
            (_responses.deepseek_responses_web_search_codec(web_search),)
        ),
        freeform_runtime_tool_names=frozenset({"apply_patch"}),
        exact_runtime_tool_choice_kinds=frozenset({RuntimeToolKind.STRUCTURED}),
        emit_freeform_runtime_tool_description=False,
        allowed_models=frozenset({"deepseek-v4-flash"}),
        extra_request_body=extra_request_body,
        finish_reason_map={
            "max_output_tokens": "length",
            "content_filter": "content_filter",
        },
    )


def _thinking_request_body(
    *,
    thinking: bool,
    effort: DeepSeekThinkingEffort | None,
) -> dict[str, object]:
    body: dict[str, object] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    if effort is not None:
        body["reasoning_effort"] = effort
    return body


def _validate_options(thinking: object, effort: object) -> bool:
    if not isinstance(thinking, bool):
        raise ValueError("thinking must be a bool")
    if effort is not None and not thinking:
        raise ValueError("effort is only valid when thinking=True")
    if effort is not None and effort not in _THINKING_EFFORTS:
        expected = ", ".join(sorted(_THINKING_EFFORTS))
        raise ValueError(f"effort must be one of: {expected}")
    return thinking


def _profile_name(prefix: str, thinking: bool) -> str:
    mode = "thinking" if thinking else "nonthinking"
    return f"{prefix}-{mode}"
