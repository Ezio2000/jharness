"""Compact offline executable contract for DeepSeek's Responses profile."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal, TypedDict, TypeVar

from jharness.kernel import (
    ContentPart,
    FreeformToolCall,
    FreeformToolSpec,
    Message,
    ModelError,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    ResponseFormat,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
    thaw_json_value,
)
from jharness.models.deepseek import (
    DEEPSEEK_RESPONSES_WEB_SEARCH,
    deepseek_openai_responses_profile,
    deepseek_responses_web_search,
)
from jharness.models.openai import OpenAIResponsesCodec, OpenAIResponsesError

JsonObject = dict[str, Any]
Category = Literal["request", "response", "rejection"]
ErrorT = TypeVar("ErrorT", bound=Exception)

MODEL = "deepseek-v4-flash"
_LOOKUP = StructuredToolSpec(
    "lookup",
    "Look up one fact.",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)
_APPLY_PATCH = FreeformToolSpec("apply_patch", "Apply one unified patch.")
_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class OfflineContractReport(TypedDict):
    """Case names derived only from checks completed by this run."""

    status: Literal["passed"]
    protocol: str
    model: str
    request_checks: tuple[str, ...]
    response_checks: tuple[str, ...]
    rejection_checks: tuple[str, ...]
    total_checks: int


@dataclass(slots=True)
class _Recorder:
    completed: dict[Category, list[str]] = field(
        default_factory=lambda: {
            "request": [],
            "response": [],
            "rejection": [],
        }
    )
    _names: set[str] = field(default_factory=set[str])

    def equal(self, category: Category, name: str, actual: object, expected: object) -> None:
        if actual != expected:
            raise AssertionError(
                f"offline {category} case {name!r} failed: expected {expected!r}, got {actual!r}"
            )
        self._record(category, name)

    def reject(
        self,
        name: str,
        error_type: type[ErrorT],
        message: str,
        operation: Callable[[], object],
    ) -> ErrorT:
        error = _expect_error(error_type, message, operation)
        self._record("rejection", name)
        return error

    def report(self) -> OfflineContractReport:
        return {
            "status": "passed",
            "protocol": "deepseek-native-responses",
            "model": MODEL,
            "request_checks": tuple(self.completed["request"]),
            "response_checks": tuple(self.completed["response"]),
            "rejection_checks": tuple(self.completed["rejection"]),
            "total_checks": len(self._names),
        }

    def _record(self, category: Category, name: str) -> None:
        if not name or name in self._names:
            raise ValueError(f"invalid or duplicate offline contract case name: {name!r}")
        self._names.add(name)
        self.completed[category].append(name)


def run_offline_contract() -> OfflineContractReport:
    """Execute the locally observable DeepSeek Responses profile contract."""

    recorder = _Recorder()
    _check_requests(recorder)
    _check_responses(recorder)
    _check_rejections(recorder)
    return recorder.report()


def _check_requests(recorder: _Recorder) -> None:
    profile = deepseek_openai_responses_profile(effort="none")
    codec = OpenAIResponsesCodec(model=MODEL, profile=profile)
    web_search = deepseek_responses_web_search()
    payload = codec.encode_request(
        ModelRequest(
            messages=(Message.system("Be precise."), Message.user("Answer with evidence.")),
            runtime_tools=(_LOOKUP, _APPLY_PATCH),
            provider_tools=(web_search,),
            options=ModelOptions(
                model=MODEL,
                temperature=0.2,
                top_p=0.8,
                max_output_tokens=512,
            ),
            response_format=ResponseFormat("json_schema", schema=_SCHEMA, strict=True),
        )
    )
    recorder.equal(
        "request",
        "request:complete_stateless_payload",
        {
            "profile_state": (profile.store, profile.include),
            "forbidden": tuple(
                key
                for key in ("store", "include", "previous_response_id", "parallel_tool_calls")
                if key in payload
            ),
            "input": payload["input"],
            "options": tuple(
                payload[key]
                for key in ("model", "temperature", "top_p", "max_output_tokens", "reasoning")
            ),
            "tools": payload["tools"],
            "choice": payload["tool_choice"],
            "format": payload["text"],
        },
        {
            "profile_state": (None, frozenset[str]()),
            "forbidden": (),
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Be precise."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Answer with evidence."}],
                },
            ],
            "options": (MODEL, 0.2, 0.8, 512, {"effort": "none"}),
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up one fact.",
                    "parameters": thaw_json_value(_LOOKUP.input_schema),
                },
                {"type": "custom", "name": "apply_patch"},
                {"type": "web_search"},
            ],
            "choice": "auto",
            "format": {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": _SCHEMA,
                    "strict": True,
                }
            },
        },
    )

    format_cases = (
        ("request:format_text", ResponseFormat("text"), {"type": "text"}),
        (
            "request:format_json_object",
            ResponseFormat("json_object"),
            {"type": "json_object"},
        ),
    )
    for name, response_format, expected in format_cases:
        encoded = codec.encode_request(
            ModelRequest(messages=(Message.user("format"),), response_format=response_format)
        )
        recorder.equal("request", name, encoded["text"], {"format": expected})

    choice_cases: tuple[tuple[str, ToolChoice, str | JsonObject], ...] = (
        ("request:choice_none", ToolChoice(type="none"), "none"),
        ("request:choice_required", ToolChoice(type="required"), "required"),
        (
            "request:choice_runtime",
            ToolChoice(type="runtime", name="lookup"),
            {"type": "function", "name": "lookup"},
        ),
        (
            "request:choice_provider",
            ToolChoice(type="provider", provider_tool=DEEPSEEK_RESPONSES_WEB_SEARCH),
            {"type": "web_search"},
        ),
    )
    for name, choice, expected in choice_cases:
        encoded = codec.encode_request(
            ModelRequest(
                messages=(Message.user("choose"),),
                runtime_tools=(_LOOKUP, _APPLY_PATCH),
                provider_tools=(web_search,),
                tool_choice=choice,
            )
        )
        recorder.equal("request", name, encoded["tool_choice"], expected)

    for variant in ("web_search", "web_search_2025_08_26"):
        encoded = codec.encode_request(
            ModelRequest(
                messages=(Message.user("search"),),
                provider_tools=(deepseek_responses_web_search({"variant": variant}),),
                tool_choice=ToolChoice(
                    type="provider", provider_tool=DEEPSEEK_RESPONSES_WEB_SEARCH
                ),
            )
        )
        recorder.equal(
            "request",
            f"request:web_variant_{variant}",
            (encoded["tools"], encoded["tool_choice"]),
            ([{"type": variant}], {"type": variant}),
        )


def _check_responses(recorder: _Recorder) -> None:
    codec = OpenAIResponsesCodec(
        model=MODEL,
        profile=deepseek_openai_responses_profile(effort="none"),
    )
    response = codec.decode_response(_wire_response(_rich_output()))
    recorder.equal(
        "response",
        "response:rich_terminal",
        {
            "fields": tuple(item.name for item in fields(ModelResponse)),
            "terminal": (
                response.finish_reason,
                response.model_id,
                response.response_id,
                response.provider_turn_pending,
            ),
            "usage_fields": tuple(item.name for item in fields(ModelUsage)),
            "usage": response.usage,
            "outputs": tuple(_output_snapshot(item) for item in response.output),
        },
        {
            "fields": (
                "output",
                "finish_reason",
                "usage",
                "model_id",
                "response_id",
                "provider_turn_pending",
                "metadata",
            ),
            "terminal": ("tool_calls", MODEL, "response-contract", False),
            "usage_fields": (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            ),
            "usage": ModelUsage(21, 13, 34, 5, 8, 3),
            "outputs": (
                ("text", "Evidence", "https://example.test/source"),
                ("reasoning", "Check facts."),
                ("refusal", "Cannot disclose that."),
                ("structured", "function-call", "lookup", {"query": "protocol"}),
                ("freeform", "custom-call", "apply_patch", "patch body"),
                (
                    "provider",
                    "web-search-call",
                    DEEPSEEK_RESPONSES_WEB_SEARCH,
                    ProviderToolStatus.COMPLETED,
                    {"type": "search", "query": "current protocol"},
                ),
            ),
        },
    )

    incomplete = codec.decode_response(
        _wire_response(
            [
                {
                    "id": "message-incomplete",
                    "type": "message",
                    "status": "incomplete",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Partial", "annotations": []}],
                }
            ],
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )
    )
    recorder.equal(
        "response",
        "response:incomplete_terminal",
        (
            incomplete.finish_reason,
            incomplete.metadata["incomplete_details"],
            tuple(part.text for part in incomplete.visible_parts()),
        ),
        ("length", {"reason": "max_output_tokens"}, ("Partial",)),
    )

    failed = _expect_error(
        ModelError,
        "synthetic provider failure",
        lambda: codec.decode_response(
            _wire_response(
                [],
                status="failed",
                error={"code": "synthetic_failure", "message": "synthetic provider failure"},
            )
        ),
    )
    recorder.equal(
        "response",
        "response:failed_terminal",
        (
            failed.info.code,
            failed.info.provider,
            failed.info.retryable,
            failed.info.metadata,
        ),
        (
            "synthetic_failure",
            "deepseek-openai-responses",
            False,
            {"response_id": "response-contract", "status": "failed"},
        ),
    )


def _check_rejections(recorder: _Recorder) -> None:
    profile = deepseek_openai_responses_profile(effort="none")
    codec = OpenAIResponsesCodec(model=MODEL, profile=profile)
    thinking_codec = OpenAIResponsesCodec(
        model=MODEL,
        profile=deepseek_openai_responses_profile(effort="high"),
    )
    cases: tuple[tuple[str, type[Exception], str, Callable[[], object]], ...] = (
        (
            "rejection:codec_wrong_model",
            ValueError,
            "only supports models",
            lambda: OpenAIResponsesCodec(model="deepseek-v4-pro", profile=profile),
        ),
        (
            "rejection:request_wrong_model",
            ValueError,
            "only supports models",
            lambda: codec.encode_request(
                ModelRequest(
                    messages=(Message.user("wrong model"),),
                    options=ModelOptions(model="deepseek-v4-pro"),
                )
            ),
        ),
        (
            "rejection:image_input",
            OpenAIResponsesError,
            "does not support image input",
            lambda: codec.encode_request(_part_request("image")),
        ),
        (
            "rejection:file_input",
            OpenAIResponsesError,
            "does not support file input",
            lambda: codec.encode_request(_part_request("file")),
        ),
        (
            "rejection:image_generation",
            OpenAIResponsesError,
            "does not support provider tool",
            lambda: codec.encode_request(
                ModelRequest(
                    messages=(Message.user("generate"),),
                    provider_tools=(
                        ProviderToolSpec(ProviderToolId("openai.responses", "image_generation")),
                    ),
                )
            ),
        ),
        (
            "rejection:unmodeled_provider_tool",
            OpenAIResponsesError,
            "does not support provider tool",
            lambda: codec.encode_request(
                ModelRequest(
                    messages=(Message.user("provider"),),
                    provider_tools=(
                        ProviderToolSpec(ProviderToolId("deepseek.responses", "file_search")),
                    ),
                )
            ),
        ),
        (
            "rejection:seed",
            OpenAIResponsesError,
            "does not support seed",
            lambda: codec.encode_request(
                ModelRequest(messages=(Message.user("seed"),), options=ModelOptions(seed=7))
            ),
        ),
        (
            "rejection:stop",
            OpenAIResponsesError,
            "does not support stop sequences",
            lambda: codec.encode_request(
                ModelRequest(messages=(Message.user("stop"),), options=ModelOptions(stop=("END",)))
            ),
        ),
        (
            "rejection:custom_tool",
            OpenAIResponsesError,
            "does not support freeform runtime tool",
            lambda: codec.encode_request(
                ModelRequest(
                    messages=(Message.user("custom"),),
                    runtime_tools=(FreeformToolSpec("shell", "Unsupported."),),
                )
            ),
        ),
        (
            "rejection:exact_apply_patch_choice",
            OpenAIResponsesError,
            "exact freeform runtime tool choice",
            lambda: codec.encode_request(
                ModelRequest(
                    messages=(Message.user("patch"),),
                    runtime_tools=(_APPLY_PATCH,),
                    tool_choice=ToolChoice(type="runtime", name="apply_patch"),
                )
            ),
        ),
        (
            "rejection:disable_parallel_calls",
            OpenAIResponsesError,
            "cannot disable parallel runtime tool calls",
            lambda: codec.encode_request(
                ModelRequest(
                    messages=(Message.user("serial"),),
                    runtime_tools=(_LOOKUP,),
                    tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
                )
            ),
        ),
        (
            "rejection:web_variant",
            OpenAIResponsesError,
            "web_search variant must be one of",
            lambda: codec.encode_request(
                ModelRequest(
                    messages=(Message.user("variant"),),
                    provider_tools=(deepseek_responses_web_search({"variant": "web_search_beta"}),),
                )
            ),
        ),
        (
            "rejection:thinking_required_choice",
            OpenAIResponsesError,
            "does not support tool_choice='required'",
            lambda: thinking_codec.encode_request(
                ModelRequest(
                    messages=(Message.user("thinking"),),
                    provider_tools=(deepseek_responses_web_search(),),
                    tool_choice=ToolChoice(type="required"),
                )
            ),
        ),
    )
    for name, error_type, message, operation in cases:
        recorder.reject(name, error_type, message, operation)


def _rich_output() -> list[JsonObject]:
    return [
        {
            "id": "message-text",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Evidence",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "start_index": 0,
                            "end_index": 8,
                            "url": "https://example.test/source",
                            "title": "Example source",
                        }
                    ],
                }
            ],
        },
        {
            "id": "reasoning-1",
            "type": "reasoning",
            "status": "completed",
            "content": [{"type": "reasoning_text", "text": "Check facts."}],
        },
        {
            "id": "message-refusal",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "Cannot disclose that."}],
        },
        {
            "id": "function-item",
            "type": "function_call",
            "status": "completed",
            "call_id": "function-call",
            "name": "lookup",
            "arguments": '{"query":"protocol"}',
        },
        {
            "id": "custom-item",
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": "custom-call",
            "name": "apply_patch",
            "input": "patch body",
        },
        {
            "id": "web-search-call",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": "current protocol"},
        },
    ]


def _output_snapshot(item: object) -> tuple[object, ...]:
    if isinstance(item, ContentPart):
        if item.type == "text":
            return (item.type, item.text, _citation_url(item))
        return (item.type, item.text)
    if isinstance(item, StructuredToolCall):
        return ("structured", item.id, item.name, item.arguments)
    if isinstance(item, FreeformToolCall):
        return ("freeform", item.id, item.name, item.input)
    if isinstance(item, ProviderToolCall):
        return ("provider", item.id, item.tool, item.status, item.arguments)
    raise AssertionError(f"unexpected decoded output type: {type(item).__name__}")


def _citation_url(part: ContentPart) -> str:
    value: object = thaw_json_value(part.metadata)
    for key in ("responses", "content", "annotations"):
        if not isinstance(value, dict) or key not in value:
            raise AssertionError(f"decoded text metadata is missing {key!r}")
        value = value[key]
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        raise AssertionError("decoded text metadata has no URL annotation")
    url = value[0].get("url")
    if not isinstance(url, str):
        raise AssertionError("decoded URL annotation has no string URL")
    return url


def _part_request(part_type: Literal["image", "file"]) -> ModelRequest:
    return ModelRequest(
        messages=(
            Message(
                "user",
                (ContentPart(type=part_type, uri=f"https://example.test/{part_type}"),),
            ),
        )
    )


def _wire_response(
    output: list[JsonObject],
    *,
    status: str = "completed",
    incomplete_reason: str | None = None,
    error: Mapping[str, object] | str | None = None,
) -> JsonObject:
    return {
        "id": "response-contract",
        "object": "response",
        "created_at": 1_700_000_000,
        "completed_at": 1_700_000_001,
        "status": status,
        "error": error,
        "incomplete_details": (
            None if incomplete_reason is None else {"reason": incomplete_reason}
        ),
        "model": MODEL,
        "output": output,
        "previous_response_id": None,
        "store": False,
        "usage": {
            "input_tokens": 21,
            "output_tokens": 13,
            "total_tokens": 34,
            "input_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 5},
        },
    }


def _expect_error(
    error_type: type[ErrorT],
    message_fragment: str,
    operation: Callable[[], object],
) -> ErrorT:
    try:
        operation()
    except error_type as error:
        if message_fragment not in str(error):
            raise AssertionError(
                f"expected {error_type.__name__} containing {message_fragment!r}, got {error!s}"
            ) from error
        return error
    except Exception as error:
        raise AssertionError(
            f"expected {error_type.__name__}, got {type(error).__name__}: {error}"
        ) from error
    raise AssertionError(f"expected {error_type.__name__}: {message_fragment}")


if __name__ == "__main__":
    print(json.dumps(run_offline_contract(), ensure_ascii=False, indent=2))
