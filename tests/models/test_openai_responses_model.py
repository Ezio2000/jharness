from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from time import time
from typing import Any, cast

import httpx
import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    FreeformToolCall,
    FreeformToolSpec,
    Message,
    ModelCapabilities,
    ModelDelta,
    ModelError,
    ModelErrorInfo,
    ModelProviderToolCallDelta,
    ModelReasoningDelta,
    ModelRequest,
    ModelRuntimeToolCallDelta,
    ModelUsageDelta,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    RunContext,
    RuntimeToolKind,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
    ToolSuccess,
)
from jharness.models.deepseek import deepseek_responses_profile
from jharness.models.openai import (
    OpenAIResponsesArtifactStore,
    OpenAIResponsesCodec,
    OpenAIResponsesError,
    OpenAIResponsesImageGenerationTool,
    OpenAIResponsesModel,
    OpenAIResponsesProfile,
    OpenAIResponsesProviderToolCodec,
    OpenAIResponsesProviderToolRegistry,
    OpenAIResponsesProviderToolStreamUpdate,
    OpenAIResponsesWebSearchTool,
)
from jharness.models.openai.responses.stream import OpenAIResponsesStreamDecoder

_DEEPSEEK_WEB = ProviderToolId("deepseek.responses", "web_search")
_OPENAI_IMAGE = ProviderToolId("openai.responses", "image_generation")
_JPEG_BYTES = b"\xff\xd8\xffjpeg-payload"
_JPEG_BASE64 = base64.b64encode(_JPEG_BYTES).decode("ascii")


@dataclass(frozen=True, slots=True)
class _SyntheticComputerTool(OpenAIResponsesProviderToolCodec):
    tool: ProviderToolId
    output_item_type: str = field(default="computer_call", init=False)
    event_prefix: str = field(default="response.computer_call.", init=False)

    @property
    def declaration_types(self) -> frozenset[str]:
        return frozenset({"computer_use_preview"})

    def encode_declaration(self, spec: ProviderToolSpec) -> dict[str, Any]:
        if spec.tool != self.tool:
            raise OpenAIResponsesError("synthetic codec identity mismatch")
        return {"type": "computer_use_preview", **dict(spec.configuration)}

    def decode_call(
        self,
        item: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> ProviderToolCall:
        del response
        return ProviderToolCall(
            id=cast(str, item["id"]),
            tool=self.tool,
            status=ProviderToolStatus(cast(str, item["status"])),
            arguments=cast(Mapping[str, Any], item.get("action", {})),
        )

    def encode_history(self, call: ProviderToolCall) -> dict[str, Any]:
        if call.tool != self.tool:
            raise OpenAIResponsesError("synthetic codec identity mismatch")
        return {
            "type": self.output_item_type,
            "id": call.id,
            "status": call.status.value,
            "action": dict(call.arguments),
        }

    def stream_event_update(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> OpenAIResponsesProviderToolStreamUpdate:
        del value
        if event_type != "response.computer_call.completed":
            raise OpenAIResponsesError("unsupported synthetic provider event")
        return OpenAIResponsesProviderToolStreamUpdate(ProviderToolStatus.COMPLETED)


def _openai_feature_profile(
    *,
    image_input: bool = False,
    image_generation: bool = False,
) -> OpenAIResponsesProfile:
    default = OpenAIResponsesProfile()
    provider_tools = frozenset({_OPENAI_IMAGE}) if image_generation else frozenset[ProviderToolId]()
    return OpenAIResponsesProfile(
        capabilities=replace(
            default.capabilities,
            tool_choice_types=(
                default.capabilities.tool_choice_types | {"provider"}
                if provider_tools
                else default.capabilities.tool_choice_types
            ),
            input_modalities=(
                frozenset({"text", "image", "file"})
                if image_input
                else default.capabilities.input_modalities
            ),
            provider_tools=provider_tools,
        ),
        provider_tool_registry=(
            OpenAIResponsesProviderToolRegistry(
                (
                    OpenAIResponsesImageGenerationTool(
                        tool=_OPENAI_IMAGE,
                        configuration_fields=frozenset({"output_format"}),
                    ),
                )
            )
            if image_generation
            else OpenAIResponsesProviderToolRegistry()
        ),
    )


class _MemoryOpenAIResponsesArtifactStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.saved: list[str] = []
        self.loaded: list[str] = []

    async def save_image(
        self,
        data: bytes,
        *,
        media_type: str,
        call_id: str,
        context: RunContext,
    ) -> ArtifactRef:
        del call_id, context
        digest = sha256(data).hexdigest()
        ref = f"artifact:sha256:{digest}"
        self.values[ref] = data
        self.saved.append(ref)
        return ArtifactRef(
            ref,
            media_type=media_type,
            size_bytes=len(data),
            sha256=digest,
        )

    async def load_image(
        self,
        artifact: ArtifactRef,
        *,
        call_id: str,
        context: RunContext,
    ) -> bytes:
        del call_id, context
        self.loaded.append(artifact.ref)
        return self.values[artifact.ref]


class _StaticArtifactStore(_MemoryOpenAIResponsesArtifactStore):
    def __init__(self, artifact: object) -> None:
        super().__init__()
        self.artifact = artifact

    async def save_image(
        self,
        data: bytes,
        *,
        media_type: str,
        call_id: str,
        context: RunContext,
    ) -> ArtifactRef:
        del data, media_type, call_id, context
        return cast(ArtifactRef, self.artifact)


class _FailingArtifactStore(_MemoryOpenAIResponsesArtifactStore):
    async def save_image(
        self,
        data: bytes,
        *,
        media_type: str,
        call_id: str,
        context: RunContext,
    ) -> ArtifactRef:
        del data, media_type, call_id, context
        raise OSError("artifact save failed")

    async def load_image(
        self,
        artifact: ArtifactRef,
        *,
        call_id: str,
        context: RunContext,
    ) -> bytes:
        del artifact, call_id, context
        raise OSError("artifact load failed")


def _terminal_response(
    output: list[dict[str, Any]],
    *,
    model: str = "gpt-test",
    status: str = "completed",
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp-1",
        "object": "response",
        "created_at": 1,
        "completed_at": 2,
        "status": status,
        "error": None,
        "incomplete_details": ({"reason": "max_output_tokens"} if status == "incomplete" else None),
        "model": model,
        "output": output,
        "previous_response_id": None,
        "store": False,
        "tools": [] if tools is None else tools,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    }


def _stream_event(
    decoder: OpenAIResponsesStreamDecoder,
    event_type: str,
    sequence_number: int,
    **fields: Any,
) -> tuple[bool, list[ModelDelta]]:
    event = {"type": event_type, "sequence_number": sequence_number, **fields}
    return decoder.apply_event(event_type, event)


def test_openai_responses_default_profile_is_conservative_and_stateless() -> None:
    profile = OpenAIResponsesProfile()
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)
    payload = codec.encode_request(ModelRequest(messages=(Message.user("hello"),)))

    assert profile.capabilities.input_modalities == frozenset({"text"})
    assert profile.capabilities.provider_tools == frozenset()
    assert profile.capabilities.structured_output is False
    assert profile.capabilities.json_mode is False
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in payload

    stored = OpenAIResponsesProfile(store=True, include=frozenset())
    stored_payload = OpenAIResponsesCodec(model="gpt-test", profile=stored).encode_request(
        ModelRequest(messages=(Message.user("hello"),))
    )
    assert stored_payload["store"] is True
    assert "include" not in stored_payload

    with pytest.raises(ValueError, match="stateless reasoning history"):
        OpenAIResponsesProfile(store=False, include=frozenset())

    reasoning = {
        "id": "reasoning-1",
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "summary"}],
    }
    with pytest.raises(OpenAIResponsesError, match="encrypted_content"):
        codec.decode_response(_terminal_response([reasoning]))
    reasoning["encrypted_content"] = "encrypted-state"
    response = codec.decode_response(_terminal_response([reasoning]))
    assert response.metadata["provider"] == "openai-responses"
    replay = codec.encode_request(
        ModelRequest(messages=(Message.user("hello"), response.to_assistant_message()))
    )
    assert cast(list[dict[str, Any]], replay["input"])[1]["encrypted_content"] == (
        "encrypted-state"
    )

    with pytest.raises(OpenAIResponsesError, match="reserved request field: store"):
        OpenAIResponsesCodec(
            model="gpt-test",
            profile=OpenAIResponsesProfile(extra_request_body={"store": True}),
        ).encode_request(ModelRequest(messages=(Message.user("hello"),)))


def test_deepseek_responses_profile_and_request_encode_native_responses() -> None:
    profile = deepseek_responses_profile(effort="none")
    codec = OpenAIResponsesCodec(model="deepseek-v4-flash", profile=profile)
    request = ModelRequest(
        messages=(Message.system("policy"), Message.user("question")),
        runtime_tools=(StructuredToolSpec("lookup", "lookup", {"type": "object"}),),
        provider_tools=(ProviderToolSpec(_DEEPSEEK_WEB),),
        tool_choice=ToolChoice(
            type="provider",
            provider_tool=_DEEPSEEK_WEB,
            allow_parallel_runtime_tool_calls=False,
        ),
    )

    payload = codec.encode_request(request)

    assert profile.capabilities.input_modalities == frozenset({"text"})
    assert profile.capabilities.output_modalities == frozenset({"text"})
    assert profile.capabilities.provider_tools == frozenset({_DEEPSEEK_WEB})
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["tool_choice"] == {"type": "web_search"}
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "lookup",
            "parameters": {"type": "object"},
        },
        {"type": "web_search"},
    ]
    assert [item["role"] for item in cast(list[dict[str, Any]], payload["input"])] == [
        "system",
        "user",
    ]
    assert "store" not in payload
    assert "previous_response_id" not in payload
    assert "parallel_tool_calls" not in payload

    provider_only_payload = codec.encode_request(
        ModelRequest(
            messages=(Message.user("search"),),
            provider_tools=(ProviderToolSpec(_DEEPSEEK_WEB),),
            tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
        )
    )
    assert "parallel_tool_calls" not in provider_only_payload

    versioned_search = codec.encode_request(
        ModelRequest(
            messages=(Message.user("search"),),
            provider_tools=(
                ProviderToolSpec(
                    _DEEPSEEK_WEB,
                    {"variant": "web_search_2025_08_26"},
                ),
            ),
            tool_choice=ToolChoice(
                type="provider",
                provider_tool=_DEEPSEEK_WEB,
            ),
        )
    )
    assert versioned_search["tools"] == [{"type": "web_search_2025_08_26"}]
    assert versioned_search["tool_choice"] == {"type": "web_search_2025_08_26"}

    custom = codec.encode_request(
        ModelRequest(
            messages=(Message.user("patch"),),
            runtime_tools=(FreeformToolSpec("apply_patch", "must not reach wire"),),
            tool_choice=ToolChoice(type="required"),
        )
    )
    assert custom["tools"] == [{"type": "custom", "name": "apply_patch"}]
    with pytest.raises(OpenAIResponsesError, match="freeform runtime tool"):
        codec.encode_request(
            ModelRequest(
                messages=(Message.user("run"),),
                runtime_tools=(FreeformToolSpec("shell", "unsupported"),),
            )
        )

    thinking_profile = deepseek_responses_profile()
    thinking_codec = OpenAIResponsesCodec(
        model="deepseek-v4-flash",
        profile=thinking_profile,
    )
    with pytest.raises(OpenAIResponsesError, match="does not support tool_choice='required'"):
        thinking_codec.encode_request(
            ModelRequest(
                messages=(Message.user("question"),),
                provider_tools=(ProviderToolSpec(_DEEPSEEK_WEB),),
                tool_choice=ToolChoice(type="required"),
            )
        )
    with pytest.raises(ValueError, match="input modality"):
        OpenAIResponsesProfile(
            capabilities=replace(
                OpenAIResponsesProfile().capabilities,
                input_modalities=frozenset({"audio"}),
            )
        )
    with pytest.raises(ValueError, match="output modality"):
        OpenAIResponsesProfile(
            capabilities=replace(
                OpenAIResponsesProfile().capabilities,
                output_modalities=frozenset({"image"}),
            )
        )
    with pytest.raises(ValueError, match="registry must exactly match"):
        OpenAIResponsesProfile(
            capabilities=ModelCapabilities(
                provider_tools=frozenset({ProviderToolId("test", "computer")}),
                tool_choice_types=frozenset({"auto", "none", "required", "runtime", "provider"}),
            ),
        )


def test_deepseek_responses_custom_tool_terminal_history_and_output_round_trip() -> None:
    profile = deepseek_responses_profile(effort="none")
    codec = OpenAIResponsesCodec(model="deepseek-v4-flash", profile=profile)
    wire_item = {
        "id": "ct-item-1",
        "type": "custom_tool_call",
        "status": "completed",
        "call_id": "ct-call-1",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
    }

    response = codec.decode_response(_terminal_response([wire_item], model="deepseek-v4-flash"))

    assert response.runtime_tool_calls() == (
        FreeformToolCall(
            "ct-call-1",
            "apply_patch",
            "*** Begin Patch\n*** End Patch",
        ),
    )
    payload = codec.encode_request(
        ModelRequest(
            messages=(
                Message.user("patch"),
                response.to_assistant_message(),
                Message.tool(
                    "ct-call-1",
                    ToolSuccess((ContentPart.text_part("Done!"),)),
                ),
            ),
            runtime_tools=(FreeformToolSpec("apply_patch", "not emitted"),),
            tool_choice=ToolChoice(type="none"),
        )
    )

    assert payload["tools"] == [{"type": "custom", "name": "apply_patch"}]
    assert cast(list[dict[str, Any]], payload["input"])[1:] == [
        {
            "type": "custom_tool_call",
            "call_id": "ct-call-1",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "ct-call-1",
            "output": "Done!",
        },
    ]


def test_deepseek_responses_custom_tool_stream_round_trip() -> None:
    profile = deepseek_responses_profile(effort="none")
    decoder = OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="deepseek-v4-flash", profile=profile),
        profile,
    )
    final_item = {
        "id": "ct-item-1",
        "type": "custom_tool_call",
        "status": "completed",
        "call_id": "ct-call-1",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
    }
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _, added = _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={**final_item, "status": "in_progress", "input": ""},
    )
    _, streamed = _stream_event(
        decoder,
        "response.custom_tool_call_input.delta",
        2,
        item_id="ct-item-1",
        output_index=0,
        delta="*** Begin Patch\n*** End Patch",
    )
    _stream_event(
        decoder,
        "response.custom_tool_call_input.done",
        3,
        item_id="ct-item-1",
        output_index=0,
        input="*** Begin Patch\n*** End Patch",
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        4,
        output_index=0,
        item=final_item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        5,
        response=_terminal_response([final_item], model="deepseek-v4-flash"),
    )

    assert terminal is True
    assert added == [
        ModelRuntimeToolCallDelta(
            output_index=0,
            input_kind=RuntimeToolKind.FREEFORM,
            input_delta="",
            id="ct-call-1",
            name="apply_patch",
        )
    ]
    assert streamed == [
        ModelRuntimeToolCallDelta(
            output_index=0,
            input_kind=RuntimeToolKind.FREEFORM,
            input_delta="*** Begin Patch\n*** End Patch",
        )
    ]
    assert decoder.completed_response().runtime_tool_calls() == (
        FreeformToolCall(
            "ct-call-1",
            "apply_patch",
            "*** Begin Patch\n*** End Patch",
        ),
    )


def test_openai_responses_provider_registry_is_open_for_synthetic_tool_dialects() -> None:
    synthetic_tool = ProviderToolId("synthetic.responses", "computer_use")
    default = OpenAIResponsesProfile()
    profile = OpenAIResponsesProfile(
        capabilities=replace(
            default.capabilities,
            provider_tools=frozenset({synthetic_tool}),
            tool_choice_types=default.capabilities.tool_choice_types | {"provider"},
        ),
        provider_tool_registry=OpenAIResponsesProviderToolRegistry(
            (_SyntheticComputerTool(tool=synthetic_tool),)
        ),
    )
    codec = OpenAIResponsesCodec(model="synthetic-model", profile=profile)
    request = ModelRequest(
        messages=(Message.user("search"),),
        provider_tools=(ProviderToolSpec(synthetic_tool, {"display_width": 1024}),),
        tool_choice=ToolChoice(type="provider", provider_tool=synthetic_tool),
    )

    payload = codec.encode_request(request)
    assert payload["tools"] == [{"type": "computer_use_preview", "display_width": 1024}]
    assert payload["tool_choice"] == {"type": "computer_use_preview"}
    response = codec.decode_response(
        _terminal_response(
            [
                {
                    "id": "computer-synthetic",
                    "type": "computer_call",
                    "status": "completed",
                    "action": {"type": "click", "x": 1, "y": 2},
                }
            ],
            model="synthetic-model",
        )
    )
    assert response.provider_tool_calls()[0].tool == synthetic_tool
    replay = codec.encode_request(
        replace(
            request,
            messages=(Message.user("search"), response.to_assistant_message()),
        )
    )
    assert cast(list[dict[str, Any]], replay["input"])[1] == {
        "type": "computer_call",
        "id": "computer-synthetic",
        "status": "completed",
        "action": {"type": "click", "x": 1, "y": 2},
    }

    decoder = OpenAIResponsesStreamDecoder(codec, profile)
    final_item = {
        "id": "computer-stream",
        "type": "computer_call",
        "status": "completed",
        "action": {"type": "click", "x": 3, "y": 4},
    }
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _, added = _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={**final_item, "status": "in_progress"},
    )
    _, completed = _stream_event(
        decoder,
        "response.computer_call.completed",
        2,
        item_id="computer-stream",
        output_index=0,
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        3,
        output_index=0,
        item=final_item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        4,
        response=_terminal_response([final_item], model="synthetic-model"),
    )
    assert terminal is True
    assert [cast(ModelProviderToolCallDelta, delta).tool for delta in added + completed] == [
        synthetic_tool,
        synthetic_tool,
    ]

    with pytest.raises(OpenAIResponsesError, match="does not support provider tool"):
        codec.encode_request(
            replace(
                request,
                provider_tools=(ProviderToolSpec(_DEEPSEEK_WEB),),
                tool_choice=ToolChoice(type="auto"),
            )
        )


def test_openai_responses_parallel_control_applies_only_when_runtime_calls_can_be_parallel() -> (
    None
):
    runtime_tool = StructuredToolSpec("lookup", "lookup", {"type": "object"})
    request = ModelRequest(
        messages=(Message.user("question"),),
        runtime_tools=(runtime_tool,),
        tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
    )

    controllable = OpenAIResponsesCodec(model="gpt-test").encode_request(request)
    assert controllable["parallel_tool_calls"] is False

    default_profile = OpenAIResponsesProfile()
    serial_profile = replace(
        default_profile,
        capabilities=replace(
            default_profile.capabilities,
            parallel_runtime_tool_calls=False,
            parallel_runtime_tool_call_control=False,
        ),
    )
    inherently_serial = OpenAIResponsesCodec(
        model="gpt-test",
        profile=serial_profile,
    ).encode_request(request)
    assert "parallel_tool_calls" not in inherently_serial

    uncontrollable_profile = replace(
        default_profile,
        capabilities=replace(
            default_profile.capabilities,
            parallel_runtime_tool_call_control=False,
        ),
    )
    with pytest.raises(OpenAIResponsesError, match="parallel runtime tool calls"):
        OpenAIResponsesCodec(
            model="gpt-test",
            profile=uncontrollable_profile,
        ).encode_request(request)


def test_openai_responses_encodes_non_native_assistant_history_as_easy_input() -> None:
    payload = OpenAIResponsesCodec(model="gpt-test").encode_request(
        ModelRequest(
            messages=(
                Message.user("question"),
                Message.assistant((ContentPart.text_part("answer"),)),
            ),
        )
    )

    assert cast(list[dict[str, Any]], payload["input"])[1] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "input_text", "text": "answer"}],
    }


def test_deepseek_responses_rejects_exact_custom_tool_choice() -> None:
    codec = OpenAIResponsesCodec(
        model="deepseek-v4-flash",
        profile=deepseek_responses_profile(effort="none"),
    )

    with pytest.raises(OpenAIResponsesError, match="exact freeform runtime tool choice"):
        codec.encode_request(
            ModelRequest(
                messages=(Message.user("patch"),),
                runtime_tools=(FreeformToolSpec("apply_patch", "apply a patch"),),
                tool_choice=ToolChoice(type="runtime", name="apply_patch"),
            )
        )


async def test_deepseek_responses_nonstream_client_preserves_interleaved_output_order() -> None:
    captured: dict[str, object] = {}
    wire_response = _terminal_response(
        [
            {
                "id": "ws-1",
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "first"},
            },
            {
                "id": "msg-1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "one", "annotations": []}],
            },
            {
                "id": "ws-2",
                "type": "web_search_call",
                "status": "failed",
                "action": {"type": "search", "query": "second"},
            },
            {
                "id": "msg-2",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "two", "annotations": []}],
            },
        ],
        model="deepseek-v4-flash",
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        captured["url"] = str(raw.url)
        captured["body"] = json.loads(raw.content)
        return httpx.Response(200, json=wire_response, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://api.deepseek.test/v1",
            api_key="secret",
            model="deepseek-v4-flash",
            profile=deepseek_responses_profile(effort="none"),
            client=client,
        )
        response = await model.invoke(
            ModelRequest(
                messages=(Message.user("question"),),
                provider_tools=(ProviderToolSpec(_DEEPSEEK_WEB),),
            ),
            RunContext("run-1", time()),
            stream=False,
            emit_delta=None,
        )

    assert captured["url"] == "https://api.deepseek.test/v1/responses"
    assert "store" not in cast(dict[str, object], captured["body"])
    assert [type(item) for item in response.output] == [
        ProviderToolCall,
        ContentPart,
        ProviderToolCall,
        ContentPart,
    ]
    assert [part.text for part in response.visible_parts()] == ["one", "two"]
    first, _, failed, _ = response.output
    assert isinstance(first, ProviderToolCall)
    assert first.status is ProviderToolStatus.COMPLETED
    assert isinstance(failed, ProviderToolCall)
    assert failed.status is ProviderToolStatus.FAILED
    assert failed.error is not None and failed.error.code == "web_search_failed"


def test_openai_responses_terminal_function_call_rejects_incomplete_execution() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")

    for status in ("in_progress", "incomplete"):
        with pytest.raises(OpenAIResponsesError, match="must be completed"):
            codec.decode_response(
                _terminal_response(
                    [
                        {
                            "id": "fc-1",
                            "type": "function_call",
                            "status": status,
                            "call_id": "call-1",
                            "name": "lookup",
                            "arguments": "{}",
                        }
                    ]
                )
            )

    response = codec.decode_response(
        _terminal_response(
            [
                {
                    "id": "fc-1",
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ]
        )
    )
    assert response.runtime_tool_calls() == (StructuredToolCall("call-1", "lookup", {}),)

    with pytest.raises(OpenAIResponsesError, match="incomplete Responses"):
        codec.decode_response(
            _terminal_response(
                [
                    {
                        "id": "fc-1",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": "{}",
                    }
                ],
                status="incomplete",
            )
        )


def test_deepseek_responses_reasoning_sse_tracks_open_part_and_uses_terminal_response() -> None:
    profile = deepseek_responses_profile()
    codec = OpenAIResponsesCodec(model="deepseek-v4-flash", profile=profile)
    decoder = OpenAIResponsesStreamDecoder(codec, profile)
    reasoning_item = {
        "id": "rs-1",
        "type": "reasoning",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": "分析"}],
        "summary": [],
    }

    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={**reasoning_item, "status": "in_progress", "content": []},
    )
    _stream_event(
        decoder,
        "response.content_part.added",
        2,
        item_id="rs-1",
        output_index=0,
        content_index=0,
        part={"type": "reasoning_text", "text": ""},
    )
    _, reasoning_deltas = _stream_event(
        decoder,
        "response.reasoning_text.delta",
        3,
        item_id="rs-1",
        output_index=0,
        content_index=0,
        delta="分析",
    )
    _stream_event(
        decoder,
        "response.reasoning_text.done",
        4,
        item_id="rs-1",
        output_index=0,
        content_index=0,
        text="分析",
    )
    with pytest.raises(OpenAIResponsesError, match="emitted twice"):
        _stream_event(
            decoder,
            "response.reasoning_text.done",
            5,
            item_id="rs-1",
            output_index=0,
            content_index=0,
            text="分析",
        )
    _stream_event(
        decoder,
        "response.content_part.done",
        7,
        item_id="rs-1",
        output_index=0,
        content_index=0,
        part={"type": "reasoning_text", "text": "分析"},
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        8,
        output_index=0,
        item=reasoning_item,
    )
    terminal, usage_deltas = _stream_event(
        decoder,
        "response.completed",
        9,
        response=_terminal_response([reasoning_item], model="deepseek-v4-flash"),
    )

    assert len(reasoning_deltas) == 1
    assert isinstance(reasoning_deltas[0], ModelReasoningDelta)
    assert reasoning_deltas[0].text_delta == "分析"
    assert terminal is True
    assert len(usage_deltas) == 1 and isinstance(usage_deltas[0], ModelUsageDelta)
    completed = decoder.completed_response()
    reasoning = completed.output[0]
    assert isinstance(reasoning, ContentPart)
    assert reasoning.type == "reasoning"
    assert reasoning.text == "分析"


@pytest.mark.parametrize("lifecycle_status", ["completed", "incomplete", "failed"])
def test_deepseek_responses_web_search_keeps_lifecycle_status_provisional_until_output_item(
    lifecycle_status: str,
) -> None:
    profile = deepseek_responses_profile(effort="none")
    decoder = OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="deepseek-v4-flash", profile=profile),
        profile,
    )
    action = {"type": "search", "query": "JHarness"}
    final_item = {
        "id": "ws-1",
        "type": "web_search_call",
        "status": "failed",
        "action": action,
        "error": {"code": "search_failed", "message": "failed"},
    }
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _, added = _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={
            "id": "ws-1",
            "type": "web_search_call",
            "status": "searching",
        },
    )
    _, lifecycle = _stream_event(
        decoder,
        f"response.web_search_call.{lifecycle_status}",
        2,
        item_id="ws-1",
        output_index=0,
        action=action,
    )
    _, failed = _stream_event(
        decoder,
        "response.output_item.done",
        3,
        output_index=0,
        item=final_item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        4,
        response=_terminal_response([final_item], model="deepseek-v4-flash"),
    )

    deltas = [added[0], lifecycle[0], failed[0]]
    assert all(isinstance(delta, ModelProviderToolCallDelta) for delta in deltas)
    assert [cast(ModelProviderToolCallDelta, delta).status for delta in deltas] == [
        ProviderToolStatus.IN_PROGRESS,
        ProviderToolStatus.IN_PROGRESS,
        ProviderToolStatus.FAILED,
    ]
    assert cast(ModelProviderToolCallDelta, lifecycle[0]).event == (
        f"response.web_search_call.{lifecycle_status}"
    )
    assert cast(ModelProviderToolCallDelta, lifecycle[0]).data == {"action": action}
    assert terminal is True
    completed_call = decoder.completed_response().output[0]
    assert isinstance(completed_call, ProviderToolCall)
    assert completed_call.status is ProviderToolStatus.FAILED


def _generic_web_search_stream_decoder() -> OpenAIResponsesStreamDecoder:
    web_search = ProviderToolId("test.responses", "web_search")
    default = OpenAIResponsesProfile()
    profile = OpenAIResponsesProfile(
        capabilities=replace(
            default.capabilities,
            provider_tools=frozenset({web_search}),
        ),
        provider_tool_registry=OpenAIResponsesProviderToolRegistry(
            (OpenAIResponsesWebSearchTool(tool=web_search),)
        ),
    )
    return OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="gpt-test", profile=profile),
        profile,
    )


def test_openai_responses_provider_lifecycle_rejects_conflicting_terminal_statuses() -> None:
    decoder = _generic_web_search_stream_decoder()
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={"id": "ws-1", "type": "web_search_call", "status": "searching"},
    )
    _stream_event(
        decoder,
        "response.web_search_call.completed",
        2,
        item_id="ws-1",
        output_index=0,
    )
    with pytest.raises(OpenAIResponsesError, match="cannot return to in_progress"):
        _stream_event(
            decoder,
            "response.web_search_call.in_progress",
            3,
            item_id="ws-1",
            output_index=0,
        )
    with pytest.raises(OpenAIResponsesError, match="conflicting terminal statuses"):
        _stream_event(
            decoder,
            "response.output_item.done",
            4,
            output_index=0,
            item={
                "id": "ws-1",
                "type": "web_search_call",
                "status": "failed",
                "error": {"code": "search_failed", "message": "failed"},
            },
        )


def test_openai_responses_provider_output_item_done_requires_a_terminal_status() -> None:
    decoder = _generic_web_search_stream_decoder()
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={"id": "ws-1", "type": "web_search_call", "status": "searching"},
    )

    with pytest.raises(OpenAIResponsesError, match="requires a terminal status"):
        _stream_event(
            decoder,
            "response.output_item.done",
            2,
            output_index=0,
            item={"id": "ws-1", "type": "web_search_call", "status": "searching"},
        )


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
def test_openai_responses_terminal_response_accepts_provider_status_matching_output_item_done(
    status: str,
) -> None:
    decoder = _generic_web_search_stream_decoder()
    final_item = {
        "id": "ws-1",
        "type": "web_search_call",
        "status": status,
        "action": {"type": "search", "query": "JHarness"},
        "error": {"code": "search_failed", "message": "failed"},
    }
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={"id": "ws-1", "type": "web_search_call", "status": "searching"},
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        2,
        output_index=0,
        item=final_item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        3,
        response=_terminal_response([final_item], model="gpt-test"),
    )

    assert terminal is True
    call = decoder.completed_response().output[0]
    assert isinstance(call, ProviderToolCall)
    assert call.status is ProviderToolStatus(status)


@pytest.mark.parametrize(
    ("done_status", "terminal_status"),
    [("failed", "completed"), ("completed", "failed")],
)
def test_openai_responses_terminal_response_rejects_provider_status_mismatching_output_item_done(
    done_status: str,
    terminal_status: str,
) -> None:
    decoder = _generic_web_search_stream_decoder()
    done_item = {
        "id": "ws-1",
        "type": "web_search_call",
        "status": done_status,
        "action": {"type": "search", "query": "JHarness"},
        "error": {"code": "search_failed", "message": "failed"},
    }
    terminal_item = {**done_item, "status": terminal_status}
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={"id": "ws-1", "type": "web_search_call", "status": "searching"},
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        2,
        output_index=0,
        item=done_item,
    )

    with pytest.raises(OpenAIResponsesError, match=r"does not match output_item\.done"):
        _stream_event(
            decoder,
            "response.completed",
            3,
            response=_terminal_response([terminal_item], model="gpt-test"),
        )


def test_openai_responses_output_text_annotation_event_validates_the_open_message_part() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    decoder = OpenAIResponsesStreamDecoder(codec, codec.profile)
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={
            "id": "msg-1",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
    )
    _stream_event(
        decoder,
        "response.content_part.added",
        2,
        item_id="msg-1",
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": ""},
    )
    _, deltas = _stream_event(
        decoder,
        "response.output_text.annotation.added",
        7,
        item_id="msg-1",
        output_index=0,
        content_index=0,
        annotation_index=0,
        annotation={
            "type": "url_citation",
            "start_index": 0,
            "end_index": 4,
            "url": "https://example.test/source",
            "title": "source",
        },
    )

    assert deltas == []
    with pytest.raises(OpenAIResponsesError, match="annotation"):
        _stream_event(
            decoder,
            "response.output_text.annotation.added",
            8,
            item_id="msg-1",
            output_index=0,
            content_index=0,
            annotation_index=1,
            annotation=None,
        )


def test_openai_responses_provider_only_terminal_response_is_valid() -> None:
    profile = deepseek_responses_profile()
    response = OpenAIResponsesCodec(
        model="deepseek-v4-flash",
        profile=profile,
    ).decode_response(
        _terminal_response(
            [
                {
                    "id": "ws-1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "only"},
                }
            ],
            model="deepseek-v4-flash",
        )
    )

    assert len(response.output) == 1
    assert isinstance(response.output[0], ProviderToolCall)
    assert response.visible_parts() == ()
    assert response.finish_reason == "stop"

    with pytest.raises(OpenAIResponsesError, match="in-progress provider tools"):
        OpenAIResponsesCodec(
            model="deepseek-v4-flash",
            profile=profile,
        ).decode_response(
            _terminal_response(
                [
                    {
                        "id": "ws-2",
                        "type": "web_search_call",
                        "status": "searching",
                        "action": {"type": "search", "query": "pending"},
                    }
                ],
                model="deepseek-v4-flash",
            )
        )


def test_openai_responses_vision_inputs_encode_url_base64_and_artifact_with_media_validation() -> (
    None
):
    codec = OpenAIResponsesCodec(
        model="gpt-test",
        profile=_openai_feature_profile(image_input=True),
    )
    request = ModelRequest(
        messages=(
            Message(
                "user",
                (
                    ContentPart(type="image", uri="https://images.test/cat.png"),
                    ContentPart(type="image", data={"base64": _JPEG_BASE64}),
                    ContentPart.artifact_part(ArtifactRef("file-image", media_type="image/png")),
                ),
            ),
        )
    )

    content = cast(list[dict[str, Any]], codec.encode_request(request)["input"])[0]["content"]
    assert content == [
        {"type": "input_image", "image_url": "https://images.test/cat.png"},
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{_JPEG_BASE64}",
        },
        {"type": "input_image", "file_id": "file-image"},
    ]

    with pytest.raises(OpenAIResponsesError, match="does not match"):
        codec.encode_request(
            ModelRequest(
                messages=(
                    Message(
                        "user",
                        (
                            ContentPart(
                                type="image",
                                media_type="image/png",
                                data={"base64": _JPEG_BASE64},
                            ),
                        ),
                    ),
                )
            )
        )
    with pytest.raises(OpenAIResponsesError, match="valid base64"):
        codec.encode_request(
            ModelRequest(
                messages=(
                    Message(
                        "user",
                        (
                            ContentPart(
                                type="image",
                                media_type="image/jpeg",
                                data={"base64": f"{_JPEG_BASE64}!"},
                            ),
                        ),
                    ),
                )
            )
        )


def test_openai_responses_image_generation_decodes_media_type_and_replays_base64_history() -> None:
    codec = OpenAIResponsesCodec(
        model="gpt-test",
        profile=_openai_feature_profile(image_generation=True),
    )
    wire = _terminal_response(
        [
            {
                "id": "ig-1",
                "type": "image_generation_call",
                "status": "completed",
                "result": _JPEG_BASE64,
            }
        ],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    response = codec.decode_response(wire)
    call = response.provider_tool_calls()[0]
    assert call.tool == _OPENAI_IMAGE
    assert call.output[0].type == "image"
    assert call.output[0].media_type == "image/jpeg"
    assert call.output[0].data["base64"] == _JPEG_BASE64
    native_item = cast(dict[str, Any], call.metadata["responses"])["item"]
    assert "result" not in native_item

    payload = codec.encode_request(
        ModelRequest(
            messages=(Message.user("draw"), response.to_assistant_message()),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
        )
    )
    replay = cast(list[dict[str, Any]], payload["input"])[1]
    assert replay == {
        "id": "ig-1",
        "type": "image_generation_call",
        "status": "completed",
        "result": _JPEG_BASE64,
    }

    mismatched = _terminal_response(
        cast(list[dict[str, Any]], wire["output"]),
        tools=[{"type": "image_generation", "output_format": "png"}],
    )
    with pytest.raises(OpenAIResponsesError, match="does not match"):
        codec.decode_response(mismatched)


async def test_openai_responses_image_generation_externalizes_and_hydrates_artifact_history() -> (
    None
):
    profile = _openai_feature_profile(image_generation=True)
    store = _MemoryOpenAIResponsesArtifactStore()
    captured: list[dict[str, Any]] = []
    responses = [
        _terminal_response(
            [
                {
                    "id": "ig-1",
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": _JPEG_BASE64,
                }
            ],
            tools=[{"type": "image_generation", "output_format": "jpeg"}],
        ),
        _terminal_response(
            [
                {
                    "id": "msg-2",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "saved", "annotations": []}],
                }
            ]
        ),
    ]

    async def handler(raw: httpx.Request) -> httpx.Response:
        captured.append(json.loads(raw.content))
        return httpx.Response(200, json=responses.pop(0), request=raw)

    assert isinstance(store, OpenAIResponsesArtifactStore)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        without_store = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            client=client,
        )
        image_request = ModelRequest(
            messages=(Message.user("draw"),),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
        )
        with pytest.raises(ValueError, match="OpenAIResponsesArtifactStore"):
            await without_store.invoke(
                image_request,
                RunContext("run-unsafe", time()),
                stream=False,
                emit_delta=None,
            )

        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=store,
            client=client,
        )
        context = RunContext("run-1", time())
        generated = await model.invoke(
            image_request,
            context,
            stream=False,
            emit_delta=None,
        )
        call = generated.provider_tool_calls()[0]
        assert call.output[0].type == "artifact"
        digest = sha256(_JPEG_BYTES).hexdigest()
        artifact_ref = f"artifact:sha256:{digest}"
        assert call.output[0].artifact == ArtifactRef(
            artifact_ref,
            media_type="image/jpeg",
            size_bytes=len(_JPEG_BYTES),
            sha256=digest,
        )
        assert "base64" not in call.output[0].data

        completed = await model.invoke(
            ModelRequest(
                messages=(
                    Message.user("draw"),
                    generated.to_assistant_message(),
                    Message.user("confirm"),
                ),
                provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
            ),
            context,
            stream=False,
            emit_delta=None,
        )

    replay = cast(list[dict[str, Any]], captured[1]["input"])[1]
    assert replay["result"] == _JPEG_BASE64
    assert store.saved == [artifact_ref]
    assert store.loaded == [artifact_ref]
    assert completed.visible_parts()[0].text == "saved"


async def test_openai_responses_unrequested_inline_image_result_requires_artifact_store() -> None:
    profile = _openai_feature_profile(image_generation=True)
    wire = _terminal_response(
        [
            {
                "id": "ig-unrequested",
                "type": "image_generation_call",
                "status": "completed",
                "result": _JPEG_BASE64,
            }
        ],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=wire, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            client=client,
        )
        with pytest.raises(ModelError, match="without an OpenAIResponsesArtifactStore") as caught:
            await model.invoke(
                ModelRequest(messages=(Message.user("text only"),)),
                RunContext("run-unrequested", time()),
                stream=False,
                emit_delta=None,
            )
    assert caught.value.info.code == "codec_error"


@pytest.mark.parametrize(
    ("artifact", "error_type", "message"),
    (
        ("not-an-artifact", TypeError, "must return ArtifactRef"),
        (
            ArtifactRef("artifact:missing-size", media_type="image/jpeg"),
            ValueError,
            "requires size_bytes",
        ),
        (
            ArtifactRef(
                "artifact:missing-sha",
                media_type="image/jpeg",
                size_bytes=len(_JPEG_BYTES),
            ),
            ValueError,
            "requires sha256",
        ),
        (
            ArtifactRef(
                "artifact:wrong-media",
                media_type="image/png",
                size_bytes=len(_JPEG_BYTES),
                sha256=sha256(_JPEG_BYTES).hexdigest(),
            ),
            ValueError,
            "media_type must match",
        ),
        (
            ArtifactRef(
                "artifact:wrong-size",
                media_type="image/jpeg",
                size_bytes=len(_JPEG_BYTES) + 1,
                sha256=sha256(_JPEG_BYTES).hexdigest(),
            ),
            ValueError,
            "size_bytes does not match",
        ),
        (
            ArtifactRef(
                "artifact:wrong-sha",
                media_type="image/jpeg",
                size_bytes=len(_JPEG_BYTES),
                sha256="0" * 64,
            ),
            ValueError,
            "sha256 does not match",
        ),
    ),
)
async def test_openai_responses_image_artifact_store_return_is_fully_validated(
    artifact: object,
    error_type: type[Exception],
    message: str,
) -> None:
    profile = _openai_feature_profile(image_generation=True)
    wire = _terminal_response(
        [
            {
                "id": "ig-invalid-artifact",
                "type": "image_generation_call",
                "status": "completed",
                "result": _JPEG_BASE64,
            }
        ],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=wire, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=_StaticArtifactStore(artifact),
            client=client,
        )
        request = ModelRequest(
            messages=(Message.user("draw"),),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
        )
        with pytest.raises(error_type, match=message):
            await model.invoke(
                request,
                RunContext("run-invalid-artifact", time()),
                stream=False,
                emit_delta=None,
            )


async def test_openai_responses_image_artifact_save_failure_aborts_the_model_response() -> None:
    profile = _openai_feature_profile(image_generation=True)
    wire = _terminal_response(
        [
            {
                "id": "ig-save-failure",
                "type": "image_generation_call",
                "status": "completed",
                "result": _JPEG_BASE64,
            }
        ],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=wire, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=_FailingArtifactStore(),
            client=client,
        )
        with pytest.raises(OSError, match="artifact save failed"):
            await model.invoke(
                ModelRequest(
                    messages=(Message.user("draw"),),
                    provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
                ),
                RunContext("run-save-failure", time()),
                stream=False,
                emit_delta=None,
            )


async def test_openai_responses_image_artifact_hydration_rejects_corrupt_stored_bytes() -> None:
    profile = _openai_feature_profile(image_generation=True)
    digest = sha256(_JPEG_BYTES).hexdigest()
    artifact = ArtifactRef(
        f"artifact:sha256:{digest}",
        media_type="image/jpeg",
        size_bytes=len(_JPEG_BYTES),
        sha256=digest,
    )
    store = _MemoryOpenAIResponsesArtifactStore()
    store.values[artifact.ref] = b"x" * len(_JPEG_BYTES)
    history = ProviderToolCall(
        "ig-corrupt",
        _OPENAI_IMAGE,
        ProviderToolStatus.COMPLETED,
        output=(ContentPart.artifact_part(artifact),),
    )

    async def unexpected_handler(raw: httpx.Request) -> httpx.Response:
        raise AssertionError(f"corrupt artifact must fail before HTTP: {raw.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=store,
            client=client,
        )
        request = ModelRequest(
            messages=(
                Message.user("draw"),
                Message.assistant((history,)),
                Message.user("continue"),
            ),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
        )
        with pytest.raises(ValueError, match="sha256 does not match"):
            await model.invoke(
                request,
                RunContext("run-corrupt", time()),
                stream=False,
                emit_delta=None,
            )


async def test_openai_responses_image_artifact_load_failure_aborts_before_http() -> None:
    profile = _openai_feature_profile(image_generation=True)
    digest = sha256(_JPEG_BYTES).hexdigest()
    artifact = ArtifactRef(
        f"artifact:sha256:{digest}",
        media_type="image/jpeg",
        size_bytes=len(_JPEG_BYTES),
        sha256=digest,
    )
    history = ProviderToolCall(
        "ig-load-failure",
        _OPENAI_IMAGE,
        ProviderToolStatus.COMPLETED,
        output=(ContentPart.artifact_part(artifact),),
    )

    async def unexpected_handler(raw: httpx.Request) -> httpx.Response:
        raise AssertionError(f"load failure must abort before HTTP: {raw.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=_FailingArtifactStore(),
            client=client,
        )
        with pytest.raises(OSError, match="artifact load failed"):
            await model.invoke(
                ModelRequest(
                    messages=(
                        Message.user("draw"),
                        Message.assistant((history,)),
                        Message.user("continue"),
                    ),
                    provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
                ),
                RunContext("run-load-failure", time()),
                stream=False,
                emit_delta=None,
            )


@pytest.mark.parametrize("status", ["incomplete", "failed"])
async def test_openai_responses_terminal_partial_or_failed_image_results_are_externalized(
    status: str,
) -> None:
    profile = _openai_feature_profile(image_generation=True)
    item: dict[str, Any] = {
        "id": f"ig-{status}",
        "type": "image_generation_call",
        "status": status,
        "result": _JPEG_BASE64,
    }
    if status == "failed":
        item["error"] = {"code": "image_failed", "message": "partial result"}
    wire = _terminal_response(
        [item],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=wire, request=raw)

    store = _MemoryOpenAIResponsesArtifactStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=store,
            client=client,
        )
        response = await model.invoke(
            ModelRequest(
                messages=(Message.user("draw"),),
                provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
            ),
            RunContext(f"run-{status}", time()),
            stream=False,
            emit_delta=None,
        )

    call = response.provider_tool_calls()[0]
    assert call.status.value == status
    assert call.output[0].type == "artifact"


async def test_openai_responses_streamed_image_result_is_externalized_after_live_partial_data() -> (
    None
):
    profile = _openai_feature_profile(image_generation=True)
    final_item = {
        "id": "ig-stream",
        "type": "image_generation_call",
        "status": "completed",
        "result": _JPEG_BASE64,
    }
    terminal = _terminal_response(
        [final_item],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp-1", "object": "response", "status": "in_progress"},
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "ig-stream",
                "type": "image_generation_call",
                "status": "generating",
            },
        },
        {
            "type": "response.image_generation_call.partial_image",
            "sequence_number": 2,
            "item_id": "ig-stream",
            "output_index": 0,
            "partial_image_index": 0,
            "partial_image_b64": _JPEG_BASE64,
        },
        {
            "type": "response.image_generation_call.completed",
            "sequence_number": 3,
            "item_id": "ig-stream",
            "output_index": 0,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "output_index": 0,
            "item": final_item,
        },
        {
            "type": "response.completed",
            "sequence_number": 5,
            "response": terminal,
        },
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=raw,
        )

    deltas: list[ModelDelta] = []

    async def emit_delta(delta: ModelDelta) -> None:
        deltas.append(delta)

    store = _MemoryOpenAIResponsesArtifactStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=store,
            client=client,
        )
        response = await model.invoke(
            ModelRequest(
                messages=(Message.user("draw"),),
                provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
            ),
            RunContext("run-stream-image", time()),
            stream=True,
            emit_delta=emit_delta,
        )
        unsafe_model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            client=client,
        )
        with pytest.raises(ModelError, match="without an OpenAIResponsesArtifactStore") as caught:
            await unsafe_model.invoke(
                ModelRequest(messages=(Message.user("text only"),)),
                RunContext("run-stream-unrequested", time()),
                stream=True,
                emit_delta=None,
            )

    assert caught.value.info.code == "codec_error"
    call = response.provider_tool_calls()[0]
    assert call.output[0].type == "artifact"
    assert any(
        isinstance(delta, ModelProviderToolCallDelta) and delta.data.get("base64") == _JPEG_BASE64
        for delta in deltas
    )


async def test_openai_responses_stream_rejects_done_sentinel() -> None:
    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: [DONE]\n\n",
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            client=client,
        )
        with pytest.raises(ModelError, match="typed response event") as caught:
            await model.invoke(
                ModelRequest(messages=(Message.user("question"),)),
                RunContext("run-1", time()),
                stream=True,
                emit_delta=None,
            )

    assert caught.value.info.code == "codec_error"


async def test_openai_responses_failed_response_has_same_nonstream_and_stream_semantics() -> None:
    failed_response: dict[str, Any] = {
        "id": "resp-failed",
        "object": "response",
        "model": "gpt-test",
        "status": "failed",
        "error": {"code": "generation_failed", "message": "generation failed"},
        "output": [],
    }

    async def invoke_failed(*, stream: bool) -> ModelErrorInfo:
        async def handler(raw: httpx.Request) -> httpx.Response:
            if not stream:
                return httpx.Response(
                    200,
                    headers={"x-request-id": "request-1"},
                    json=failed_response,
                    request=raw,
                )
            created = {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp-failed",
                    "object": "response",
                    "status": "in_progress",
                },
            }
            failed: dict[str, Any] = {
                "type": "response.failed",
                "sequence_number": 1,
                "response": failed_response,
            }
            body = (
                f"event: response.created\ndata: {json.dumps(created)}\n\n"
                f"event: response.failed\ndata: {json.dumps(failed)}\n\n"
            )
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream",
                    "x-request-id": "request-1",
                },
                content=body,
                request=raw,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAIResponsesModel(
                base_url="https://provider.test/v1",
                api_key="secret",
                model="gpt-test",
                client=client,
            )
            with pytest.raises(ModelError) as caught:
                await model.invoke(
                    ModelRequest(messages=(Message.user("question"),)),
                    RunContext("run-failed", time()),
                    stream=stream,
                    emit_delta=None,
                )
        return caught.value.info

    nonstream = await invoke_failed(stream=False)
    streamed = await invoke_failed(stream=True)

    assert nonstream == streamed
    assert nonstream.code == "generation_failed"
    assert nonstream.provider == "openai-responses"
    assert nonstream.status_code is None
    assert nonstream.request_id == "request-1"
    assert nonstream.metadata == {"response_id": "resp-failed", "status": "failed"}

    async def envelope_handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"code": "plain_error", "message": "plain envelope"}},
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(envelope_handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            client=client,
        )
        with pytest.raises(ModelError) as caught:
            await model.invoke(
                ModelRequest(messages=(Message.user("question"),)),
                RunContext("run-envelope", time()),
                stream=False,
                emit_delta=None,
            )
    assert caught.value.info.code == "plain_error"
