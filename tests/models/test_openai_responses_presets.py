from __future__ import annotations

from time import time
from typing import Any

import httpx
import pytest

from jharness.kernel import Message, ModelRequest, RunContext, ToolChoice
from jharness.models.openai import (
    OPENAI_RESPONSES_IMAGE_GENERATION,
    OPENAI_RESPONSES_WEB_SEARCH,
    OpenAIResponsesCodec,
    OpenAIResponsesError,
    OpenAIResponsesImageGenerationTool,
    OpenAIResponsesModel,
    OpenAIResponsesProfile,
    OpenAIResponsesWebSearchTool,
    openai_responses_image_generation,
    openai_responses_profile,
    openai_responses_web_search,
)


def test_openai_responses_hosted_tool_declarations_use_stable_identities() -> None:
    web_search = openai_responses_web_search()
    image_generation = openai_responses_image_generation(
        {"quality": "high", "output_format": "png"}
    )

    assert OPENAI_RESPONSES_WEB_SEARCH.namespace == "openai.responses"
    assert OPENAI_RESPONSES_WEB_SEARCH.type == "web_search"
    assert web_search.tool is OPENAI_RESPONSES_WEB_SEARCH
    assert web_search.configuration == {}
    assert OPENAI_RESPONSES_IMAGE_GENERATION.namespace == "openai.responses"
    assert OPENAI_RESPONSES_IMAGE_GENERATION.type == "image_generation"
    assert image_generation.tool is OPENAI_RESPONSES_IMAGE_GENERATION
    assert image_generation.configuration == {
        "quality": "high",
        "output_format": "png",
    }


def test_openai_responses_hosted_tool_declarations_deep_freeze_configuration() -> None:
    web_configuration: dict[str, Any] = {
        "filters": {"allowed_domains": ["example.com"]},
    }
    image_configuration: dict[str, Any] = {
        "input_image_mask": {
            "image_url": "https://example.com/mask.png",
            "metadata": {"layers": [1]},
        }
    }

    web_search = openai_responses_web_search(web_configuration)
    image_generation = openai_responses_image_generation(image_configuration)
    web_configuration["filters"]["allowed_domains"].append("mutated.example")
    image_configuration["input_image_mask"]["metadata"]["layers"].append(2)

    assert web_search.configuration == {
        "filters": {"allowed_domains": ["example.com"]},
    }
    assert image_generation.configuration == {
        "input_image_mask": {
            "image_url": "https://example.com/mask.png",
            "metadata": {"layers": [1]},
        }
    }
    with pytest.raises(TypeError, match="provider tool configuration is immutable"):
        web_search.configuration["filters"]["allowed_domains"].append("blocked.example")
    with pytest.raises(TypeError, match="provider tool configuration is immutable"):
        image_generation.configuration["input_image_mask"]["metadata"]["layers"].append(2)


def test_openai_responses_official_profile_installs_both_hosted_tool_codecs() -> None:
    profile = openai_responses_profile()

    assert profile.name == "openai-responses"
    assert profile.capabilities.provider_tools == frozenset(
        {
            OPENAI_RESPONSES_WEB_SEARCH,
            OPENAI_RESPONSES_IMAGE_GENERATION,
        }
    )
    assert "provider" in profile.capabilities.tool_choice_types
    assert tuple(type(codec) for codec in profile.provider_tool_registry.codecs) == (
        OpenAIResponsesWebSearchTool,
        OpenAIResponsesImageGenerationTool,
    )
    assert (
        profile.provider_tool_registry.codec_for_tool(OPENAI_RESPONSES_WEB_SEARCH).tool
        is OPENAI_RESPONSES_WEB_SEARCH
    )
    assert (
        profile.provider_tool_registry.codec_for_tool(OPENAI_RESPONSES_IMAGE_GENERATION).tool
        is OPENAI_RESPONSES_IMAGE_GENERATION
    )


def test_openai_responses_base_profile_remains_provider_neutral() -> None:
    profile = OpenAIResponsesProfile()

    assert profile.capabilities.provider_tools == frozenset()
    assert "provider" not in profile.capabilities.tool_choice_types
    assert profile.provider_tool_registry.codecs == ()


def test_generic_openai_responses_profile_rejects_hosted_tool_presets() -> None:
    request = ModelRequest(
        messages=(Message.user("search the web"),),
        provider_tools=(openai_responses_web_search(),),
    )

    with pytest.raises(
        OpenAIResponsesError,
        match=r"profile does not support provider tool: openai\.responses/web_search",
    ):
        OpenAIResponsesCodec(
            model="gpt-test",
            profile=OpenAIResponsesProfile(),
        ).encode_request(request)


def test_openai_responses_official_profile_encodes_only_explicitly_enabled_tools() -> None:
    profile = openai_responses_profile()
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)

    without_tools = codec.encode_request(ModelRequest(messages=(Message.user("hello"),)))
    with_web_search = codec.encode_request(
        ModelRequest(
            messages=(Message.user("find current information"),),
            provider_tools=(openai_responses_web_search({"search_context_size": "high"}),),
            tool_choice=ToolChoice(
                type="provider",
                provider_tool=OPENAI_RESPONSES_WEB_SEARCH,
            ),
        )
    )

    assert "tools" not in without_tools
    assert "tool_choice" not in without_tools
    assert with_web_search["tools"] == [{"type": "web_search", "search_context_size": "high"}]
    assert with_web_search["tool_choice"] == {"type": "web_search"}


async def test_openai_responses_image_artifacts_are_required_only_when_explicitly_enabled() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "object": "response",
                "created_at": 1,
                "completed_at": 2,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "model": "gpt-test",
                "output": [
                    {
                        "id": "msg-1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hello",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "previous_response_id": None,
                "store": False,
                "tools": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=openai_responses_profile(),
            client=client,
        )
        context = RunContext("run-1", time())
        response = await model.invoke(
            ModelRequest(messages=(Message.user("hello"),)),
            context,
            stream=False,
            emit_delta=None,
        )

        assert response.visible_parts()[0].text == "hello"
        assert len(requests) == 1

        with pytest.raises(ValueError, match="OpenAIResponsesArtifactStore"):
            await model.invoke(
                ModelRequest(
                    messages=(Message.user("draw a lighthouse"),),
                    provider_tools=(openai_responses_image_generation(),),
                ),
                context,
                stream=False,
                emit_delta=None,
            )

    assert len(requests) == 1
