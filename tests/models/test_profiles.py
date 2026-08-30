from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest

from jharness.kernel import ModelCapabilities
from jharness.models._http import model_client_config
from jharness.models.anthropic import AnthropicMessagesModel, AnthropicMessagesProfile
from jharness.models.openai import (
    OpenAIChatModel,
    OpenAIChatProfile,
    OpenAIResponsesProfile,
)


@pytest.mark.parametrize("model_type", [OpenAIChatModel, AnthropicMessagesModel])
def test_model_clients_share_constructor_validation(model_type: type[object]) -> None:
    constructor = cast(Any, model_type)
    for keywords, pattern in (
        ({"base_url": "", "api_key": "secret", "model": "model"}, "base_url"),
        ({"base_url": "https://x", "api_key": "", "model": "model"}, "api_key"),
        ({"base_url": "https://x", "api_key": "secret", "model": ""}, "model"),
    ):
        with pytest.raises(ValueError, match=pattern):
            constructor(**keywords)
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        constructor(
            base_url="https://provider.test",
            api_key="secret",
            model="model",
            unknown=True,
        )
    configured = constructor(
        base_url="https://provider.test/",
        api_key="secret",
        model="model",
    )
    assert configured.base_url == "https://provider.test"
    assert not hasattr(configured, "api_key")
    assert configured._api_key == "secret"
    assert isinstance(configured._timeout, httpx.Timeout)
    assert configured._timeout.connect == 10.0
    assert configured._timeout.read == 60.0

    without_transport_timeout = constructor(
        base_url="https://provider.test",
        api_key="secret",
        model="model",
        timeout=None,
    )
    assert without_transport_timeout._timeout is None

    for keywords, pattern in (
        ({"max_response_body_bytes": 0}, "max_response_body_bytes"),
        ({"max_sse_line_bytes": 0}, "max_sse_line_bytes"),
        (
            {"max_sse_line_bytes": 20, "max_sse_event_bytes": 10},
            "max_sse_event_bytes",
        ),
    ):
        with pytest.raises(ValueError, match=pattern):
            constructor(
                base_url="https://provider.test",
                api_key="secret",
                model="model",
                **keywords,
            )


def test_shared_transport_config_repr_redacts_api_key() -> None:
    config = model_client_config(
        base_url="https://provider.test",
        api_key="repr-secret",
        model="model",
        options={},
        default_profile=object(),
        constructor_name="test",
    )

    assert config.api_key == "repr-secret"
    assert "repr-secret" not in repr(config)


@pytest.mark.parametrize(
    ("factory", "expected_name"),
    (
        pytest.param(OpenAIChatProfile, "openai-chat", id="openai-chat"),
        pytest.param(OpenAIResponsesProfile, "openai-responses", id="openai-responses"),
        pytest.param(
            AnthropicMessagesProfile,
            "anthropic-messages",
            id="anthropic-messages",
        ),
    ),
)
def test_profile_names_are_short_and_protocol_consistent(
    factory: Callable[[], Any],
    expected_name: str,
) -> None:
    assert factory().name == expected_name


def test_openai_chat_profile_validates_every_configuration_family() -> None:
    invalid: tuple[tuple[dict[str, Any], type[Exception], str], ...] = (
        ({"name": ""}, ValueError, "profile name"),
        ({"capabilities": 1}, TypeError, "must be ModelCapabilities"),
        ({"json_schema_name": ""}, ValueError, "json_schema_name"),
        (
            {
                "capabilities": replace(
                    OpenAIChatProfile().capabilities,
                    input_modalities=frozenset({"video"}),
                )
            },
            ValueError,
            "input modality",
        ),
    )
    for keywords, error, pattern in invalid:
        with pytest.raises(error, match=pattern):
            OpenAIChatProfile(**cast(Any, keywords))


def test_anthropic_messages_profile_validates_every_configuration_family() -> None:
    invalid: tuple[tuple[dict[str, Any], type[Exception], str], ...] = (
        ({"name": ""}, ValueError, "profile name"),
        ({"anthropic_version": ""}, ValueError, "anthropic_version"),
        ({"capabilities": 1}, TypeError, "must be ModelCapabilities"),
        (
            {
                "capabilities": replace(
                    AnthropicMessagesProfile().capabilities,
                    seed=True,
                )
            },
            ValueError,
            "does not support seed",
        ),
        ({"default_max_tokens": True}, TypeError, "must be an integer"),
        ({"default_max_tokens": 0}, ValueError, "must be >= 1"),
        ({"json_object_schema": 1}, TypeError, "must be a mapping"),
    )
    for keywords, error, pattern in invalid:
        with pytest.raises(error, match=pattern):
            AnthropicMessagesProfile(**cast(Any, keywords))


def test_profiles_expose_only_the_new_capability_contract() -> None:
    responses = OpenAIResponsesProfile()
    assert responses.capabilities.input_modalities == frozenset({"text"})
    assert responses.capabilities.provider_tools == frozenset()
    assert responses.capabilities.structured_output is False
    assert responses.store is False
    assert responses.include == frozenset({"reasoning.encrypted_content"})

    profiles = (
        OpenAIChatProfile(),
        responses,
        AnthropicMessagesProfile(),
    )
    removed_fields = (
        "supports_tools",
        "supports_tool_choice",
        "supports_image_input",
        "supports_file_input",
        "supports_json_schema",
        "supports_seed",
    )

    for profile in profiles:
        assert isinstance(profile.capabilities, ModelCapabilities)
        assert all(not hasattr(profile, field) for field in removed_fields)
