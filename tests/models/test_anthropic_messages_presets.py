from __future__ import annotations

from typing import Any

import pytest

from jharness.kernel import Message, ModelRequest, ToolChoice
from jharness.models.anthropic import (
    ANTHROPIC_MESSAGES_WEB_SEARCH,
    AnthropicMessagesCodec,
    AnthropicMessagesError,
    AnthropicMessagesProfile,
    anthropic_messages_profile,
    anthropic_messages_web_search,
)


def test_anthropic_messages_preset_installs_web_search_without_changing_generic_profile() -> None:
    generic = AnthropicMessagesProfile()
    official = anthropic_messages_profile()

    assert generic.capabilities.provider_tools == frozenset()
    assert "provider" not in generic.capabilities.tool_choice_types
    assert generic.server_tools.tools == frozenset()

    assert official.name == "anthropic-messages"
    assert official.capabilities.provider_tools == frozenset({ANTHROPIC_MESSAGES_WEB_SEARCH})
    assert official.capabilities.tool_choice_types == (
        generic.capabilities.tool_choice_types | {"provider"}
    )
    assert official.server_tools.tools == frozenset({ANTHROPIC_MESSAGES_WEB_SEARCH})


def test_anthropic_messages_web_search_factory_owns_configuration() -> None:
    configuration: dict[str, Any] = {
        "allowed_domains": ["example.com"],
        "max_uses": 2,
    }
    spec = anthropic_messages_web_search(configuration)
    configuration["allowed_domains"] = ["changed.example"]

    assert spec.tool == ANTHROPIC_MESSAGES_WEB_SEARCH
    assert spec.configuration == {
        "allowed_domains": ("example.com",),
        "max_uses": 2,
    }
    assert anthropic_messages_web_search().configuration == {}


def test_anthropic_messages_preset_encodes_web_search_and_exact_choice() -> None:
    profile = anthropic_messages_profile()
    spec = anthropic_messages_web_search(
        {
            "allowed_domains": ["example.com"],
            "max_uses": 2,
        }
    )
    request = ModelRequest(
        messages=(Message.user("Search the web"),),
        provider_tools=(spec,),
        tool_choice=ToolChoice(
            type="provider",
            provider_tool=ANTHROPIC_MESSAGES_WEB_SEARCH,
        ),
    )

    payload = AnthropicMessagesCodec(model="claude", profile=profile).encode_request(request)

    assert payload["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["example.com"],
            "max_uses": 2,
        }
    ]
    assert payload["tool_choice"] == {"type": "tool", "name": "web_search"}


@pytest.mark.parametrize(
    "variant",
    (
        "web_search_20250305",
        "web_search_20260209",
        "web_search_20260318",
    ),
)
def test_anthropic_messages_preset_supports_current_web_search_variants(
    variant: str,
) -> None:
    configuration: dict[str, Any] = {"variant": variant}
    if variant == "web_search_20260318":
        configuration["response_inclusion"] = "all"
    request = ModelRequest(
        messages=(Message.user("Search the web"),),
        provider_tools=(anthropic_messages_web_search(configuration),),
    )

    payload = AnthropicMessagesCodec(
        model="claude",
        profile=anthropic_messages_profile(),
    ).encode_request(request)

    expected = {"type": variant, "name": "web_search"}
    if variant == "web_search_20260318":
        expected["response_inclusion"] = "all"
    assert payload["tools"] == [expected]


def test_anthropic_messages_profile_does_not_enable_web_search_by_itself() -> None:
    payload = AnthropicMessagesCodec(
        model="claude",
        profile=anthropic_messages_profile(),
    ).encode_request(ModelRequest(messages=(Message.user("Hello"),)))

    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]


def test_generic_anthropic_messages_profile_rejects_web_search_spec() -> None:
    request = ModelRequest(
        messages=(Message.user("Search the web"),),
        provider_tools=(anthropic_messages_web_search(),),
    )

    with pytest.raises(
        AnthropicMessagesError,
        match=r"profile does not support provider tool: anthropic\.messages/web_search",
    ):
        AnthropicMessagesCodec(
            model="claude",
            profile=AnthropicMessagesProfile(),
        ).encode_request(request)
