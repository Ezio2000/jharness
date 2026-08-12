from __future__ import annotations

import importlib
import subprocess
import sys
from importlib.util import find_spec
from typing import get_args

import conformance
import jharness.kernel as kernel
import jharness.kernel.diagnostics as diagnostics
import jharness.kernel.wire as wire
import jharness.models as models
import jharness.repository as repository
import jharness.toolkit as toolkit
import jharness.tools as tools


def test_package_all_exports_exist() -> None:
    for module in (kernel, toolkit, tools, repository, diagnostics, wire, conformance):
        exports = set(module.__all__)
        assert exports
        assert all(hasattr(module, name) for name in exports)
    assert models.__all__ == []


def test_repository_root_exports_all_supported_backends() -> None:
    assert set(repository.__all__) == {
        "MemoryRunRepository",
        "MySQLRunRepository",
        "MySQLTLS",
        "RedisRunRepository",
        "SQLiteRunRepository",
    }


def test_repository_base_import_and_embedded_backends_need_no_optional_drivers() -> None:
    program = """
import asyncio
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "pymysql" or name == "redis" or name.startswith("redis.")):
        raise AssertionError(f"base repository import loaded optional driver: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from jharness.repository import (
    MemoryRunRepository,
    MySQLRunRepository,
    MySQLTLS,
    RedisRunRepository,
    SQLiteRunRepository,
)

async def verify():
    memory = MemoryRunRepository()
    assert await memory.get_head("missing") is None
    sqlite = SQLiteRunRepository(":memory:")
    await sqlite.initialize()
    assert await sqlite.get_head("missing") is None
    await sqlite.close()

asyncio.run(verify())
assert MySQLTLS(ca="mysql-ca.pem").verify_identity is True
assert MySQLRunRepository and RedisRunRepository
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_kernel_root_contains_the_documented_protocol_families() -> None:
    required = {
        "Runtime",
        "Invocation",
        "Checkpoint",
        "RunSnapshot",
        "Planning",
        "ToolsPending",
        "Suspended",
        "Completed",
        "Failed",
        "Limited",
        "RunRepository",
        "Model",
        "ToolCatalogProvider",
        "ApprovalPolicy",
        "HistoryReducer",
        "BatchPolicy",
    }
    assert required <= set(kernel.__all__)
    assert {"build_trace", "verify_trace", "RunTrace"} <= set(diagnostics.__all__)
    assert {"encode_checkpoint", "decode_checkpoint", "StartRequest"} <= set(wire.__all__)


def test_only_documented_model_namespaces_are_public() -> None:
    decorators = importlib.import_module("jharness.models.decorators")
    assert set(decorators.__all__) == {"FallbackModel", "RetryingModel"}
    expected_exports = {
        "jharness.models.openai": {
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
        "jharness.models.anthropic": {
            "AnthropicMessagesCodec",
            "AnthropicMessagesError",
            "AnthropicMessagesModel",
            "AnthropicMessagesProfile",
            "AnthropicMessagesServerToolCodec",
            "AnthropicMessagesServerToolRegistry",
            "anthropic_messages_web_search_codec",
        },
        "jharness.models.deepseek": {
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
    }
    for namespace, expected in expected_exports.items():
        module = importlib.import_module(namespace)
        assert set(module.__all__) == expected
        assert all(hasattr(module, name) for name in expected)

    deepseek = importlib.import_module("jharness.models.deepseek")
    assert frozenset(get_args(deepseek.DeepSeekThinkingEffort)) == frozenset({"high", "max"})
    assert frozenset(get_args(deepseek.DeepSeekResponsesEffort)) == frozenset(
        {"none", "low", "high", "xhigh", "max"}
    )

    legacy_exports = {
        "jharness.models.openai": {
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
        "jharness.models.anthropic": {
            "AnthropicCodec",
            "AnthropicError",
            "AnthropicModel",
            "AnthropicProfile",
            "AnthropicServerToolCodec",
            "AnthropicServerToolRegistry",
            "anthropic_web_search_codec",
        },
        "jharness.models.deepseek": {
            "DEEPSEEK_ANTHROPIC_WEB_SEARCH",
            "deepseek_anthropic_profile",
            "deepseek_anthropic_web_search",
            "deepseek_openai_chat_profile",
            "deepseek_openai_responses_profile",
        },
    }
    for namespace, legacy in legacy_exports.items():
        module = importlib.import_module(namespace)
        assert all(not hasattr(module, name) for name in legacy)

    for implementation in (
        "jharness.models.anthropic.messages",
        "jharness.models.openai.chat",
        "jharness.models.openai.responses",
    ):
        assert importlib.import_module(implementation).__all__ == []

    for legacy_module in (
        "jharness.models.anthropic.errors",
        "jharness.models.anthropic.messages_api",
        "jharness.models.anthropic.profiles",
        "jharness.models.openai.chat_completions",
        "jharness.models.openai.errors",
        "jharness.models.openai.profiles",
        "jharness.models.openai.responses_api",
    ):
        assert find_spec(legacy_module) is None
