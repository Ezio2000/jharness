"""DeepSeek provider profiles."""

from jharness.models.deepseek.profiles import (
    DeepSeekResponsesEffort,
    deepseek_anthropic_profile,
    deepseek_openai_chat_profile,
    deepseek_openai_responses_profile,
)
from jharness.models.deepseek.tools import (
    DEEPSEEK_ANTHROPIC_WEB_SEARCH,
    DEEPSEEK_RESPONSES_WEB_SEARCH,
    deepseek_anthropic_web_search,
    deepseek_responses_web_search,
)

__all__ = [
    "DEEPSEEK_ANTHROPIC_WEB_SEARCH",
    "DEEPSEEK_RESPONSES_WEB_SEARCH",
    "DeepSeekResponsesEffort",
    "deepseek_anthropic_profile",
    "deepseek_anthropic_web_search",
    "deepseek_openai_chat_profile",
    "deepseek_openai_responses_profile",
    "deepseek_responses_web_search",
]
