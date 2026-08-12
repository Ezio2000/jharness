"""Run one invocation against an OpenAI Chat Completions endpoint.

Required environment variables:

  OPENAI_CHAT_BASE_URL   Example: https://api.example.com/v1
  OPENAI_CHAT_API_KEY
  OPENAI_CHAT_MODEL

Optional environment variables:

  OPENAI_CHAT_PROFILE_NAME
  OPENAI_CHAT_PROMPT
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace

from jharness.kernel import Completed, Message, Runtime
from jharness.models.openai import OpenAIChatModel, OpenAIChatProfile


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


async def main() -> None:
    default_profile = OpenAIChatProfile()
    model = OpenAIChatModel(
        base_url=env_required("OPENAI_CHAT_BASE_URL"),
        api_key=env_required("OPENAI_CHAT_API_KEY"),
        model=env_required("OPENAI_CHAT_MODEL"),
        profile=OpenAIChatProfile(
            name=os.environ.get("OPENAI_CHAT_PROFILE_NAME", "openai-chat"),
            capabilities=replace(
                default_profile.capabilities,
                input_modalities=frozenset({"text"}),
            ),
        ),
    )
    prompt = os.environ.get("OPENAI_CHAT_PROMPT", "Say hello in one short sentence.")
    checkpoint = await Runtime(model=model).start((Message.user(prompt),)).result()
    state = checkpoint.snapshot.state
    if not isinstance(state, Completed):
        raise RuntimeError(f"run stopped with {checkpoint.snapshot.status}")
    print("".join(part.text or "" for part in state.parts))


if __name__ == "__main__":
    asyncio.run(main())
