# JHarness

JHarness is a provider-neutral Python 3.11+ runtime for durable model and tool
execution. Its five distributions are independently installable and released at the
same version.

## Install

Install only what the application uses:

| Distribution | Command | Namespace | Purpose |
| --- | --- | --- | --- |
| `jharness-kernel` | `uv add jharness-kernel` | `jharness.kernel` | Runtime, state, ports, checkpoints, and wire codecs |
| `jharness-toolkit` | `uv add jharness-toolkit` | `jharness.toolkit` | Tool registry, function adapters, validation, and policies |
| `jharness-models` | `uv add jharness-models` | `jharness.models` | OpenAI Chat/Responses and Anthropic Messages adapters, DeepSeek profiles, retry, and fallback |
| `jharness-repository` | `uv add jharness-repository` | `jharness.repository` | Memory, SQLite, MySQL, and Redis persistence |
| `jharness-tools` | `uv add jharness-tools` | `jharness.tools` | Filesystem, shell, interaction, and child-agent tools |

Every non-kernel distribution installs the exact matching kernel. MySQL and Redis
drivers are opt-in:

```bash
uv add "jharness-repository[mysql]"
uv add "jharness-repository[redis]"
```

## Quick Start

```python
import asyncio
import os

from jharness.kernel import Completed, Message, Runtime
from jharness.models.openai import OpenAIChatCompletionsModel


async def main() -> None:
    model = OpenAIChatCompletionsModel(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["OPENAI_MODEL"],
    )
    checkpoint = await Runtime(model=model).start(
        (Message.user("Say hello in one short sentence."),)
    ).result()
    state = checkpoint.snapshot.state
    if not isinstance(state, Completed):
        raise RuntimeError(f"run stopped with {checkpoint.snapshot.status}")
    print("".join(part.text or "" for part in state.parts))


asyncio.run(main())
```

Models and tools compose at the kernel boundary:

```python
from pathlib import Path

from jharness.kernel import Runtime
from jharness.models.decorators import FallbackModel, RetryingModel
from jharness.toolkit import ToolRegistry
from jharness.tools import GlobTool, GrepTool, LsTool, ReadTool

model = FallbackModel(
    RetryingModel(primary_model, max_attempts=3),
    RetryingModel(backup_model, max_attempts=2),
)
root = Path.cwd()
tools = ToolRegistry((ReadTool(root), LsTool(root), GlobTool(root), GrepTool(root)))
runtime = Runtime(model=model, tools=tools)
```

Model output is one ordered sequence of content, runtime-owned tool calls, and
provider-owned tool calls. Only runtime-owned `RuntimeToolCall` values enter JHarness
tool scheduling; hosted calls such as Responses image generation or web search are
executed by the provider and recorded as `ProviderToolCall` values. Exact input/output
modalities are advertised independently by each model profile.

Provider setup is covered in [model adapters](docs/model-adapters.md); additional
runnable examples live in [`examples`](examples/).

## Durability

Every durable boundary creates an immutable `Checkpoint`; `Invocation.result()` returns
the last committed one. `jharness.kernel.wire` encodes portable JSON, while an optional
`RunRepository` stores atomic, revision-checked commits. See the
[v0 contracts](contracts/v0/README.md) for normative behavior and
[repository implementations](docs/repositories.md) for storage choices.

## Core Documentation

| Need | Read |
| --- | --- |
| Architecture and extension boundaries | [Architecture](docs/architecture.md) |
| Provider adapters, profiles, retry, and fallback | [Model adapters](docs/model-adapters.md) |
| Memory, SQLite, MySQL, and Redis | [Repositories](docs/repositories.md) |
| Portable behavior | [Contracts](contracts/v0/README.md) |
