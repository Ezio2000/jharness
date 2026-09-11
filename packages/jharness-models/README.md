# jharness-models

OpenAI Chat (Chat Completions API), OpenAI Responses, and Anthropic Messages adapters,
plus retry/fallback composition for the JHarness kernel.

```bash
uv add jharness-models
```

| Import namespace | Adapter | Runtime tools | Official hosted-tool presets |
| --- | --- | --- | --- |
| `jharness.models.openai` | `OpenAIChatModel` | Function calls | None |
| `jharness.models.openai` | `OpenAIResponsesModel` | Function and custom calls | Web search, image generation |
| `jharness.models.anthropic` | `AnthropicMessagesModel` | Client `tool_use` | Web search |

All adapters return one ordered `ModelResponse.output` of content, runtime calls, and
provider calls. JHarness executes runtime calls; hosted calls record provider execution.
Native model modalities are declared separately from hosted-tool output.

```python
from jharness.kernel import Runtime
from jharness.models.openai import (
    OpenAIResponsesModel,
    openai_responses_profile,
    openai_responses_web_search,
)

model = OpenAIResponsesModel(..., profile=openai_responses_profile())
runtime = Runtime(model=model, provider_tools=(openai_responses_web_search(),))
```

Each profile owns the immutable `ModelCapabilities` returned by its client. Generic
Responses and Messages profiles enable no hosted tools. Official profile factories
declare hosted-tool support; `provider_tools` explicitly enables tools for the run.
Capabilities are host assertions for the selected endpoint and model, not live discovery.
Profiles configure documented protocol behavior and cannot inject alternate codecs.

Responses defaults to `store=false` with encrypted reasoning for native-history replay.
Hosted image generation requires a host-owned `OpenAIResponsesArtifactStore`: saves
must be durable and idempotent, references stable through recovery, and uncommitted
saves eventually collected. Checkpoints retain integrity-bearing `ArtifactRef` values.

Retry and fallback compose around any kernel `Model`:

```python
from jharness.models.decorators import FallbackModel, RetryingModel

model = FallbackModel(
    RetryingModel(primary_model, max_attempts=3),
    RetryingModel(backup_model, max_attempts=2),
)
```

Installation includes the exact matching `jharness-kernel` version. See the
[model adapter guide](https://github.com/Ezio2000/jharness/blob/main/docs/model-adapters.md)
for provider setup, capability validation, media mappings, artifact storage, and transport rules.
