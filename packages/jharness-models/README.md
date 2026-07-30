# jharness-models

OpenAI Chat Completions and Anthropic Messages adapters, DeepSeek profiles, and
provider-neutral model composition for the JHarness kernel.

```bash
uv add jharness-models
```

```python
from jharness.models.openai import OpenAIChatCompletionsModel
```

Retry and fallback compose directly around model values:

```python
from jharness.models.decorators import FallbackModel, RetryingModel

model = FallbackModel(
    RetryingModel(primary_model, max_attempts=3),
    RetryingModel(backup_model, max_attempts=2),
)
```

Installing this distribution installs the exact matching `jharness-kernel` version.
Provider configuration and composition details are in the
[model adapter guide](https://github.com/Ezio2000/jharness/blob/main/docs/model-adapters.md).
