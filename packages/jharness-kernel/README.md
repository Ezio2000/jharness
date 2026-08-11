# jharness-kernel

The dependency-free JHarness runtime kernel, including lifecycle values, execution,
checkpoints, diagnostics, extension ports, and portable wire codecs.

```bash
uv add jharness-kernel
```

```python
from jharness.kernel import Message, Runtime
```

Kernel owns persistent `RunHistory`, incremental `DurableCommit`, cursor-based pending
runtime tool calls, and complete-history model requests. It has no runtime dependency.

Its model boundary keeps media, execution ownership, and protocol order separate:

| Concern | Public values | Meaning |
| --- | --- | --- |
| Modalities | `ModelCapabilities.input_modalities` and `output_modalities` | What media the model itself can understand or produce |
| Tool selection | `ModelCapabilities.tool_choice_types`, `parallel_tool_calls`, and `parallel_tool_call_control` | Which selection policies and parallel behavior the model can honor |
| Host-executed tools | `ToolSpec` through `ModelRequest.runtime_tools`, followed by `ToolCall` | JHarness schedules approval and execution through the host tool catalog |
| Provider-executed tools | `ProviderToolId`, `ProviderToolSpec`, and `ProviderToolCall` | The remote provider runs the tool; JHarness validates and records the result but never schedules it |
| Response order | `ModelResponse.output` and assistant `Message.output` | `ContentPart`, `ToolCall`, and `ProviderToolCall` remain in the order emitted by the protocol |

The kernel also advertises structured output, JSON mode, seed, streaming, and usage
reporting directly on `ModelCapabilities`. The runtime validates requests against this
single immutable declaration before model invocation.

Streaming deltas identify their destination with `output_index` and optional
`content_index`; the terminal `ModelResponse` remains the authoritative durable value.
Only runtime-owned `ToolCall` items can move a run into `ToolsPending`.

The source and portable contracts are maintained in the
[JHarness repository](https://github.com/Ezio2000/jharness).
