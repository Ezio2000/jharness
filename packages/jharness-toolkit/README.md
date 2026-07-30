# jharness-toolkit

Tool registration, Python function adaptation, JSON Schema validation, retry, and
circuit breaking for JHarness.

```bash
uv add jharness-toolkit
```

```python
from jharness.toolkit import ToolRegistry, function_tool
```

`ToolRegistry` validates tool arguments and, when a tool declares an output schema,
the `structured_content` of every non-failure result. A `ToolFailure` is a framework
error envelope rather than a successful business payload, so its structured content
is not checked against the success schema. `RetryingTool` retries only selected
implementation exceptions and requires an idempotent tool when more than one attempt
is configured. `CircuitBreakingTool` is an in-process policy, not a distributed rate
limiter.

Installing this distribution installs the exact matching `jharness-kernel` version.
