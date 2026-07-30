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
its result's `structured_content`. `RetryingTool` retries only selected implementation
exceptions and requires an idempotent tool when more than one attempt is configured.
`CircuitBreakingTool` is an in-process policy, not a distributed rate limiter.

Installing this distribution installs the exact matching `jharness-kernel` version.
