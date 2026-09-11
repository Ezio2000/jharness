# jharness-toolkit

Tool registration, Python function adaptation, JSON Schema validation, retry, and
circuit breaking for JHarness.

```bash
uv add jharness-toolkit
```

```python
from jharness.kernel import Runtime, ToolContext
from jharness.toolkit import ToolRegistry, function_tool


@function_tool(
    name="query_order",
    description="Look up an order's shipping status.",
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
    output_schema={"type": "string"},
    context_parameter="ctx",
)
async def query_order(order_id: str, *, ctx: ToolContext) -> str:
    await ctx.emit_progress({"message": "Looking up the order"})
    return "shipped" if order_id == "A1001" else "not found"


# Use an already configured Model implementation.
runtime = Runtime(model=model, tools=ToolRegistry((query_order,)))
```

## Function adapters

`function_tool` passes a fresh, mutable copy of the JSON object arguments to an async
function by parameter name. Positional-or-keyword and keyword-only parameters are
supported; positional-only parameters, `*args`, and `**kwargs` are rejected when the
adapter is constructed. Omitted optional parameters use Python defaults. JSON Schema
`default` annotations do not insert values. Missing required parameters and unexpected
arguments fail before the business function executes.

Keep `name`, `description`, and `input_schema` explicit. Function annotations neither
generate schemas nor coerce inputs. `ToolRegistry` validates the declared input schema
before approval; the adapter checks Python argument binding when invoked. It does not
attempt to prove equivalence between a JSON Schema and a Python signature. A direct
`tool.invoke()` call performs argument binding and result adaptation; use a registry
binding when JSON Schema validation is required.

Context injection is opt-in. Set `context_parameter="ctx"` and declare a keyword-only
`ctx` parameter to receive the current `ToolContext`; omit both when context is not
needed. Names and type annotations do not trigger injection. The input schema must
not declare this parameter: direct `properties` / `required` declarations are rejected
at construction, and any supplied argument with that name is rejected at invocation,
including when schemas use references or composition. The adapter never overwrites a
model-supplied argument with privileged context.

Both adapters use the same return rules:

| Return value | Model-visible success content | `structured_content` |
| --- | --- | --- |
| `str`, including an empty string | The exact string | The string |
| Other JSON-compatible values, including `None` | Compact JSON with sorted keys and unescaped Unicode | The value, frozen by kernel |
| `SettledResult` or `WaitingResult` | Passed through unchanged | Passed through unchanged |

JSON compatibility follows kernel's copying/freezing rules, including mappings with
string keys and JSON-compatible sequences. Bytes, sets, arbitrary objects, non-finite
numbers, and cyclic values are rejected; they are never converted with `str()`.
Unwrapped outcomes such as `ToolSuccess` are not results: wrap them in `SettledResult`.
Native results preserve failures, background acknowledgements, multimodal content, and
waiting suspensions. Business exceptions propagate to retry/circuit-breaker decorators
and the runtime; cancellation remains control flow.

`freeform_tool` accepts exactly one named business parameter containing the original
input string, with the same optional context injection and return rules:

```python
from jharness.toolkit import freeform_tool


@freeform_tool(name="count_lines", description="Count lines in the supplied text.")
async def count_lines(text: str) -> dict[str, int]:
    return {"lines": len(text.splitlines())}
```

Direct construction uses `FunctionTool(spec, function, context_parameter="ctx")` or
`FreeformFunctionTool(spec, function, context_parameter="ctx")`; `context_parameter`
defaults to `None`. Both accept the public `ToolFunction` callable type. The adapter's
`invoke(call, context)` remains the kernel-facing execution port. Business callbacks
receive only their named input values and explicitly injected context. Tools that need
the raw call ID or metadata implement `Tool` or `FreeformTool` directly.

Runnable examples cover [a basic tool loop](../../examples/basic_tool_loop.py) and
[native waiting results with checkpoint recovery](../../examples/pause_resume_trace.py).

## Validation and execution policies

`ToolRegistry` validates tool arguments and, when a tool declares an output schema,
the `structured_content` of every non-failure result. A `ToolFailure` is a framework
error envelope rather than a successful business payload, so its structured content
is not checked against the success schema. `RetryingTool` retries only selected
implementation exceptions and requires an idempotent tool when more than one attempt
is configured. `CircuitBreakingTool` is an in-process policy, not a distributed rate
limiter.

Installing this distribution installs the exact matching `jharness-kernel` version.
