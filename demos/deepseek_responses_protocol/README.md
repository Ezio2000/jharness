# DeepSeek Responses protocol demo

This demo exercises the complete intersection between JHarness's provider-neutral
response protocol and DeepSeek's native Responses API for `deepseek-v4-flash`.
It intentionally separates deterministic offline contract assertions from real,
billable end-to-end calls.

The live run covers:

- non-streaming terminal responses and all seven `ModelResponse` fields;
- the six portable usage counters and v0 response/checkpoint/event wire round-trips;
- SSE reasoning, text, runtime-tool, provider-tool, and usage deltas;
- JSON object and strict JSON Schema output;
- a structured function call followed by local execution and full history replay;
- the only supported freeform custom tool, `apply_patch`, executed in memory without
  modifying the filesystem;
- provider-hosted `web_search`, decoded search lifecycle/actions and source URLs, plus
  a second stateless turn that replays the complete ordered history;
- trace construction and deterministic verification for every Runtime invocation.

The offline contract adds deterministic coverage for request options, all supported
tool choices, both web-search wire variants, refusal/incomplete/failed terminal shapes,
ordered output, and the profile's documented local capability rejections. Each reported
case name is added only after its executable check succeeds; section and total counts are
derived from that run rather than maintained separately. It does not claim that synthetic
failure/refusal paths were observed from the live service.

## Run

Use the `uv`-managed Python environment from the repository root. Never put a key in
this directory or in a checked-in `.env` file.

```bash
export DEEPSEEK_API_KEY='your-key'
uv run python -m demos.deepseek_responses_protocol
```

DeepSeek's canonical base URL is `https://api.deepseek.com`. Override it only for a
compatible proxy:

```bash
DEEPSEEK_BASE_URL='https://api.deepseek.com' \
  uv run python -m demos.deepseek_responses_protocol
```

Run one live scenario while developing:

```bash
uv run python -m demos.deepseek_responses_protocol --only web-search
```

Available scenario names are `basic`, `reasoning`, `structured-output`,
`function-tool`, `apply-patch`, and `web-search`.

Run only the deterministic contract and tests without credentials or network access:

```bash
uv run python -m demos.deepseek_responses_protocol --offline-only
uv run python -O -m demos.deepseek_responses_protocol --offline-only
uv run pytest -q -p no:cacheprovider tests/demos/test_deepseek_responses_protocol.py
```

## Expected result

The command exits nonzero as soon as a required protocol invariant or live coverage
category is missing. A successful full run prints a JSON report whose
`live_coverage.delta_kinds` contains:

```text
content, provider_tool_call, reasoning, tool_call, usage
```

and whose output coverage contains text, reasoning, structured/freeform runtime calls,
and provider-hosted web-search calls. Search results are provider-owned: JHarness keeps
each `web_search_call` in ordered history and preserves URL citation annotations in the
following text part's Responses metadata when DeepSeek emits them. The live scenario
also verifies the canonical source URL in visible output because annotations are not
guaranteed. Likewise, the live service may omit tool-specific lifecycle events; the
scenario always requires the terminal `output_item.done` delta, while the offline
raw-SSE regression covers DeepSeek's `web_search_call.completed` variant. Provider calls
are not scheduled as local tool work.

## Protocol boundary

DeepSeek Responses is stateless. The adapter omits `store`, `include`, and
`previous_response_id`; every subsequent request resends the complete ordered history,
including reasoning, runtime tool calls/results, provider search items, and native
message metadata.

The strict JHarness DeepSeek profile supports text input/output, structured functions,
the exact freeform tool name `apply_patch`, JSON modes, and hosted web search. It rejects
image/file input, `seed`, stop sequences, other custom tools, exact custom-tool choice,
and requests to disable parallel runtime calls. Thinking efforts other than `none` only
allow automatic or disabled tool choice, so forced tool/search scenarios use
`effort="none"` while the reasoning scenario uses `effort="high"`.

Upstream references:

- [DeepSeek Responses API guide](https://api-docs.deepseek.com/guides/responses_api/)
- [DeepSeek Create Response API reference](https://api-docs.deepseek.com/api/create-response)
- [JHarness model adapter documentation](../../docs/model-adapters.md)
