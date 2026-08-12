# Changelog

All notable changes to the JHarness Python project are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Treated DeepSeek Responses web-search terminal lifecycle events as advisory until
  `response.output_item.done` finalizes the live status, allowing failed `open_page`
  and `find_in_page` actions to coexist with successful search actions in one strict
  SSE stream. Responses streaming now also rejects nonterminal
  `output_item.done` statuses and terminal provider-tool statuses that contradict the
  closed item.

## [0.6.0] - 2026-08-12

### Added

- Added an OpenAI Responses adapter with complete and SSE transports, runtime
  structured/freeform tools, provider-hosted image generation and web search, and a
  strict stateless `deepseek-v4-flash` Responses profile.
- Added protocol-owned hosted-tool registries, DeepSeek Responses `apply_patch` and
  versioned web search, and Anthropic server web-search request/terminal/SSE/history
  support including provider-owned continuation.

### Changed

- Replaced split model content/tool-call results with one ordered output sequence and
  separated exact model modalities, host-executed `RuntimeToolCall` values, and remote
  `ProviderToolCall` values at the kernel boundary.
- Replaced provider-profile capability booleans with one immutable
  `ModelCapabilities` declaration, including exact tool-choice types, parallel-control
  and seed support. Rewrote the OpenAI, Anthropic, and DeepSeek defaults and moved
  supplier-specific Responses reasoning and hosted-tool configuration into profile
  wire policy.
- Made the default OpenAI Responses profile conservative and text-only. Model-specific
  modalities, structured output, JSON mode, and hosted tools now require explicit
  profile opt-in.
- Moved repository backends to the isolated `v3` physical namespace and history-digest
  domain for ordered assistant output. Obsolete `v1` and `v2` runs are not read or
  migrated.

### Fixed

- Updated release validation to Twine 7 for Core Metadata 2.5 and expanded isolated
  distribution smoke checks to cover the Responses public API and DeepSeek profile.
- Made OpenAI Responses storage explicit: the default sends `store=false` with
  encrypted reasoning history, while compatible profiles can omit storage fields.
- Required a durable, integrity-bearing host-owned `ResponsesArtifactStore` for hosted
  image generation, rejected unexpected inline results without one, and kept durable
  history free of image base64.
- Aligned the visible-content contract with terminal incomplete provider-tool output.

## [0.5.0] - 2026-08-10

### Added

- Added the workspace-scoped `LsTool` filesystem preset for bounded, sorted,
  non-recursive directory listings.

## [0.4.0] - 2026-08-10

### Added

- Added `InMemoryAgentBackend`, an in-process default child-Agent supervisor with
  concurrent idempotent creation, parent-run authorization, waiting, cancellation,
  nested context propagation, and explicit terminal-state mapping.

## [0.3.4] - 2026-07-31

### Changed

- Removed unused conformance observation and schema-validation surfaces, unreachable
  request and workspace-cleanup branches, and redundant repository lifecycle and
  decoded-head state without changing public behavior or persisted formats.
- Replaced callback-heavy invalid-response test setup with direct parametrized data
  while preserving every boundary case and test identifier, and kept Markdown link
  validation under the authoritative specification validator.

## [0.3.3] - 2026-07-30

### Changed

- Consolidated AskQuestion metadata and exact numeric rules into one private contract
  owner, and consolidated SQLite/MySQL history-manifest policy into one pure shared
  implementation.
- Expanded release artifact checks to cover every public model namespace and made all
  release scripts use the locked uv environment.

### Fixed

- Preserved explicitly supplied falsey repositories and framework tool failures through
  runtime execution.
- Bounded complete JSON and HTTP error response accumulation with a configurable limit.

## [0.3.2] - 2026-07-30

### Added

- Added provider-neutral `RetryingModel` and `FallbackModel` composition with
  bounded server-aware jitter, monotonic deadline accounting, safe capability
  intersection, and no retry or fallback after a delta reaches the host.
- Added reusable installed-artifact API verification to coordinated TestPyPI and PyPI
  release checks.

### Changed

- Consolidated user, architecture, contract, and repository documentation around
  authoritative sources while removing duplicate guides and case-name inventories.
- Clarified portable JSON depth, lexical integer, safe number conversion, opaque-data,
  and trace-verification boundaries.

## [0.3.1] - 2026-07-23

### Added

- Added OpenAI Chat Completions profile controls for seed support, reasoning-content
  round trips, required reasoning on tool calls, and non-null assistant tool-call
  content, plus an Anthropic profile control for redacted-thinking replay.

### Fixed

- Enabled DeepSeek OpenAI-format thinking models to use tools while omitting the
  unsupported `tool_choice` parameter and preserving the required
  `reasoning_content` and non-null assistant `content` across tool-call turns.
- Rejected unsupported DeepSeek `seed` requests and mapped its top-level
  `prompt_cache_hit_tokens` usage field to `cache_read_tokens`.
- Prevented DeepSeek Anthropic-format history from replaying unsupported
  `redacted_thinking` blocks.

## [0.3.0] - 2026-07-19

### Changed

- Replaced tuple-backed run history with structurally shared `RunHistory`, including
  persistent tool-call linkage proofs and cursor-based pending tool calls.
- Replaced `RunRepository.commit(checkpoint)` with validated incremental
  `DurableCommit` values and run-scoped checkpoint idempotency.
- Replaced full-checkpoint repository writes with shared in-memory values and v2
  incremental history chunks for SQLite, MySQL, and Redis; obsolete v1 storage is not
  read or migrated.

### Performance

- Fixed-size append runs now perform linear cumulative history, persistence, and
  pending-tool work instead of repeatedly scanning or encoding old state. Model requests
  intentionally continue to contain the complete current history.

## [0.2.2] - 2026-07-19

### Added

- Added the coordinated `jharness-repository` distribution with memory, SQLite,
  MySQL, and Redis implementations of the kernel checkpoint repository protocol.

### Changed

- Made MySQL and Redis repository drivers strict opt-in extras; the base repository
  install now depends only on the coordinated kernel and supports Memory and SQLite.

### Fixed

- Closed MySQL and SQLite repository executors when asynchronous context initialization
  fails, and made real MySQL and Redis integration tests remove their generated data.

## [0.2.1] - 2026-07-18

### Fixed

- Made release artifact counting ignore non-package files created by the build tool.

## [0.2.0] - 2026-07-18

### Changed

- Consolidated runtime code, model adapters, reusable tools, portable contracts,
  conformance cases, tests, examples, benchmarks, documentation, and release automation
  into one Python repository.
- Established one uv workspace that publishes the coordinated `jharness-kernel`,
  `jharness-toolkit`, `jharness-models`, and `jharness-tools` distributions.
- Established `jharness.models` as the sole model-adapter package across source paths,
  imports, documentation, tests, and release metadata, without compatibility aliases.
- Established one-way component dependencies on the kernel and exact coordinated
  kernel version pins in the other three distributions.
- Made contracts and conformance fixtures local project inputs, removing external
  synchronization and revision-pin workflows.

### Added

- OpenAI Chat Completions, Anthropic Messages, and DeepSeek model profiles.
- Ready-to-use filesystem, shell, structured interaction, and child-agent tools.
- Four-distribution namespace ownership, non-overlap, isolated-install verification,
  and coordinated release documentation.

## [0.1.0] - 2026-07-14

### Added

- Portable v0 JSON schemas and normative runtime documentation.
- Sixty-six deterministic conformance cases and a standard tool catalog.
- Provider-neutral lifecycle, model, tool, event, wire, and trace contracts.

[Unreleased]: https://github.com/Ezio2000/jharness/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Ezio2000/jharness/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Ezio2000/jharness/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Ezio2000/jharness/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/Ezio2000/jharness/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/Ezio2000/jharness/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Ezio2000/jharness/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Ezio2000/jharness/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Ezio2000/jharness/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/Ezio2000/jharness/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ezio2000/jharness/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ezio2000/jharness/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ezio2000/jharness/releases/tag/v0.1.0
