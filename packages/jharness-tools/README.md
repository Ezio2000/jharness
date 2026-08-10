# jharness-tools

Ready-to-use filesystem, shell, interaction, and child-agent tools implementing the
JHarness kernel tool contracts.

```bash
uv add jharness-tools
```

```python
from jharness.tools import GlobTool, GrepTool, ReadTool
```

Filesystem tools are rooted in one workspace and reject path escapes. `BashTool` uses
a bounded non-interactive Bash process and a minimal environment by default, but it is
not an operating-system sandbox: commands retain the filesystem and network access
granted by the host. `inherit_environment=True` explicitly exposes the full host
environment.

Workspace path checks are not a mount namespace; mutually untrusted writers need
dedicated filesystem isolation and hard-link controls. Process-tree cleanup is best
effort, so containers—especially PID 1—must forward signals and reap child processes.

Interaction tools suspend for a host response. Child-agent tools accept the narrow
`AgentBackend` protocol. For a single-process host, `InMemoryAgentBackend` provides
concurrent supervision, parent-run authorization, idempotent creation, waiting, and
cancellation:

```python
from jharness.kernel import Runtime
from jharness.tools.agent import InMemoryAgentBackend

child_runtime = Runtime(model=model, tools=child_tool_catalog)
agent_backend = InMemoryAgentBackend(
    child_runtime,
    system_prompt="Follow the delegated task and return a concise result.",
)
```

Pass `agent_backend` to `AgentTool`, `AgentGetTool`, `AgentWaitTool`, and
`AgentCancelTool`. After a parent suspension, the host can await the in-memory
implementation's `wait_for_terminal(agent_id, requester=parent_context)` and deliver
the snapshot with `resume_agent`; other backend implementations may use their own
durable notification mechanism. The in-memory backend retains idempotency records for
its lifetime, stores them only in its process, and must be used from one asyncio event
loop. Hosts requiring bounded retention, crash recovery, or shared workers should
implement `AgentBackend` with durable storage. The host remains responsible for
choosing trusted child Runtime configuration and recording telemetry.

Installing this distribution installs the exact matching `jharness-kernel` version.
