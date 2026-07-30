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

Interaction tools suspend for a host response. Child-agent tools require a host-owned
`AgentBackend`; the host remains responsible for authorization, idempotency,
supervision, and telemetry.

Installing this distribution installs the exact matching `jharness-kernel` version.
