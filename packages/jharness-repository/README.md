# jharness-repository

Official memory, SQLite, MySQL, and Redis implementations of the JHarness kernel
repository protocol.

```bash
uv add jharness-repository
uv add "jharness-repository[mysql]"
uv add "jharness-repository[redis]"
```

```python
from jharness.repository import (
    MemoryRunRepository,
    MySQLRunRepository,
    RedisRunRepository,
    SQLiteRunRepository,
)
```

Memory and SQLite require no optional driver. MySQL and Redis load only the selected
extra when that backend initializes. All implementations consume atomic
`DurableCommit` values and expose `get_head(run_id)` for complete recovery.

Backend selection, lifecycle, MySQL TLS, and Redis Cluster setup are documented in the
[repository guide](https://github.com/Ezio2000/jharness/blob/main/docs/repositories.md).
Installing this distribution installs the exact matching `jharness-kernel` version.
