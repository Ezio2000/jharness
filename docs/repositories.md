# Repository Implementations

`jharness-repository` provides four implementations of the kernel
`RunRepository` protocol:

| Class | Storage | Extra | Lifecycle |
| --- | --- | --- | --- |
| `MemoryRunRepository` | Current process | None | No close required |
| `SQLiteRunRepository` | SQLite | None | Async context manager or `await close()` |
| `MySQLRunRepository` | MySQL/InnoDB | `mysql` | Async context manager or `await close()` |
| `RedisRunRepository` | Redis or Redis Cluster | `redis` | Async context manager or `await close()` |

```bash
uv add jharness-repository
uv add "jharness-repository[mysql]"
uv add "jharness-repository[redis]"
```

The base package keeps remote drivers optional and lazy. All four classes remain
importable, but initializing MySQL or Redis without its extra raises
`RepositoryError`.

## Choose a Backend

Memory is thread-safe and useful for tests or process-lifetime state:

```python
from jharness.kernel import Runtime
from jharness.repository import MemoryRunRepository

repository = MemoryRunRepository()
runtime = Runtime(model=model, repository=repository)
```

The `async with` snippets below are fragments for use inside an existing `async def`.

SQLite needs no external service and moves blocking work to a repository-owned worker:

```python
from jharness.repository import SQLiteRunRepository

async with SQLiteRunRepository("runs.sqlite3") as repository:
    runtime = Runtime(model=model, repository=repository)
    checkpoint = await runtime.start(messages).result()
```

Use MySQL when several application processes share an InnoDB database. The database
must already exist and the user must be able to create and update tables:

```python
from jharness.repository import MySQLRunRepository, MySQLTLS

async with MySQLRunRepository(
    host="mysql.internal",
    user="jharness",
    password=mysql_password,
    database="jharness",
    tls=MySQLTLS(ca="/etc/jharness/mysql-ca.pem"),
) as repository:
    runtime = Runtime(model=model, repository=repository)
```

`MySQLTLS` verifies the certificate and server identity by default. Client `cert` and
`key` must be supplied together for mutual TLS. Omitting `tls` leaves transport policy
to the endpoint and driver defaults.

Use Redis for an async shared backend. Cluster mode selects redis-py's cluster-aware
client:

```python
from jharness.repository import RedisRunRepository

async with RedisRunRepository(
    "redis://redis-cluster.internal:6379",
    cluster=True,
    key_prefix="production-agent",
) as repository:
    runtime = Runtime(model=model, repository=repository)
```

Redis persistence, replication, authentication, TLS, and backups remain deployment
responsibilities. The adapter does not set a TTL.

## Shared Semantics

Every backend atomically commits one validated `DurableCommit`, checks the expected
revision and parent, accepts an exact idempotent retry, and rejects checkpoint-ID
reuse with different content. `await repository.get_head(run_id)` returns the complete
recovery checkpoint.

Persistent backends store explicit history deltas while preserving complete recovery.
Their physical namespace is version `v2`; obsolete `v1` data is not read or migrated.
See the normative [repository contract](../contracts/v0/repository.md).

Database repositories initialize lazily, so an explicit `await initialize()` call is
optional. Prefer an async context manager, or call `await close()` after the last
operation. Closing rejects new work and settles already accepted work. Backend and
runtime timeouts should be configured together because cancellation cannot safely
detach a commit whose outcome is still unknown.
