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
Their physical namespace is version `v3`; obsolete `v1` and `v2` data is not read or
migrated. Version `v3` isolates the ordered assistant-output history format and its
history-digest domain from earlier releases.
See the normative [repository contract](../contracts/v0/repository.md).

### Retire Obsolete Physical Namespaces

A `v3` deployment starts with no readable runs from earlier namespaces. Before cleanup,
stop writers, back up the backend, verify that rollback to an older JHarness release is
not required, and retain any old data needed for audit or export. Cleanup is an explicit
operator action; repository initialization never deletes old data.

For SQLite, remove the old tables from each database after the checks above:

```sql
DROP TABLE IF EXISTS jharness_v2_history_chunks;
DROP TABLE IF EXISTS jharness_v2_checkpoint_ledger;
DROP TABLE IF EXISTS jharness_v2_run_heads;
DROP TABLE IF EXISTS jharness_v1_checkpoint_ids;
DROP TABLE IF EXISTS jharness_v1_run_heads;
```

MySQL uses the same suffixes with the configured `table_prefix`, for example
`jharness_v2_run_heads`. Drop the history and checkpoint-id/ledger tables before their
corresponding run-head table. Confirm the selected database and expanded table names
instead of applying a wildcard drop.

Redis derives `<namespace>` as the lowercase SHA-256 hex digest of `key_prefix`. The old
key shapes are `jharness:{<namespace>}:v1:state` and
`jharness:v2:{<namespace>:<run-hash>}:head|ledger|history`. Enumerate only those patterns
with incremental `SCAN`, review the resulting keys, and remove them with `UNLINK`; do
not use blocking `KEYS` or delete the `jharness:v3:` namespace. In Redis Cluster, scan
each primary node because no single node owns the complete keyspace.

Database repositories initialize lazily, so an explicit `await initialize()` call is
optional. Prefer an async context manager, or call `await close()` after the last
operation. Closing rejects new work and settles already accepted work. Backend and
runtime timeouts should be configured together because cancellation cannot safely
detach a commit whose outcome is still unknown.
