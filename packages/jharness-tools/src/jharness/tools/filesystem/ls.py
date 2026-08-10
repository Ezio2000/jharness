"""The workspace-scoped Ls preset."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from jharness.kernel import ToolCall, ToolContext, ToolExecution, ToolResult, ToolRisk, ToolSpec
from jharness.tools.filesystem._common import (
    FilesystemFailure,
    OperationCancelled,
    PathInput,
    SearchBudget,
    Workspace,
    cancelled,
    failure,
    is_reserved_temporary_name,
    nullable_output,
    positive_float,
    positive_int,
    run_blocking,
    secure_scandir,
    success,
)

_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1_000
_MAX_SEARCH_SECONDS = 10.0
_MAX_SCANNED_ENTRIES = 100_000


@dataclass(frozen=True, slots=True, init=False)
class LsTool:
    """List direct children of one directory inside a workspace."""

    workspace: Workspace
    default_limit: int
    max_limit: int
    max_search_seconds: float
    max_scanned_entries: int
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        root: PathInput,
        *,
        default_limit: int = _DEFAULT_LIMIT,
        max_limit: int = _MAX_LIMIT,
        max_search_seconds: float = _MAX_SEARCH_SECONDS,
        max_scanned_entries: int = _MAX_SCANNED_ENTRIES,
    ) -> None:
        default_limit = positive_int(default_limit, "default_limit")
        max_limit = positive_int(max_limit, "max_limit")
        max_search_seconds = positive_float(max_search_seconds, "max_search_seconds")
        max_scanned_entries = positive_int(max_scanned_entries, "max_scanned_entries")
        if default_limit > max_limit:
            raise ValueError("default_limit cannot exceed max_limit")
        object.__setattr__(self, "workspace", Workspace.create(root))
        object.__setattr__(self, "default_limit", default_limit)
        object.__setattr__(self, "max_limit", max_limit)
        object.__setattr__(self, "max_search_seconds", max_search_seconds)
        object.__setattr__(self, "max_scanned_entries", max_scanned_entries)
        object.__setattr__(self, "spec", _spec(default_limit, max_limit))

    @property
    def root(self) -> Path:
        return self.workspace.root

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        path = cast(str, call.arguments.get("path", "."))
        limit = cast(int, call.arguments.get("limit", self.default_limit))
        try:
            return await run_blocking(
                lambda cancelled_check: self._ls(path, limit, cancelled_check),
                lambda: context.cancel_requested,
            )
        except OperationCancelled:
            return cancelled("Ls")
        except FilesystemFailure as exc:
            return failure(exc)

    def _ls(
        self,
        path: str,
        limit: int,
        cancelled_check: Callable[[], bool],
    ) -> ToolResult:
        budget = SearchBudget.create(
            cancelled_check,
            self.max_search_seconds,
            self.max_scanned_entries,
        )
        budget.checkpoint()
        directory = self.workspace.directory(path)
        try:
            smallest = heapq.nsmallest(
                limit + 1,
                self._entries(directory, budget),
                key=lambda value: (value.casefold(), value),
            )
        except OSError as exc:
            raise FilesystemFailure(
                "filesystem_error",
                f"Cannot list directory: {path}",
            ) from exc
        truncated = len(smallest) > limit
        entries = smallest[:limit]
        text = "\n".join(entries) if entries else "Directory is empty."
        return success(
            text,
            {
                "path": self.workspace.display(directory),
                "entries": entries,
                "truncated": truncated,
            },
        )

    def _entries(self, directory: Path, budget: SearchBudget) -> Iterator[str]:
        with secure_scandir(self.workspace, directory) as entries:
            for entry in entries:
                budget.consume_entry()
                if is_reserved_temporary_name(entry.name):
                    continue
                suffix = "/" if entry.is_dir(follow_symlinks=False) else ""
                yield f"{entry.name}{suffix}"


def _spec(default_limit: int, max_limit: int) -> ToolSpec:
    output = {
        "type": "object",
        "required": ["path", "entries", "truncated"],
        "properties": {
            "path": {"type": "string"},
            "entries": {"type": "array", "items": {"type": "string"}},
            "truncated": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    return ToolSpec(
        name="Ls",
        description=(
            "List direct children of a directory within the configured workspace. "
            "Directories have a trailing '/'. "
            f"Results are sorted and limit defaults to {default_limit}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "default": ".",
                    "description": "Directory inside the workspace to list.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_limit,
                    "default": default_limit,
                    "description": "Maximum number of sorted entries to return.",
                },
            },
            "additionalProperties": False,
        },
        output_schema=nullable_output(output),
        execution=ToolExecution(concurrency="parallel", read_only=True, idempotent=True),
        risk=ToolRisk(filesystem="read", destructive=False),
    )
