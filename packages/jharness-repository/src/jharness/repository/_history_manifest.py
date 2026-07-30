"""Backend-neutral policy for advancing durable history manifests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from jharness.kernel import (
    HistoryAppend,
    HistoryReplace,
    HistoryUnchanged,
    RepositoryError,
)

from ._codec import CommitIdentity, EncodedHistoryChunk


class HistoryManifest(Protocol):
    """The head fields used by the shared history policy."""

    @property
    def checkpoint_id(self) -> str: ...

    @property
    def history_generation(self) -> int: ...

    @property
    def history_chunk_count(self) -> int: ...

    @property
    def history_message_count(self) -> int: ...

    @property
    def history_digest(self) -> bytes: ...


def validate_new_base(
    head: HistoryManifest | None,
    identity: CommitIdentity,
) -> None:
    """Require a new commit to extend the authoritative history base."""

    if head is None:
        if (
            identity.parent_checkpoint_id is not None
            or identity.base_history_count is not None
            or identity.base_history_digest is not None
        ):
            raise RepositoryError("first durable commit has an invalid history base")
        return
    if identity.parent_checkpoint_id != head.checkpoint_id:
        raise RepositoryError("parent checkpoint does not match the authoritative head")
    if (
        identity.base_history_count != head.history_message_count
        or identity.base_history_digest != head.history_digest
    ):
        raise RepositoryError("history change base does not match the authoritative head")


def next_history_manifest(
    head: HistoryManifest | None,
    identity: CommitIdentity,
    added_chunks: int,
) -> tuple[int, int, int]:
    """Return generation, first chunk index, and resulting chunk count."""

    change = identity.commit.history
    if head is None or isinstance(change, HistoryReplace):
        return identity.revision, 0, added_chunks
    if isinstance(change, HistoryAppend):
        return (
            head.history_generation,
            head.history_chunk_count,
            head.history_chunk_count + added_chunks,
        )
    if not isinstance(change, HistoryUnchanged):
        raise RepositoryError("advanced durable commit has an invalid history mutation")
    return head.history_generation, head.history_chunk_count, head.history_chunk_count


def validate_encoded_history(
    head: HistoryManifest | None,
    identity: CommitIdentity,
    chunks: Sequence[EncodedHistoryChunk],
) -> None:
    """Require encoded chunks to represent the declared history mutation."""

    added_messages = sum(chunk.message_count for chunk in chunks)
    change = identity.commit.history
    if head is None or isinstance(change, HistoryReplace):
        if not chunks or added_messages != identity.history_count:
            raise RepositoryError("encoded replacement history is inconsistent")
    elif isinstance(change, HistoryAppend):
        if not chunks or head.history_message_count + added_messages != identity.history_count:
            raise RepositoryError("encoded appended history is inconsistent")
    elif chunks or head.history_message_count != identity.history_count:
        raise RepositoryError("encoded unchanged history is inconsistent")
