"""Host-owned artifact persistence for Responses image-generation history."""

from __future__ import annotations

import base64
import binascii
from dataclasses import replace
from hashlib import sha256
from typing import Protocol, cast, runtime_checkable

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    ProviderToolCall,
    RunContext,
)
from jharness.models.openai.responses.errors import OpenAIResponsesError


@runtime_checkable
class OpenAIResponsesArtifactStore(Protocol):
    """Durably persist provider images outside checkpoint history.

    ``call_id`` is provider-controlled and scoped only to its response; it must not be
    trusted as a path or global key. A successful save must be durable, and repeated
    saves of the same bytes must be idempotent. A reference must never be reassigned to
    different bytes. Returned references must be
    stable across process restarts and include exact ``size_bytes`` and ``sha256``
    integrity metadata.

    Image persistence precedes the checkpoint commit. Implementations therefore own
    retention and garbage collection for saves that never become reachable from a
    committed checkpoint.
    """

    async def save_image(
        self,
        data: bytes,
        *,
        media_type: str,
        call_id: str,
        context: RunContext,
    ) -> ArtifactRef:
        """Durably save bytes and return a stable, integrity-bearing reference."""

        ...

    async def load_image(
        self,
        artifact: ArtifactRef,
        *,
        call_id: str,
        context: RunContext,
    ) -> bytes:
        """Load the exact referenced bytes or fail without substituting content."""

        ...


async def externalize_image_call(
    call: ProviderToolCall,
    store: OpenAIResponsesArtifactStore,
    context: RunContext,
) -> ProviderToolCall:
    if not call.output:
        return call
    part = _single_image_output(call)
    if part.type == "artifact":
        return call
    raw_base64 = part.data.get("base64")
    if not isinstance(raw_base64, str) or not raw_base64:
        raise OpenAIResponsesError(
            "image_generation result requires non-empty base64 data before persistence"
        )
    try:
        data = base64.b64decode(raw_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise OpenAIResponsesError("image_generation result contains invalid base64") from exc
    media_type = part.media_type
    if not isinstance(media_type, str) or not media_type.startswith("image/"):
        raise OpenAIResponsesError("image_generation result requires an image media_type")
    artifact = await store.save_image(
        data,
        media_type=media_type,
        call_id=call.id,
        context=context,
    )
    _validate_artifact(artifact, data, media_type=media_type)
    return replace(
        call,
        output=(ContentPart.artifact_part(artifact, metadata=part.metadata),),
    )


async def hydrate_image_call(
    call: ProviderToolCall,
    store: OpenAIResponsesArtifactStore,
    context: RunContext,
) -> ProviderToolCall:
    if not call.output:
        return call
    part = _single_image_output(call)
    if part.type == "image":
        return call
    artifact = part.artifact
    if artifact is None:
        raise OpenAIResponsesError("image_generation artifact output requires an artifact")
    media_type = artifact.media_type
    if not isinstance(media_type, str) or not media_type.startswith("image/"):
        raise OpenAIResponsesError("image_generation artifact requires an image media_type")
    raw_data = cast(object, await store.load_image(artifact, call_id=call.id, context=context))
    if not isinstance(raw_data, bytes):
        raise TypeError("OpenAIResponsesArtifactStore.load_image() must return bytes")
    data = raw_data
    _validate_artifact(artifact, data, media_type=media_type)
    return replace(
        call,
        output=(
            ContentPart(
                type="image",
                media_type=media_type,
                data={"base64": base64.b64encode(data).decode("ascii")},
                metadata=part.metadata,
            ),
        ),
    )


def _single_image_output(call: ProviderToolCall) -> ContentPart:
    if len(call.output) != 1 or call.output[0].type not in {"image", "artifact"}:
        raise OpenAIResponsesError(
            "image_generation history requires exactly one image or artifact output"
        )
    return call.output[0]


def image_call_has_artifact(call: ProviderToolCall) -> bool:
    """Return whether an image call contains a durable artifact reference."""

    return any(part.type == "artifact" for part in call.output)


def image_call_has_inline_result(call: ProviderToolCall) -> bool:
    """Return whether an image call still contains provider-owned inline bytes."""

    return any(part.type == "image" for part in call.output)


def _validate_artifact(
    artifact: object,
    data: bytes,
    *,
    media_type: str,
) -> ArtifactRef:
    if not isinstance(artifact, ArtifactRef):
        raise TypeError("OpenAIResponsesArtifactStore.save_image() must return ArtifactRef")
    if artifact.media_type != media_type:
        raise ValueError("image artifact media_type must match the provider result")
    if artifact.size_bytes is None:
        raise ValueError("image artifact requires size_bytes integrity metadata")
    if artifact.size_bytes != len(data):
        raise ValueError("image artifact size_bytes does not match stored data")
    if artifact.sha256 is None:
        raise ValueError("image artifact requires sha256 integrity metadata")
    if artifact.sha256.lower() != sha256(data).hexdigest():
        raise ValueError("image artifact sha256 does not match stored data")
    return artifact
