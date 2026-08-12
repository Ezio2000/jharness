from __future__ import annotations

from jharness.kernel import (
    ModelContentDelta,
    ModelRuntimeToolCallDelta,
    RuntimeToolKind,
    StructuredToolCall,
)
from jharness.models._stream import DeltaAccumulator


def test_delta_accumulator_handles_many_tiny_chunks_without_changing_the_result() -> None:
    accumulator = DeltaAccumulator(ValueError)
    assert accumulator.has_output is False
    chunk_count = 4_096
    for _ in range(chunk_count):
        accumulator.apply(ModelContentDelta(output_index=0, text_delta="x", content_index=0))
    assert accumulator.has_output is True

    encoded_arguments = '{"value":"' + ("y" * chunk_count) + '"}'
    for index, chunk in enumerate(encoded_arguments):
        accumulator.apply(
            ModelRuntimeToolCallDelta(
                output_index=1,
                input_kind=RuntimeToolKind.STRUCTURED,
                input_delta=chunk,
                id="call-1" if index == 0 else None,
                name="search" if index == 0 else None,
            )
        )

    response = accumulator.response(
        finish_reason="tool_calls",
        model_id="model-1",
        response_id="response-1",
        metadata={},
    )

    assert response.visible_parts()[0].text == "x" * chunk_count
    call = response.runtime_tool_calls()[0]
    assert isinstance(call, StructuredToolCall)
    assert call.arguments == {"value": "y" * chunk_count}
