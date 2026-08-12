"""Focused test for the executable DeepSeek Responses offline contract."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from demos.deepseek_responses_protocol.demo import extract_text_urls
from demos.deepseek_responses_protocol.offline_contract import run_offline_contract

ROOT = Path(__file__).resolve().parents[2]


def test_markdown_wrapped_source_urls_are_normalized() -> None:
    assert extract_text_urls("Source: **https://example.test/guide/**") == (
        "https://example.test/guide/",
    )


def test_offline_contract() -> None:
    report = run_offline_contract()

    assert report["status"] == "passed"
    assert report["protocol"] == "deepseek-native-responses"
    assert report["model"] == "deepseek-v4-flash"
    _assert_report_is_self_consistent(report)
    assert {
        "rejection:image_generation",
        "rejection:unmodeled_provider_tool",
    }.issubset(report["rejection_checks"])


def test_optimized_module_entrypoint_runs_offline_contract() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "demos.deepseek_responses_protocol",
            "--offline-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    report = cast(dict[str, object], json.loads(completed.stdout))
    assert completed.stderr == "[offline] DeepSeek Responses profile and codec contract\n"
    assert report["live"] == []
    offline = cast(dict[str, object], report["offline_contract"])
    assert offline["status"] == "passed"
    _assert_report_is_self_consistent(offline)


def _assert_report_is_self_consistent(report: Mapping[str, object]) -> None:
    sections = (
        cast(Sequence[str], report["request_checks"]),
        cast(Sequence[str], report["response_checks"]),
        cast(Sequence[str], report["rejection_checks"]),
    )
    executed_names = tuple(name for section in sections for name in section)

    assert all(sections)
    assert all(executed_names)
    assert len(executed_names) == len(set(executed_names))
    assert report["total_checks"] == len(executed_names)
