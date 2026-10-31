# Copyright SUSE LLC
"""Unit tests for openqa_llm_investigate.verdict."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from openqa_llm_investigate import verdict
from tests.test_openqa_llm_investigate import CHAT_REPLIES, llm_investigate, setup_mock_client

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

_TEST = "lvm-encrypt-separate-boot"
_REPLY = "Classification: Product bug\n**BISECT: YES** \u2014 isolated\nRestart: YES \u2014 flaky network"
_NO_LAST_GOOD = "BISECT: NO \u2014 openQA has no last good job of this scenario"
# investigation_ajax `last_good` of o3 6304039 (none) and 6304175 (5252327, softfailed 92 failures ago)
_NOT_FOUND = {"first_bad": {"link": "/tests/5125495", "text": 5125495, "type": "link"}, "last_good": "not found"}
_OUTSIDE_WINDOW = {"last_good": {"link": "/tests/5252327", "text": 5252327, "type": "link"}}
_SOFTFAILED_IN_WINDOW = {"last_good": {"link": "/tests/6304100", "text": 6304100, "type": "link"}}


def _restart_no(streak: int) -> str:
    return _REPLY.replace("YES \u2014 flaky network", f"NO \u2014 failed at least {streak} times in a row")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Classification: Test issue\n**BISECT: YES** \u2014 isolated", True),
        ("- bisect: yes \u2014 isolated", True),
        ("BISECT: YES?\nBISECT: NO", False),
        ("`BISECT: NO` \u2014 widespread\n", False),
        ("**BISECT:** YES \u2014 isolated", True),
        ("**BISECT**: NO", False),
        ("BISECT: **YES**", True),
        ("__BISECT__: _YES_", True),
        ("### BISECT: YES", True),
        ("1. **BISECT:** NO \u2014 known", False),
        ("BISECT: NOT sure", None),
        ("Step 3. BISECT: YES", None),
        ("Classification: Test issue\nRestart: NO", None),
        ("Other jobs: BISECT: YES would be wrong.", None),
        ("", None),
    ],
)
def test_bisect_verdict_reads_the_last_bisect_line(reply: str, *, expected: bool | None) -> None:
    assert verdict.bisect_verdict(reply) is expected


@pytest.mark.parametrize(
    ("streak", "investigation", "expected"),
    [
        (95, _NOT_FOUND, f"Classification: Product bug\n{_NO_LAST_GOOD}\n{_restart_no(95).splitlines()[-1]}"),
        (92, _OUTSIDE_WINDOW, _restart_no(92)),
        (4, _SOFTFAILED_IN_WINDOW, _restart_no(4)),
        (1, _NOT_FOUND, _REPLY),
        (2, _NOT_FOUND, _REPLY),
        (None, _NOT_FOUND, _REPLY),
        (3, {}, _restart_no(3)),
    ],
    ids=[
        "no-last-good",
        "last-good-outside-the-window",
        "softfailed-last-good",
        "first-failure",
        "two-failures",
        "not-in-history",
        "investigation-failed",
    ],
)
def test_enforce_history_rules(streak: int | None, investigation: dict[str, Any], expected: str) -> None:
    assert verdict.enforce_history_rules(_REPLY, streak, investigation=investigation, test=_TEST) == expected


def test_enforce_history_rules_keeps_the_reason_of_a_no() -> None:
    reply = "BISECT: NO \u2014 1 other job(s) in this build fail in the same module\nRestart: NO \u2014 needle mismatch"
    assert verdict.enforce_history_rules(reply, 25, investigation=_NOT_FOUND, test=_TEST) == reply


def test_enforce_history_rules_leaves_investigation_jobs_to_the_model() -> None:
    test = f"{_TEST}:investigate:retry"
    assert verdict.enforce_history_rules(_REPLY, 95, investigation=_NOT_FOUND, test=test) == _REPLY


@pytest.mark.parametrize(
    ("bisect", "restart"),
    [
        ("**BISECT:** YES \u2014 isolated", "**Restart:** YES"),
        ("**BISECT**: YES", "**Restart**: YES"),
        ("BISECT: **YES**", "Restart: **YES** \u2014 flaky"),
        ("__BISECT__: _YES_", "_Restart_: _YES_"),
        ("### BISECT: YES", "2. Restart: YES"),
    ],
)
def test_enforce_history_rules_rewrites_emphasized_lines(bisect: str, restart: str) -> None:
    out = verdict.enforce_history_rules(f"{bisect}\n{restart}", 3, investigation=_NOT_FOUND, test=_TEST)
    assert out == f"{_NO_LAST_GOOD}\nRestart: NO \u2014 failed at least 3 times in a row"


def test_investigate_without_a_bisect_line_falls_back_on_a_job_reviewed_meanwhile(mocker: MockerFixture) -> None:
    mock_post = mocker.patch("llm_investigate.post_comment")
    mock_print = mocker.patch("llm_investigate.typer.echo")
    reviewed = mocker.patch("llm_investigate.is_reviewed", side_effect=[False, True])
    mocker.patch.dict(CHAT_REPLIES, {5: "Classification: Product bug"})
    setup_mock_client(mocker)
    llm_investigate.investigate("123")
    mock_print.assert_called_once_with("https://openqa.opensuse.org/tests/123")
    mock_post.assert_not_called()
    assert reviewed.call_count == 1


def test_investigate_forces_no_on_a_copied_template_when_other_jobs_fail_in_the_same_module(
    mocker: MockerFixture,
) -> None:
    mock_post = mocker.patch("llm_investigate.post_comment")
    mock_print = mocker.patch("llm_investigate.typer.echo")
    mocker.patch("llm_investigate._build_failures", return_value=(3, 2))
    mocker.patch.dict(CHAT_REPLIES, {5: "Classification: Test issue\nBISECT: YES | NO \u2014 your own short reason"})
    setup_mock_client(mocker)
    llm_investigate.investigate("123")
    mock_print.assert_not_called()
    assert "BISECT: NO \u2014 2 other job(s) in this build fail in the same module" in mock_post.call_args[0][2]
