# Copyright SUSE LLC
"""Read the BISECT verdict and override BISECT and Restart where the job's history rules them out."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .http import JsonDict

BISECT_RE = re.compile(r"^(?:[\s*_`>#-]|\d+\.)*BISECT[*_`]*:[\s*_`]*(YES|NO)(?![a-z0-9])", re.IGNORECASE)
_RESTART_RE = re.compile(r"^(?:[\s*_`>#-]|\d+\.)*Restart[*_`]*:[\s*_`]*(YES|NO)(?![a-z0-9])", re.IGNORECASE)
_DETERMINISTIC_STREAK = 3
_NO_LAST_GOOD = "not found"


def bisect_verdict(verdict: str) -> bool | None:
    """Return whether the last `BISECT: YES|NO` line of the verdict is YES, or None if it has none."""
    for ln in reversed(verdict.splitlines()):
        if m := BISECT_RE.match(ln):
            return m.group(1).upper() == "YES"
    return None


def _yes_to_no(verdict: str, pattern: re.Pattern[str], no: str) -> str:
    lines = verdict.splitlines()
    return "\n".join(no if (m := pattern.match(ln)) and m.group(1).upper() == "YES" else ln for ln in lines)


def enforce_history_rules(verdict: str, streak: int | None, *, investigation: JsonDict, test: str) -> str:
    """Rewrite a BISECT or Restart YES to NO where the job's history rules it out; small models ignore it.

    `streak` is the failures in a row up to the job, None if the job is not in the fetched history. `investigation` is
    openQA's investigation of the job, whose `last_good` is "not found" if openQA knows no good job to bisect against.
    Jobs of openqa-investigate itself are left to the model, so that it still reports their result.
    """
    if ":investigate:" in test or (streak or 0) < _DETERMINISTIC_STREAK:
        return verdict
    if investigation.get("last_good") == _NO_LAST_GOOD:
        verdict = _yes_to_no(verdict, BISECT_RE, "BISECT: NO \u2014 openQA has no last good job of this scenario")
    return _yes_to_no(verdict, _RESTART_RE, f"Restart: NO \u2014 failed at least {streak} times in a row")
