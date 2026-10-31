# Copyright SUSE LLC
"""Ask openQA whether a job is already reviewed."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .http import AnalyzerError, OpenQA
from .text import redact

if TYPE_CHECKING:
    import httpx

log = logging.getLogger(__name__)


def is_reviewed(client: httpx.Client, server: str, job_id: str) -> bool:
    """Return whether openQA shows the job as reviewed: one of its comments carries a bugref or a label.

    openQA decides this itself, so comments with flag:verbatim, like our own summaries, do not count.
    The answer is fetched afresh on every call (False when openQA cannot tell).
    """
    try:
        page = OpenQA(client, server).json(f"/tests/{job_id}/ajax?previous_limit=0&next_limit=0")
    except AnalyzerError as e:
        return _cannot_tell(job_id, e)
    try:
        current = next(j for j in page["data"] if j.get("iscurrent"))
    except (KeyError, TypeError, AttributeError, StopIteration) as e:
        return _cannot_tell(job_id, e)
    comment_data = current.get("comment_data") or {}
    return bool(comment_data.get("reviewed"))


def _cannot_tell(job_id: str, error: Exception) -> bool:
    log.warning("Could not check whether job %s is reviewed: %s", job_id, redact(repr(error)))
    return False
