# Copyright SUSE LLC
"""Tests for skipping jobs that openQA shows as reviewed."""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, call

import pytest

from openqa_llm_investigate.reviews import is_reviewed
from tests.test_openqa_llm_investigate import llm_investigate, setup_mock_client

if TYPE_CHECKING:
    import httpx
    from pytest_mock import MockerFixture

_REVIEWED = {"bugs": [{"content": "boo#1288770"}], "comments": None, "label": None, "reviewed": 1}
_LABELLED = {"bugs": [], "label": "x", "reviewed": 1}
_COMMENTED = {"bugs": [], "comments": 1, "label": None, "reviewed": None}
_URL = "https://openqa.opensuse.org/tests/123"


def _page(current: Any, latest: Any = 0) -> dict[str, Any]:
    """Return openQA's next & previous results with the comment data of the current and the latest job."""
    return {
        "info": [],
        "data": [{"id": 2, "islatest": 1, "comment_data": latest}, {"id": 1, "iscurrent": 1, "comment_data": current}],
    }


@pytest.mark.parametrize(
    ("comment_data", "reviewed"),
    [
        pytest.param(_REVIEWED, True, id="bugref"),
        pytest.param(_LABELLED, True, id="label"),
        pytest.param(_COMMENTED, False, id="comments-without-bugref-or-label"),
        pytest.param(0, False, id="no-comments"),
    ],
)
def test_is_reviewed_follows_openqa(make_openqa: Any, comment_data: Any, *, reviewed: bool) -> None:
    oqa = make_openqa({"/tests/1/ajax": json.dumps(_page(comment_data))})
    assert is_reviewed(oqa.client, oqa.server, "1") is reviewed


def test_is_reviewed_ignores_the_review_of_another_job(make_openqa: Any) -> None:
    oqa = make_openqa({"/tests/1/ajax": json.dumps(_page(_COMMENTED, latest=_REVIEWED))})
    assert not is_reviewed(oqa.client, oqa.server, "1")


def test_is_reviewed_asks_openqa_afresh_on_each_call(make_openqa: Any) -> None:
    answers = iter([_page(_COMMENTED), _page(_REVIEWED)])
    params = []

    def route(request: httpx.Request) -> str:
        params.append(dict(request.url.params))
        return json.dumps(next(answers))

    oqa = make_openqa({"/tests/1/ajax": route})
    assert not is_reviewed(oqa.client, oqa.server, "1")
    assert is_reviewed(oqa.client, oqa.server, "1")
    assert params == [{"previous_limit": "0", "next_limit": "0"}] * 2


@pytest.mark.parametrize(
    "routes",
    [
        pytest.param({}, id="not-found"),
        pytest.param({"/tests/1/ajax": "[]"}, id="malformed-page"),
        pytest.param(
            {"/tests/1/ajax": json.dumps({"data": [{"id": 2, "comment_data": _REVIEWED}]})}, id="no-current-entry"
        ),
    ],
)
def test_is_reviewed_warns_and_says_no_when_openqa_cannot_tell(
    make_openqa: Any, routes: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    oqa = make_openqa(routes)
    with caplog.at_level(logging.WARNING):
        assert not is_reviewed(oqa.client, oqa.server, "1")
    assert "Could not check whether job 1 is reviewed" in caplog.text


def test_investigate_leaves_a_reviewed_job_to_the_standard_investigation(mocker: MockerFixture) -> None:
    mock_post = mocker.patch("llm_investigate.post_comment")
    mock_print = mocker.patch("llm_investigate.typer.echo")
    client = setup_mock_client(mocker, overrides={"/ajax?": _page(_REVIEWED)})

    llm_investigate.investigate("123")

    mock_print.assert_called_once_with(_URL)
    client.post.assert_not_called()
    client.stream.assert_not_called()
    mock_post.assert_not_called()
    fetched = [f"{c.args[0]}?{urllib.parse.urlencode(c.kwargs.get('params') or {})}" for c in client.get.call_args_list]
    assert not [url for url in fetched if "details_ajax" in url or "/api/v1/jobs?test=" in url]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"api/v1/jobs": {"job": {"id": 123, "result": "passed"}}}, id="passed"),
        pytest.param({"comments": [{"text": "**LLM Investigation summary:** done"}]}, id="llm-summary"),
    ],
)
def test_investigate_exits_before_the_review_check(mocker: MockerFixture, overrides: dict[str, Any]) -> None:
    mock_print = mocker.patch("llm_investigate.typer.echo")
    client = setup_mock_client(mocker, overrides={"/ajax?": _page(_REVIEWED), **overrides})

    with pytest.raises(SystemExit) as exc:
        llm_investigate.investigate("123")

    assert exc.value.code == 0
    mock_print.assert_not_called()
    assert not any("/ajax?" in c.args[0] for c in client.get.call_args_list)


@pytest.mark.parametrize(
    ("verdict", "printed"),
    [
        pytest.param("BISECT: YES \u2014 isolated", True, id="bisect-yes"),
        pytest.param("BISECT: NO \u2014 known issue", False, id="bisect-no"),
    ],
)
def test_investigate_keeps_the_bisect_gate_but_does_not_post_when_a_review_arrives_meanwhile(
    mocker: MockerFixture, verdict: str, *, printed: bool
) -> None:
    mock_post = mocker.patch("llm_investigate.post_comment")
    mock_print = mocker.patch("llm_investigate.typer.echo")
    client = setup_mock_client(mocker)
    pages = iter([_page(_COMMENTED), _page(_REVIEWED)])
    replies = {1: "* isosize: died", 3: "The module died.", 5: f"Classification: Product bug\n{verdict}"}
    serve = client.get.side_effect

    def get(url: str, *args: Any, **kwargs: Any) -> Any:
        return Mock(text=json.dumps(next(pages))) if "/ajax?" in url else serve(url, *args, **kwargs)

    def post(*_args: Any, **kwargs: Any) -> Mock:
        reply = replies[len(kwargs["json"]["messages"])]
        return Mock(json=Mock(return_value={"choices": [{"message": {"content": reply}}]}))

    client.get.side_effect = get
    client.post.side_effect = post

    llm_investigate.investigate("123")

    assert client.post.call_count == 3
    assert mock_print.call_args_list == ([call(_URL)] if printed else [])
    mock_post.assert_not_called()


@pytest.mark.parametrize(
    ("option", "output"),
    [
        pytest.param("dry", "**LLM Investigation summary:**", id="dry"),
        pytest.param("print_prompt", "===== Prompt 1/3 =====", id="print-prompt"),
    ],
)
def test_investigate_dry_and_print_prompt_skip_the_review_check(
    mocker: MockerFixture, option: str, output: str
) -> None:
    mock_post = mocker.patch("llm_investigate.post_comment")
    mock_print = mocker.patch("llm_investigate.typer.echo")
    client = setup_mock_client(mocker, overrides={"/ajax?": _page(_REVIEWED)})

    llm_investigate.investigate("123", **{option: True})

    assert any(output in c.args[0] for c in mock_print.call_args_list)
    assert not any("/ajax?" in c.args[0] for c in client.get.call_args_list)
    mock_post.assert_not_called()


def test_investigate_force_posts_on_a_reviewed_job(mocker: MockerFixture) -> None:
    mock_post = mocker.patch("llm_investigate.post_comment")
    mocker.patch("llm_investigate.typer.echo")
    client = setup_mock_client(mocker, overrides={"/ajax?": _page(_REVIEWED)})

    llm_investigate.investigate("123", force=True)

    mock_post.assert_called_once()
    assert not any("/ajax?" in c.args[0] for c in client.get.call_args_list)
