"""Tests for walking the numbered result pages of a people or company search.

The capture is scripted per result page, so each test says what LinkedIn
showed on which page and checks only what the walk made of it: which pages it
asked for, what it returned, and where its cursor resumes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

from linkedin_mcp_server.core import pacing
from linkedin_mcp_server.core.pacing import PacingRefusedError
from linkedin_mcp_server.linkedin import paging
from linkedin_mcp_server.linkedin.capture import CaptureMode, SectionCapture
from linkedin_mcp_server.linkedin.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    ExtractedSection,
)
from linkedin_mcp_server.linkedin.link_metadata import Reference
from linkedin_mcp_server.linkedin.paging import walk_result_pages
from linkedin_mcp_server.linkedin.search_urls import (
    build_people_search_url,
    read_cursor,
)


def search_page(page: int) -> str:
    return build_people_search_url("engineer", network=["F"], page=page)


def people(*names: str) -> list[Reference]:
    return [
        {"kind": "person", "url": f"/in/{name}/", "text": name.title()}
        for name in names
    ]


def shown(text: str, references: list[Reference] | None = None) -> ExtractedSection:
    return ExtractedSection(text=text, references=references or [])


class ScriptedPages:
    """A capture answering each result page from a script.

    An answer may be an exception, raised the way the real capture raises a
    refusal from the pacing gate. *seconds* is how long each page takes on
    *clock*, of which *gate_wait* is the gate's own pause.
    """

    def __init__(
        self,
        answers: dict[int, ExtractedSection | Exception],
        *,
        clock: list[float] | None = None,
        seconds: float = 0.0,
        gate: Any = None,
        gate_wait: float = 0.0,
    ):
        self.answers = answers
        self.asked: list[int] = []
        self.plans: list[Any] = []
        self._clock = clock
        self._seconds = seconds
        self._gate = gate
        self._gate_wait = gate_wait

    async def capture(self, url: str, *, section_name: str, plan: Any):
        assert section_name == "search_results"
        page = int(parse_qs(urlparse(url).query).get("page", ["1"])[0])
        self.asked.append(page)
        self.plans.append(plan)
        if self._clock is not None:
            self._clock[0] += self._seconds
        if self._gate is not None:
            self._gate.waited_seconds += self._gate_wait
        answer = self.answers[page]
        if isinstance(answer, Exception):
            raise answer
        return answer


async def walk(pages: ScriptedPages, **kwargs: Any) -> dict[str, Any]:
    return await walk_result_pages(
        cast(SectionCapture, pages), search_page, result_kind="person", **kwargs
    )


def resumes_at(result: dict[str, Any]) -> int:
    """The page the result's cursor resumes at, read the way the tool reads it."""
    cursor = result["pagination"]["next_cursor"]
    assert cursor is not None
    return read_cursor(cursor, search_page(1), first=1)


class TestReadingPages:
    async def test_pages_are_read_in_order_and_joined(self):
        pages = ScriptedPages(
            {
                1: shown("Ada\nBob", people("ada", "bob")),
                2: shown("Cy\nDee", people("cy", "dee")),
                3: shown("Eve", people("eve")),
            }
        )

        result = await walk(pages, max_pages=3)

        assert pages.asked == [1, 2, 3]
        assert all(plan.mode is CaptureMode.SEARCH_RESULTS for plan in pages.plans)
        assert result["url"] == search_page(1)
        assert result["sections"] == {
            "search_results": "Ada\nBob\n---\nCy\nDee\n---\nEve"
        }
        assert [ref["url"] for ref in result["references"]["search_results"]] == [
            "/in/ada/",
            "/in/bob/",
            "/in/cy/",
            "/in/dee/",
            "/in/eve/",
        ]
        assert result["pagination"]["pages_fetched"] == 3
        assert result["pagination"]["reason"] == "max_pages"
        assert resumes_at(result) == 4

    async def test_one_page_is_the_default(self):
        pages = ScriptedPages({1: shown("Ada", people("ada"))})

        result = await walk(pages)

        assert pages.asked == [1]
        assert result["pagination"]["reason"] == "max_pages"
        assert resumes_at(result) == 2

    async def test_a_cursor_starts_the_walk_at_its_page(self):
        pages = ScriptedPages(
            {4: shown("Ada", people("ada")), 5: shown("Bob", people("bob"))}
        )

        result = await walk(pages, first_page=4, max_pages=2)

        assert pages.asked == [4, 5]
        assert result["url"] == search_page(4)
        assert resumes_at(result) == 6

    async def test_a_page_past_the_last_ends_the_walk_without_a_cursor(self):
        """LinkedIn's no-results page links to nobody.

        Its text is dropped after real results, and the walk ends: a cursor
        here would buy another page load for the same empty answer.
        """
        pages = ScriptedPages(
            {1: shown("Ada", people("ada")), 2: shown("No results found")}
        )

        result = await walk(pages, max_pages=5)

        assert pages.asked == [1, 2]
        assert result["sections"] == {"search_results": "Ada"}
        assert result["pagination"] == {
            "pages_fetched": 2,
            "next_cursor": None,
            "reason": "complete",
        }

    async def test_a_search_with_no_results_keeps_linkedins_words_for_it(self):
        pages = ScriptedPages({1: shown("No results found")})

        result = await walk(pages, max_pages=3)

        assert pages.asked == [1]
        assert result["sections"] == {"search_results": "No results found"}
        assert result["pagination"]["reason"] == "complete"
        assert result["pagination"]["next_cursor"] is None

    async def test_an_empty_page_ends_the_walk(self):
        pages = ScriptedPages({1: shown("Ada", people("ada")), 2: shown("")})

        result = await walk(pages, max_pages=3)

        assert pages.asked == [1, 2]
        assert result["pagination"]["reason"] == "complete"
        assert result["pagination"]["next_cursor"] is None

    async def test_links_to_other_kinds_are_not_results(self):
        """A people page that links only to companies has no people on it."""
        company: list[Reference] = [{"kind": "company", "url": "/company/acme/"}]
        pages = ScriptedPages(
            {1: shown("Ada", people("ada")), 2: shown("Acme", company)}
        )

        result = await walk(pages, max_pages=3)

        assert pages.asked == [1, 2]
        assert result["pagination"]["reason"] == "complete"


class TestRepeatedResults:
    async def test_a_page_of_people_already_read_ends_the_walk(self):
        """LinkedIn serving an earlier page again, as past its last page.

        The repeat is neither returned nor resumable: the same people twice
        would read as more results, and a cursor would loop.
        """
        pages = ScriptedPages(
            {
                1: shown("Ada\nBob", people("ada", "bob")),
                2: shown("Cy", people("cy")),
                3: shown("Ada again", people("bob", "ada")),
            }
        )

        result = await walk(pages, max_pages=5)

        assert pages.asked == [1, 2, 3]
        assert result["sections"] == {"search_results": "Ada\nBob\n---\nCy"}
        assert result["pagination"] == {
            "pages_fetched": 3,
            "next_cursor": None,
            "reason": "no_new_results",
        }

    async def test_a_page_with_one_new_person_is_still_read(self):
        pages = ScriptedPages(
            {
                1: shown("Ada\nBob", people("ada", "bob")),
                2: shown("Bob\nCy", people("bob", "cy")),
            }
        )

        result = await walk(pages, max_pages=2)

        assert result["sections"] == {"search_results": "Ada\nBob\n---\nBob\nCy"}
        assert result["pagination"]["reason"] == "max_pages"


class TestRateLimits:
    async def test_a_refusal_after_a_page_keeps_it_and_resumes_at_the_refused_one(
        self,
    ):
        refusal = PacingRefusedError("Rate limit: 60 page visits an hour.", 1800)
        pages = ScriptedPages(
            {1: shown("Ada", people("ada")), 2: shown("Bob", people("bob")), 3: refusal}
        )

        result = await walk(pages, max_pages=5)

        assert pages.asked == [1, 2, 3]
        assert result["sections"] == {"search_results": "Ada\n---\nBob"}
        error = result["section_errors"]["search_results"]
        assert error["error_type"] == "rate_limit"
        assert error["retry_after"] == 1800
        assert "60 page visits an hour" in error["error_message"]
        assert result["pagination"]["reason"] == "rate_limited"
        assert result["pagination"]["pages_fetched"] == 2
        assert resumes_at(result) == 3

    async def test_a_refusal_before_any_page_is_the_answer(self):
        refusal = PacingRefusedError("Rate limit: stopped.", 900)
        pages = ScriptedPages({1: refusal})

        with pytest.raises(PacingRefusedError) as raised:
            await walk(pages, max_pages=3)

        assert raised.value is refusal

    async def test_linkedins_rate_limit_page_stops_the_walk_where_it_showed(self):
        pages = ScriptedPages(
            {1: shown("Ada", people("ada")), 2: shown(RATE_LIMITED_SECTION_TEXT)}
        )

        result = await walk(pages, max_pages=3)

        assert pages.asked == [1, 2]
        assert result["sections"] == {"search_results": "Ada"}
        assert result["section_errors"]["search_results"]["error_type"] == "rate_limit"
        assert result["pagination"]["reason"] == "rate_limited"
        assert resumes_at(result) == 2


class TestFailures:
    async def test_a_failed_page_keeps_what_was_read_and_says_why(self):
        error = {"error_type": "navigation_error", "error_message": "timeout"}
        pages = ScriptedPages(
            {
                1: shown("Ada", people("ada")),
                2: ExtractedSection(text="", references=[], error=error),
            }
        )

        result = await walk(pages, max_pages=3)

        assert result["sections"] == {"search_results": "Ada"}
        assert result["section_errors"] == {"search_results": error}
        # A failure is not a reason to stop that a cursor can answer.
        assert "pagination" not in result


class TestTimeBudget:
    @pytest.fixture
    def clock(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        now = [1000.0]
        monkeypatch.setattr(paging, "time", SimpleNamespace(monotonic=lambda: now[0]))
        return now

    @pytest.fixture
    def gate(self, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
        stub = SimpleNamespace(waited_seconds=0.0, expected_wait=lambda: 5.0)
        monkeypatch.setattr(pacing, "gate", lambda: stub)
        return stub

    def _pages(self, count: int, **kwargs: Any) -> ScriptedPages:
        return ScriptedPages(
            {n: shown(f"p{n}", people(f"person-{n}")) for n in range(1, count + 1)},
            **kwargs,
        )

    async def test_the_walk_stops_while_it_can_still_answer(self, clock, gate):
        """No page starts that would not end inside the tool's timeout.

        A tool FastMCP cancels returns nothing, so the pages already read are
        handed back with a cursor at the first unread one. Pages take 30 s and
        the gate will pause 5 s: after two, a third would end at 95 s of the
        80 s a 100 s timeout allows.
        """
        pages = self._pages(5, clock=clock, seconds=30.0)

        result = await walk(pages, max_pages=5, tool_timeout=100.0)

        assert pages.asked == [1, 2]
        assert result["pagination"]["reason"] == "time_budget"
        assert resumes_at(result) == 3

    async def test_the_gates_pauses_are_not_counted_as_page_time(self, clock, gate):
        """The pause is predicted separately, so it is taken out of the page.

        Here 20 of each page's 30 s are the gate's. Counted as page time, the
        prediction would add them twice and stop a page early.
        """
        pages = self._pages(5, clock=clock, seconds=30.0, gate=gate, gate_wait=20.0)

        result = await walk(pages, max_pages=5, tool_timeout=100.0)

        assert pages.asked == [1, 2, 3]
        assert result["pagination"]["reason"] == "time_budget"
        assert resumes_at(result) == 4

    async def test_the_first_page_is_always_tried(self, clock, gate):
        pages = self._pages(1, clock=clock, seconds=30.0)

        result = await walk(pages, max_pages=1, tool_timeout=1.0)

        assert pages.asked == [1]
        assert result["pagination"]["reason"] == "max_pages"
