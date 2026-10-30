"""Tests for walking the numbered result pages of a people or company search.

The capture is scripted per result page, so each test says what LinkedIn
showed on which page and checks only what the walk made of it: which pages it
asked for, what it returned, and where its cursor resumes.
"""

from __future__ import annotations

import time
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
    results_digest,
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
        on_capture: Any = None,
    ):
        self.answers = answers
        self._on_capture = on_capture
        self.asked: list[int] = []
        self.plans: list[Any] = []
        self.deadlines: list[Any] = []
        self._clock = clock
        self._seconds = seconds
        self._gate = gate
        self._gate_wait = gate_wait

    async def capture(
        self, url: str, *, section_name: str, plan: Any, deadline: Any = None
    ):
        assert section_name == "search_results"
        self.deadlines.append(deadline)
        page = int(parse_qs(urlparse(url).query).get("page", ["1"])[0])
        self.asked.append(page)
        self.plans.append(plan)
        if self._clock is not None:
            self._clock[0] += self._seconds
        if self._gate is not None:
            self._gate.waited_seconds += self._gate_wait
        if self._on_capture is not None:
            self._on_capture(page)
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
    return read_cursor(cursor, search_page(1), first=1).position


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

    async def test_two_pages_without_results_end_the_walk_without_a_cursor(self):
        """LinkedIn's no-results page links to nobody, and neither does the next.

        One such page after real results could be members shown without a
        profile link, so it takes a second in a row to be the end. A cursor
        there would buy another page load for the same empty answer.
        """
        pages = ScriptedPages(
            {
                1: shown("Ada", people("ada")),
                2: shown("No results found"),
                3: shown("No results found"),
            }
        )

        result = await walk(pages, max_pages=5)

        assert pages.asked == [1, 2, 3]
        assert result["sections"] == {"search_results": "Ada\n---\nNo results found"}
        assert result["pagination"] == {
            "pages_fetched": 3,
            "next_cursor": None,
            "reason": "complete",
        }

    async def test_a_page_of_members_without_profile_links_is_passed(self):
        """Out-of-network members show as cards with no link to a profile.

        Their page is kept and the walk goes on to the people behind it,
        where reading it as the end lost every page after it.
        """
        pages = ScriptedPages(
            {
                1: shown("Ada", people("ada")),
                2: shown("LinkedIn Member\nLinkedIn Member"),
                3: shown("Cy", people("cy")),
            }
        )

        result = await walk(pages, max_pages=3)

        assert pages.asked == [1, 2, 3]
        assert result["sections"] == {
            "search_results": "Ada\n---\nLinkedIn Member\nLinkedIn Member\n---\nCy"
        }
        assert result["pagination"]["reason"] == "max_pages"
        assert resumes_at(result) == 4

    async def test_a_resumed_page_without_results_after_one_ends_the_walk(self):
        """The cursor remembers that the page before had none either."""
        first = ScriptedPages(
            {1: shown("Ada", people("ada")), 2: shown("No results found")}
        )
        result = await walk(first, max_pages=2)
        assert result["pagination"]["reason"] == "max_pages"
        resume = read_cursor(
            result["pagination"]["next_cursor"], search_page(1), first=1
        )

        then = ScriptedPages({3: shown("No results found")})
        ended = await walk(
            then, first_page=resume.position, previous=resume.last_results
        )

        assert then.asked == [3]
        assert ended["pagination"] == {
            "pages_fetched": 1,
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
            {
                1: shown("Ada", people("ada")),
                2: shown("Acme", company),
                3: shown("Acme again", company),
            }
        )

        result = await walk(pages, max_pages=5)

        assert pages.asked == [1, 2, 3]
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


class TestOutOfTime:
    async def test_no_time_left_for_the_gates_wait_is_the_time_budget(self):
        """Not LinkedIn and not a limit, so no rate_limit error goes with it."""
        out_of_time = PacingRefusedError(
            "Rate limit: this call has no time left", 5, kind="time"
        )
        pages = ScriptedPages({1: shown("Ada", people("ada")), 2: out_of_time})

        result = await walk(pages, max_pages=3)

        assert "section_errors" not in result
        assert result["pagination"]["reason"] == "time_budget"
        assert resumes_at(result) == 2


def hold_back_loads() -> None:
    """Make the gate refuse the in-page loads a page asks for, as a stop does."""
    from linkedin_mcp_server.config.schema import RateLimitConfig

    gate = pacing.Pacing(RateLimitConfig(), None)
    gate.record_pushback("http_429")
    pacing.install_gate(gate)
    assert gate.loads_allowed(1) == 0


class TestAPageCutShort:
    async def test_a_page_the_rate_limit_stopped_loading_is_read_again(self):
        """Its results are kept, and the cursor reads it again in full."""

        def on_capture(page: int) -> None:
            if page == 2:
                hold_back_loads()

        pages = ScriptedPages(
            {
                1: shown("Ada", people("ada")),
                2: shown("Bob", people("bob")),
                3: shown("Cy", people("cy")),
            },
            on_capture=on_capture,
        )

        with pacing.tool_call(deadline=None):
            result = await walk(pages, max_pages=3)

        assert pages.asked == [1, 2]
        assert result["sections"] == {"search_results": "Ada\n---\nBob"}
        error = result["section_errors"]["search_results"]
        assert error["error_type"] == "rate_limit"
        assert error["retry_after"] > 0
        assert result["pagination"]["reason"] == "rate_limited"
        assert resumes_at(result) == 2
        # The page before it is what the re-read compares with, so the page
        # read again is not taken for a repeat of itself.
        resume = read_cursor(
            result["pagination"]["next_cursor"], search_page(1), first=1
        )
        assert resume.last_results == results_digest(["/in/ada/"])


class TestACallerPagingOnACursor:
    async def test_paging_one_page_at_a_time_stops_on_a_repeated_page(self):
        """LinkedIn serving the last page again past the end ends the loop.

        Each call starts afresh, so only the cursor can remember what the page
        before it held; without that, every call took the repeat for new
        people and handed out another cursor.
        """
        repeated = shown("Bob", people("bob"))
        pages = ScriptedPages(
            {1: shown("Ada", people("ada")), **{n: repeated for n in range(2, 11)}}
        )
        cursor = None
        returned = []
        for _ in range(10):
            resume = read_cursor(cursor, search_page(1), first=1)
            result = await walk(
                pages, first_page=resume.position, previous=resume.last_results
            )
            returned.append(result["sections"].get("search_results"))
            cursor = result["pagination"]["next_cursor"]
            if cursor is None:
                break

        assert pages.asked == [1, 2, 3]
        assert returned == ["Ada", "Bob", None]
        assert result["pagination"]["reason"] == "no_new_results"


class TestWhereAPageLands:
    async def test_a_page_linkedin_answers_with_another_is_a_failure(self):
        """A dropped `page` serves page one again, under page two's number."""
        pages = ScriptedPages(
            {1: shown("Ada", people("ada")), 2: shown("Ada", people("ada"))}
        )

        result = await walk(pages, max_pages=3, landed_url=lambda: search_page(1))

        assert pages.asked == [1, 2]
        assert result["sections"] == {"search_results": "Ada"}
        assert result["section_errors"]["search_results"]["error_type"] == (
            "dropped_page"
        )
        assert "pagination" not in result

    @pytest.mark.parametrize(
        ("landed", "page", "on_page"),
        [
            (search_page(2) + "&sid=x&origin=GLOBAL_SEARCH_HEADER", 2, True),
            (search_page(1), 1, True),
            (search_page(1) + "&page=1", 1, True),
            (search_page(1), 2, False),
            (search_page(3), 2, False),
            (
                "https://www.linkedin.com/search/results/companies/?keywords=engineer&page=2",
                2,
                False,
            ),
            ("https://portal.invalid/search/results/people/?page=2", 2, False),
        ],
    )
    def test_a_page_is_its_route_and_its_number(self, landed, page, on_page):
        assert paging._on_page(landed, search_page(page), page) is on_page


class TestTheLastPage:
    async def test_a_cursor_past_the_last_page_reads_nothing(self):
        pages = ScriptedPages({})

        result = await walk(pages, first_page=101)

        assert pages.asked == []
        assert result["pagination"] == {
            "pages_fetched": 0,
            "next_cursor": None,
            "reason": "complete",
        }

    async def test_the_walk_ends_at_the_last_page(self):
        pages = ScriptedPages({100: shown("Ada", people("ada"))})

        result = await walk(pages, first_page=100, max_pages=5)

        assert pages.asked == [100]
        assert result["pagination"]["reason"] == "complete"
        assert result["pagination"]["next_cursor"] is None


class TestLinkedInsOwnRateLimitPage:
    @pytest.fixture
    def stopped(self):
        """A gate that LinkedIn's rate-limit page has just stopped."""
        from linkedin_mcp_server.config.schema import RateLimitConfig

        gate = pacing.Pacing(RateLimitConfig(), None)
        pacing.install_gate(gate)
        return gate

    async def test_raised_after_a_page_keeps_it_and_says_when(self, stopped):
        from linkedin_mcp_server.core.exceptions import RateLimitError

        def on_capture(page: int) -> None:
            if page == 2:
                stopped.record_pushback("slow_down_page")

        pages = ScriptedPages(
            {
                1: shown("Ada", people("ada")),
                2: RateLimitError("Rate limit detected", suggested_wait_time=300),
            },
            on_capture=on_capture,
        )

        result = await walk(pages, max_pages=3)

        assert result["sections"] == {"search_results": "Ada"}
        error = result["section_errors"]["search_results"]
        assert (error["error_type"], error["retry_after"]) == ("rate_limit", 900)
        assert result["pagination"]["reason"] == "rate_limited"
        assert resumes_at(result) == 2

    async def test_raised_before_any_page_is_the_answer(self, stopped):
        from linkedin_mcp_server.core.exceptions import RateLimitError

        pages = ScriptedPages({1: RateLimitError("Rate limit detected")})

        with pytest.raises(RateLimitError):
            await walk(pages, max_pages=3)

    async def test_its_empty_page_says_when_to_come_back(self, stopped):
        def on_capture(page: int) -> None:
            if page == 2:
                stopped.record_pushback("empty_page")

        pages = ScriptedPages(
            {1: shown("Ada", people("ada")), 2: shown(RATE_LIMITED_SECTION_TEXT)},
            on_capture=on_capture,
        )

        result = await walk(pages, max_pages=3)

        assert result["section_errors"]["search_results"]["retry_after"] == 300
        assert resumes_at(result) == 2


class TestTheWalksDeadline:
    async def test_every_page_is_bounded_by_the_walks_time(self):
        """Its navigations, and the retry of an empty page, end before the
        call's own timeout would cancel everything read."""
        pages = ScriptedPages(
            {1: shown("Ada", people("ada")), 2: shown("Bob", people("bob"))}
        )
        before = time.monotonic()

        await walk(pages, max_pages=2, tool_timeout=100.0)

        assert all(deadline is not None for deadline in pages.deadlines)
        assert 0 < pages.deadlines[0] - before <= 80.0 + 1.0

    async def test_the_gates_wait_decides_whether_a_page_starts(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(paging, "time", SimpleNamespace(monotonic=lambda: now[0]))
        stub = SimpleNamespace(waited_seconds=0.0, expected_wait=lambda: 10.0)
        monkeypatch.setattr(pacing, "gate", lambda: stub)
        pages = ScriptedPages(
            {n: shown(f"p{n}", people(f"person-{n}")) for n in range(1, 4)},
            clock=now,
            seconds=25.0,
        )

        result = await walk(pages, max_pages=3, tool_timeout=100.0)

        # The third page would start at 50s, wait 10s and take 25s: 85s of
        # the 80s allowed. Without the wait it fits, at 75s.
        assert pages.asked == [1, 2]
        assert result["pagination"]["reason"] == "time_budget"
