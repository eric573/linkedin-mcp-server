"""Tests for core utility functions (rate-limit detection, scrolling, modals)."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from patchright.async_api import Page

from linkedin_mcp_server.core.exceptions import (
    AccountRestrictedError,
    AuthenticationError,
    RateLimitError,
)
from linkedin_mcp_server.core import utils
from linkedin_mcp_server.core.utils import (
    detect_rate_limit,
    scroll_job_sidebar,
    scroll_to_bottom,
)


@pytest.fixture
def mock_page():
    """Create a mock Patchright page for rate-limit tests."""
    page = MagicMock()
    page.url = "https://www.linkedin.com/in/testuser/details/experience/"

    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_locator.inner_text = AsyncMock(return_value="")
    page.locator = MagicMock(return_value=mock_locator)
    return page


class TestDetectRateLimit:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/checkpoint/challenge/123",
            "https://www.linkedin.com/authwall?trk=login",
        ],
    )
    async def test_challenge_url_needs_login(self, mock_page, url):
        """Waiting clears no challenge, so it must not read as a rate limit."""
        mock_page.url = url
        with pytest.raises(AuthenticationError, match="--login"):
            await detect_rate_limit(mock_page)

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/flagship-web/login/login-restriction/",
            # Also a `/login` route, so only the order keeps it a restriction.
            "https://www.linkedin.com/login/login-restriction/",
        ],
    )
    async def test_restriction_url_is_not_a_login(self, mock_page, url):
        mock_page.url = url
        with pytest.raises(AccountRestrictedError):
            await detect_rate_limit(mock_page)

    async def test_challenge_word_in_slug_passes(self, mock_page):
        """A false match here would retire a working session."""
        mock_page.url = "https://www.linkedin.com/in/authwall-checkpoint/"
        await detect_rate_limit(mock_page)

    async def test_normal_page_with_main_skips_body_heuristic(self, mock_page):
        """A normal page with <main> should NOT trigger body text checks."""
        main_locator = MagicMock()
        main_locator.count = AsyncMock(return_value=1)

        body_locator = MagicMock()
        # Body contains a phrase that would false-positive
        body_locator.inner_text = AsyncMock(
            return_value="Helping SaaS teams slow down churn with data-driven retention"
        )

        def locator_side_effect(selector):
            if selector == "main":
                return main_locator
            if selector == "body":
                return body_locator
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        # Should NOT raise — the page has <main>, so body heuristic is skipped
        await detect_rate_limit(mock_page)

    async def test_error_page_without_main_triggers_heuristic(self, mock_page):
        """A short error page without <main> with rate-limit text should raise."""
        main_locator = MagicMock()
        main_locator.count = AsyncMock(return_value=0)

        body_locator = MagicMock()
        body_locator.inner_text = AsyncMock(
            return_value="Too many requests. Slow down."
        )

        def locator_side_effect(selector):
            if selector == "main":
                return main_locator
            if selector == "body":
                return body_locator
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        with pytest.raises(RateLimitError, match="Rate limit message"):
            await detect_rate_limit(mock_page)

    async def test_long_body_without_main_does_not_trigger(self, mock_page):
        """A page without <main> but with long body text (>2000 chars) is not an error page."""
        main_locator = MagicMock()
        main_locator.count = AsyncMock(return_value=0)

        body_locator = MagicMock()
        # Long body with a matching phrase buried in content
        body_locator.inner_text = AsyncMock(
            return_value="x" * 2000 + " try again later"
        )

        def locator_side_effect(selector):
            if selector == "main":
                return main_locator
            if selector == "body":
                return body_locator
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        # Should NOT raise — body is too long to be an error page
        await detect_rate_limit(mock_page)

    async def test_normal_url_no_error_passes(self, mock_page):
        """A clean normal page passes all checks without raising."""
        main_locator = MagicMock()
        main_locator.count = AsyncMock(return_value=1)

        def locator_side_effect(selector):
            if selector == "main":
                return main_locator
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        await detect_rate_limit(mock_page)


class TestScrollDeadline:
    """A spent scroll budget must not turn into no deadline at all."""

    @staticmethod
    def _page() -> MagicMock:
        page = MagicMock()
        page.url = "https://www.linkedin.com/jobs/search/?keywords=python"
        page.wait_for_selector = AsyncMock()
        page.evaluate = AsyncMock(return_value={"status": "gone"})
        return page

    async def test_a_spent_budget_skips_the_scroll(self):
        """Patchright reads a zero timeout as no timeout.

        A search that has spent its budget would then wait on a page with no
        job card until the tool is cancelled, and cancellation throws away
        every page gathered before it.
        """
        page = self._page()

        assert await scroll_job_sidebar(page, deadline=0) is False
        page.wait_for_selector.assert_not_called()

    async def test_a_sliver_of_budget_is_still_a_timeout(self):
        """`int(0.0004 * 1000)` is zero, which the guard above does not catch."""
        page = self._page()

        await scroll_job_sidebar(page, deadline=0.0004)

        assert page.wait_for_selector.await_args.kwargs["timeout"] == 1

    async def test_a_deadline_spent_releasing_the_rail_still_ends_the_call(self):
        """Releasing the rail handle is shielded, so a tool deadline that falls
        inside it would otherwise be swallowed and the scroll return normally
        after the caller's time was up."""
        import asyncio

        import anyio

        page = self._page()
        holder = MagicMock()

        async def slow_dispose() -> None:
            await asyncio.sleep(0.3)

        holder.dispose = AsyncMock(side_effect=slow_dispose)
        page.evaluate_handle = AsyncMock(return_value=holder)

        with pytest.raises(TimeoutError):
            with anyio.fail_after(0.1):
                await scroll_job_sidebar(page)

        holder.dispose.assert_awaited_once()


@pytest.fixture
def gate():
    """A gate with room for two visits an hour, ten loads, and no pause."""
    from linkedin_mcp_server.config.schema import RateLimitConfig
    from linkedin_mcp_server.core import pacing

    gate = pacing.Pacing(
        RateLimitConfig(visits_per_hour=2, pause_min_seconds=0, pause_max_seconds=0),
        None,
    )
    pacing.install_gate(gate)
    return gate


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Every pause the scrolls take, taken at once."""
    import asyncio

    taken: list[float] = []

    async def sleep(seconds: float) -> None:
        taken.append(seconds)

    class _Asyncio:
        """The real module, with only its sleep replaced."""

        def __getattr__(self, name: str):
            return sleep if name == "sleep" else getattr(asyncio, name)

    monkeypatch.setattr(utils, "asyncio", _Asyncio())
    return taken


class _GrowingPage:
    """A page body that grows by one batch per scroll, *batches* times."""

    def __init__(self, batches: int, on_scroll=None):
        self.url = "https://www.linkedin.com/in/someone/"
        self.height = 1000
        self.batches = batches
        self.scrolls = 0
        self._on_scroll = on_scroll

    async def evaluate(self, script: str):
        if script.startswith("window.scrollTo"):
            self.scrolls += 1
            if self.batches > 0:
                self.batches -= 1
                self.height += 500
            if self._on_scroll is not None:
                self._on_scroll(self.scrolls)
            return None
        return self.height


class TestBodyScrollsAreCounted:
    """A scroll that grows the page made LinkedIn send more, on any page."""

    async def test_each_scroll_that_grows_the_page_is_counted(self, gate, sleeps):
        page = _GrowingPage(batches=3)

        loaded = await scroll_to_bottom(
            cast(Page, page), pause_time=0.5, max_scrolls=10
        )

        assert loaded == 3
        assert page.scrolls == 4
        assert gate.loads_allowed(20) == 7

    async def test_the_scroll_that_finds_the_page_complete_costs_nothing(
        self, gate, sleeps
    ):
        page = _GrowingPage(batches=0)

        assert (
            await scroll_to_bottom(cast(Page, page), pause_time=0.5, max_scrolls=10)
            == 0
        )
        assert page.scrolls == 1
        assert gate.loads_allowed(20) == 10

    async def test_scrolling_ends_where_the_budget_does(self, gate, sleeps):
        page = _GrowingPage(batches=50)

        loaded = await scroll_to_bottom(
            cast(Page, page), pause_time=0.5, max_scrolls=20
        )

        assert loaded == 10
        assert page.scrolls == 10
        assert gate.loads_allowed(1) == 0

    async def test_a_stop_ends_the_scrolling(self, gate, sleeps):
        """LinkedIn refusing a batch stops the scrolls that would have followed."""

        def refuse_the_second(scrolls: int) -> None:
            if scrolls == 2:
                gate.record_pushback("http_429", retry_after=None)

        page = _GrowingPage(batches=50, on_scroll=refuse_the_second)

        await scroll_to_bottom(cast(Page, page), pause_time=0.5, max_scrolls=20)

        assert page.scrolls == 2

    async def test_loads_are_never_closer_than_a_second(self, gate, sleeps):
        page = _GrowingPage(batches=4)

        await scroll_to_bottom(cast(Page, page), pause_time=0.5, max_scrolls=10)

        assert len(sleeps) == 5
        assert all(1.0 <= pause <= 1.5 for pause in sleeps)


class TestRailScrollsAreCounted:
    """Each round of the job rail asks LinkedIn for the next batch of cards."""

    @staticmethod
    def _rail(monkeypatch, batches: int, state: dict | None = None) -> MagicMock:
        """A rail whose cards arrive after each scroll, *batches* times.

        *state* counts the scrolls that asked for a batch.
        """
        state = state if state is not None else {}
        state.update(cards=7, pending=0, batches=batches, asked=0)

        async def step(page, holder, *, scroll):
            if scroll:
                state["asked"] += 1
                if state["batches"] > 0:
                    state["batches"] -= 1
                    state["pending"] += 4
            else:
                state["cards"] += state["pending"]
                state["pending"] = 0
            return {"status": "ok", "cards": state["cards"], "height": state["cards"]}

        monkeypatch.setattr(utils, "_rail_step", step)
        page = MagicMock()
        page.url = "https://www.linkedin.com/jobs/search/?keywords=python"
        page.wait_for_selector = AsyncMock()
        holder = MagicMock()
        holder.dispose = AsyncMock()
        page.evaluate_handle = AsyncMock(return_value=holder)
        return page

    async def test_each_round_that_brings_cards_is_counted_and_spaced(
        self, gate, sleeps, monkeypatch
    ):
        page = self._rail(monkeypatch, batches=3)

        await scroll_job_sidebar(page, max_scrolls=10)

        assert gate.loads_allowed(20) == 7
        # A load's pause before every round after one that loaded; the polls
        # for a batch are a fraction of a second.
        assert len([pause for pause in sleeps if pause >= 1.0]) == 3

    async def test_the_rail_stops_where_the_budget_does(
        self, gate, sleeps, monkeypatch
    ):
        gate.record_loads(8)
        state: dict = {}
        page = self._rail(monkeypatch, batches=10, state=state)

        await scroll_job_sidebar(page, max_scrolls=10)

        # Room for two more loads: two rounds, and no third scroll at all.
        assert state["asked"] == 2
        assert gate.loads_allowed(1) == 0
