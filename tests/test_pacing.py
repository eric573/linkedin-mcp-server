"""The gate in front of everything the server makes LinkedIn do.

Driven by a fake clock, so waits are recorded rather than slept, and by a
random source pinned to one point of each range, so a pause is a known number.
"""

from __future__ import annotations

import ast
import json
import os
import random
import stat
from pathlib import Path
from typing import Any, cast

import pytest

from linkedin_mcp_server.config.schema import RateLimitConfig
from linkedin_mcp_server.core import pacing
from linkedin_mcp_server.core.pacing import Clock, Pacing, PacingRefusedError

_START = 1_793_286_000.0  # 2026-10-29 15:00 UTC
_SEARCH = "https://www.linkedin.com/search/results/people/?keywords=x"
_PROFILE = "https://www.linkedin.com/in/someone/"


class FakeClock:
    def __init__(self, start: float = _START) -> None:
        self.now = start
        self.mono = 5_000.0
        self.slept: list[float] = []

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.mono += seconds

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)

    def clock(self) -> Clock:
        return Clock(
            wall=lambda: self.now, monotonic=lambda: self.mono, sleep=self.sleep
        )


class PinnedRandom(random.Random):
    """``uniform`` always lands *fraction* of the way through its range."""

    def __init__(self, fraction: float) -> None:
        super().__init__()
        self.fraction = fraction

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.fraction


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def record_path(profile_dir: Path) -> Path:
    # Beside the saved session, where the server keeps it.
    return profile_dir.parent / pacing.RECORD_FILE


def make_gate(
    clock: FakeClock, path: Path | None, *, fraction: float = 0.5, **limits: Any
) -> Pacing:
    settings = RateLimitConfig(**limits)
    settings.validate()
    return Pacing(settings, path, clock=clock.clock(), rng=PinnedRandom(fraction))


async def visit(
    gate: Pacing, url: str = _PROFILE, *, deadline: float | None = None
) -> None:
    await gate.before_visit(url, deadline=deadline)
    gate.after_visit()


class TestPause:
    async def test_the_first_visit_does_not_wait(self, clock, record_path):
        gate = make_gate(clock, record_path)
        with pacing.tool_call(deadline=None):
            await gate.before_visit(_PROFILE)
        assert clock.slept == []

    async def test_the_next_visit_waits_out_the_pause_from_when_the_page_loaded(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)  # pause 3-6, pinned to 4.5
        with pacing.tool_call(deadline=None):
            await visit(gate)
            clock.advance(1.0)
            await gate.before_visit(_PROFILE)
        assert clock.slept == [pytest.approx(3.5)]

    async def test_time_spent_reading_the_page_counts_toward_the_pause(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)
        with pacing.tool_call(deadline=None):
            await visit(gate)
            clock.advance(10.0)
            await gate.before_visit(_PROFILE)
        assert clock.slept == []

    @pytest.mark.parametrize(("fraction", "pause"), [(0.0, 2.0), (1.0, 8.0)])
    async def test_the_pause_is_drawn_from_the_configured_range(
        self, clock, record_path, fraction, pause
    ):
        gate = make_gate(
            clock,
            record_path,
            fraction=fraction,
            pause_min_seconds=2.0,
            pause_max_seconds=8.0,
        )
        await visit(gate)
        await gate.before_visit(_PROFILE)
        assert clock.slept == [pytest.approx(pause)]


class TestPauseAfterInPageLoads:
    """The next page visit is paced from the last thing that loaded."""

    async def test_a_thread_opened_after_the_page_restarts_the_full_pause(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)  # pause 3-6, pinned to 4.5
        with pacing.tool_call(deadline=None):
            await visit(gate)
            clock.advance(10.0)
            gate.record_conversation_opens(1)
            clock.advance(1.0)
            await gate.before_visit(_PROFILE)
        assert clock.slept == [pytest.approx(3.5)]

    async def test_a_load_after_the_page_restarts_a_loads_pause(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)  # load pause 1.0 x 1.25 at midpoint
        with pacing.tool_call(deadline=None):
            await visit(gate)
            clock.advance(10.0)
            gate.record_loads(1)
            clock.advance(0.5)
            await gate.before_visit(_PROFILE)
        assert clock.slept == [pytest.approx(0.75)]

    async def test_a_load_never_shortens_a_longer_pause(self, clock, record_path):
        gate = make_gate(clock, record_path)
        with pacing.tool_call(deadline=None):
            await visit(gate)
            gate.record_loads(1)
            await gate.before_visit(_PROFILE)
        assert clock.slept == [pytest.approx(4.5)]


class TestBudget:
    async def test_a_call_is_refused_once_the_hour_is_used_up(self, clock, record_path):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=3,
        )
        with pacing.tool_call(deadline=None):
            for _ in range(3):
                await visit(gate)
                clock.advance(60)
            with pytest.raises(
                PacingRefusedError, match="3 of the 3 page visits"
            ) as refused:
                await gate.before_visit(_PROFILE)
        # The first visit leaves the hour 3600s after it was made, 180s ago.
        assert refused.value.retry_after == 3600 - 180
        assert gate.refusal_for_new_call() is not None

    async def test_the_hour_frees_up_as_old_visits_age_out(self, clock, record_path):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=2,
        )
        with pacing.tool_call(deadline=None):
            await visit(gate)
            await visit(gate)
            clock.advance(3601)
            await gate.before_visit(_PROFILE)
        assert gate.status()["visits"]["last_hour"] == 1

    async def test_a_short_wait_for_budget_is_waited_out_rather_than_refused(
        self, clock, record_path
    ):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=2,
        )
        with pacing.tool_call(deadline=None):
            await visit(gate)
            await visit(gate)
            clock.advance(3600 - 10)
            assert gate.refusal_for_new_call() is None
            await gate.before_visit(_PROFILE)
        assert clock.slept == [pytest.approx(10)]

    async def test_the_day_limit_holds_after_the_hour_frees_up(
        self, clock, record_path
    ):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=0,
            visits_per_day=2,
        )
        with pacing.tool_call(deadline=None):
            await visit(gate)
            await visit(gate)
            clock.advance(7200)
            with pytest.raises(PacingRefusedError, match="any 24 hours") as refused:
                await gate.before_visit(_PROFILE)
        assert refused.value.retry_after == 86400 - 7200

    async def test_zero_turns_both_limits_off(self, clock, record_path):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=0,
            visits_per_day=0,
        )
        with pacing.tool_call(deadline=None):
            for _ in range(500):
                await visit(gate)

    async def test_five_scrolls_cost_one_visit(self, clock, record_path):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=2,
        )
        assert gate.loads_allowed(20) == 10
        gate.record_loads(5)
        assert gate.loads_allowed(20) == 5
        with pacing.tool_call(deadline=None):
            await visit(gate)
            with pytest.raises(PacingRefusedError):
                await gate.before_visit(_PROFILE)
        assert gate.loads_allowed(3) == 0

    async def test_opening_a_conversation_costs_a_visit(self, clock, record_path):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=3,
        )
        assert gate.conversation_opens_allowed(10) == 3
        assert gate.conversation_opens_allowed(None) == 3
        gate.record_conversation_opens(3)
        assert gate.conversation_opens_allowed(10) == 0
        with pacing.tool_call(deadline=None):
            with pytest.raises(PacingRefusedError):
                await gate.before_visit(_PROFILE)

    async def test_outside_a_tool_call_a_short_wait_is_waited_out(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)
        await visit(gate)
        await visit(gate)
        assert gate.status()["visits"]["last_hour"] == 2
        assert clock.slept == [pytest.approx(4.5)]

    async def test_outside_a_tool_call_a_spent_limit_refuses(self, clock, record_path):
        """The CLI and a sign-in repair spend the same budget as a tool call."""
        gate = make_gate(
            clock,
            record_path,
            visits_per_hour=1,
            pause_min_seconds=0,
            pause_max_seconds=0,
        )
        await visit(gate)
        with pytest.raises(PacingRefusedError) as refused:
            await gate.before_visit(_PROFILE)
        assert refused.value.kind == "limit"
        assert gate.status()["visits"]["last_hour"] == 1

    async def test_outside_a_tool_call_a_stop_refuses(self, clock, record_path):
        """A stop holds for every process and every path, not only tool calls."""
        gate = make_gate(clock, record_path)
        gate.record_pushback("http_429")
        with pytest.raises(PacingRefusedError) as refused:
            await gate.before_visit(_PROFILE)
        assert refused.value.kind == "stop"
        assert gate.status()["visits"]["last_hour"] == 0

    async def test_once_a_change_is_submitted_the_call_is_only_paced(
        self, clock, record_path
    ):
        """Accepting an invitation counts toward no limit and still must be confirmed."""
        gate = make_gate(clock, record_path)
        with pacing.tool_call(deadline=None):
            await visit(gate)
            pacing.mark_write_submitted()
            gate.record_pushback("http_429")
            await gate.before_visit(_PROFILE)
        assert clock.slept == [pytest.approx(4.5)]


class TestAStopWhileWaiting:
    async def test_a_stop_that_starts_during_the_pause_refuses_the_visit(
        self, clock, record_path
    ):
        """The page left open keeps receiving answers while a visit waits."""
        gate = make_gate(clock, record_path)
        sleep = clock.sleep

        async def refused_while_waiting(seconds: float) -> None:
            await sleep(seconds)
            gate.record_pushback("http_429")

        gate._clock = Clock(
            wall=lambda: clock.now,
            monotonic=lambda: clock.mono,
            sleep=refused_while_waiting,
        )
        with pacing.tool_call(deadline=None):
            await visit(gate)
            with pytest.raises(PacingRefusedError, match="HTTP 429"):
                await gate.before_visit(_PROFILE)

        assert gate.status()["visits"]["last_hour"] == 1


class TestDeadline:
    async def test_a_call_without_time_for_the_wait_and_the_page_is_refused(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)
        with pacing.tool_call(deadline=clock.mono + 600):
            await visit(gate)
        with pacing.tool_call(deadline=clock.mono + 10):
            with pytest.raises(PacingRefusedError, match="no time left") as refused:
                await gate.before_visit(_PROFILE)
        assert refused.value.retry_after == 5
        assert clock.slept == []

    async def test_the_navigation_deadline_counts_as_well_as_the_calls(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)
        with pacing.tool_call(deadline=clock.mono + 600):
            await visit(gate)
            with pytest.raises(PacingRefusedError, match="no time left"):
                await gate.before_visit(_PROFILE, deadline=clock.mono + 12)

    async def test_conversations_opened_stop_where_the_call_runs_out_of_time(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)
        # Each open takes its 1.5-3s pause, 2.25s on average, and 1.2s to settle.
        with pacing.tool_call(deadline=clock.mono + 15 + 4 * 3.45 + 1):
            assert gate.conversation_opens_allowed(20) == 4


class TestWrites:
    async def test_invitations_stop_at_the_daily_limit(self, clock, record_path):
        gate = make_gate(clock, record_path, invites_per_day=2)
        for _ in range(2):
            gate.check_write("invitation")
            gate.record_write("invitation")
            clock.advance(100)
        with pytest.raises(
            PacingRefusedError, match="2 connection invitations"
        ) as refused:
            gate.check_write("invitation")
        assert refused.value.retry_after == 86400 - 200

    async def test_invitations_stop_at_the_weekly_limit(self, clock, record_path):
        gate = make_gate(clock, record_path, invites_per_day=0, invites_per_week=3)
        for _ in range(3):
            gate.check_write("invitation")
            gate.record_write("invitation")
            clock.advance(86400)
        with pytest.raises(PacingRefusedError, match="last 7 days"):
            gate.check_write("invitation")
        # Messages have a limit of their own.
        gate.check_write("message")

    async def test_messages_stop_at_the_daily_limit(self, clock, record_path):
        gate = make_gate(clock, record_path, messages_per_day=1)
        gate.check_write("message")
        gate.record_write("message")
        with pytest.raises(PacingRefusedError, match="1 messages"):
            gate.check_write("message")
        gate.check_write("invitation")

    async def test_after_a_write_the_call_is_paced_but_never_refused(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path, visits_per_hour=1)
        with pacing.tool_call(deadline=None):
            await visit(gate)
            gate.record_write("invitation")
            # The hour is used up, and LinkedIn even pushed back: the read that
            # confirms the invitation still happens.
            gate.record_pushback("http_429")
            await gate.before_visit(_PROFILE)
        assert clock.slept == [pytest.approx(4.5)]
        with pacing.tool_call(deadline=None):
            with pytest.raises(PacingRefusedError):
                await gate.before_visit(_PROFILE)


class TestStops:
    async def test_a_429_stops_every_kind_of_request(self, clock, record_path):
        gate = make_gate(clock, record_path)
        gate.record_pushback("http_429")

        assert gate.stop_active()
        assert gate.stop_seconds_left() == pytest.approx(900)
        refusal = gate.refusal_for_new_call()
        assert refusal is not None and refusal.retry_after == 900
        assert "HTTP 429" in str(refusal)
        with pacing.tool_call(deadline=None):
            with pytest.raises(PacingRefusedError):
                await gate.before_visit(_PROFILE)
        with pytest.raises(PacingRefusedError):
            gate.check_write("message")
        assert gate.loads_allowed(5) == 0
        assert gate.conversation_opens_allowed(5) == 0

        clock.advance(901)
        assert not gate.stop_active()
        with pacing.tool_call(deadline=None):
            await gate.before_visit(_PROFILE)

    @pytest.mark.parametrize(("asked", "stop"), [(3000, 3000), (60, 900)])
    async def test_the_longer_of_linkedins_wait_and_the_servers_wins(
        self, clock, record_path, asked, stop
    ):
        gate = make_gate(clock, record_path)
        gate.record_pushback("http_999", retry_after=asked)
        assert gate.stop_seconds_left() == pytest.approx(stop)

    async def test_each_repeat_within_a_day_doubles_the_stop(self, clock, record_path):
        gate = make_gate(clock, record_path)
        lengths = []
        for _ in range(3):
            gate.record_pushback("http_429")
            lengths.append(gate.stop_seconds_left())
            clock.advance(gate.stop_seconds_left() + 1)
        assert lengths == [pytest.approx(900), pytest.approx(1800), pytest.approx(3600)]

        clock.advance(86400)
        gate.record_pushback("http_429")
        assert gate.stop_seconds_left() == pytest.approx(900)

    async def test_pushback_that_keeps_coming_backs_off_however_spaced(
        self, clock, record_path
    ):
        """Each stop within a day of the one before doubles that one."""
        gate = make_gate(clock, record_path)
        lengths = []
        for _ in range(4):
            gate.record_pushback("http_429")
            lengths.append(gate.stop_seconds_left())
            clock.advance(12 * 3600)
        assert lengths == [
            pytest.approx(900),
            pytest.approx(1800),
            pytest.approx(3600),
            pytest.approx(7200),
        ]

    async def test_linkedins_wait_is_capped_at_the_longest_stop(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)
        gate.record_pushback("http_429", retry_after=10**9)
        assert gate.stop_seconds_left() == pytest.approx(86400)
        refusal = gate.refusal_for_new_call()
        assert refusal is not None and refusal.retry_after == 86400

    async def test_a_burst_of_refusals_is_one_stop(self, clock, record_path):
        gate = make_gate(clock, record_path)
        for _ in range(5):
            gate.record_pushback("http_429")
        assert gate.stop_seconds_left() == pytest.approx(900)

        clock.advance(901)
        gate.record_pushback("http_429")
        assert gate.stop_seconds_left() == pytest.approx(1800)

    async def test_doubling_never_runs_past_a_day(self, clock, record_path):
        gate = make_gate(clock, record_path)
        for _ in range(12):
            gate.record_pushback("http_429")
            clock.advance(gate.stop_seconds_left() + 1)
            # Keep every stop inside one rolling day.
            clock.now -= gate.stop_seconds_left()
        gate.record_pushback("http_429")
        assert gate.stop_seconds_left() <= 86400

    @pytest.mark.parametrize(
        ("kind", "stop"),
        [("challenge", 86400), ("restricted", 86400), ("empty_page", 300)],
    )
    async def test_each_signal_stops_for_its_own_time(
        self, clock, record_path, kind, stop
    ):
        gate = make_gate(clock, record_path)
        gate.record_pushback(kind)
        assert gate.stop_seconds_left() == pytest.approx(stop)

    async def test_a_challenge_lengthens_a_shorter_stop_already_running(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)
        gate.record_pushback("http_429")
        gate.record_pushback("challenge")
        assert gate.stop_seconds_left() == pytest.approx(86400)
        assert gate.status()["stop"]["reason"] == "challenge"

    async def test_a_stop_switched_off_stops_nothing(self, clock, record_path):
        gate = make_gate(clock, record_path, stop_minutes=0, challenge_stop_hours=0)
        gate.record_pushback("http_429", retry_after=600)
        gate.record_pushback("challenge")
        assert not gate.stop_active()


class TestAClockSetBack:
    """No limit outlasts what it reports, whatever the clock did."""

    async def test_visits_counted_while_the_clock_ran_ahead_age_out_from_now(
        self, clock, record_path
    ):
        gate = make_gate(
            clock,
            record_path,
            visits_per_hour=2,
            pause_min_seconds=0,
            pause_max_seconds=0,
        )
        clock.now += 8 * 3600
        await visit(gate)
        await visit(gate)
        clock.now -= 8 * 3600

        refusal = gate.refusal_for_new_call()
        assert refusal is not None and refusal.retry_after <= 3600
        clock.advance(3601)
        assert gate.refusal_for_new_call() is None

    async def test_a_stop_set_while_the_clock_ran_ahead_ends_after_its_length(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path)
        clock.now += 3 * 86400
        gate.record_pushback("http_429")
        clock.now -= 3 * 86400

        assert gate.stop_seconds_left() == pytest.approx(900)
        refusal = gate.refusal_for_new_call()
        assert refusal is not None and refusal.retry_after == 900
        clock.advance(901)
        assert not gate.stop_active()


class TestRecord:
    async def test_it_survives_a_restart(self, clock, record_path):
        first = make_gate(clock, record_path)
        await visit(first)
        first.record_write("invitation")
        first.record_pushback("http_429")

        second = make_gate(clock, record_path)
        second.refresh()

        status = second.status()
        assert status["visits"]["last_hour"] == 1
        assert status["invitations"]["last_24_hours"] == 1
        assert status["stop"]["reason"] == "http_429"

    async def test_another_processs_visits_are_adopted_on_refresh(
        self, clock, record_path
    ):
        mine = make_gate(clock, record_path)
        mine.refresh()
        theirs = make_gate(clock, record_path)
        for _ in range(3):
            await visit(theirs)
            clock.advance(10)

        mine.refresh()
        assert mine.status()["visits"]["last_hour"] == 3

    async def test_a_damaged_record_keeps_the_history_this_process_holds(
        self, clock, record_path
    ):
        """The process holding the lease had the newest record in memory."""
        gate = make_gate(clock, record_path, invites_per_day=2)
        await visit(gate)
        gate.record_write("invitation")
        gate.record_write("invitation")
        record_path.write_text("{not json", encoding="utf-8")

        gate.refresh()

        assert gate.stop_seconds_left() == pytest.approx(900)
        saved = json.loads(record_path.read_text())
        assert saved["stop"]["reason"] == "damaged_record"
        assert len(saved["invitations"]) == 2
        clock.advance(901)
        with pytest.raises(PacingRefusedError, match="2 connection invitations"):
            gate.check_write("invitation")

    async def test_a_damaged_record_with_no_history_stops_a_day_and_holds_invitations(
        self, clock, record_path
    ):
        """Nothing to fall back on: every window it can no longer count is spent."""
        record_path.write_text("", encoding="utf-8")
        gate = make_gate(clock, record_path)

        gate.refresh()

        assert gate.stop_seconds_left() == pytest.approx(86400)
        clock.advance(86401)
        gate.check_write("message")
        with pytest.raises(PacingRefusedError, match="damaged") as refused:
            gate.check_write("invitation")
        assert refused.value.retry_after == pytest.approx(6 * 86400, abs=2)
        assert "held_until" in gate.status()["invitations"]
        # The hold is part of the record, so a restart keeps it.
        restarted = make_gate(clock, record_path)
        restarted.refresh()
        with pytest.raises(PacingRefusedError, match="damaged"):
            restarted.check_write("invitation")

    async def test_a_record_from_a_newer_server_is_left_alone(self, clock, record_path):
        newer = json.dumps({"version": 2, "visits": [], "anything": "else"})
        record_path.write_text(newer, encoding="utf-8")
        gate = make_gate(clock, record_path)

        gate.refresh()
        gate.record_loads(1)

        assert gate.stop_active()
        assert gate.status()["stop"]["reason"] == "newer_record"
        with pytest.raises(PacingRefusedError, match="newer version"):
            await gate.before_visit(_PROFILE)
        assert record_path.read_text(encoding="utf-8") == newer

    async def test_reading_the_status_never_writes_the_record(
        self, clock, record_path, monkeypatch
    ):
        """get_session_status runs without the lease; only the lease holder repairs.

        Through the gate the process builds for itself on first use, which is
        what get_session_status meets in a process that has served no call.
        """
        record_path.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(pacing, "_gate", None)

        status = pacing.status_on_disk()

        assert "unreadable" in status["record"]
        assert record_path.read_text(encoding="utf-8") == "{not json"

    async def test_an_idle_process_never_writes_an_old_copy_over_a_newer_one(
        self, clock, record_path
    ):
        """A frontend's gate built days ago reads the owner's record before saving."""
        idle = make_gate(clock, record_path)
        idle.refresh()
        busy = make_gate(clock, record_path)
        for _ in range(3):
            await visit(busy)

        idle.record_loads(1)

        saved = json.loads(record_path.read_text())
        assert len(saved["visits"]) == 3
        assert len(saved["loads"]) == 1

    async def test_a_failed_save_is_tried_again_with_the_next_change(
        self, clock, record_path, monkeypatch
    ):
        from linkedin_mcp_server import common_utils

        gate = make_gate(clock, record_path)
        await visit(gate)
        real_write = common_utils.secure_write_text
        failures = iter([OSError("disk busy")])

        def flaky_write(path, text, *args, **kwargs):
            failure = next(failures, None)
            if failure is not None:
                raise failure
            return real_write(path, text, *args, **kwargs)

        monkeypatch.setattr(common_utils, "secure_write_text", flaky_write)
        gate.record_pushback("http_429")
        assert json.loads(record_path.read_text())["stop"] is None

        # Another process writes the file meanwhile. The stop this one could
        # not save is newer than that, so it is not read over: it is written.
        other = make_gate(clock, record_path)
        other.refresh()
        other.record_loads(1)
        gate.refresh()
        assert gate.stop_active()
        assert json.loads(record_path.read_text())["stop"]["reason"] == "http_429"

    async def test_a_missing_record_keeps_what_this_process_knows(
        self, clock, record_path
    ):
        gate = make_gate(clock, record_path, visits_per_hour=2)
        await visit(gate)
        record_path.unlink()
        gate.refresh()
        await visit(gate)

        assert json.loads(record_path.read_text())["visits"] == [
            pytest.approx(_START),
            pytest.approx(_START + 4.5),
        ]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    async def test_only_this_account_can_read_it(self, clock, record_path):
        gate = make_gate(clock, record_path)
        await visit(gate)
        assert stat.S_IMODE(record_path.stat().st_mode) == 0o600

    async def test_what_no_limit_looks_at_is_dropped(self, clock, record_path):
        gate = make_gate(clock, record_path)
        await visit(gate)
        gate.record_write("invitation")
        clock.advance(2 * 86400)
        await visit(gate)

        written = json.loads(record_path.read_text())
        # The visit from two days ago is gone; the invitation counts for a week.
        assert len(written["visits"]) == 1
        assert len(written["invitations"]) == 1

    async def test_setting_the_clock_back_frees_no_budget(self, clock, record_path):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=2,
        )
        await visit(gate)
        await visit(gate)
        clock.now -= 7200
        with pacing.tool_call(deadline=None):
            with pytest.raises(PacingRefusedError):
                await gate.before_visit(_PROFILE)

    async def test_nothing_is_written_under_a_root_this_server_does_not_own(
        self, clock, record_path, monkeypatch
    ):
        from linkedin_mcp_server.profile_claim import ProfileRootRefusedError

        def refuse(_root):
            raise ProfileRootRefusedError("not ours")

        monkeypatch.setattr(
            "linkedin_mcp_server.profile_claim.require_profile_claim", refuse
        )
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            visits_per_hour=1,
        )
        with pacing.tool_call(deadline=None):
            await visit(gate)
            # Still enforced, from memory.
            with pytest.raises(PacingRefusedError):
                await gate.before_visit(_PROFILE)
        assert not record_path.exists()


class TestMonthlySearch:
    async def test_people_search_pages_stop_at_the_monthly_limit(
        self, clock, record_path
    ):
        gate = make_gate(
            clock,
            record_path,
            pause_max_seconds=0,
            pause_min_seconds=0,
            search_pages_per_month=2,
        )
        with pacing.tool_call(deadline=None):
            await visit(gate, _SEARCH)
            await visit(gate, _SEARCH)
            with pytest.raises(
                PacingRefusedError, match="people-search pages"
            ) as refused:
                await gate.before_visit(_SEARCH)
            # Anything but a people search still goes.
            await gate.before_visit(_PROFILE)
        # 2026-10-29 15:00 UTC to 2026-11-01 00:00 UTC.
        assert refused.value.retry_after == (2 * 86400) + (9 * 3600)

    async def test_it_is_off_unless_set(self, clock, record_path):
        gate = make_gate(clock, record_path, pause_max_seconds=0, pause_min_seconds=0)
        with pacing.tool_call(deadline=None):
            for _ in range(20):
                await visit(gate, _SEARCH)
        assert gate.status()["people_search_pages"] == {
            "this_month": 20,
            "per_month": 0,
        }


class TestLoadPause:
    @pytest.mark.parametrize(
        ("fraction", "asked", "paused"),
        [(0.0, 0.5, 1.0), (1.0, 0.5, 1.5), (0.0, 2.0, 2.0), (1.0, 2.0, 3.0)],
    )
    def test_it_never_drops_below_a_second_and_is_stretched_at_random(
        self, clock, fraction, asked, paused
    ):
        gate = make_gate(clock, None, fraction=fraction)
        assert gate.load_pause(asked) == pytest.approx(paused)


class TestDisabled:
    async def test_it_lets_everything_through_at_once(self, clock):
        gate = Pacing.disabled()
        gate.record_pushback("http_429")
        with pacing.tool_call(deadline=0.0):
            for _ in range(100):
                await visit(gate)
        assert gate.refusal_for_new_call() is None
        assert gate.loads_allowed(50) == 50
        assert gate.conversation_opens_allowed(None) is None
        assert gate.load_pause(0.5) == 0.5
        assert gate.conversation_pause() == 0.0
        assert gate.status() == {"enabled": False}


class TestSignals:
    @pytest.fixture
    def installed(self, clock, record_path):
        gate = make_gate(clock, record_path)
        pacing.install_gate(gate)
        return gate

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/checkpoint/challenge/AgF1x",
            "https://www.linkedin.com/uas/consumer-email-challenge",
        ],
    )
    def test_a_checkpoint_during_a_tool_call_stops_for_a_day(self, installed, url):
        with pacing.tool_call(deadline=None):
            pacing.note_auth_route(url)
        assert installed.stop_seconds_left() == pytest.approx(86400)

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/login",
            "https://www.linkedin.com/authwall?trk=x",
            "https://www.linkedin.com/in/checkpoint-fan/",
        ],
    )
    def test_being_signed_out_is_not_a_challenge(self, installed, url):
        with pacing.tool_call(deadline=None):
            pacing.note_auth_route(url)
        assert not installed.stop_active()

    def test_a_checkpoint_while_you_sign_in_yourself_stops_nothing(self, installed):
        pacing.note_auth_route("https://www.linkedin.com/checkpoint/challenge/AgF1x")
        pacing.note_account_restricted()
        assert not installed.stop_active()

    def test_a_refusal_between_calls_still_stops(self, installed):
        pacing.note_linkedin_refusal(999, None)
        assert installed.status()["stop"]["reason"] == "http_999"


def _own_calls(function: ast.AST):
    """Every call made in *function* itself, not in a function nested inside it."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


_NAVIGATIONS = {
    "goto",
    "reload",
    "go_back",
    "go_forward",
    "goto_reporting_proxy_errors",
}


class TestEveryPageVisitIsPaced:
    """A navigation that skips the gate is LinkedIn traffic nothing counts.

    Every function in the package that opens a page has to ask the gate in the
    same function. A new navigation added anywhere else fails here, which is
    the point: the budget is only as good as its one way in.
    """

    #: Navigations that are deliberately not visits the gate paces.
    ALLOWED = {
        # The wrapper every paced navigation goes through, not a visit itself.
        "linkedin_mcp_server/core/proxy_errors.py",
        # The sign-in window, which is you signing in rather than automation.
        "linkedin_mcp_server/setup.py",
    }

    def test_every_function_that_navigates_asks_the_gate_first(self):
        import linkedin_mcp_server

        package = Path(linkedin_mcp_server.__file__).parent
        navigating: set[str] = set()
        unpaced: list[str] = []
        for path in sorted(package.rglob("*.py")):
            relative = path.relative_to(package.parent).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                names = {_called_name(call) for call in _own_calls(function)}
                if not names & _NAVIGATIONS:
                    continue
                navigating.add(relative)
                if relative not in self.ALLOWED and "before_visit" not in names:
                    unpaced.append(f"{relative}:{function.name}")

        assert unpaced == []
        # The scan has to see the navigations it guards, or it proves nothing.
        assert {
            "linkedin_mcp_server/linkedin/navigation.py",
            "linkedin_mcp_server/drivers/browser.py",
        } <= navigating


class TestTheMiddleware:
    """Where every browser call meets the gate before it can reach LinkedIn."""

    @pytest.fixture
    def lease(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        lease = MagicMock()
        lease.try_acquire.return_value = True
        monkeypatch.setattr(
            "linkedin_mcp_server.sequential_tool_middleware.get_profile_lease",
            lambda: lease,
        )
        self.started = []
        monkeypatch.setattr(
            "linkedin_mcp_server.drivers.browser.note_call_started",
            lambda: self.started.append(True),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.drivers.browser.note_activity", lambda: None
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.drivers.browser.release_profile_if_idle_or_requested",
            AsyncMock(),
        )
        return lease

    @staticmethod
    def _context(tool: str):
        from unittest.mock import MagicMock

        context = MagicMock()
        context.message.name = tool
        context.fastmcp_context = None
        return context

    async def test_a_call_during_a_stop_is_refused_before_the_browser_is_touched(
        self, lease, clock, record_path
    ):
        from unittest.mock import AsyncMock

        from linkedin_mcp_server.error_handler import CodedToolError, ErrorCode
        from linkedin_mcp_server.sequential_tool_middleware import (
            SequentialToolExecutionMiddleware,
        )

        gate = make_gate(clock, record_path)
        gate.record_pushback("http_429")
        pacing.install_gate(gate)
        call_next = AsyncMock(return_value="ran")

        with pytest.raises(CodedToolError) as refused:
            await SequentialToolExecutionMiddleware().on_call_tool(
                self._context("get_person_profile"), call_next
            )

        assert refused.value.code is ErrorCode.RATE_LIMITED
        assert refused.value.retry_after == 900
        call_next.assert_not_awaited()
        assert self.started == []
        lease.release.assert_called_once()

    async def test_a_stop_another_process_recorded_is_honoured(
        self, lease, clock, record_path
    ):
        from unittest.mock import AsyncMock

        from linkedin_mcp_server.error_handler import CodedToolError
        from linkedin_mcp_server.sequential_tool_middleware import (
            SequentialToolExecutionMiddleware,
        )

        mine = make_gate(clock, record_path)
        mine.refresh()
        pacing.install_gate(mine)
        theirs = make_gate(clock, record_path)
        theirs.record_pushback("http_999")

        with pytest.raises(CodedToolError):
            await SequentialToolExecutionMiddleware().on_call_tool(
                self._context("search_people"), AsyncMock()
            )

    async def test_closing_the_browser_is_never_refused(
        self, lease, clock, record_path
    ):
        from unittest.mock import AsyncMock

        from linkedin_mcp_server.sequential_tool_middleware import (
            SequentialToolExecutionMiddleware,
        )

        gate = make_gate(clock, record_path)
        gate.record_pushback("challenge")
        pacing.install_gate(gate)
        call_next = AsyncMock(return_value="closed")

        result = await SequentialToolExecutionMiddleware().on_call_tool(
            self._context("close_session"), call_next
        )

        assert result == "closed"

    async def test_the_tool_runs_as_a_paced_call_ending_at_the_tool_timeout(
        self, lease
    ):
        import time

        from linkedin_mcp_server.sequential_tool_middleware import (
            SequentialToolExecutionMiddleware,
        )

        seen = {}

        async def call_next(context):
            scope = pacing._scope.get()
            seen["deadline"] = scope.deadline if scope else None
            return "ok"

        before = time.monotonic()
        await SequentialToolExecutionMiddleware(tool_timeout=42.0).on_call_tool(
            self._context("get_feed"), call_next
        )

        assert before + 42.0 <= seen["deadline"] <= time.monotonic() + 42.0
        assert not pacing.in_tool_call()


class TestStopClosesTheBrowser:
    """The page left open keeps talking to LinkedIn until it closes."""

    @pytest.fixture
    def open_browser(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        from linkedin_mcp_server.drivers import browser as driver

        lease = MagicMock()
        lease.handoff_requested.return_value = False
        monkeypatch.setattr(driver, "_browser", MagicMock())
        monkeypatch.setattr(driver, "_browser_lease", lease)
        monkeypatch.setattr(driver, "_calls_in_flight", 0)
        # Idle for a second only, so the idle timeout is not what closes it.
        monkeypatch.setattr(driver, "_last_activity", __import__("time").monotonic())
        close = AsyncMock(return_value=True)
        monkeypatch.setattr(driver, "_close_unless_a_call_arrived", close)
        return close

    async def test_a_stop_closes_the_browser_once_no_call_runs(
        self, open_browser, clock, record_path
    ):
        from linkedin_mcp_server.drivers.browser import (
            release_profile_if_idle_or_requested,
        )

        gate = make_gate(clock, record_path)
        pacing.install_gate(gate)
        assert await release_profile_if_idle_or_requested() is False
        open_browser.assert_not_awaited()

        gate.record_pushback("http_429")
        assert await release_profile_if_idle_or_requested() is True
        open_browser.assert_awaited_once()

    async def test_a_call_still_running_keeps_its_browser(
        self, open_browser, clock, record_path, monkeypatch
    ):
        from linkedin_mcp_server.drivers import browser as driver

        gate = make_gate(clock, record_path)
        gate.record_pushback("http_429")
        pacing.install_gate(gate)
        monkeypatch.setattr(driver, "_calls_in_flight", 1)

        assert await driver.release_profile_if_idle_or_requested() is False
        open_browser.assert_not_awaited()


class TestSignalsFromThePage:
    """What the page shows that means LinkedIn wants fewer requests."""

    @pytest.fixture
    def gate(self, clock, record_path):
        gate = make_gate(clock, record_path)
        pacing.install_gate(gate)
        return gate

    @pytest.mark.parametrize(("empty_reads", "stopped"), [(2, True), (1, False)])
    async def test_a_section_still_empty_after_its_retry_stops_requests(
        self, gate, mock_page, empty_reads, stopped
    ):
        from unittest.mock import AsyncMock, patch

        from linkedin_mcp_server.linkedin.capture import CapturePlan, SectionCapture
        from linkedin_mcp_server.linkedin.content import PageContentReader
        from linkedin_mcp_server.linkedin.contracts import (
            RATE_LIMITED_SECTION_TEXT,
            ExtractedSection,
        )
        from linkedin_mcp_server.linkedin.navigation import PageNavigator
        from linkedin_mcp_server.linkedin.session import PageSession

        session = PageSession(mock_page)
        capture = SectionCapture(
            session, PageNavigator(session), PageContentReader(session)
        )
        empty = ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        reads = [empty] * empty_reads + [
            ExtractedSection(text="Profile", references=[])
        ]

        with (
            patch.object(capture, "_capture_once", AsyncMock(side_effect=reads)),
            patch.object(PageSession, "delay", AsyncMock()),
        ):
            await capture.capture(
                "https://www.linkedin.com/in/someone/", "main_profile", CapturePlan()
            )

        assert gate.stop_active() is stopped
        if stopped:
            assert gate.stop_seconds_left() == pytest.approx(300)

    @pytest.mark.parametrize("page", ["search", "saved"])
    async def test_a_job_page_still_empty_after_its_retry_stops_requests(
        self, gate, mock_page, page
    ):
        from unittest.mock import AsyncMock, patch

        from linkedin_mcp_server.linkedin import job_pages
        from linkedin_mcp_server.linkedin.content import PageContentReader
        from linkedin_mcp_server.linkedin.contracts import (
            RATE_LIMITED_SECTION_TEXT,
            ExtractedSection,
        )
        from linkedin_mcp_server.linkedin.job_pages import JobPageReader
        from linkedin_mcp_server.linkedin.navigation import PageNavigator
        from linkedin_mcp_server.linkedin.session import PageSession

        session = PageSession(mock_page)
        reader = JobPageReader(
            session, PageNavigator(session), PageContentReader(session)
        )
        empty = ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        once = (
            "_extract_search_page_once"
            if page == "search"
            else "_extract_saved_jobs_page_once"
        )
        with (
            patch.object(reader, once, AsyncMock(return_value=empty)),
            patch.object(job_pages.asyncio, "sleep", AsyncMock()),
        ):
            if page == "search":
                await reader._extract_search_page(
                    "https://www.linkedin.com/jobs/search/?keywords=x", "search_results"
                )
            else:
                await reader._extract_saved_jobs_page(
                    "https://www.linkedin.com/jobs-tracker/", "saved_jobs"
                )

        assert gate.status()["stop"]["reason"] == "empty_page"

    async def test_a_rate_limit_page_stops_requests(self, gate):
        from unittest.mock import AsyncMock, MagicMock

        from linkedin_mcp_server.core.exceptions import RateLimitError
        from linkedin_mcp_server.core.utils import detect_rate_limit

        page = MagicMock()
        page.url = "https://www.linkedin.com/in/someone/"
        main = MagicMock(count=AsyncMock(return_value=0))
        body = MagicMock(
            inner_text=AsyncMock(return_value="Too many requests. Slow down.")
        )
        page.locator = MagicMock(
            side_effect=lambda selector: main if selector == "main" else body
        )

        with pytest.raises(RateLimitError):
            await detect_rate_limit(page)

        assert gate.status()["stop"]["reason"] == "slow_down_page"

    async def test_the_error_says_how_long_the_stop_lasts_not_the_pages_hint(
        self, gate
    ):
        from linkedin_mcp_server.core.exceptions import RateLimitError
        from linkedin_mcp_server.error_handler import CodedToolError, raise_tool_error

        gate.record_pushback("slow_down_page")
        with pytest.raises(CodedToolError) as shaped:
            raise_tool_error(
                RateLimitError(
                    "Rate limit message detected on page.", suggested_wait_time=30
                ),
                "get_person_profile",
            )

        assert shaped.value.retry_after == 900
        assert "Wait 900 seconds" in str(shaped.value)

    async def test_a_refusal_from_the_gate_keeps_its_own_words(self, gate):
        from linkedin_mcp_server.error_handler import (
            CodedToolError,
            ErrorCode,
            raise_tool_error,
        )

        refusal = pacing.PacingRefusedError(
            "Rate limit: 60 of the 60 page visits ...", 321
        )
        with pytest.raises(CodedToolError) as shaped:
            raise_tool_error(refusal, "search_people")

        assert shaped.value.code is ErrorCode.RATE_LIMITED
        assert shaped.value.retry_after == 321
        assert str(shaped.value) == "Rate limit: 60 of the 60 page visits ..."


class TestCheckpointsMetByATool:
    """A challenge met during a tool call stops everything for a day."""

    CHECKPOINT = "https://www.linkedin.com/checkpoint/challenge/AgF1x"

    @pytest.fixture
    def gate(self, clock, record_path):
        gate = make_gate(clock, record_path)
        pacing.install_gate(gate)
        return gate

    @pytest.mark.parametrize("in_a_call", [True, False])
    async def test_a_navigation_that_lands_on_a_checkpoint(
        self, gate, mock_page, in_a_call
    ):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock, patch

        from linkedin_mcp_server.core.exceptions import AuthenticationError
        from linkedin_mcp_server.linkedin.navigation import PageNavigator
        from linkedin_mcp_server.linkedin.session import PageSession

        async def land_on_checkpoint(url, **kwargs):
            mock_page.url = self.CHECKPOINT

        mock_page.goto = AsyncMock(side_effect=land_on_checkpoint)
        navigator = PageNavigator(PageSession(mock_page))
        scope = pacing.tool_call(deadline=None) if in_a_call else nullcontext()

        with (
            scope,
            patch(
                "linkedin_mcp_server.linkedin.navigation.detect_auth_barrier_quick",
                AsyncMock(return_value="checkpoint"),
            ),
            pytest.raises(AuthenticationError),
        ):
            await navigator._navigate_to_page("https://www.linkedin.com/in/someone/")

        assert gate.stop_active() is in_a_call
        if in_a_call:
            assert gate.stop_seconds_left() == pytest.approx(86400)

    async def test_the_session_check_meeting_a_checkpoint(self, gate, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        from linkedin_mcp_server.drivers import browser as driver

        browser = MagicMock()
        browser.page.url = self.CHECKPOINT
        for name in (
            "_open_feed",
            "stabilize_navigation",
            "record_page_trace",
            "_refuse_a_landing_off_linkedin",
            "_log_feed_failure_context",
        ):
            monkeypatch.setattr(driver, name, AsyncMock())
        monkeypatch.setattr(
            driver, "resolve_remember_me_prompt", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(
            driver, "detect_auth_barrier_quick", AsyncMock(return_value="checkpoint")
        )

        with pacing.tool_call(deadline=None):
            assert await driver._feed_auth_succeeds(browser) is False

        assert gate.status()["stop"]["reason"] == "challenge"

    async def test_a_refused_session_check_is_not_read_as_an_expired_login(
        self, gate, monkeypatch
    ):
        """False would have the caller retire a valid login for a rate limit."""
        from unittest.mock import AsyncMock, MagicMock

        from linkedin_mcp_server.drivers import browser as driver

        gate.record_pushback("http_429")
        goto = AsyncMock()
        monkeypatch.setattr(driver, "goto_reporting_proxy_errors", goto)

        with pacing.tool_call(deadline=None), pytest.raises(pacing.PacingRefusedError):
            await driver._feed_auth_succeeds(MagicMock())

        goto.assert_not_awaited()


class TestWritesInTheTools:
    """Invitations and messages are capped where they are sent."""

    PROFILE = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
    SENT = "Jane\n\n· 3rd\n\nEngineer\n\nMessage\nPending\nMore\nAbout\n"

    @staticmethod
    def _actions(mock_page, *texts):
        from unittest.mock import AsyncMock

        from linkedin_mcp_server.linkedin.connection_actions import ConnectionActions
        from linkedin_mcp_server.linkedin.navigation import PageNavigator
        from linkedin_mcp_server.linkedin.session import PageSession

        pages = [
            {
                "url": "https://www.linkedin.com/in/jane/",
                "sections": {"main_profile": t},
            }
            for t in texts
        ]
        session = PageSession(mock_page)
        return ConnectionActions(
            session, PageNavigator(session), AsyncMock(side_effect=pages)
        )

    @staticmethod
    def _signals(invite: bool):
        from linkedin_mcp_server.linkedin.connection import ActionSignals

        return ActionSignals(
            has_invite_anchor=invite,
            has_compose_anchor_in_action_root=False,
            has_edit_intro_anchor=False,
            has_labeled_action_button=False,
            has_labeled_action_anchor=False,
            has_incoming_action_row=False,
        )

    async def test_an_accepted_invitation_is_confirmed_whatever_stops_after_it(
        self, mock_page, clock, record_path
    ):
        """Accepting cannot be undone, so the read that confirms it is never refused.

        LinkedIn answers the accept itself with a 429 here, which starts a stop.
        """
        from unittest.mock import AsyncMock, patch

        from linkedin_mcp_server.linkedin.connection import ActionSignals
        from linkedin_mcp_server.linkedin.connection_actions import ConnectionActions
        from linkedin_mcp_server.linkedin.navigation import PageNavigator
        from linkedin_mcp_server.linkedin.session import PageSession

        gate = make_gate(clock, record_path)
        pacing.install_gate(gate)
        texts = iter(["Jane\n\n· 2nd\n\nAccept\nIgnore\n", "Jane\n\n· 1st\n"])

        async def read_main_profile(username, *args, **kwargs):
            # The real read is a page visit like any other.
            await pacing.gate().before_visit(f"https://www.linkedin.com/in/{username}/")
            pacing.gate().after_visit()
            return {"url": "u", "sections": {"main_profile": next(texts)}}

        def signals(incoming: bool) -> ActionSignals:
            return ActionSignals(
                has_invite_anchor=False,
                has_compose_anchor_in_action_root=not incoming,
                has_edit_intro_anchor=False,
                has_labeled_action_button=False,
                has_labeled_action_anchor=False,
                has_incoming_action_row=incoming,
            )

        async def accept() -> bool:
            gate.record_pushback("http_429")
            return True

        session = PageSession(mock_page)
        actions = ConnectionActions(session, PageNavigator(session), read_main_profile)
        with (
            patch.object(
                actions,
                "_read_action_signals",
                AsyncMock(side_effect=[signals(True), signals(False)]),
            ),
            patch.object(
                actions, "_click_incoming_accept", AsyncMock(side_effect=accept)
            ),
            pacing.tool_call(deadline=None),
        ):
            result = await actions.connect_with_person("jane")

        assert result["status"] == "accepted"
        assert gate.stop_active()

    async def test_an_invitation_over_the_limit_never_opens_the_invite_page(
        self, mock_page, clock, record_path
    ):
        from unittest.mock import AsyncMock, patch

        from linkedin_mcp_server.linkedin.navigation import PageNavigator

        gate = make_gate(clock, record_path, invites_per_day=1)
        gate.record_write("invitation")
        pacing.install_gate(gate)
        actions = self._actions(mock_page, self.PROFILE)

        with (
            patch.object(
                actions,
                "_read_action_signals",
                AsyncMock(return_value=self._signals(True)),
            ),
            patch.object(PageNavigator, "_navigate_to_page", AsyncMock()) as navigate,
            pytest.raises(PacingRefusedError, match="connection invitations"),
        ):
            await actions.connect_with_person("jane")

        navigate.assert_not_awaited()

    async def test_an_invitation_is_counted_before_its_send_is_clicked(
        self, mock_page, clock, record_path
    ):
        from unittest.mock import AsyncMock, patch

        from linkedin_mcp_server.linkedin.navigation import PageNavigator

        gate = make_gate(clock, record_path)
        pacing.install_gate(gate)
        actions = self._actions(mock_page, self.PROFILE, self.SENT)
        at_click = {}

        async def click():
            scope = pacing._scope.get()
            at_click["counted"] = gate.status()["invitations"]["last_24_hours"]
            at_click["submitted"] = scope is not None and scope.write_submitted
            return True

        with (
            pacing.tool_call(deadline=None),
            patch.object(
                actions,
                "_read_action_signals",
                AsyncMock(side_effect=[self._signals(True), self._signals(False)]),
            ),
            patch.object(PageNavigator, "_navigate_to_page", AsyncMock()),
            patch.object(actions, "_dialog_is_open", AsyncMock(return_value=True)),
            patch.object(actions, "_click_dialog_primary_button", side_effect=click),
        ):
            result = await actions.connect_with_person("jane")

        assert result["status"] == "connected"
        assert at_click == {"counted": 1, "submitted": True}

    async def test_a_message_over_the_limit_is_refused_before_the_browser(
        self, clock, record_path, monkeypatch
    ):
        from unittest.mock import AsyncMock, MagicMock

        from fastmcp import FastMCP
        from fastmcp.tools import FunctionTool

        from linkedin_mcp_server.tools import messaging

        gate = make_gate(clock, record_path, messages_per_day=1)
        gate.record_write("message")
        pacing.install_gate(gate)
        get_extractor = AsyncMock()
        monkeypatch.setattr(messaging, "get_ready_extractor", get_extractor)

        mcp = FastMCP("test")
        messaging.register_messaging_tools(mcp)
        send = cast(FunctionTool, await mcp.get_tool("send_message")).fn
        ctx = MagicMock(report_progress=AsyncMock())

        with pytest.raises(Exception, match="messages were sent"):
            await send(
                linkedin_username="jane", message="Hi", confirm_send=True, ctx=ctx
            )
        get_extractor.assert_not_awaited()

        # A dry run sends nothing, so the limit does not stand in its way.
        get_extractor.return_value = MagicMock(send_message=AsyncMock(return_value={}))
        await send(linkedin_username="jane", message="Hi", confirm_send=False, ctx=ctx)
        get_extractor.assert_awaited_once()
