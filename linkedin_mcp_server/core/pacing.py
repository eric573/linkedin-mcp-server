"""How hard the server may use LinkedIn, decided in one place.

Every page the server opens asks :meth:`Pacing.before_visit` first. Every
in-page load that fetches more is counted here too: opening a conversation, a
scroll that pulls the next batch, a "Show more" click. So is every invitation
and message sent. One budget covers all of it, because LinkedIn judges what
the account does, not which tool asked. The decision is recorded in
``docs/decisions/2026-10-30-pace-every-linkedin-request.md``.

Weights. A page visit costs one unit, and so does opening a conversation,
which loads a thread the way a page does. A scroll that grows the page (a
profile's lazy sections, an endless list, the job rail) or a "Show more" click
costs a fifth, the five-scrolls-to-a-page the content search already uses.
The scroll that finds the page complete costs nothing. Each load asks first
and is counted as it happens, never after its loop, and the next page visit
waits its pause after the last thing that loaded.

Stops. When LinkedIn pushes back -- an HTTP 429 or 999, a rate-limit page, a
page that comes back empty even when retried, a checkpoint -- every request
stops for a while, for every tool and every process that shares the session.

Waiting versus refusing. A short wait is waited out and a long one refuses, so
the client is told when to come back rather than having a call held open, and
a stop refuses everything, inside a tool call or outside one (the CLI, a
sign-in repair). After a change has been submitted (an invitation, a message,
an accepted invitation) the rest of that call is paced but never refused:
refusing the read that confirms it would hide whether it went out.

The record. What happened is kept as timestamps in ``rate-limit.json`` beside
the saved session, never as remaining counts, so a changed limit applies to
the history at once. It is written while the profile lease is held, which
every browser call holds, and atomically, so a reader without the lease always
sees a whole version. It is read again before every change, so a process that
has been idle never writes an old copy over a newer one. A timestamp a clock
set back left in the future is pulled back to now, so no limit outlasts what
it reports.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from .exceptions import RateLimitError

if TYPE_CHECKING:
    from linkedin_mcp_server.config.schema import RateLimitConfig

logger = logging.getLogger(__name__)

RECORD_FILE = "rate-limit.json"
_RECORD_VERSION = 1

_HOUR = 3600.0
_DAY = 86400.0
_WEEK = 7 * _DAY

#: A scroll or "Show more" click that loads more items, as a fraction of a visit.
LOAD_WEIGHT = 0.2
#: The pause between in-page loads never drops below this.
_LOAD_PAUSE_FLOOR_SECONDS = 1.0
#: ...and is stretched by a random factor from this range.
_LOAD_PAUSE_JITTER = (1.0, 1.5)
#: The pause before opening each conversation, drawn from this range, or from
#: half the configured page pause when that is longer: a thread loads less
#: than a page, and a slower page pause slows the opens with it.
CONVERSATION_PAUSE_SECONDS = (1.5, 3.0)
#: What opening one conversation takes besides its pause: the click settling.
_CONVERSATION_OPEN_SECONDS = 1.2
#: The longest wait a tool call sits through rather than being refused, unless
#: the configured pause itself is longer.
MAX_WAIT_IN_CALL_SECONDS = 30.0
#: What a page needs after its wait to load, be read and be answered.
_PAGE_RESERVE_SECONDS = 15.0
#: A doubled stop never runs past this.
_MAX_DOUBLED_STOP_SECONDS = _DAY
_EPSILON = 1e-9

PushbackKind = Literal[
    "http_429",
    "http_999",
    "slow_down_page",
    "empty_page",
    "challenge",
    "restricted",
    "damaged_record",
    "newer_record",
]
WriteKind = Literal["invitation", "message"]
#: Why a request is refused: LinkedIn pushed back, a limit is spent, or the
#: tool call has no time left for the wait.
RefusalKind = Literal["stop", "limit", "time"]

#: Routes where LinkedIn challenges the session rather than signing it out.
#: Matched by path, never by page text, so they hold in every language.
_CHALLENGE_PATHS = ("/checkpoint", "/challenge", "/uas/consumer-email-challenge")
_PEOPLE_SEARCH_PATH = "/search/results/people"

_PUSHBACK_TEXT: dict[str, str] = {
    "http_429": "LinkedIn answered HTTP 429 (too many requests)",
    "http_999": "LinkedIn answered HTTP 999 (request denied)",
    "slow_down_page": "LinkedIn showed a rate-limit page",
    "empty_page": "a LinkedIn page came back empty, the usual sign of throttling",
    "challenge": "LinkedIn showed a security checkpoint during a tool call",
    "restricted": "LinkedIn restricted the account",
    "damaged_record": "the rate limit record could not be read",
    "newer_record": (
        "the rate limit record was written by a newer version of this server, "
        "which this one cannot read; update it, or stop the newer one and "
        "delete rate-limit.json"
    ),
}


class PacingRefusedError(RateLimitError):
    """The server will not ask LinkedIn for anything more right now.

    ``kind`` says why. A ``time`` refusal is the tool call running out of time
    for a wait, not LinkedIn or a limit, and a reader that budgets its own time
    reports it as such.
    """

    def __init__(self, message: str, retry_after: int, *, kind: RefusalKind = "limit"):
        super().__init__(message, suggested_wait_time=retry_after)
        self.retry_after = retry_after
        self.kind: RefusalKind = kind


async def _sleep(seconds: float) -> None:
    # Looked up at call time, so a test that patches `asyncio.sleep` sees it.
    await asyncio.sleep(seconds)


@dataclass(frozen=True)
class Clock:
    """Where the gate reads the time and waits. A fake one drives the tests."""

    wall: Callable[[], float] = field(default=lambda: time.time())
    monotonic: Callable[[], float] = field(default=lambda: time.monotonic())
    sleep: Callable[[float], Awaitable[None]] = field(default=_sleep)


@dataclass
class _CallScope:
    """What the gate knows about the tool call it is serving."""

    deadline: float | None
    write_submitted: bool = False
    #: In-page loads the gate held back during the call.
    loads_held_back: int = 0


_scope: ContextVar[_CallScope | None] = ContextVar("pacing_call_scope", default=None)


@contextmanager
def tool_call(deadline: float | None) -> Iterator[None]:
    """Mark the enclosed work as one tool call ending at *deadline* (monotonic).

    Entered by the serializing middleware around the tool body. Only inside it
    does the gate refuse; outside, it paces and counts.
    """
    token = _scope.set(_CallScope(deadline))
    try:
        yield
    finally:
        _scope.reset(token)


def in_tool_call() -> bool:
    """Whether the running task is serving a tool call."""
    return _scope.get() is not None


def loads_held_back() -> int:
    """How many in-page loads the gate has held back during the running call.

    A reader compares this before and after a section to learn that the
    section was cut short, which it then says, rather than return a shorter
    list that reads as complete.
    """
    scope = _scope.get()
    return 0 if scope is None else scope.loads_held_back


def cut_short_retry_after() -> int:
    """Whole seconds until the gate allows what it held back."""
    return _whole_seconds(gate().seconds_until_next_visit())


def stop_retry_after() -> int | None:
    """Whole seconds until a running stop ends, or None when none runs.

    For a reader answering LinkedIn's own rate-limit page, which starts the
    stop, so the caller is told when to come back and not just that it was
    refused.
    """
    left = gate().stop_seconds_left()
    return _whole_seconds(left) if left > 0 else None


def mark_write_submitted() -> None:
    """Note that the running call has made a change it still has to confirm.

    For a change no limit counts, such as accepting an invitation: the rest
    of the call is paced but never refused, as after a counted write.
    """
    scope = _scope.get()
    if scope is not None:
        scope.write_submitted = True


@dataclass
class _Record:
    """Everything the gate remembers, as it is written to disk."""

    visits: list[float] = field(default_factory=list)
    conversation_opens: list[float] = field(default_factory=list)
    loads: list[float] = field(default_factory=list)
    invitations: list[float] = field(default_factory=list)
    messages: list[float] = field(default_factory=list)
    people_search_pages: dict[str, int] = field(default_factory=dict)
    pause_until: float = 0.0
    stop_until: float = 0.0
    stop_reason: str | None = None
    #: How long the latest stop was set to last, which the next one doubles.
    stop_length: float = 0.0
    stops_started: list[float] = field(default_factory=list)
    #: No invitation before this, after a record whose week was lost.
    invitations_held_until: float = 0.0

    def has_history(self) -> bool:
        return any(
            (
                self.visits,
                self.conversation_opens,
                self.loads,
                self.invitations,
                self.messages,
                self.stops_started,
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": _RECORD_VERSION,
            "visits": self.visits,
            "conversation_opens": self.conversation_opens,
            "loads": self.loads,
            "invitations": self.invitations,
            "messages": self.messages,
            "people_search_pages": self.people_search_pages,
            "pause_until": self.pause_until,
            "stop": (
                {
                    "until": self.stop_until,
                    "reason": self.stop_reason,
                    "length": self.stop_length,
                }
                if self.stop_until
                else None
            ),
            "stops_started": self.stops_started,
            "invitations_held_until": self.invitations_held_until,
        }

    @classmethod
    def from_json(cls, raw: object) -> _Record:
        """Parse strictly: a record that does not parse is a damaged one."""
        if not isinstance(raw, dict) or raw.get("version") != _RECORD_VERSION:
            raise ValueError("not a version 1 rate limit record")
        stop = raw.get("stop")
        if stop is not None and not isinstance(stop, dict):
            raise ValueError("stop is not an object")
        months = raw.get("people_search_pages", {})
        if not isinstance(months, dict) or not all(
            isinstance(key, str) and _is_count(value) for key, value in months.items()
        ):
            raise ValueError("people_search_pages is not a map of counts")
        reason = stop.get("reason") if stop else None
        if reason is not None and not isinstance(reason, str):
            raise ValueError("stop reason is not text")
        return cls(
            visits=_stamps(raw, "visits"),
            conversation_opens=_stamps(raw, "conversation_opens"),
            loads=_stamps(raw, "loads"),
            invitations=_stamps(raw, "invitations"),
            messages=_stamps(raw, "messages"),
            people_search_pages=dict(months),
            pause_until=_number(raw.get("pause_until", 0.0)),
            stop_until=_number(stop.get("until", 0.0)) if stop else 0.0,
            stop_reason=reason,
            stop_length=_number(stop.get("length", 0.0)) if stop else 0.0,
            stops_started=_stamps(raw, "stops_started"),
            invitations_held_until=_number(raw.get("invitations_held_until", 0.0)),
        )


def _is_newer(raw: object) -> bool:
    """Whether *raw* is a record of a later format than this server writes."""
    if not isinstance(raw, dict):
        return False
    version = raw.get("version")
    return (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version > _RECORD_VERSION
    )


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not a number")
    if not math.isfinite(value):
        raise ValueError("not a finite number")
    return float(value)


def _stamps(raw: dict[str, Any], name: str) -> list[float]:
    values = raw.get(name, [])
    if not isinstance(values, list):
        raise ValueError(f"{name} is not a list")
    return [_number(value) for value in values]


class Pacing:
    """The gate. One per process; :func:`gate` returns it."""

    def __init__(
        self,
        settings: RateLimitConfig | None,
        path: Path | None,
        *,
        clock: Clock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """*settings* None makes a gate that paces nothing, for tests.

        *path* None keeps the record in memory only.
        """
        self._settings = settings
        self._path = path
        self._clock = clock or Clock()
        self._rng = rng or random.Random()
        self._record = _Record()
        self._seen: tuple[int, int] | None = None
        self._owned: bool | None = None
        self._warned = False
        #: The record holds a change a save failed to write.
        self._dirty = False
        #: The file belongs to a newer server: never written over.
        self._read_only = False
        #: Every second spent waiting before a visit, so a caller timing its
        #: own pages can take the waits back out.
        self.waited_seconds = 0.0

    @classmethod
    def disabled(cls) -> Pacing:
        """A gate that lets everything through at once."""
        return cls(None, None)

    @property
    def enabled(self) -> bool:
        return self._settings is not None

    # -- the record --------------------------------------------------------

    def refresh(self) -> None:
        """Adopt the record on disk, which another process may have written.

        Called with the profile lease held, and before every change, so an
        idle process never writes an old copy over a newer one. A missing
        file keeps what this process knows: it is written again with the next
        change, as is a change an earlier save failed to write, which is newer
        than the file. A file that does not parse never hands out a new
        budget: this process keeps the history it holds and stops for a
        while, and one with no history stops for a day and holds invitations
        for a week, the windows it can no longer count. A file from a newer
        version of the server is left alone, and this one sends nothing.
        """
        if not self.enabled or self._path is None:
            return
        if self._dirty:
            self._save()
            if self._dirty:
                return
        seen = self._stat()
        if seen is None or seen == self._seen:
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if _is_newer(raw):
                self._stand_back(seen)
                return
            record = _Record.from_json(raw)
        except (OSError, ValueError, TypeError) as exc:
            self._repair(exc)
            return
        self._record = record
        self._read_only = False
        self._seen = seen

    def _repair(self, exc: Exception) -> None:
        """Replace a record that does not parse, without handing out budget."""
        assert self._path is not None
        self._read_only = False
        if self._seen is not None or self._record.has_history():
            logger.warning(
                "The rate limit record %s could not be read (%s); keeping the "
                "history this process holds and starting a stop",
                self._path,
                exc,
            )
            self.record_pushback("damaged_record")
        else:
            logger.warning(
                "The rate limit record %s could not be read (%s), and this "
                "process has no history of its own: stopping for a day and "
                "holding invitations for a week. If the account has been "
                "quiet, stop the server and delete the file to start afresh.",
                self._path,
                exc,
            )
            self._record = _Record()
            self.record_pushback("damaged_record", retry_after=_DAY)
            self._record.invitations_held_until = self._clock.wall() + _WEEK
        # Written even when stops are switched off, or every later call
        # would read the damaged file again and warn again.
        self._save()

    def _stand_back(self, seen: tuple[int, int]) -> None:
        """A newer server's record: read nothing from it, write nothing over it."""
        if not self._read_only:
            logger.warning(
                "The rate limit record %s was written by a newer version of "
                "this server. This one sends LinkedIn nothing and leaves the "
                "file alone until it is updated.",
                self._path,
            )
        self._read_only = True
        self._seen = seen
        self._record = _Record()
        self.record_pushback("newer_record", retry_after=self._longest_stop())

    def _stat(self) -> tuple[int, int] | None:
        assert self._path is not None
        try:
            info = self._path.stat()
        except FileNotFoundError:
            return None
        except OSError:
            # Unreadable metadata is as good as a changed file: try to read it,
            # and treat that failure as damage.
            return (-1, -1)
        return (info.st_mtime_ns, info.st_size)

    def _save(self) -> None:
        if self._path is None or self._read_only:
            return
        from linkedin_mcp_server.common_utils import secure_write_text

        try:
            self._require_owned_root()
            secure_write_text(
                self._path,
                json.dumps(self._record.to_json(), indent=1, sort_keys=True) + "\n",
            )
            self._seen = self._stat()
            self._dirty = False
        except Exception as exc:
            # Tried again before the next change, and before any file is read
            # over it: what this process holds is the newer record.
            self._dirty = True
            stopping = self._record.stop_until > self._clock.wall()
            if stopping or not self._warned:
                self._warned = True
                logger.warning(
                    "Could not save the rate limit record to %s (%s). The limits "
                    "still apply in this process%s, and the save is tried again "
                    "with the next change.",
                    self._path,
                    exc,
                    ", including the stop it holds" if stopping else "",
                )

    def _require_owned_root(self) -> None:
        """Write only under a session root this server owns.

        The Profile Safety Rules in AGENTS.md: the atomic write replaces a file,
        and ``USER_DATA_DIR`` accepts any path, so a mistyped one must not
        have a file of ours written over whatever it names.
        """
        if self._owned:
            return
        from linkedin_mcp_server.profile_claim import require_profile_claim
        from linkedin_mcp_server.session_state import get_source_profile_dir

        require_profile_claim(get_source_profile_dir())
        self._owned = True

    def _prune(self, now: float) -> None:
        """Forget what no limit looks at any more."""
        record = self._record
        day_ago = now - _DAY
        week_ago = now - _WEEK
        record.visits = [t for t in record.visits if t > day_ago]
        record.conversation_opens = [
            t for t in record.conversation_opens if t > day_ago
        ]
        record.loads = [t for t in record.loads if t > day_ago]
        record.messages = [t for t in record.messages if t > day_ago]
        record.invitations = [t for t in record.invitations if t > week_ago]
        record.stops_started = [t for t in record.stops_started if t > day_ago]
        month = _month(now)
        record.people_search_pages = {
            key: value
            for key, value in record.people_search_pages.items()
            if key == month
        }

    def _now(self) -> float:
        now = self._clock.wall()
        self._normalize(now)
        self._prune(now)
        return now

    def _normalize(self, now: float) -> None:
        """Pull back what a clock set back left in the future.

        Every timestamp is when something happened, so none is later than now;
        one that is would count against its window for as long as the clock
        was off, past every wait the gate reports. A stop keeps its own length
        from now, and a pause its longest.
        """
        record = self._record
        changed = False
        for name in (
            "visits",
            "conversation_opens",
            "loads",
            "invitations",
            "messages",
            "stops_started",
        ):
            stamps: list[float] = getattr(record, name)
            if any(stamp > now for stamp in stamps):
                setattr(record, name, [min(stamp, now) for stamp in stamps])
                changed = True
        stop_cap = now + (
            record.stop_length if record.stop_length > 0 else self._longest_stop()
        )
        if record.stop_until > stop_cap:
            record.stop_until = stop_cap
            changed = True
        pause_cap = now + self._longest_pause()
        if record.pause_until > pause_cap:
            record.pause_until = pause_cap
            changed = True
        if record.invitations_held_until > now + _WEEK:
            record.invitations_held_until = now + _WEEK
            changed = True
        if changed:
            # Written with the next change, not here: reading the gate is
            # not a reason to write the record.
            self._dirty = True

    # -- what has been spent -----------------------------------------------

    def _spent(self, now: float, window: float) -> list[tuple[float, float]]:
        """Every weighted entry in the window, oldest first.

        An entry dated after *now*, which a clock set back produces, still
        counts: moving the clock must never free up budget.
        """
        since = now - window
        record = self._record
        entries = [(t, 1.0) for t in record.visits if t > since]
        entries += [(t, 1.0) for t in record.conversation_opens if t > since]
        entries += [(t, LOAD_WEIGHT) for t in record.loads if t > since]
        entries.sort()
        return entries

    def _windows(self) -> tuple[tuple[float, int, str], ...]:
        assert self._settings is not None
        return (
            (_HOUR, self._settings.visits_per_hour, "hour"),
            (_DAY, self._settings.visits_per_day, "24 hours"),
        )

    def _budget_block(self, now: float, weight: float) -> tuple[float, str] | None:
        """How long until *weight* more fits under every window, and why."""
        worst: tuple[float, str] | None = None
        for window, cap, label in self._windows():
            if cap <= 0:
                continue
            entries = self._spent(now, window)
            total = sum(w for _, w in entries)
            if total + weight <= cap + _EPSILON:
                continue
            wait = window
            remaining = total
            for stamp, entry_weight in entries:
                remaining -= entry_weight
                if remaining + weight <= cap + _EPSILON:
                    wait = min(window, max(0.0, stamp + window - now))
                    break
            reason = (
                f"{total:g} of the {cap} page visits allowed in any {label} are used"
            )
            if worst is None or wait > worst[0]:
                worst = (wait, reason)
        return worst

    def _room(self, now: float) -> float:
        """How many units fit under every window right now."""
        room = math.inf
        for window, cap, _ in self._windows():
            if cap <= 0:
                continue
            spent = sum(w for _, w in self._spent(now, window))
            room = min(room, cap - spent)
        return max(0.0, room)

    def _stop_left(self, now: float) -> float:
        # Bounded by `_normalize`, so this is when the stop really ends.
        return max(0.0, self._record.stop_until - now)

    def _longest_stop(self) -> float:
        """The longest any stop may run, LinkedIn's Retry-After included."""
        assert self._settings is not None
        return max(
            _MAX_DOUBLED_STOP_SECONDS, self._settings.challenge_stop_hours * 3600
        )

    def _longest_pause(self) -> float:
        assert self._settings is not None
        return max(
            self._settings.pause_max_seconds,
            _LOAD_PAUSE_FLOOR_SECONDS * _LOAD_PAUSE_JITTER[1],
        )

    def _pause_left(self, now: float) -> float:
        left = self._record.pause_until - now
        if left <= 0:
            return 0.0
        return min(left, self._longest_pause())

    def _search_block(self, now: float, url: str) -> tuple[float, str] | None:
        assert self._settings is not None
        cap = self._settings.search_pages_per_month
        if cap <= 0 or not _is_people_search(url):
            return None
        used = self._record.people_search_pages.get(_month(now), 0)
        if used < cap:
            return None
        return (
            _seconds_to_next_month(now),
            f"{used} of the {cap} people-search pages allowed this month are used",
        )

    def _longest_wait_in_call(self) -> float:
        assert self._settings is not None
        return max(MAX_WAIT_IN_CALL_SECONDS, self._settings.pause_max_seconds)

    def _stop_error(self, now: float) -> PacingRefusedError:
        left = self._stop_left(now)
        reason = _PUSHBACK_TEXT.get(
            self._record.stop_reason or "", "LinkedIn pushed back"
        )
        until = datetime.fromtimestamp(now + left, UTC).strftime("%H:%M UTC")
        return PacingRefusedError(
            f"Rate limit stop: {reason}, so the server is not sending LinkedIn "
            f"anything until {until} (in {_duration(left)}). Try again then.",
            _whole_seconds(left),
            kind="stop",
        )

    # -- page visits -------------------------------------------------------

    async def before_visit(self, url: str, *, deadline: float | None = None) -> None:
        """Wait for, or refuse, the page visit to *url*, then count it.

        *deadline* is a monotonic reading the visit has to finish by, besides
        the tool call's own.

        Raises:
            PacingRefusedError: When LinkedIn asked for a stop, a limit is used
                up for longer than a call should wait, or the call has no time
                left for the wait and the page. Inside a tool call or outside
                one alike, except once the call has submitted a change.
        """
        if not self.enabled:
            return
        self.refresh()
        now = self._now()
        scope = _scope.get()
        enforced = scope is None or not scope.write_submitted
        pause = self._pause_left(now)
        blocks = [
            block
            for block in (
                self._budget_block(now, 1.0),
                self._search_block(now, url),
            )
            if block is not None
        ]
        if enforced:
            if self._stop_left(now) > 0:
                raise self._stop_error(now)
            longest = max(blocks, key=lambda item: item[0]) if blocks else None
            if longest is not None and longest[0] > self._longest_wait_in_call():
                raise PacingRefusedError(
                    f"Rate limit: {longest[1]}. The next page visit is allowed "
                    f"in {_duration(longest[0])}.",
                    _whole_seconds(longest[0]),
                )
            wait = max(pause, longest[0] if longest else 0.0)
            limit = _earliest(scope.deadline if scope else None, deadline)
            # With no pause left to wait as well: a page started this close to
            # the end is cancelled with the call, and everything read before
            # it with it.
            if (
                limit is not None
                and self._clock.monotonic() + wait + _PAGE_RESERVE_SECONDS > limit
            ):
                waiting = f"to wait {_duration(wait)} " if wait > 0 else ""
                raise PacingRefusedError(
                    f"Rate limit: this call has no time left {waiting}for its "
                    "next page. Call again to continue.",
                    _whole_seconds(wait),
                    kind="time",
                )
        else:
            # Once a change went out: paced, never refused.
            wait = pause
        # Milliseconds are as fine as a pause needs to be, and keeping the float
        # noise of the subtraction out keeps the logs and traces readable.
        wait = round(wait, 3)
        if wait > 0:
            logger.debug("Pacing: waiting %.1fs before %s", wait, _path_of(url))
            await self._clock.sleep(wait)
            self.waited_seconds += wait
            now = self._now()
            # LinkedIn may have asked for a stop while this visit waited: the
            # page left open keeps receiving answers in the meantime.
            if enforced and self._stop_left(now) > 0:
                raise self._stop_error(now)
        self._record.visits.append(now)
        if _is_people_search(url):
            month = _month(now)
            pages = self._record.people_search_pages
            pages[month] = pages.get(month, 0) + 1
        self._save()

    def after_visit(self) -> None:
        """Note that a page finished loading; the next pause runs from now."""
        if not self.enabled:
            return
        assert self._settings is not None
        self.refresh()
        now = self._now()
        self._record.pause_until = now + self._rng.uniform(
            self._settings.pause_min_seconds, self._settings.pause_max_seconds
        )
        self._save()

    def expected_wait(self) -> float:
        """Seconds the next page visit will wait, when it is not refused."""
        if not self.enabled:
            return 0.0
        now = self._now()
        block = self._budget_block(now, 1.0)
        return max(self._pause_left(now), block[0] if block else 0.0)

    def refusal_for_new_call(self) -> PacingRefusedError | None:
        """Why a tool call should not start at all, or None.

        Asked before the browser is touched, so a call that would be refused
        at its first page costs nothing, not even a browser start.
        """
        if not self.enabled:
            return None
        now = self._now()
        if self._stop_left(now) > 0:
            return self._stop_error(now)
        block = self._budget_block(now, 1.0)
        if block is not None and block[0] > self._longest_wait_in_call():
            return PacingRefusedError(
                f"Rate limit: {block[1]}. The next page visit is allowed in "
                f"{_duration(block[0])}.",
                _whole_seconds(block[0]),
            )
        return None

    # -- loads inside a page -----------------------------------------------

    def loads_allowed(self, wanted: int) -> int:
        """How many of *wanted* scrolls or "Show more" clicks may load now.

        Fewer than *wanted* is noted on the running call, see
        :func:`loads_held_back`.
        """
        if not self.enabled or wanted <= 0:
            return max(0, wanted)
        now = self._now()
        if self._stop_left(now) > 0:
            allowed = 0
        else:
            room = self._room(now)
            allowed = (
                wanted
                if math.isinf(room)
                else max(0, min(wanted, math.floor(room / LOAD_WEIGHT + _EPSILON)))
            )
        scope = _scope.get()
        if scope is not None and allowed < wanted:
            scope.loads_held_back += wanted - allowed
        return allowed

    def record_loads(self, count: int) -> None:
        """Count *count* scrolls or clicks that loaded, or tried to load, more.

        The next page visit then waits a load's pause after this one, rather
        than only the pause after the page the load was made on.
        """
        if not self.enabled or count <= 0:
            return
        self.refresh()
        now = self._now()
        self._record.loads.extend([now] * count)
        self._pause_until_at_least(now + self.load_pause(0.0))
        self._save()

    def load_pause(self, seconds: float) -> float:
        """The pause after a load the caller would have waited *seconds* for."""
        if not self.enabled:
            return seconds
        return max(seconds, _LOAD_PAUSE_FLOOR_SECONDS) * self._rng.uniform(
            *_LOAD_PAUSE_JITTER
        )

    def conversation_opens_allowed(self, wanted: int | None) -> int | None:
        """How many conversations may be opened now: *wanted*, or fewer.

        None means no limit of the caller's own, and comes back None only when
        the gate sets none either.
        """
        if not self.enabled:
            return wanted
        now = self._now()
        if self._stop_left(now) > 0:
            return 0
        allowed = self._room(now)
        scope = _scope.get()
        if scope is not None and scope.deadline is not None:
            left = scope.deadline - self._clock.monotonic() - _PAGE_RESERVE_SECONDS
            each = (
                sum(self._conversation_pause_range()) / 2 + _CONVERSATION_OPEN_SECONDS
            )
            allowed = min(allowed, max(0.0, left) / each)
        if math.isinf(allowed):
            return wanted
        count = math.floor(allowed + _EPSILON)
        return count if wanted is None else min(wanted, count)

    def opens_refusal(self) -> PacingRefusedError:
        """Why no conversation may be opened now, as the error to raise."""
        now = self._now()
        if self._stop_left(now) > 0:
            return self._stop_error(now)
        block = self._budget_block(now, 1.0)
        if block is not None:
            return PacingRefusedError(
                f"Rate limit: {block[1]}. The next is allowed in "
                f"{_duration(block[0])}.",
                _whole_seconds(block[0]),
            )
        return PacingRefusedError(
            "Rate limit: this call has no time left to open a conversation. "
            "Call again to continue.",
            _whole_seconds(self._pause_left(now)),
            kind="time",
        )

    def seconds_until_next_visit(self) -> float:
        """How long until a visit would go ahead: a stop, a limit or the pause."""
        if not self.enabled:
            return 0.0
        now = self._now()
        block = self._budget_block(now, 1.0)
        return max(
            self._stop_left(now), self._pause_left(now), block[0] if block else 0.0
        )

    def conversation_pause(self) -> float:
        """Seconds to wait before opening the next conversation, drawn at random
        so the clicks keep no fixed rhythm."""
        if not self.enabled:
            return 0.0
        return self._rng.uniform(*self._conversation_pause_range())

    def _conversation_pause_range(self) -> tuple[float, float]:
        assert self._settings is not None
        low, high = CONVERSATION_PAUSE_SECONDS
        return (
            max(low, self._settings.pause_min_seconds / 2),
            max(high, self._settings.pause_max_seconds / 2),
        )

    def record_conversation_opens(self, count: int) -> None:
        """Count *count* conversations opened, or clicked open.

        A thread loads like a page, so the next page visit waits a full pause
        after it, as it would after a page.
        """
        if not self.enabled or count <= 0:
            return
        assert self._settings is not None
        self.refresh()
        now = self._now()
        self._record.conversation_opens.extend([now] * count)
        self._pause_until_at_least(
            now
            + self._rng.uniform(
                self._settings.pause_min_seconds, self._settings.pause_max_seconds
            )
        )
        self._save()

    def _pause_until_at_least(self, until: float) -> None:
        if until > self._record.pause_until:
            self._record.pause_until = until

    # -- writes ------------------------------------------------------------

    def check_write(self, kind: WriteKind) -> None:
        """Refuse a write that a stop or its own limit does not allow.

        Raises:
            PacingRefusedError: When it is not allowed now.
        """
        if not self.enabled:
            return
        assert self._settings is not None
        self.refresh()
        now = self._now()
        if self._stop_left(now) > 0:
            raise self._stop_error(now)
        held = self._record.invitations_held_until - now
        if kind == "invitation" and held > 0:
            until = datetime.fromtimestamp(now + held, UTC).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            raise PacingRefusedError(
                "Rate limit: invitations are held until "
                f"{until}, because the rate limit record was damaged and this "
                "week's invitations could not be counted.",
                _whole_seconds(held),
            )
        settings = self._settings
        if kind == "invitation":
            stamps = self._record.invitations
            limits = (
                (_DAY, settings.invites_per_day, "24 hours"),
                (_WEEK, settings.invites_per_week, "7 days"),
            )
            noun = "connection invitations"
        else:
            stamps = self._record.messages
            limits = ((_DAY, settings.messages_per_day, "24 hours"),)
            noun = "messages"
        for window, cap, label in limits:
            if cap <= 0:
                continue
            recent = sorted(t for t in stamps if t > now - window)
            if len(recent) < cap:
                continue
            wait = min(window, max(0.0, recent[len(recent) - cap] + window - now))
            raise PacingRefusedError(
                f"Rate limit: {len(recent)} {noun} were sent in the last {label}, "
                f"and this server is set to send at most {cap}. The next is "
                f"allowed in {_duration(wait)}.",
                _whole_seconds(wait),
            )

    def record_write(self, kind: WriteKind) -> None:
        """Count a write about to be submitted, whatever comes of it.

        From here on the call is never refused, only paced: the read that
        confirms the write is the only way to learn whether it went out.
        """
        mark_write_submitted()
        if not self.enabled:
            return
        self.refresh()
        now = self._now()
        if kind == "invitation":
            self._record.invitations.append(now)
        else:
            self._record.messages.append(now)
        self._save()

    # -- LinkedIn pushing back ----------------------------------------------

    def record_pushback(
        self, kind: PushbackKind, *, retry_after: float | None = None
    ) -> None:
        """Stop every LinkedIn request for a while, because LinkedIn asked.

        Each stop that starts within a day of the end of the one before lasts
        twice as long as that one, so pushback that keeps coming backs off
        further however it is spaced, up to a day. Signals that arrive while a stop runs
        extend it by its own length rather than counting as a repeat, so one
        burst of refusals is one stop. LinkedIn's ``Retry-After`` wins when it
        asks for longer, up to the longest stop there is.
        """
        if not self.enabled:
            return
        assert self._settings is not None
        settings = self._settings
        # Both of these are recorded by `refresh` itself.
        if kind not in ("damaged_record", "newer_record"):
            self.refresh()
        now = self._now()
        active = self._stop_left(now) > 0
        record = self._record
        if kind in ("challenge", "restricted"):
            length = settings.challenge_stop_hours * 3600
        else:
            base = (
                settings.empty_page_stop_minutes
                if kind == "empty_page"
                else settings.stop_minutes
            ) * 60
            doubled = max(base, _MAX_DOUBLED_STOP_SECONDS)
            if base <= 0 or kind in ("damaged_record", "newer_record"):
                length = base
            elif active:
                length = min(max(base, record.stop_length), doubled)
            elif record.stop_length > 0 and now - record.stop_until < _DAY:
                # LinkedIn pushing back again within a day of the end of the
                # last stop: twice as long as that one.
                length = min(max(base, 2 * record.stop_length), doubled)
            else:
                length = base
            if retry_after is not None and length > 0:
                length = max(length, min(retry_after, self._longest_stop()))
        if length <= 0:
            return
        until = now + length
        if until > record.stop_until:
            record.stop_until = until
            record.stop_reason = kind
            record.stop_length = length
        if not active:
            self._record.stops_started.append(now)
            logger.warning(
                "%s; no LinkedIn requests for %s",
                _PUSHBACK_TEXT[kind][0].upper() + _PUSHBACK_TEXT[kind][1:],
                _duration(self._stop_left(now)),
            )
        self._save()

    def stop_refusal(self) -> PacingRefusedError | None:
        """What a running stop refuses with, or None when none runs."""
        if not self.enabled:
            return None
        now = self._now()
        return self._stop_error(now) if self._stop_left(now) > 0 else None

    def stop_active(self) -> bool:
        """Whether every LinkedIn request is stopped right now."""
        return self.enabled and self._stop_left(self._now()) > 0

    def stop_seconds_left(self) -> float:
        """Seconds until the current stop ends, 0 when none runs."""
        return self._stop_left(self._now()) if self.enabled else 0.0

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """What has been spent and what is left, for ``get_session_status``."""
        if not self.enabled:
            return {"enabled": False}
        assert self._settings is not None
        settings = self._settings
        now = self._now()
        record = self._record

        def within(stamps: list[float], window: float) -> int:
            return sum(1 for t in stamps if t > now - window)

        stop_left = self._stop_left(now)
        block = self._budget_block(now, 1.0)
        return {
            "enabled": True,
            "visits": {
                "last_hour": round(sum(w for _, w in self._spent(now, _HOUR)), 1),
                "per_hour": settings.visits_per_hour,
                "last_24_hours": round(sum(w for _, w in self._spent(now, _DAY)), 1),
                "per_24_hours": settings.visits_per_day,
            },
            "invitations": {
                "last_24_hours": within(record.invitations, _DAY),
                "per_24_hours": settings.invites_per_day,
                "last_7_days": within(record.invitations, _WEEK),
                "per_7_days": settings.invites_per_week,
                **(
                    {
                        "held_until": datetime.fromtimestamp(
                            record.invitations_held_until, UTC
                        ).isoformat()
                    }
                    if record.invitations_held_until > now
                    else {}
                ),
            },
            "messages": {
                "last_24_hours": within(record.messages, _DAY),
                "per_24_hours": settings.messages_per_day,
            },
            "people_search_pages": {
                "this_month": record.people_search_pages.get(_month(now), 0),
                "per_month": settings.search_pages_per_month,
            },
            # A stop included: an agent checking before a long run is told
            # when the next visit goes ahead, not that it could now.
            "next_visit_in_seconds": round(
                max(stop_left, self._pause_left(now), block[0] if block else 0.0), 1
            ),
            "stop": (
                {
                    "reason": record.stop_reason,
                    "until": datetime.fromtimestamp(now + stop_left, UTC).isoformat(),
                    "seconds_left": _whole_seconds(stop_left),
                }
                if stop_left > 0
                else None
            ),
        }


def status_on_disk() -> dict[str, Any]:
    """The process gate's ``status()`` as the record on disk has it.

    For ``get_session_status``, which answers without the profile lease: the
    writes are atomic, so the file read is a whole version, and another process
    may be the one writing it. A damaged record is reported rather than
    repaired, because repairing it writes, and only the lease holder writes.
    """
    current = gate()
    if not current.enabled or current._path is None:
        return current.status()
    try:
        raw = json.loads(current._path.read_text(encoding="utf-8"))
        record = _Record.from_json(raw)
    except FileNotFoundError:
        return current.status()
    except (OSError, ValueError, TypeError):
        return {
            "enabled": True,
            "record": "unreadable; the next LinkedIn tool call replaces it and "
            "starts with a stop",
        }
    view = copy.copy(current)
    view._record = record
    # A view never writes, whatever it finds to correct.
    view._read_only = True
    return view.status()


def _earliest(*deadlines: float | None) -> float | None:
    present = [deadline for deadline in deadlines if deadline is not None]
    return min(present) if present else None


def _whole_seconds(seconds: float) -> int:
    return max(1, math.ceil(seconds))


def _duration(seconds: float) -> str:
    seconds = _whole_seconds(seconds)
    if seconds < 120:
        return f"{seconds} seconds"
    minutes = math.ceil(seconds / 60)
    if minutes < 120:
        return f"{minutes} minutes"
    return f"{minutes / 60:.1f} hours"


def _path_of(url: str) -> str:
    try:
        return urlsplit(url).path or "/"
    except ValueError:
        return "/"


def _is_people_search(url: str) -> bool:
    return _path_of(url).rstrip("/") == _PEOPLE_SEARCH_PATH


def _month(now: float) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y-%m")


def _seconds_to_next_month(now: float) -> float:
    current = datetime.fromtimestamp(now, UTC)
    if current.month == 12:
        following = current.replace(year=current.year + 1, month=1, day=1)
    else:
        following = current.replace(month=current.month + 1, day=1)
    following = following.replace(hour=0, minute=0, second=0, microsecond=0)
    return max(0.0, following.timestamp() - now)


# -- the process's gate ------------------------------------------------------

_gate: Pacing | None = None


def gate() -> Pacing:
    """The process's gate, built from the configuration on first use."""
    global _gate
    if _gate is None:
        _gate = _build_gate()
    return _gate


def _build_gate() -> Pacing:
    from linkedin_mcp_server.config import get_config
    from linkedin_mcp_server.session_state import auth_root_dir

    settings = get_config().rate_limit
    path: Path | None
    try:
        path = auth_root_dir() / RECORD_FILE
    except Exception:
        logger.warning(
            "The session folder could not be resolved, so the rate limit record "
            "is kept in memory only",
            exc_info=True,
        )
        path = None
    # Not refreshed here: reading the record may repair it, which writes, and
    # this can run without the lease (get_session_status). Every change
    # refreshes first, and the middleware refreshes under the lease.
    return Pacing(settings, path)


def install_gate(replacement: Pacing) -> None:
    """Make *replacement* the process's gate. For tests and their harnesses."""
    global _gate
    _gate = replacement


def reset_pacing_for_testing() -> None:
    """Install a gate that paces nothing, so unrelated tests run at full speed.

    Tests about pacing install a real one with a fake clock.
    """
    install_gate(Pacing.disabled())


# -- signals from elsewhere, which must never raise into their callers ---------


def note_linkedin_refusal(status: int, retry_after: float | None) -> None:
    """LinkedIn answered 429 or 999, during a call or between calls."""
    try:
        gate().record_pushback(
            "http_999" if status == 999 else "http_429", retry_after=retry_after
        )
    except Exception:
        logger.debug("Could not record a LinkedIn refusal", exc_info=True)


def note_slow_down_page() -> None:
    """LinkedIn served a rate-limit page."""
    try:
        gate().record_pushback("slow_down_page")
    except Exception:
        logger.debug("Could not record a rate-limit page", exc_info=True)


def note_empty_page() -> None:
    """A page came back empty even after its retry, the usual sign of throttling."""
    try:
        gate().record_pushback("empty_page")
    except Exception:
        logger.debug("Could not record an empty page", exc_info=True)


def note_auth_route(url: str) -> None:
    """A tool call met a sign-in route; a challenge among them starts the long stop.

    Only during a tool call: a checkpoint met while you sign in yourself is
    yours to answer, and stopping your tools for a day after it would punish
    the sign-in rather than the automation.
    """
    if not in_tool_call():
        return
    path = _path_of(url).rstrip("/") or "/"
    if not any(
        path == route or path.startswith(f"{route}/") for route in _CHALLENGE_PATHS
    ):
        return
    try:
        gate().record_pushback("challenge")
    except Exception:
        logger.debug("Could not record a checkpoint", exc_info=True)


def note_account_restricted() -> None:
    """A tool call met LinkedIn's account-restriction route."""
    if not in_tool_call():
        return
    try:
        gate().record_pushback("restricted")
    except Exception:
        logger.debug("Could not record a restriction", exc_info=True)
