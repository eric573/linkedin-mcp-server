# Pace every LinkedIn request

- Date: 2026-10-30
- Supersedes: [Throttling is a status](2026-09-23-throttling-is-a-status.md)

Nothing here limited how much the server asked of LinkedIn. A call paused two
seconds between its own pages and retried one empty page after five, but the
next call started at full speed, a refusal changed nothing about the call after
it, and opening conversations, scrolling endless lists and clicking "Show more"
fetched from LinkedIn about once a second with nothing counting. An account
that is throttled often is an account that gets restricted, and the server is
the one party that sees all of its own traffic.

So one gate, `core/pacing.py`, stands in front of everything the server makes
LinkedIn do. It holds one budget for the account rather than one per tool,
because LinkedIn judges what the account does, not which tool asked.

- Every page visit asks it first, waits out a random pause measured from the
  end of the previous load, and counts against limits per hour and per day. A
  navigation that does not ask fails `tests/test_pacing.py`.
- Opening a conversation counts as a visit, because LinkedIn loads the thread
  like a page. A scroll that grows the page, on a profile, a list or the job
  rail, or a "Show more" click counts a fifth; the scroll that finds the page
  complete costs nothing. Each asks before it is made and is counted as it
  happens, so a stop ends the loop at once and a cancelled call has counted
  what it sent, and a section the gate cut short says so instead of reading
  as complete. The next page visit waits its pause after the last load.
- Invitations and messages have limits of their own, counted at the click
  whatever comes of it.
- When LinkedIn pushes back, every request stops for a while: an HTTP 429 or
  999 from its hosts at any time, its rate-limit page, a page still empty after
  its retry, a checkpoint or restriction met during a tool call. The longer of
  LinkedIn's `Retry-After` (up to a day) and the server's own wait wins, and
  each stop that starts within a day of the end of the one before lasts twice
  as long as that one, up to a day, so pushback that keeps coming backs off
  further however it is spaced. The browser closes for it, because the page left open keeps talking
  to LinkedIn.
- A short wait is waited out and a long one refuses with `rate_limited` and
  `retry_after`, keeping whatever pages were read, and a stop refuses
  everything, inside a tool call and outside one (the CLI, a sign-in repair).
  After a write the call is never refused, because the read that confirms it
  is the only way to learn whether it went out; accepting an invitation counts
  as one.

What the gate remembers is kept as timestamps in `rate-limit.json` beside the
session, written atomically while the profile lease is held, so restarts and
every process sharing the session spend one budget. It is read again before
every change, so an idle process never writes an old copy over a newer one,
and a save that fails is tried again rather than lose a stop. A file that does
not parse never hands out a new budget: a process keeps the history it holds
and stops for a while, and one with none stops for a day and holds invitations
for a week. A file from a newer version of the server is left alone. A
timestamp a clock set back left in the future is pulled back to now, so no
limit outlasts what it reports.

This supersedes the earlier record in one respect. A refusal is still read from
the status and never from words on the page, still counts only from LinkedIn's
own hosts, and is still appended to a failure that happened anyway. What
changes is that the record is no longer evidence alone: every refusal now stops
requests, whichever call it answered. A refused beacon may have cost its page
nothing, and stopping on it anyway is deliberate. A false stop costs minutes;
the opposite mistake can cost the account.

LinkedIn publishes none of its limits, so the defaults are cautious guesses and
every one is a setting.
