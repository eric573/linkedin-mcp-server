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
  like a page. A scroll or "Show more" click that loads more items counts a
  fifth. Scrolling a page that is already loaded is part of its visit.
- Invitations and messages have limits of their own, counted at the click
  whatever comes of it.
- When LinkedIn pushes back, every request stops for a while: an HTTP 429 or
  999 from its hosts at any time, its rate-limit page, a page still empty after
  its retry, a checkpoint or restriction met during a tool call. The longer of
  LinkedIn's `Retry-After` and the server's own wait wins, and each stop that
  starts within a day of another lasts twice as long. The browser closes for
  it, because the page left open keeps talking to LinkedIn.
- Inside a tool call a short wait is waited out and a long one refuses with
  `rate_limited` and `retry_after`, keeping whatever pages were read. Outside a
  call nothing is refused, only paced. After a write the call is never
  refused, because the read that confirms it is the only way to learn whether
  it went out.

What the gate remembers is kept as timestamps in `rate-limit.json` beside the
session, written atomically while the profile lease is held, so restarts and
every process sharing the session spend one budget.

This supersedes the earlier record in one respect. A refusal is still read from
the status and never from words on the page, still counts only from LinkedIn's
own hosts, and is still appended to a failure that happened anyway. What
changes is that the record is no longer evidence alone: every refusal now stops
requests, whichever call it answered. A refused beacon may have cost its page
nothing, and stopping on it anyway is deliberate. A false stop costs minutes;
the opposite mistake can cost the account.

LinkedIn publishes none of its limits, so the defaults are cautious guesses and
every one is a setting.
