"""Walk the numbered result pages of a people or company search."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core import pacing
from linkedin_mcp_server.core.exceptions import RateLimitError
from linkedin_mcp_server.core.pacing import PacingRefusedError
from linkedin_mcp_server.linkedin.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.linkedin.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    cut_short_section_error,
    rate_limited_section_error,
)
from linkedin_mcp_server.linkedin.link_metadata import (
    Reference,
    ReferenceKind,
    dedupe_references,
)
from linkedin_mcp_server.linkedin.search_urls import (
    LAST_RESULT_PAGE,
    next_cursor,
    results_digest,
)

logger = logging.getLogger(__name__)

# The share of the tool's timeout a walk may spend, as job search does: a
# cancelled tool returns nothing, so the walk stops itself while there is still
# time to hand back the pages it read.
_TIME_FRACTION = 0.8

# What a page with no result on it digests to.
_NO_RESULTS = results_digest(())


def dropped_page_section_error(page: int, landed_url: str) -> dict[str, str]:
    """The entry for a result page LinkedIn answered with a different one."""
    return {
        "error_type": "dropped_page",
        "error_message": (
            f"LinkedIn answered result page {page} with {landed_url}, so paging "
            "stopped rather than return another page's results as this one's."
        ),
    }


def _on_page(landed_url: str, requested_url: str, page: int) -> bool:
    """Whether *landed_url* is result page *page* of the search requested."""
    landed = urlsplit(landed_url)
    requested = urlsplit(requested_url)
    landed_page = parse_qs(landed.query).get("page", ["1"])[0]
    return (
        landed.netloc == requested.netloc
        and landed.path.rstrip("/") == requested.path.rstrip("/")
        and landed_page == str(page)
    )


async def walk_result_pages(
    capture: SectionCapture,
    page_url: Callable[[int], str],
    *,
    result_kind: ReferenceKind,
    first_page: int = 1,
    max_pages: int = 1,
    previous: str | None = None,
    landed_url: Callable[[], str] | None = None,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Read result pages ``first_page`` onward, at most ``max_pages`` of them.

    *page_url* builds the address of one result page, and *landed_url* reads
    where a page actually landed. *result_kind* is what a result links to,
    ``person`` or ``company``. *previous* is the digest of the results on the
    page before ``first_page``, which a cursor carries.

    The walk ends at a page whose results were all read already, the page
    before it included, which is LinkedIn serving an earlier page again: the
    same people twice would read as more results, and a cursor would loop on
    them. A page that links to none is past the last one when it is the
    search's first page or follows another such page. One between pages of
    results is kept and passed: members LinkedIn shows without a profile link
    leave a page like that.

    Every page is a navigation, so the pacing gate spaces and counts each one.
    A page it refuses, or that LinkedIn answers with its rate-limit page, ends
    the walk with the pages already read, and the cursor resumes at that page.

    Returns:
        ``{url, sections: {search_results}, references?, section_errors?,
        pagination?}``, where ``pagination`` is ``{pages_fetched,
        next_cursor, reason}`` unless a failure stopped the walk, which
        ``section_errors`` then explains.
    """
    gate = pacing.gate()
    url = page_url(first_page)
    started = time.monotonic()
    budget = tool_timeout * _TIME_FRACTION
    deadline = started + budget
    slowest = 0.0
    texts: list[str] = []
    references: list[Reference] = []
    seen: set[str] = set()
    section_errors: dict[str, dict[str, Any]] = {}
    reason: str | None = None
    pages_fetched = 0
    next_page = first_page
    last_page = min(first_page + max_pages - 1, LAST_RESULT_PAGE)

    for page in range(first_page, last_page + 1):
        next_page = page
        elapsed = time.monotonic() - started
        # Costed from the slowest page so far plus the wait the gate will add,
        # the same prediction job search makes.
        if pages_fetched and elapsed + gate.expected_wait() + slowest > budget:
            logger.debug("Stopping before page %d: %.1fs spent", page, elapsed)
            reason = "time_budget"
            break

        page_started = time.monotonic()
        waited_before = gate.waited_seconds
        held_back = pacing.loads_held_back()
        try:
            extracted = await capture.capture(
                page_url(page),
                section_name="search_results",
                plan=CapturePlan(CaptureMode.SEARCH_RESULTS),
                deadline=deadline,
            )
        except RateLimitError as refused:
            # The gate's refusal, or LinkedIn's own rate-limit page, which
            # starts a stop. Nothing read means the refusal is the answer.
            if not pages_fetched:
                raise
            # Out of time for the gate's wait is this walk's own time budget,
            # not LinkedIn or a limit.
            if isinstance(refused, PacingRefusedError) and refused.kind == "time":
                reason = "time_budget"
                break
            retry_after = (
                refused.retry_after
                if isinstance(refused, PacingRefusedError)
                else pacing.stop_retry_after()
            )
            section_errors["search_results"] = rate_limited_section_error(
                str(refused), retry_after=retry_after
            )
            reason = "rate_limited"
            break
        pages_fetched += 1
        # The gate's waits come back out, or the prediction above would
        # count them twice.
        page_seconds = time.monotonic() - page_started
        slowest = max(slowest, page_seconds - (gate.waited_seconds - waited_before))

        # A failure and LinkedIn's rate-limit page say more than where the page
        # ended up, so they come first.
        if not extracted.text and extracted.error:
            section_errors["search_results"] = extracted.error
            break
        if extracted.text == RATE_LIMITED_SECTION_TEXT:
            section_errors["search_results"] = rate_limited_section_error(
                retry_after=pacing.stop_retry_after()
            )
            reason = "rate_limited"
            break
        if landed_url is not None:
            landed = landed_url()
            if not _on_page(landed, page_url(page), page):
                logger.debug("Result page %d landed on %s, stopping", page, landed)
                section_errors["search_results"] = dropped_page_section_error(
                    page, landed
                )
                break
        if not extracted.text:
            reason = "complete"
            break

        results = {
            reference["url"]
            for reference in extracted.references
            if reference["kind"] == result_kind
        }
        if not results:
            if page == 1 or previous == _NO_RESULTS:
                # Past the last page. Kept when it is all this call read,
                # because it says so in LinkedIn's words.
                if not texts:
                    texts.append(extracted.text)
                reason = "complete"
                break
            # Results with no profile link, between pages that have them.
            texts.append(extracted.text)
            references.extend(extracted.references)
            previous = _NO_RESULTS
            next_page = page + 1
            continue
        digest = results_digest(results)
        if results <= seen or digest == previous:
            logger.debug("Result page %d repeats earlier results, stopping", page)
            reason = "no_new_results"
            break

        seen |= results
        texts.append(extracted.text)
        references.extend(extracted.references)
        # The rate limit stopped this page loading all of its results. What it
        # showed is kept, and the cursor reads the page again in full rather
        # than skip the ones it held back, with the page before it as the one
        # to compare with.
        if pacing.loads_held_back() > held_back:
            section_errors["search_results"] = cut_short_section_error(
                pacing.cut_short_retry_after()
            )
            reason = "rate_limited"
            break
        previous = digest
        next_page = page + 1
    else:
        reason = "complete" if next_page > LAST_RESULT_PAGE else "max_pages"

    result: dict[str, Any] = {
        "url": url,
        "sections": {"search_results": "\n---\n".join(texts)} if texts else {},
    }
    if references:
        result["references"] = {"search_results": dedupe_references(references)}
    if section_errors:
        result["section_errors"] = section_errors
    if reason is not None:
        result["pagination"] = {
            "pages_fetched": pages_fetched,
            # Only the stops that leave results unread get one, and they all
            # leave `next_page` at the first of those and `previous` at what
            # the page before it held.
            "next_cursor": next_cursor(reason, next_page, url, previous),
            "reason": reason,
        }
    return result
