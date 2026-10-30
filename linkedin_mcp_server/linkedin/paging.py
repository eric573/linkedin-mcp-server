"""Walk the numbered result pages of a people or company search."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core import pacing
from linkedin_mcp_server.core.pacing import PacingRefusedError
from linkedin_mcp_server.linkedin.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.linkedin.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    rate_limited_section_error,
)
from linkedin_mcp_server.linkedin.link_metadata import (
    Reference,
    ReferenceKind,
    dedupe_references,
)
from linkedin_mcp_server.linkedin.search_urls import next_cursor

logger = logging.getLogger(__name__)

# The share of the tool's timeout a walk may spend, as job search does: a
# cancelled tool returns nothing, so the walk stops itself while there is still
# time to hand back the pages it read.
_TIME_FRACTION = 0.8


async def walk_result_pages(
    capture: SectionCapture,
    page_url: Callable[[int], str],
    *,
    result_kind: ReferenceKind,
    first_page: int = 1,
    max_pages: int = 1,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Read result pages ``first_page`` onward, at most ``max_pages`` of them.

    *page_url* builds the address of one result page. *result_kind* is what a
    result links to, ``person`` or ``company``: a page that links to none of
    them is past the last one, and a page whose results were all read already
    is LinkedIn serving an earlier page again, so either ends the walk rather
    than return the same people twice.

    Every page is a navigation, so the pacing gate spaces and counts each one.
    A page it refuses ends the walk with the pages already read, and the
    cursor resumes at the refused one.

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
    slowest = 0.0
    texts: list[str] = []
    references: list[Reference] = []
    seen: set[str] = set()
    section_errors: dict[str, dict[str, Any]] = {}
    reason: str | None = None
    pages_fetched = 0
    next_page = first_page

    for page in range(first_page, first_page + max_pages):
        next_page = page
        elapsed = time.monotonic() - started
        # Costed from the slowest page so far plus the wait the gate will add,
        # the same prediction job search makes.
        if texts and elapsed + gate.expected_wait() + slowest > budget:
            logger.debug("Stopping before page %d: %.1fs spent", page, elapsed)
            reason = "time_budget"
            break

        page_started = time.monotonic()
        waited_before = gate.waited_seconds
        try:
            extracted = await capture.capture(
                page_url(page),
                section_name="search_results",
                plan=CapturePlan(CaptureMode.SEARCH_RESULTS),
            )
        except PacingRefusedError as refused:
            # Nothing read means the refusal is the answer.
            if not texts:
                raise
            section_errors["search_results"] = rate_limited_section_error(
                str(refused), retry_after=refused.retry_after
            )
            reason = "rate_limited"
            break
        pages_fetched += 1
        # The gate's waits come back out, or the prediction above would
        # count them twice.
        page_seconds = time.monotonic() - page_started
        slowest = max(slowest, page_seconds - (gate.waited_seconds - waited_before))

        if extracted.text == RATE_LIMITED_SECTION_TEXT:
            section_errors["search_results"] = rate_limited_section_error()
            reason = "rate_limited"
            break
        if not extracted.text:
            if extracted.error:
                section_errors["search_results"] = extracted.error
            else:
                reason = "complete"
            break

        results = {
            reference["url"]
            for reference in extracted.references
            if reference["kind"] == result_kind
        }
        if not results:
            # LinkedIn's own page for no results, past the last one. Kept when
            # it is all this call read, because it says so in LinkedIn's words.
            if not texts:
                texts.append(extracted.text)
            reason = "complete"
            break
        if results <= seen:
            logger.debug("Result page %d repeats earlier results, stopping", page)
            reason = "no_new_results"
            break

        seen |= results
        texts.append(extracted.text)
        references.extend(extracted.references)
        next_page = page + 1
    else:
        reason = "max_pages"

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
            # Only the stops that leave results unread get one, and they
            # all leave `next_page` at the first of those.
            "next_cursor": next_cursor(reason, next_page, url),
            "reason": reason,
        }
    return result
