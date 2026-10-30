"""URL grammar and filter validation for LinkedIn's four search surfaces."""

from __future__ import annotations

from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit

import hashlib
import json
import re

from linkedin_mcp_server.core.exceptions import InvalidReferenceError
from linkedin_mcp_server.linkedin.contracts import FilterValidationError

# Normalization maps for job search filters. Job search encodes recency as
# ``f_TPR=r<seconds>``; content search uses named tokens, hence the separate
# ``CONTENT_DATE_POSTED_MAP`` below.
JOB_DATE_POSTED_MAP = {
    "past_hour": "r3600",
    "past_24_hours": "r86400",
    "past_week": "r604800",
    "past_month": "r2592000",
}

EXPERIENCE_LEVEL_MAP = {
    "internship": "1",
    "entry": "2",
    "associate": "3",
    "mid_senior": "4",
    "director": "5",
    "executive": "6",
}

JOB_TYPE_MAP = {
    "full_time": "F",
    "part_time": "P",
    "contract": "C",
    "temporary": "T",
    "volunteer": "V",
    "internship": "I",
    "other": "O",
}

WORK_TYPE_MAP = {"on_site": "1", "remote": "2", "hybrid": "3"}

SORT_BY_MAP = {"date": "DD", "relevance": "R"}

# Content (post) search uses literal ``datePosted`` tokens inside a JSON-list
# facet, e.g. ``datePosted=["past-week"]`` — unlike job search, which uses
# ``f_TPR=r<seconds>`` codes. The three hyphenated values are LinkedIn's
# complete set, verified live: the filter dropdown offers exactly Past 24
# hours / week / month, and anything else is ignored while still being echoed
# back in the url, so a near-miss spelling returns unfiltered results that
# look filtered. The underscore keys are this server's own spelling, carried
# over so ``date_posted`` reads the same here as in ``search_jobs``
# (``JOB_DATE_POSTED_MAP``); ``past_hour`` has no content-search equivalent.
CONTENT_DATE_POSTED_MAP = {
    "past-24h": "past-24h",
    "past_24_hours": "past-24h",
    "past-week": "past-week",
    "past_week": "past-week",
    "past-month": "past-month",
    "past_month": "past-month",
}

# Content-search ``contentType`` facet: the "Content type" menu's options,
# keyed by their label, valued by the token LinkedIn writes into the URL.
# Measured on the live menu; the labels are not the tokens ("Images" writes
# "photos", "Job posts" writes "jobs"). There is no Articles option.
CONTENT_TYPE_MAP = {
    "videos": "videos",
    "images": "photos",
    "job_posts": "jobs",
    "live_videos": "liveVideos",
    "documents": "documents",
}

# A member's profile URN id, as ``get_person_profile`` returns it, with or
# without the URN prefix. Narrower than what
# ``identifiers.normalize_profile_urn`` accepts, because a vanity name has the
# same characters and LinkedIn ignores one in the connections filter, which
# answers with everyone the rest of the search matches. Every member id read
# so far begins ``ACoA``, and a vanity name that does is not worth the guess.
_PROFILE_URN_PREFIX = "urn:li:fsd_profile:"
_MEMBER_ID = re.compile(r"^ACoA[A-Za-z0-9_-]+$")

# Valid tokens for the people-search ``network`` facet.
# LinkedIn accepts "F" (1st-degree), "S" (2nd-degree), "O" (3rd-degree and beyond).
NETWORK_TOKENS = ("F", "S", "O")


def _normalize_csv(value: str, mapping: dict[str, str]) -> str:
    """Normalize and encode each token in a comma-separated filter value."""
    parts = [v.strip() for v in value.split(",")]
    return ",".join(quote_plus(mapping.get(p, p), safe="") for p in parts)


def _encode_list_facet(values: list[str]) -> str:
    """Encode a list of string values for a LinkedIn search list facet.

    LinkedIn's people- and content-search URLs use JSON-list encoded facets of
    the form ``["A","B"]``. This helper URL-encodes the rendered JSON so the
    final URL contains e.g. ``%5B%22F%22%5D`` for ``["F"]``.
    """
    return quote_plus(json.dumps(values, separators=(",", ":")))


def build_job_search_url(
    keywords: str,
    location: str | None = None,
    date_posted: str | None = None,
    job_type: str | None = None,
    experience_level: str | None = None,
    work_type: str | None = None,
    easy_apply: bool = False,
    sort_by: str | None = None,
) -> str:
    """Build a LinkedIn job search URL with optional filters.

    Human-readable names are normalized to LinkedIn URL codes.
    Comma-separated values are normalized individually.
    Unknown values pass through unchanged.
    """
    params = f"keywords={quote_plus(keywords)}"
    if location:
        params += f"&location={quote_plus(location)}"

    if date_posted:
        mapped = JOB_DATE_POSTED_MAP.get(date_posted.strip(), date_posted)
        params += f"&f_TPR={quote_plus(mapped)}"
    if job_type:
        params += f"&f_JT={_normalize_csv(job_type, JOB_TYPE_MAP)}"
    if experience_level:
        params += f"&f_E={_normalize_csv(experience_level, EXPERIENCE_LEVEL_MAP)}"
    if work_type:
        params += f"&f_WT={_normalize_csv(work_type, WORK_TYPE_MAP)}"
    if easy_apply:
        params += "&f_EA=true"
    if sort_by:
        mapped = SORT_BY_MAP.get(sort_by.strip(), sort_by)
        params += f"&sortBy={quote_plus(mapped)}"

    return f"https://www.linkedin.com/jobs/search/?{params}"


def build_people_search_url(
    keywords: str,
    location: str | None = None,
    network: list[str] | None = None,
    current_company: str | None = None,
    connection_of: str | None = None,
    page: int = 1,
) -> str:
    """Build a LinkedIn people search URL, refusing filters LinkedIn ignores.

    Every refusal happens before a URL exists, so a workflow calling this can
    never navigate on a filter LinkedIn would swallow. An unknown ``network``
    token, a plain-text ``location`` and a plain-text ``currentCompany`` are
    each accepted by the URL and then dropped, which answers with the
    unfiltered result set while the request still reads as filtered.

    ``location`` is the same shape of mistake as ``currentCompany``: people
    search filters on the numeric geo URN id via the ``geoUrn`` list facet, and
    a ``location=`` query parameter is ignored outright -- so a search that
    reads as scoped to one city returns the whole world.

    ``connection_of`` is a member's profile URN id, the ``ACoAA...`` value
    ``get_person_profile`` returns as ``profile_urn``, and filters to that
    member's connections through LinkedIn's ``connectionOf`` facet, the one its
    own "connections" link on a profile opens. A vanity name is refused for the
    same reason as a place name: LinkedIn would ignore it.

    ``page`` is LinkedIn's own result page, ten people each; the first page
    carries no parameter.
    """
    if network is not None:
        invalid = [t for t in network if t not in NETWORK_TOKENS]
        if invalid:
            raise FilterValidationError(
                "Invalid network token(s) "
                f"{invalid!r}; expected any of {list(NETWORK_TOKENS)!r}"
            )

    if location and not re.fullmatch(r"[0-9]+", location):
        raise FilterValidationError(
            f"location must be a numeric LinkedIn geo URN id "
            f"(e.g. '103644278' for the United States); got {location!r}. "
            f"Plain-text place names are silently ignored by LinkedIn and "
            f"return the unfiltered result set. Use keywords when the geo URN "
            f"is not available."
        )

    if current_company and not re.fullmatch(r"[0-9]+", current_company):
        raise FilterValidationError(
            f"current_company must be a numeric LinkedIn company URN id "
            f"(e.g. '1115' for SAP); got {current_company!r}. Plain-text "
            f"company names are silently ignored by LinkedIn. Look up the "
            f'URN via get_company_profile -> references["about"].'
        )

    # Blank is no filter, as for the company and the place.
    member = None
    if connection_of is not None and connection_of.strip():
        member = connection_of.strip().removeprefix(_PROFILE_URN_PREFIX)
        if not _MEMBER_ID.fullmatch(member):
            raise FilterValidationError(
                f"connection_of must be a LinkedIn profile URN id, the "
                f"profile_urn get_person_profile returns (e.g. "
                f"'ACoAABcdEfG...'); got {connection_of!r}. A username or "
                f"profile URL is ignored by LinkedIn's connections filter."
            )

    params = f"keywords={quote_plus(keywords)}"
    if location:
        params += f"&geoUrn={_encode_list_facet([location])}"
    if network:
        params += f"&network={_encode_list_facet(network)}"
    if current_company:
        params += f"&currentCompany={_encode_list_facet([current_company])}"
    if member:
        params += f"&connectionOf={_encode_list_facet([member])}"
    params += _page_parameter(page)

    return f"https://www.linkedin.com/search/results/people/?{params}"


def build_company_search_url(keywords: str, page: int = 1) -> str:
    """Build a LinkedIn company search URL.

    A function even for a few parameters: every search URL this server
    navigates to is then written in one place, so the next caller has nothing
    to assemble by hand and no second spelling of the host and route to keep
    in step with the other three. ``page`` is LinkedIn's own result page, as
    in people search.
    """
    return (
        "https://www.linkedin.com/search/results/companies/"
        f"?keywords={quote_plus(keywords)}{_page_parameter(page)}"
    )


def _page_parameter(page: int) -> str:
    """``&page=N`` for a later result page, nothing for the first."""
    if page < 1:
        raise FilterValidationError(f"page must be 1 or more, got {page}")
    return "" if page == 1 else f"&page={page}"


# -- resuming a paged search -------------------------------------------------
#
# A cursor is the position the next call starts from, followed by a short
# fingerprint of the search it belongs to: "3.1f2e3d4c" resumes that search at
# position 3. Handed back to a different search, it would page through results
# the caller never asked for, so the fingerprint refuses it. The position is a
# result page for people and company search, an offset for job search and a
# page index for the saved-jobs tracker; the search it belongs to decides.

_PAGING_PARAMETERS = frozenset({"page", "start"})


def search_fingerprint(url: str) -> str:
    """Eight hex digits naming one search: its route and query, paging aside."""
    parts = urlsplit(url)
    query = sorted(
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if name not in _PAGING_PARAMETERS
    )
    identity = f"{parts.netloc}{parts.path.rstrip('/')}?{urlencode(query)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]


def make_cursor(position: int, search_url: str) -> str:
    """The cursor that resumes the search at *search_url* from *position*."""
    return f"{position}.{search_fingerprint(search_url)}"


# Why a paged read stopped, when another call can pick up where it did. A read
# that ran out of results, or met the same results again, has nothing left.
_RESUMABLE_STOPS = frozenset({"max_pages", "time_budget", "rate_limited"})


def next_cursor(reason: str, position: int, search_url: str) -> str | None:
    """The cursor for a read that stopped for *reason* before *position*.

    None when the stop leaves nothing to read, so a cursor in a result always
    means more can be had.
    """
    if reason not in _RESUMABLE_STOPS:
        return None
    return make_cursor(position, search_url)


def read_cursor(cursor: str | None, search_url: str, *, first: int) -> int:
    """Where *cursor* resumes the search at *search_url*; *first* without one.

    Raises:
        InvalidReferenceError: When the cursor is malformed or came from a
            different search.
    """
    if cursor is None or not cursor.strip():
        return first
    position_text, separator, fingerprint = cursor.strip().partition(".")
    if not separator or not position_text.isdigit() or not fingerprint:
        raise InvalidReferenceError(
            f"cursor {cursor!r} is not one this server gave out. Pass the "
            "pagination.next_cursor of the previous result exactly, or leave "
            "cursor out to start from the beginning."
        )
    if fingerprint != search_fingerprint(search_url):
        raise InvalidReferenceError(
            "cursor belongs to a different search. Repeat the search it came "
            "from with the same keywords and filters, or leave cursor out to "
            "start this one from the beginning."
        )
    position = int(position_text)
    if position < first:
        raise InvalidReferenceError(
            f"cursor {cursor!r} names a position before the first one."
        )
    return position


def build_content_search_url(
    keywords: str,
    date_posted: str | None = None,
    content_type: str | None = None,
) -> str:
    """Build a LinkedIn content (post) search URL.

    Reproduces the ``FACETED_SEARCH`` URL LinkedIn produces from the
    Posts results tab, e.g. for "Buscamos Unity" in the past week:
    ``/search/results/content/?keywords=Buscamos+Unity&origin=FACETED_SEARCH&datePosted=%5B%22past-week%22%5D``

    The ``datePosted`` facet is a one-element JSON list carrying a literal
    LinkedIn token, URL-encoded — unlike job search, which uses
    ``f_TPR=r<seconds>``. The value is mapped through
    ``CONTENT_DATE_POSTED_MAP`` so the server's own underscore spelling
    reaches LinkedIn in the form it recognizes. An unmapped value is refused
    here rather than sent, because LinkedIn ignores one instead of rejecting
    it and answers an unfiltered search that reads as a filtered one.

    ``content_type`` is refused the same way and for the same reason. It is
    written after ``datePosted``, the order the live page produces.
    """
    if (
        date_posted is not None
        and date_posted.strip()
        and date_posted.strip() not in CONTENT_DATE_POSTED_MAP
    ):
        raise FilterValidationError(
            f"Invalid date_posted {date_posted!r}; expected one of "
            f"{list(CONTENT_DATE_POSTED_MAP)!r}."
        )

    if (
        content_type is not None
        and content_type.strip()
        and content_type.strip() not in CONTENT_TYPE_MAP
    ):
        raise FilterValidationError(
            f"Invalid content_type {content_type!r}; expected one of "
            f"{list(CONTENT_TYPE_MAP)!r}."
        )

    params = f"keywords={quote_plus(keywords)}&origin=FACETED_SEARCH"
    if date_posted and date_posted.strip():
        token = CONTENT_DATE_POSTED_MAP.get(date_posted.strip(), date_posted.strip())
        params += f"&datePosted={_encode_list_facet([token])}"
    if content_type and content_type.strip():
        token = CONTENT_TYPE_MAP[content_type.strip()]
        params += f"&contentType={_encode_list_facet([token])}"
    return f"https://www.linkedin.com/search/results/content/?{params}"
