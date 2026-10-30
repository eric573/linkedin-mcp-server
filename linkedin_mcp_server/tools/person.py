"""
LinkedIn person profile reading tools.

Uses innerText extraction for resilient profile data capture
with configurable section selection.
"""

import json
import logging
import time
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import BeforeValidator, Field

from linkedin_mcp_server.callbacks import MCPContextProgressCallback
from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.dependencies import get_ready_extractor, tool_errors
from linkedin_mcp_server.error_handler import CodedToolError, ErrorCode
from linkedin_mcp_server.linkedin import parse_person_sections
from linkedin_mcp_server.linkedin.contracts import FilterValidationError
from linkedin_mcp_server.linkedin.identifiers import normalize_person_identifier
from linkedin_mcp_server.linkedin.search_urls import (
    build_people_search_url,
    read_cursor,
)

logger = logging.getLogger(__name__)


def _coerce_str_list(value: Any) -> Any:
    """Accept a string where a list of strings is declared.

    ``network`` is published as ``anyOf: [array, null]``, which is a correct
    JSON Schema. Some MCP clients collapse an ``anyOf``-with-null union to an
    untyped ``{}`` and then transmit the value as a string, so the array the
    caller wrote never arrives as one and pydantic rejects it (#739).

    Coercing at the tool boundary keeps the transport quirk here and leaves
    ``LinkedInExtractor.search_people`` strictly ``list[str]``. Only the
    container shape is repaired; token values are still validated downstream,
    so an invalid token fails with the same message it always did.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except ValueError:
            pass
        else:
            if isinstance(decoded, list):
                return decoded

    return [part.strip() for part in text.split(",") if part.strip()]


StrList = Annotated[list[str], BeforeValidator(_coerce_str_list)]


def register_person_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register all person-related tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Person Profile",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person"},
    )
    @tool_errors
    async def get_person_profile(
        linkedin_username: str,
        ctx: Context,
        sections: str | None = None,
        max_scrolls: Annotated[int, Field(ge=1, le=50)] | None = None,
    ) -> dict[str, Any]:
        """
        Get a specific person's LinkedIn profile.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates"). A full profile URL is accepted too and is reduced to the username.
            ctx: FastMCP context for progress reporting
            sections: Comma-separated list of extra sections to read.
                The main profile page is always included.
                Available sections: experience, education, interests, honors, volunteering, languages, certifications, skills, projects, contact_info, posts
                Examples: "experience,education", "contact_info", "skills,projects", "honors,languages", "posts"
                Default (None) reads only the main profile page.
            max_scrolls: Maximum pagination attempts per section to load more content.
                On detail sections (experience, certifications, skills, etc.) this
                is the max number of "Show more" button clicks. On activity/posts
                it is the max scroll-to-bottom iterations. Applies to all sections
                in this call. Default (None) uses 5 for detail sections and 10 for
                posts. Increase when a profile has many items in a section
                (e.g., 30+ certifications, max_scrolls=20). To avoid slowing down
                other sections, request heavy sections in a separate call.

        Returns:
            Dict with url, sections (name -> raw text), and optional references and section_errors.
            Sections may be absent if extraction yielded no content for that page.
            contact_info is read only from an accepted contact-overlay root. If no such root
            is found, the section is omitted and section_errors explains the failure;
            underlying profile text and links are never substituted. Existing suspected
            rate-limit retry and stop behavior is retained.
            Includes unknown_sections when unrecognised names are passed.
            The LLM should parse the raw text in each section.
        """
        linkedin_username = normalize_person_identifier(linkedin_username)
        extractor = await get_ready_extractor(ctx, tool_name="get_person_profile")
        requested, unknown = parse_person_sections(sections)

        logger.info(
            "Reading profile: %s (sections=%s)",
            linkedin_username,
            sections,
        )

        cb = MCPContextProgressCallback(ctx)
        result = await extractor.read_person(
            linkedin_username,
            requested,
            callbacks=cb,
            max_scrolls=max_scrolls,
        )

        if unknown:
            result["unknown_sections"] = unknown

        return result

    @mcp.tool(
        timeout=tool_timeout,
        title="Search People",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person", "search"},
    )
    @tool_errors
    async def search_people(
        ctx: Context,
        keywords: str | None = None,
        location: str | None = None,
        network: StrList | None = None,
        current_company: str | None = None,
        connection_of: str | None = None,
        max_pages: Annotated[int, Field(ge=1, le=10)] = 1,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for people on LinkedIn, ten per result page.

        Lists a company's employees with current_company, and a member's
        connections with connection_of; keywords are optional when a filter
        is given. Every page is a page load against the account's rate limit,
        so ask for more pages only when the first ones did not answer the
        question.

        Args:
            ctx: FastMCP context for progress reporting
            keywords: Search keywords (e.g., "software engineer", "recruiter at
                Google"). Optional when any filter below is given.
            location: Optional location filter, as LinkedIn's numeric geo URN
                id (e.g., "103644278" for the United States). Plain-text place
                names are silently ignored by LinkedIn and return unfiltered
                results, so they are refused here. The id is the geoUrn value
                in the URL of a LinkedIn people search filtered to that place;
                without one, put the place in keywords instead.
            network: Optional connection-degree filter. Each element is one of
                "F" (1st-degree), "S" (2nd-degree), "O" (3rd-degree and beyond).
                Example: ["F"] to only return 1st-degree connections. A single
                token ("F") or a comma-separated string ("F,S") is also
                accepted, for clients that cannot transmit an array.
            current_company: Optional current-employer filter. LinkedIn's
                currentCompany facet only filters on the numeric company URN id
                (e.g. "1115" for SAP); plain company names are accepted by the
                URL but ignored by LinkedIn and return the unfiltered result
                set. Look up a company's URN via get_company_profile -- it is
                exposed under references["about"]. For company-wide employee
                demographics (location/education/function breakdown) plus a
                slug-based lookup, use get_company_employees instead.
            connection_of: Optional member filter: only that member's
                connections. The member's profile URN id (e.g. "ACoAAB..."),
                which get_person_profile returns as profile_urn; a username is
                refused, because LinkedIn ignores it. LinkedIn decides how much
                of the list it shows: a member can hide their connections, and
                for someone you are not connected to it may show only the
                connections you share. Combine with network to narrow it.
            max_pages: Result pages to read in this call (1-10, default 1).
            cursor: pagination.next_cursor from an earlier call with the same
                keywords and filters, to continue where it stopped. A cursor
                from a different search is refused.

        Returns:
            Dict with url, sections (search_results -> raw text of every page
            read, joined by "---"), and optional references. The LLM should
            parse the raw text to extract individual people and their profiles.
            pagination ({pages_fetched, next_cursor, reason}) says why reading
            stopped: complete (no more results), no_new_results (LinkedIn
            served results already returned), max_pages, time_budget (the tool
            timeout was near) or rate_limited (the account's rate limit or
            LinkedIn stopped it; section_errors says when to retry). Only the
            last three come with a next_cursor, and more results may exist.
            Members LinkedIn shows without a profile link ("LinkedIn Member")
            have no URL to return, but their page is read and passed.
            pagination is absent when a failure stopped the search, which
            section_errors then explains, for example dropped_page when
            LinkedIn answered a result page with a different one.
        """
        # Before the browser, as in search_jobs: the extractor's time budget
        # is what is left of the figure FastMCP cancels this call on.
        started = time.monotonic()
        terms = keywords or ""
        # Blank is no filter, as the URL builder reads it, so it does not
        # count as one here either.
        connection_of = (connection_of or "").strip() or None
        if not terms.strip() and not (
            location or network or current_company or connection_of
        ):
            raise CodedToolError(
                "Give keywords, a filter (location, network, current_company, "
                "connection_of), or both. A search with neither lists people "
                "LinkedIn picks, at the cost of a page load each.",
                ErrorCode.INVALID_ARGUMENT,
            )
        try:
            # The builder refuses a filter LinkedIn would ignore. Doing it here
            # keeps that refusal off the browser: get_ready_extractor can install
            # Chromium or rotate a login before this call would have failed.
            search_url = build_people_search_url(
                terms,
                location=location,
                network=network,
                current_company=current_company,
                connection_of=connection_of,
            )
        except FilterValidationError as e:
            raise CodedToolError(str(e), ErrorCode.INVALID_ARGUMENT) from e
        # A cursor from another search would page through results nobody asked
        # for, so it is refused here, before a page is spent on it.
        resume = read_cursor(cursor, search_url, first=1)

        extractor = await get_ready_extractor(ctx, tool_name="search_people")
        logger.info(
            "Searching people: keywords='%s', location='%s', network=%s, "
            "current_company='%s', connection_of='%s', pages %d-%d",
            terms,
            location,
            network,
            current_company,
            connection_of,
            resume.position,
            resume.position + max_pages - 1,
        )

        await ctx.report_progress(
            progress=0, total=100, message="Starting people search"
        )

        try:
            result = await extractor.search_people(
                terms,
                location,
                network=network,
                current_company=current_company,
                connection_of=connection_of,
                max_pages=max_pages,
                first_page=resume.position,
                last_results=resume.last_results,
                # What is left of the figure FastMCP cancels this call on.
                tool_timeout=max(0.0, tool_timeout - (time.monotonic() - started)),
            )
        except FilterValidationError as e:
            # Validation messages carry actionable detail; surface
            # them as ToolError so mask_error_details doesn't reduce
            # them to "Error calling tool 'search_people'".
            raise CodedToolError(str(e), ErrorCode.INVALID_ARGUMENT) from e

        await ctx.report_progress(progress=100, total=100, message="Complete")

        return result

    @mcp.tool(
        timeout=tool_timeout,
        title="Connect With Person",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"person", "actions"},
    )
    @tool_errors
    async def connect_with_person(
        linkedin_username: str,
        ctx: Context,
        note: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a LinkedIn connection request or accept an incoming one.

        The tool is annotated with destructiveHint so MCP clients will
        prompt for user confirmation before execution.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates"). A full profile URL is accepted too and is reduced to the username.
            ctx: FastMCP context for progress reporting
            note: Optional note to include with the invitation

        Returns:
            Dict with url, status, message, and note_sent.
            Statuses: pending, already_connected, follow_only,
            connect_unavailable, unavailable, send_failed,
            note_not_supported, custom_note_limit_reached,
            connected, or accepted.

            ``connected`` means this call submitted the invitation and the
            re-read profile no longer exposes Connect; it does not mean a
            1st-degree connection. The ``message`` names the state read after
            the send, normally pending. ``pending`` means an invitation was
            already outstanding before the call, and ``accepted`` means an
            incoming invitation was accepted.

            When status is ``custom_note_limit_reached`` LinkedIn rejected
            personalized invite notes because the free note quota for the
            account is exhausted. The ``message`` is the raw Premium dialog
            text read from LinkedIn.

            A status of ``outcome_unknown`` comes from the transport rather
            than the page: the call's answer never arrived from the browser
            process, so whether the invitation was sent is unknown. It carries
            ``retry_safe: False`` and no ``note_sent``; check the profile
            before calling again, because a repeat may invite twice.
        """
        linkedin_username = normalize_person_identifier(linkedin_username)
        extractor = await get_ready_extractor(ctx, tool_name="connect_with_person")
        logger.info(
            "Connecting with person: %s (note=%s)",
            linkedin_username,
            note is not None,
        )

        await ctx.report_progress(
            progress=0,
            total=100,
            message="Starting LinkedIn connection flow",
        )

        result = await extractor.connect_with_person(
            linkedin_username,
            note=note,
        )

        await ctx.report_progress(progress=100, total=100, message="Complete")

        return result

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Sidebar Profiles",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person"},
    )
    @tool_errors
    async def get_sidebar_profiles(
        linkedin_username: str,
        ctx: Context,
    ) -> dict[str, Any]:
        """
        Get profile links from sidebar recommendation sections on a LinkedIn profile page.

        Extracts profiles from "More profiles for you", "Explore premium profiles",
        and "People you may know" sidebar sections. Follows "Show all" links to
        return the full list from each section. Sections that redirect to
        linkedin.com/premium are skipped.

        Args:
            linkedin_username: LinkedIn username of the profile page to read; a full profile URL is accepted too
                (e.g., "stickerdaniel", "williamhgates")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url and sidebar_profiles mapping section key to a list of
            /in/username/ paths. Only sections present on the page are included.
        """
        linkedin_username = normalize_person_identifier(linkedin_username)
        extractor = await get_ready_extractor(ctx, tool_name="get_sidebar_profiles")
        logger.info("Getting sidebar profiles for: %s", linkedin_username)

        await ctx.report_progress(
            progress=0, total=100, message="Extracting sidebar profiles"
        )

        result = await extractor.get_sidebar_profiles(linkedin_username)

        await ctx.report_progress(progress=100, total=100, message="Complete")

        return result

    @mcp.tool(
        timeout=tool_timeout,
        title="Get My Profile",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person"},
    )
    @tool_errors
    async def get_my_profile(
        ctx: Context,
        sections: str | None = None,
        max_scrolls: Annotated[int, Field(ge=1, le=50)] | None = None,
    ) -> dict[str, Any]:
        """
        Get the authenticated user's own LinkedIn profile.

        Navigates to /in/me/ and resolves the redirect to obtain the real
        username before reading the profile, so the url field in the result is the actual
        profile URL (e.g. linkedin.com/in/johndoe/) rather than /in/me/.

        Args:
            ctx: FastMCP context for progress reporting
            sections: Comma-separated list of extra sections to read.
                The main profile page is always included.
                Available sections: experience, education, interests, honors, volunteering, languages, certifications, skills, projects, contact_info, posts
                Examples: "experience,education", "contact_info", "skills,projects"
                Default (None) reads only the main profile page.
            max_scrolls: Maximum pagination attempts per section (same as get_person_profile).

        Returns:
            Dict with url, sections (name -> raw text), and optional references and section_errors.
            The url field reflects the resolved profile URL, revealing the real username.
            Sections may be absent if extraction yielded no content for that page.
            contact_info is read only from an accepted contact-overlay root. If no such root
            is found, the section is omitted and section_errors explains the failure;
            underlying profile text and links are never substituted. Existing suspected
            rate-limit retry and stop behavior is retained.
            Includes unknown_sections when unrecognised names are passed.
        """
        extractor = await get_ready_extractor(ctx, tool_name="get_my_profile")
        requested, unknown = parse_person_sections(sections)

        logger.info("Reading own profile (sections=%s)", sections)

        cb = MCPContextProgressCallback(ctx)
        result = await extractor.get_my_profile(
            sections=requested,
            callbacks=cb,
            max_scrolls=max_scrolls,
        )

        if unknown:
            result["unknown_sections"] = unknown

        return result
