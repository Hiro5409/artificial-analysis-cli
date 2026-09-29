"""Fetch validated model collections from the Artificial Analysis Data API."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import assert_never, get_args

import httpx2
from pydantic import ValidationError

from artificial_analysis_cli._generated import RESPONSES, Page
from artificial_analysis_cli._generated.pydantic_gen import (
    ImageArenaFreeResponse,
    ImageArenaResponse,
    LLMModelsFreeResponse,
    LLMModelsResponse,
    ModelPerformance,
    MusicArenaFreeResponse,
    MusicArenaResponse,
    Pricing,
    SpeechToSpeechFreeResponse,
    SpeechToSpeechResponse,
    SpeechToTextModelsFreeResponse,
    SpeechToTextModelsResponse,
    TextToSpeechFreeResponse,
    TextToSpeechResponse,
    VideoArenaFreeResponse,
    VideoArenaResponse,
)
from artificial_analysis_cli.catalog import (
    AccountTier,
    Family,
    FreeQuery,
    Query,
    options,
)
from artificial_analysis_cli.errors import CliError, ErrorCode
from artificial_analysis_cli.output import Collection, Metric, ModelRow, Number

BASE_URL = "https://artificialanalysis.ai"
TIMEOUT_SECONDS = 30
CACHE_MAX_AGE = timedelta(hours=1)


def fetch_collection(
    query: Query,
    *,
    api_key: str,
    cache_dir: Path,
    now: datetime,
    refresh: bool = False,
    transport: httpx2.BaseTransport | None = None,
) -> Collection:
    """Fetch every page of the query's collection, reusing a validated copy cached less than an hour ago.

    `transport` stands in for the network, as in httpx2. The HTTP client that uses it is created and closed here.
    """
    endpoint = _endpoint(query)
    response_type = RESPONSES[endpoint]
    path = cache_dir / f"{_cache_key(api_key, query)}.json"
    if not refresh and (entry := _read_cache(path, now)) is not None:
        try:
            pages = [_decode_page(response_type, body) for body in entry.bodies]
            for number, page in enumerate(pages, start=1):
                if _has_more(pages[0], page, number) != (number < len(pages)):
                    raise _invalid_response("The cache entry does not contain a complete collection.")
            return _collection(query, pages, fetched_at=entry.fetched_at, cached=True)
        except CliError:
            pass
    pages, bodies = _fetch_pages(query, endpoint, response_type, api_key, transport)
    collection = _collection(query, pages, fetched_at=now, cached=False)
    _write_cache(path, _CacheEntry(now, bodies))
    return collection


@dataclass(frozen=True)
class _FetchedPage:
    """Typed validation/display view alongside original JSON values for output."""

    model: Page
    fields: dict[str, object]
    records: list[dict[str, object]]


def _fetch_pages(
    query: Query,
    endpoint: str,
    response_type: type[Page],
    api_key: str,
    transport: httpx2.BaseTransport | None,
) -> tuple[list[_FetchedPage], list[str]]:
    pages: list[_FetchedPage] = []
    bodies: list[str] = []
    with httpx2.Client(
        base_url=BASE_URL,
        headers={"x-api-key": api_key},
        timeout=TIMEOUT_SECONDS,
        transport=transport,
    ) as client:
        while True:
            number = len(pages) + 1
            response = _request_page(client, query, endpoint, number)
            page = _decode_page(response_type, response.content)
            pages.append(page)
            bodies.append(response.text)
            if not _has_more(pages[0], page, number):
                return pages, bodies


def _has_more(first: _FetchedPage, current: _FetchedPage, number: int) -> bool:
    """Validate the page sequence and collection metadata before requesting another page."""
    if not isinstance(first.model, LLMModelsFreeResponse | LLMModelsResponse):
        return False
    if not isinstance(current.model, LLMModelsFreeResponse | LLMModelsResponse):
        raise _invalid_response("Artificial Analysis returned inconsistent response types across pages.")
    if (current.model.tier, current.model.intelligence_index_version) != (
        first.model.tier,
        first.model.intelligence_index_version,
    ):
        raise _invalid_response(
            f"Artificial Analysis returned inconsistent collection metadata on page {number}; "
            "the collection may have changed while it was fetched. Try again."
        )
    pagination = current.model.pagination
    total_pages = first.model.pagination.total_pages
    if (pagination.page, pagination.total_pages, pagination.has_more) != (
        number,
        total_pages,
        number < total_pages,
    ):
        raise _invalid_response(
            f"Artificial Analysis returned inconsistent pagination on page {number} of {total_pages}; "
            "the collection may have changed while it was fetched. Try again."
        )
    return pagination.has_more


def _request_page(client: httpx2.Client, query: Query, endpoint: str, number: int) -> httpx2.Response:
    params: dict[str, str | bool | int] = dict(options(query))
    if query.family is Family.LLM:
        params["page"] = number
    try:
        response = client.get(endpoint, params=params)
        _raise_for_status(response)
    except httpx2.TimeoutException as error:
        raise CliError(
            ErrorCode.TIMEOUT,
            f"Timed out after {TIMEOUT_SECONDS} seconds waiting for Artificial Analysis.",
            hint="Check the network, then retry the same command.",
        ) from error
    except httpx2.HTTPError as error:
        raise CliError(
            ErrorCode.NETWORK_ERROR,
            "Could not connect to Artificial Analysis.",
            hint="Check the network and Artificial Analysis service status, then retry.",
        ) from error
    return response


def _decode_page(response_type: type[Page], content: bytes | str) -> _FetchedPage:
    try:
        body = json.loads(content, parse_float=_finite_float, parse_constant=_finite_float)
    except ValueError as error:
        raise _invalid_response(
            "Artificial Analysis returned a response that is not valid JSON or contains non-finite numbers."
        ) from error
    try:
        parsed = response_type.model_validate(body, strict=True)
    except ValidationError as error:
        failure = error.errors(include_input=False, include_context=False, include_url=False)[0]
        field = str(failure["loc"][-1]) if failure["loc"] else "response"
        reason = f"missing field {field!r}" if failure["type"] == "missing" else f"{field}: {failure['msg']}"
        raise _schema_error(_one_line(reason)) from error
    return _FetchedPage(
        model=parsed,
        fields={key: value for key, value in body.items() if key not in {"data", "pagination"}},
        records=body["data"],
    )


def _finite_float(token: str) -> float:
    """Reject non-JSON constants and float overflow, including in unknown fields."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Non-finite number")
    return value


def _raise_for_status(response: httpx2.Response) -> None:
    """Report an unsuccessful status before validating its body."""
    if response.status_code == httpx2.codes.OK:
        return
    raise _status_error(response)


def _status_error(response: httpx2.Response) -> CliError:
    status = response.status_code
    match status:
        case httpx2.codes.BAD_REQUEST:
            detail = _error_detail(response)
            return CliError(
                ErrorCode.API_ERROR,
                f"Artificial Analysis rejected the request (HTTP 400){f': {detail}' if detail else ''}.",
                hint="Update artificial-analysis-cli. If the problem persists, report the command and response.",
            )
        case httpx2.codes.UNAUTHORIZED:
            return CliError(
                ErrorCode.UNAUTHORIZED,
                "Artificial Analysis rejected the API key (HTTP 401).",
                hint="Set ARTIFICIAL_ANALYSIS_API_KEY to a valid API key, then retry.",
            )
        case httpx2.codes.FORBIDDEN:
            tier = response.headers.get("X-AA-Tier")
            account = f"; account tier: {tier}" if tier in get_args(AccountTier) else ""
            return CliError(
                ErrorCode.FORBIDDEN,
                (
                    f"Artificial Analysis denied access to this endpoint (HTTP 403{account}). "
                    "Pro endpoint variants need a Pro or Commercial API key."
                ),
                hint="Use a Pro or Commercial API key, or omit --pro.",
            )
        case httpx2.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("Retry-After", "")
            retry = f" Retry after {retry_after} seconds." if retry_after.isdigit() else ""
            return CliError(
                ErrorCode.RATE_LIMITED,
                f"The Artificial Analysis API quota is exhausted (HTTP 429).{retry}",
                hint="Wait before retrying the same command.",
            )
        case _ if status >= httpx2.codes.INTERNAL_SERVER_ERROR:
            return CliError(
                ErrorCode.API_ERROR,
                f"Artificial Analysis had a server error (HTTP {status}). Try again later.",
                hint="Check the Artificial Analysis service status, then retry.",
            )
        case _:
            return CliError(
                ErrorCode.API_ERROR,
                f"Artificial Analysis returned an unexpected HTTP status {status}.",
                hint="Check the Artificial Analysis service status, then retry.",
            )


def _error_detail(response: httpx2.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    match body:
        case {"error": str(message)} if message.strip():
            return _one_line(message)
        case _:
            return None


def _one_line(text: str, limit: int = 200) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 3]}..."


def _endpoint(query: Query) -> str:
    match query.family:
        case Family.LLM:
            path = "/api/v2/language/models"
        case Family.MUSIC_INSTRUMENTAL:
            path = "/api/v2/media/music/instrumental/models"
        case Family.MUSIC_VOCAL:
            path = "/api/v2/media/music/with-vocals/models"
        case family:
            path = f"/api/v2/media/{family.value}/models"
    return f"{path}/free" if isinstance(query, FreeQuery) else path


_ELO = Metric("elo", "Elo")
_CI_95 = Metric("ci_95", "95% CI")
_SAMPLES = Metric("samples", "Samples")
_LLM_COLUMNS = (
    Metric("intelligence_index", "Intelligence"),
    Metric("input_price_per_1m_tokens", "Input $/1M"),
    Metric("output_price_per_1m_tokens", "Output $/1M"),
    Metric("output_tokens_per_second", "Output tok/s"),
)
_ARENA_FREE_COLUMNS = (_ELO, _CI_95)


def _collection(query: Query, pages: list[_FetchedPage], *, fetched_at: datetime, cached: bool) -> Collection:
    first = pages[0]
    columns, rows = _view(first.model, first.records)
    for page in pages[1:]:
        _, page_rows = _view(page.model, page.records)
        rows.extend(page_rows)
    return Collection(
        query=query,
        account_tier=first.model.tier,
        fetched_at=fetched_at,
        cached=cached,
        columns=columns,
        rows=tuple(rows),
        fields=first.fields,
    )


def _view(page: Page, records: list[dict[str, object]]) -> tuple[tuple[Metric, ...], list[ModelRow]]:
    """Display columns and rows from validated wire models."""
    values: list[tuple[Number | None, ...]]
    match page:
        case LLMModelsFreeResponse():
            columns = _LLM_COLUMNS
            values = [
                (
                    item.evaluations.artificial_analysis_intelligence_index,
                    item.pricing.price_1m_input_tokens,
                    item.pricing.price_1m_output_tokens,
                    item.performance.median_output_tokens_per_second,
                )
                for item in page.data
            ]
        case LLMModelsResponse():
            columns = _LLM_COLUMNS
            values = []
            for item in page.data:
                pricing = item.pricing
                performance = item.performance
                values.append(
                    (
                        item.evaluations.artificial_analysis_intelligence_index,
                        pricing.price_1m_input_tokens if isinstance(pricing, Pricing) else None,
                        pricing.price_1m_output_tokens if isinstance(pricing, Pricing) else None,
                        performance.median_output_tokens_per_second
                        if isinstance(performance, ModelPerformance)
                        else None,
                    )
                )
        case (
            ImageArenaFreeResponse() | VideoArenaFreeResponse() | TextToSpeechFreeResponse() | MusicArenaFreeResponse()
        ):
            columns = _ARENA_FREE_COLUMNS
            values = [(item.elo, item.ci_95) for item in page.data]
        case ImageArenaResponse():
            columns = (_ELO, _CI_95, _SAMPLES, Metric("price_per_1k_images", "$/1k images"))
            values = [(item.elo, item.ci_95, item.samples, item.price_per_1k_images) for item in page.data]
        case VideoArenaResponse():
            columns = (_ELO, _CI_95, _SAMPLES, Metric("price_per_minute", "$/min"))
            values = [(item.elo, item.ci_95, item.samples, item.price_per_minute) for item in page.data]
        case TextToSpeechResponse():
            columns = (_ELO, _CI_95, _SAMPLES, Metric("price_per_1m_characters", "$/1M chars"))
            values = [(item.elo, item.ci_95, item.samples, item.price_per_1m_characters) for item in page.data]
        case MusicArenaResponse():
            columns = (_ELO, _CI_95, _SAMPLES)
            values = [(item.elo, item.ci_95, item.samples) for item in page.data]
        case SpeechToSpeechFreeResponse() | SpeechToSpeechResponse():
            columns = (Metric("bba_score", "BBA"), Metric("fdb_score", "FDB"), Metric("tau_voice_score", "Tau-Voice"))
            values = [(item.bba_score, item.fdb_score, item.tau_voice_score) for item in page.data]
        case SpeechToTextModelsFreeResponse() | SpeechToTextModelsResponse():
            columns = (Metric("aa_wer_index", "AA-WER", higher_is_better=False),)
            values = [(item.aa_wer_index,) for item in page.data]
        case _:
            assert_never(page)
    return columns, [
        ModelRow(
            name=item.name,
            creator=None if item.model_creator is None else item.model_creator.name,
            metrics=metrics,
            record=record,
        )
        for item, record, metrics in zip(page.data, records, values, strict=True)
    ]


@dataclass(frozen=True)
class _CacheEntry:
    fetched_at: datetime
    bodies: list[str]
    """Successful response bodies in page order."""


def _cache_key(api_key: str, query: Query) -> str:
    """Name an entry after the credential, endpoint and conditions, keyed so the name does not reveal the credential."""
    conditions = {"family": query.family.name, "variant": query.variant.value, "options": options(query)}
    return hmac.new(api_key.encode(), json.dumps(conditions, sort_keys=True).encode(), hashlib.sha256).hexdigest()


def _read_cache(path: Path, now: datetime) -> _CacheEntry | None:
    """The entry when it is well-formed and younger than `CACHE_MAX_AGE`; its bodies still require validation."""
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    match document:
        case {"fetched_at": str(timestamp), "bodies": list(recorded)} if recorded:
            bodies = [body for body in recorded if isinstance(body, str)]
        case _:
            return None
    try:
        fetched_at = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if fetched_at.tzinfo is None or not timedelta(0) <= now - fetched_at < CACHE_MAX_AGE:
        return None
    if len(bodies) != len(recorded):
        return None
    return _CacheEntry(fetched_at, bodies)


def _write_cache(path: Path, entry: _CacheEntry) -> None:
    """Replace the entry atomically through a uniquely named file; a failure only costs a later request."""
    document = json.dumps({"fetched_at": entry.fetched_at.isoformat(), "bodies": entry.bodies})
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(document)
            Path(temporary).replace(path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError:
        return


def _schema_error(reason: str) -> CliError:
    return _invalid_response(f"Artificial Analysis returned data that does not match the expected schema: {reason}.")


def _invalid_response(message: str) -> CliError:
    return CliError(
        ErrorCode.INVALID_RESPONSE,
        message,
        hint="Update artificial-analysis-cli. If the problem persists, report the response shape.",
    )
