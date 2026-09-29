from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx2
import pytest

from artificial_analysis_cli.catalog import CategoryProQuery, Family, FreeQuery, Query
from artificial_analysis_cli.client import fetch_collection
from artificial_analysis_cli.errors import CliError
from artificial_analysis_cli.output import Collection
from tests.support import FakeApi, JsonObject, model, page, serving

LANGUAGE_MODELS = FreeQuery(Family.LLM)
TEXT_TO_VIDEO = FreeQuery(Family.T2V)
FETCHED_AT = datetime(2026, 9, 28, 3, 0, tzinfo=UTC)


def fetch(
    query: Query,
    api: httpx2.BaseTransport,
    cache_dir: Path,
    *,
    api_key: str = "test-key",
    at: datetime = FETCHED_AT,
    refresh: bool = False,
) -> Collection:
    return fetch_collection(query, api_key=api_key, cache_dir=cache_dir, now=at, refresh=refresh, transport=api)


def names(collection: Collection) -> list[str]:
    return [row.name for row in collection.rows]


def paged_api(pages: dict[int, JsonObject]) -> FakeApi:
    return FakeApi(lambda request: httpx2.Response(200, json=pages[int(request.url.params["page"])]))


def test_language_models_from_every_page_form_one_collection_without_pagination(tmp_path: Path) -> None:
    api = paged_api(
        {
            1: page(model("Model 1"), number=1, total_pages=2),
            2: page(model("Model 2"), number=2, total_pages=2),
        }
    )

    collection = fetch(LANGUAGE_MODELS, api, tmp_path)

    assert [row.name for row in collection.rows] == ["Model 1", "Model 2"]
    assert [request.url.params["page"] for request in api.requests] == ["1", "2"]
    assert collection.fields == {"tier": "free", "intelligence_index_version": 4.3}

    cached = fetch(LANGUAGE_MODELS, api, tmp_path)

    assert cached.cached
    assert (cached.rows, cached.fields) == (collection.rows, collection.fields)
    assert len(api.requests) == 2


@pytest.mark.parametrize(
    "second_page",
    [
        page(model("Model 2"), number=1, total_pages=2),
        page(model("Model 2"), number=2, total_pages=3),
        {
            **page(model("Model 2"), number=2, total_pages=2),
            "pagination": {"page": 2, "page_size": 200, "total_pages": 2, "has_more": True},
        },
    ],
    ids=["page-number-mismatch", "total-pages-changed", "has-more-past-last-page"],
)
def test_inconsistent_pages_fail_instead_of_returning_a_partial_collection(
    second_page: JsonObject, tmp_path: Path
) -> None:
    api = paged_api({1: page(model("Model 1"), number=1, total_pages=2), 2: second_page})

    with pytest.raises(CliError, match="inconsistent pagination"):
        fetch(LANGUAGE_MODELS, api, tmp_path)

    assert len(api.requests) == 2


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx2.Response(401, json={"error": "API key is required"}), "rejected the API key (HTTP 401)"),
        (httpx2.Response(401, text="<html>Unauthorized</html>"), "rejected the API key (HTTP 401)"),
        (
            httpx2.Response(403, json={"error": "Forbidden"}, headers={"X-AA-Tier": "free"}),
            "(HTTP 403; account tier: free)",
        ),
        (
            httpx2.Response(429, json={"error": "Rate limit exceeded"}, headers={"Retry-After": "120"}),
            "quota is exhausted (HTTP 429). Retry after 120 seconds.",
        ),
        (httpx2.Response(500, json={"error": "Internal Server Error"}), "server error (HTTP 500)"),
        (httpx2.Response(502, text="Bad gateway"), "server error (HTTP 502)"),
        (
            httpx2.Response(400, json={"error": "Invalid prompt_type"}),
            "rejected the request (HTTP 400): Invalid prompt_type",
        ),
        (httpx2.Response(404, json={"error": "Not found"}), "unexpected HTTP status 404"),
        (httpx2.Response(200, text="<html>Maintenance</html>"), "not valid JSON"),
        (httpx2.Response(200, json={"tier": "free"}), "does not match the expected schema: missing field 'data'"),
        (httpx2.Response(200, json=page(model("Model A", elo="high"))), "does not match the expected schema: elo"),
    ],
    ids=[
        "unauthorized",
        "unauthorized-html",
        "forbidden",
        "rate-limited",
        "server-error",
        "gateway-html",
        "bad-request",
        "undocumented-status",
        "not-json",
        "missing-field",
        "wrong-type",
    ],
)
def test_failed_responses_are_distinguished_without_revealing_the_api_key(
    response: httpx2.Response, expected: str, tmp_path: Path
) -> None:
    api = FakeApi(lambda _request: response)

    with pytest.raises(CliError) as failure:
        fetch(TEXT_TO_VIDEO, api, tmp_path, api_key="secret-key")

    assert expected in str(failure.value)
    assert "secret-key" not in str(failure.value)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx2.ReadTimeout("read timed out"), "Timed out after 30 seconds"),
        (httpx2.ConnectError("name resolution failed"), "Could not connect to Artificial Analysis"),
    ],
)
def test_transport_failures_are_reported(error: httpx2.HTTPError, expected: str, tmp_path: Path) -> None:
    def fail(_request: httpx2.Request) -> httpx2.Response:
        raise error

    with pytest.raises(CliError, match=expected):
        fetch(TEXT_TO_VIDEO, FakeApi(fail), tmp_path, api_key="secret-key")


def test_a_collection_fetched_within_the_hour_is_reused_from_cache(tmp_path: Path) -> None:
    api = serving(page(model("Model A")))

    fresh = fetch(TEXT_TO_VIDEO, api, tmp_path)
    cached = fetch(TEXT_TO_VIDEO, api, tmp_path, at=FETCHED_AT + timedelta(minutes=59))

    assert len(api.requests) == 1
    assert (fresh.cached, cached.cached) == (False, True)
    assert cached.fetched_at == FETCHED_AT
    assert cached.rows == fresh.rows


@pytest.mark.parametrize(
    ("api_key", "query"),
    [
        ("other-key", CategoryProQuery(Family.T2V)),
        ("test-key", FreeQuery(Family.T2V)),
        ("test-key", CategoryProQuery(Family.I2V)),
        ("test-key", CategoryProQuery(Family.T2V, include_categories=True)),
    ],
    ids=["api-key", "endpoint-variant", "family", "options"],
)
def test_cached_collections_are_not_shared_across_api_keys_or_conditions(
    api_key: str, query: Query, tmp_path: Path
) -> None:
    api = serving(page(model("Model A"), tier="pro"))
    fetch(CategoryProQuery(Family.T2V), api, tmp_path)

    other = fetch(query, api, tmp_path, api_key=api_key)

    assert len(api.requests) == 2
    assert not other.cached


def test_refresh_fetches_again_and_replaces_the_cached_collection(tmp_path: Path) -> None:
    bodies = iter([page(model("Old")), page(model("New"))])
    api = FakeApi(lambda _request: httpx2.Response(200, json=next(bodies)))
    fetch(TEXT_TO_VIDEO, api, tmp_path)

    refreshed = fetch(TEXT_TO_VIDEO, api, tmp_path, at=FETCHED_AT + timedelta(minutes=1), refresh=True)
    cached = fetch(TEXT_TO_VIDEO, api, tmp_path, at=FETCHED_AT + timedelta(minutes=2))

    assert (names(refreshed), names(cached), cached.cached) == (["New"], ["New"], True)
    assert len(api.requests) == 2


@pytest.mark.parametrize("age", [timedelta(hours=1), timedelta(seconds=-1)], ids=["expired", "from-the-future"])
def test_a_cached_collection_outside_the_hour_is_fetched_again(age: timedelta, tmp_path: Path) -> None:
    api = serving(page(model("Model A")))
    fetch(TEXT_TO_VIDEO, api, tmp_path)

    again = fetch(TEXT_TO_VIDEO, api, tmp_path, at=FETCHED_AT + age)

    assert len(api.requests) == 2
    assert not again.cached


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda original: b"{",
        lambda original: original[: len(original) // 2],
        lambda original: b'{"data": [], "cached": true}',
        lambda original: original.replace(b"1234.5", b"true"),
        lambda original: original.replace(b"1234.5", b"NaN"),
    ],
    ids=["not-json", "truncated", "unvalidated-shape", "schema-invalid-body", "nonfinite-body"],
)
def test_a_corrupt_cache_entry_is_fetched_again(corrupt: Callable[[bytes], bytes], tmp_path: Path) -> None:
    api = serving(page(model("Model A", elo=1234.5)))
    fetch(TEXT_TO_VIDEO, api, tmp_path)
    for entry in tmp_path.rglob("*"):
        if entry.is_file():
            entry.write_bytes(corrupt(entry.read_bytes()))

    again = fetch(TEXT_TO_VIDEO, api, tmp_path, at=FETCHED_AT + timedelta(minutes=1))

    assert len(api.requests) == 2
    assert (names(again), again.cached) == (["Model A"], False)


@pytest.mark.parametrize(
    "bodies",
    [
        [{**page(model("A")), "pagination": {"page": 1, "page_size": 200, "total_pages": "1", "has_more": False}}],
        [page(model("A", model_creator={"id": "creator"}))],
        [{**page(), "data": {}}],
        [page(model("A"), total_pages=2)],
        [page(model("A")), page(model("B"))],
        [page(model("A"), total_pages=2), page(model("B"), tier="pro", number=2, total_pages=2)],
        [
            page(model("A"), total_pages=2),
            {**page(model("B"), number=2, total_pages=2), "intelligence_index_version": 5.0},
        ],
    ],
    ids=["pagination", "creator", "data", "missing-page", "extra-page", "tier-changed", "index-version-changed"],
)
def test_malformed_cached_responses_are_replaced_by_a_fresh_collection(
    bodies: list[JsonObject], tmp_path: Path
) -> None:
    fetch(LANGUAGE_MODELS, serving(page(model("Old"))), tmp_path)
    for entry in tmp_path.glob("*.json"):
        document = json.loads(entry.read_text())
        document["bodies"] = [json.dumps(body) for body in bodies]
        entry.write_text(json.dumps(document))
    api = serving(page(model("Recovered")))

    recovered = fetch(LANGUAGE_MODELS, api, tmp_path)
    cached = fetch(LANGUAGE_MODELS, api, tmp_path, at=FETCHED_AT + timedelta(minutes=1))

    assert (names(recovered), recovered.cached) == (["Recovered"], False)
    assert (names(cached), cached.cached) == (["Recovered"], True)
    assert len(api.requests) == 1


@pytest.mark.parametrize("query", [LANGUAGE_MODELS, TEXT_TO_VIDEO], ids=["llm", "arena"])
def test_empty_collections_are_valid_and_cached(query: Query, tmp_path: Path) -> None:
    api = serving(page())

    fresh = fetch(query, api, tmp_path)
    cached = fetch(query, api, tmp_path)

    assert fresh.rows == cached.rows == ()
    assert (fresh.cached, cached.cached, len(api.requests)) == (False, True, 1)


def test_a_language_model_without_a_known_creator_is_valid(tmp_path: Path) -> None:
    collection = fetch(LANGUAGE_MODELS, serving(page(model("A", model_creator=None))), tmp_path)

    assert collection.rows[0].creator is None
    assert collection.rows[0].record["model_creator"] is None


@pytest.mark.parametrize("field", ["release_date", "model_creator"])
def test_required_nullable_fields_must_be_present(field: str, tmp_path: Path) -> None:
    record = model("A")
    del record[field]
    with pytest.raises(CliError, match="expected schema"):
        fetch(LANGUAGE_MODELS, serving(page(record)), tmp_path)


def test_optional_categories_cannot_be_explicitly_null(tmp_path: Path) -> None:
    with pytest.raises(CliError, match="expected schema"):
        fetch(CategoryProQuery(Family.T2I), serving(page(model("A", categories=None), tier="pro")), tmp_path)


def test_a_failed_fetch_leaves_nothing_in_the_cache(tmp_path: Path) -> None:
    first_page = page(model("Model 1"), number=1, total_pages=2)
    failing = FakeApi(
        lambda request: (
            httpx2.Response(200, json=first_page)
            if request.url.params["page"] == "1"
            else httpx2.Response(500, json={"error": "Internal Server Error"})
        )
    )
    healthy = paged_api({1: first_page, 2: page(model("Model 2"), number=2, total_pages=2)})

    with pytest.raises(CliError):
        fetch(LANGUAGE_MODELS, failing, tmp_path)
    recovered = fetch(LANGUAGE_MODELS, healthy, tmp_path, at=FETCHED_AT + timedelta(minutes=1))

    assert (names(recovered), recovered.cached, len(healthy.requests)) == (["Model 1", "Model 2"], False, 2)


def test_an_unwritable_cache_does_not_fail_a_successful_fetch(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.write_text("a file where the cache directory belongs")

    collection = fetch(TEXT_TO_VIDEO, serving(page(model("Model A"))), cache_dir)

    assert names(collection) == ["Model A"]


def test_the_cache_does_not_store_the_api_key(tmp_path: Path) -> None:
    fetch(TEXT_TO_VIDEO, serving(page(model("Model A"))), tmp_path, api_key="secret-key-123")

    entries = [entry for entry in tmp_path.rglob("*") if entry.is_file()]
    assert entries
    for entry in entries:
        assert "secret-key-123" not in str(entry)
        assert b"secret-key-123" not in entry.read_bytes()


def test_the_http_client_is_closed_after_success_and_failure(tmp_path: Path) -> None:
    succeeding = serving(page(model("Model A")))
    failing = FakeApi(lambda _request: httpx2.Response(500, json={"error": "Internal Server Error"}))

    fetch(TEXT_TO_VIDEO, succeeding, tmp_path)
    with pytest.raises(CliError):
        fetch(TEXT_TO_VIDEO, failing, tmp_path, refresh=True)

    assert (succeeding.closed, failing.closed) == (True, True)
