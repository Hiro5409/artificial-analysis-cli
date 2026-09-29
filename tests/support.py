"""Test doubles for the Artificial Analysis Data API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx2

JsonObject = dict[str, object]


def model(name: str, **fields: object) -> JsonObject:
    """A model record carrying every field that some list endpoint requires, with `fields` overriding them."""
    record: JsonObject = {
        "id": f"id-{name}",
        "name": name,
        "slug": name.lower().replace(" ", "-"),
        "release_date": "2026-01-01",
        "model_creator": {"id": "creator-id", "name": "Lab"},
        "elo": 1000,
        "ci_95": 10,
        "rank": 1,
        "samples": 500,
        "price_per_1k_images": 40,
        "price_per_minute": 3,
        "price_per_1m_characters": 15,
        "open_weights_url": None,
        "bba_score": 0.5,
        "fdb_score": 0.4,
        "tau_voice_score": 0.3,
        "aa_wer_index": 0.1,
        "aa_agenttalk": 0.2,
        "voxpopuli_cleaned_aa": 0.1,
        "earnings_22_cleaned_aa": 0.1,
        "open_weights": False,
        "providers": [],
        "evaluations": {
            "artificial_analysis_intelligence_index": 50,
            "artificial_analysis_coding_index": 40,
            "artificial_analysis_agentic_index": 30,
        },
        "artificial_analysis_intelligence_index_cost": None,
        "pricing": {
            "price_1m_blended_3_to_1": 1.5,
            "price_1m_blended_7_to_2_to_1": 1.2,
            "price_1m_input_tokens": 1,
            "price_1m_output_tokens": 3,
            "price_1m_cache_hit_tokens": None,
            "price_1m_cache_write_tokens": None,
        },
        "performance": {
            "percentile_05_output_tokens_per_second": None,
            "quartile_25_output_tokens_per_second": None,
            "median_output_tokens_per_second": 100,
            "quartile_75_output_tokens_per_second": None,
            "percentile_95_output_tokens_per_second": None,
            "percentile_05_time_to_first_token_seconds": None,
            "quartile_25_time_to_first_token_seconds": None,
            "median_time_to_first_token_seconds": 0.5,
            "quartile_75_time_to_first_token_seconds": None,
            "percentile_95_time_to_first_token_seconds": None,
            "median_time_to_first_answer_token_seconds": 1,
            "median_end_to_end_response_time_seconds": 5,
        },
    }
    record.update(fields)
    return record


def page(*models: JsonObject, tier: str = "free", number: int = 1, total_pages: int = 1) -> JsonObject:
    """A list response body; pagination fields are ignored by the endpoints that do not paginate."""
    return {
        "tier": tier,
        "intelligence_index_version": 4.3,
        "pagination": {
            "page": number,
            "page_size": 200,
            "total_pages": total_pages,
            "has_more": number < total_pages,
        },
        "data": list(models),
    }


class FakeApi(httpx2.MockTransport):
    """Answers requests with `respond` and records them, and whether the transport was closed."""

    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.requests: list[httpx2.Request] = []
        self.closed = False

        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return respond(request)

        super().__init__(handle)

    def close(self) -> None:
        self.closed = True


def serving(body: JsonObject) -> FakeApi:
    """An API that answers every request with `body`."""
    return FakeApi(lambda _request: httpx2.Response(200, json=body))


def environment(cache_home: Path, api_key: str = "test-key") -> dict[str, str]:
    return {"ARTIFICIAL_ANALYSIS_API_KEY": api_key, "XDG_CACHE_HOME": str(cache_home)}
