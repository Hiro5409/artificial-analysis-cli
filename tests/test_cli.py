from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx2
import pytest

from artificial_analysis_cli.cli import main
from tests.support import FakeApi, JsonObject, environment, model, page, serving

FETCHED_AT = datetime(2026, 9, 28, 3, 0, tzinfo=UTC)


class _ClosedPipe:
    def __init__(self, file_descriptor: int) -> None:
        self.file_descriptor = file_descriptor

    def write(self, _value: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        return self.file_descriptor


def test_version_reports_the_installed_distribution_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--version"])

    captured = capsys.readouterr()
    assert (exit_code, captured.out, captured.err) == (0, f"{version('artificial-analysis-cli')}\n", "")


def test_families_lists_every_alias_with_its_description(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["families"])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "t2v\ttext-to-video",
        "i2v\timage-to-video",
        "t2v-audio\ttext-to-video-audio",
        "i2v-audio\timage-to-video-audio",
        "t2i\ttext-to-image",
        "image-editing\timage-editing",
        "llm\tlanguage-models",
        "tts\ttext-to-speech",
        "sts\tspeech-to-speech",
        "stt\tspeech-to-text",
        "music-instrumental\tmusic-instrumental",
        "music-vocal\tmusic-with-vocals",
    ]


def test_closed_output_pipe_exits_without_a_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with (tmp_path / "closed-pipe").open("w") as sink:
        monkeypatch.setattr(sys, "stdout", _ClosedPipe(sink.fileno()))

        exit_code = main(["families"])

    assert exit_code == 1


@pytest.mark.parametrize(
    ("alias", "path", "query"),
    [
        ("t2v", "/api/v2/media/text-to-video/models", ""),
        ("i2v", "/api/v2/media/image-to-video/models", ""),
        ("t2v-audio", "/api/v2/media/text-to-video-audio/models", ""),
        ("i2v-audio", "/api/v2/media/image-to-video-audio/models", ""),
        ("t2i", "/api/v2/media/text-to-image/models", ""),
        ("image-editing", "/api/v2/media/image-editing/models", ""),
        ("llm", "/api/v2/language/models", "page=1"),
        ("tts", "/api/v2/media/text-to-speech/models", ""),
        ("sts", "/api/v2/media/speech-to-speech/models", ""),
        ("stt", "/api/v2/media/speech-to-text/models", ""),
        ("music-instrumental", "/api/v2/media/music/instrumental/models", ""),
        ("music-vocal", "/api/v2/media/music/with-vocals/models", ""),
    ],
)
@pytest.mark.parametrize(
    ("variant_flags", "path_suffix", "tier"), [((), "/free", "free"), (("--pro",), "", "pro")], ids=["free", "pro"]
)
def test_each_family_alias_fetches_its_endpoint_variant_and_shows_the_models(
    alias: str,
    path: str,
    query: str,
    variant_flags: tuple[str, ...],
    path_suffix: str,
    tier: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = serving(page(model("Model A"), tier=tier))

    exit_code = main(["list", alias, *variant_flags], environ=environment(tmp_path), transport=api)

    assert exit_code == 0
    assert [(request.url.path, request.url.query.decode()) for request in api.requests] == [(path + path_suffix, query)]
    assert api.requests[0].headers["x-api-key"] == "test-key"
    assert "Model A" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "path", "query"),
    [
        (["llm", "--pro", "--prompt-type", "100k"], "/api/v2/language/models", "prompt_type=100k&page=1"),
        (
            ["t2i", "--pro", "--categories", "--format", "json"],
            "/api/v2/media/text-to-image/models",
            "include_categories=true",
        ),
        (
            ["music-vocal", "--pro", "--genres", "--format", "json"],
            "/api/v2/media/music/with-vocals/models",
            "include_genres=true",
        ),
    ],
)
def test_family_options_are_sent_to_their_pro_endpoint(
    arguments: list[str], path: str, query: str, tmp_path: Path
) -> None:
    api = serving(page(model("Model A"), tier="pro"))

    exit_code = main(["list", *arguments], environ=environment(tmp_path), transport=api)

    assert exit_code == 0
    assert [(request.url.path, request.url.query.decode()) for request in api.requests] == [(path, query)]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["t2-v"], '"t2-v"'),
        (["llm", "--pro", "--prompt-type", "short"], '"short"'),
        (["t2v", "--pro", "--prompt-type", "long"], "--prompt-type is only available for llm."),
        (["llm", "--prompt-type", "long"], "--prompt-type requires --pro."),
        (
            ["tts", "--pro", "--categories", "--format", "json"],
            "--categories is only available for t2v, i2v, t2v-audio, i2v-audio, t2i, image-editing.",
        ),
        (["t2i", "--categories", "--format", "json"], "--categories requires --pro."),
        (["t2i", "--pro", "--categories"], "--categories requires --format json."),
        (
            ["llm", "--pro", "--genres", "--format", "json"],
            "--genres is only available for music-instrumental, music-vocal.",
        ),
        (["music-vocal", "--genres", "--format", "json"], "--genres requires --pro."),
        (["music-vocal", "--pro", "--genres", "--format", "tsv"], "--genres requires --format json."),
    ],
)
def test_invalid_arguments_exit_with_usage_error_before_any_request(
    arguments: list[str], message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = serving(page(model("Model A"), tier="pro"))

    exit_code = main(["list", *arguments], environ=environment(tmp_path), transport=api)

    captured = capsys.readouterr()
    assert (exit_code, captured.out, api.requests) == (2, "", [])
    assert message in captured.err


def test_argument_parser_uses_the_same_json_error_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list", "t2-v", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert (exit_code, captured.out) == (2, "")
    error = payload["error"]
    assert set(error) == {"code", "message", "exitCode", "hint"}
    assert (error["code"], error["exitCode"], error["hint"]) == (
        "INVALID_ARGUMENT",
        2,
        "Run artificial-analysis --help, then retry.",
    )
    assert '"t2-v"' in error["message"]


@pytest.mark.parametrize("arguments", [["wat"], ["families", "--bad"]], ids=["root", "families"])
def test_parser_errors_recommend_root_help(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert (exit_code, captured.out) == (2, "")
    assert captured.err.endswith("Hint: Run artificial-analysis --help, then retry.\n")


def test_missing_api_key_is_a_structured_error_when_json_output_was_requested(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = serving(page(model("Model A")))

    exit_code = main(
        ["list", "t2v", "--format", "json"],
        environ={"XDG_CACHE_HOME": str(tmp_path)},
        transport=api,
    )

    captured = capsys.readouterr()
    assert (exit_code, captured.out, api.requests) == (2, "", [])
    assert json.loads(captured.err) == {
        "error": {
            "code": "AUTH_REQUIRED",
            "message": "ARTIFICIAL_ANALYSIS_API_KEY is not set.",
            "exitCode": 2,
            "hint": "Set ARTIFICIAL_ANALYSIS_API_KEY, then retry.",
        }
    }


@pytest.mark.parametrize("api_key", ["secret\n", " secret", "secrét"])
def test_invalid_api_key_is_rejected_without_exposing_it(
    api_key: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = serving(page(model("Model A")))

    exit_code = main(
        ["list", "t2v", "--format", "json"],
        environ={"ARTIFICIAL_ANALYSIS_API_KEY": api_key, "XDG_CACHE_HOME": str(tmp_path)},
        transport=api,
    )

    captured = capsys.readouterr()
    assert (exit_code, captured.out, api.requests) == (2, "", [])
    error = json.loads(captured.err)["error"]
    assert (error["code"], error["exitCode"], error["hint"]) == (
        "AUTH_REQUIRED",
        2,
        "Set ARTIFICIAL_ANALYSIS_API_KEY to the API key without surrounding whitespace.",
    )
    assert error["message"] == "ARTIFICIAL_ANALYSIS_API_KEY is not a valid HTTP header value."
    assert api_key not in captured.err


def test_rejected_api_key_has_an_actionable_text_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeApi(lambda _request: httpx2.Response(401, json={"error": "API key is required"}))

    exit_code = main(["list", "t2v"], environ=environment(tmp_path), transport=api)

    captured = capsys.readouterr()
    assert (exit_code, captured.out) == (2, "")
    assert captured.err == (
        "UNAUTHORIZED: Artificial Analysis rejected the API key (HTTP 401).\n"
        "Hint: Set ARTIFICIAL_ANALYSIS_API_KEY to a valid API key, then retry.\n"
    )


@pytest.mark.parametrize(
    ("failure", "arguments", "code", "expected_exit_code", "hint"),
    [
        (
            httpx2.Response(403, json={"error": "Forbidden"}, headers={"X-AA-Tier": "free"}),
            ["t2v", "--pro"],
            "FORBIDDEN",
            3,
            "Use a Pro or Commercial API key, or omit --pro.",
        ),
        (
            httpx2.Response(429, json={"error": "Rate limit exceeded"}, headers={"Retry-After": "120"}),
            ["t2v"],
            "RATE_LIMITED",
            4,
            "Wait before retrying the same command.",
        ),
        (
            httpx2.Response(500, json={"error": "Internal Server Error"}),
            ["t2v"],
            "API_ERROR",
            1,
            "Check the Artificial Analysis service status, then retry.",
        ),
        (
            httpx2.ReadTimeout("read timed out"),
            ["t2v"],
            "TIMEOUT",
            1,
            "Check the network, then retry the same command.",
        ),
        (
            httpx2.ConnectError("name resolution failed"),
            ["t2v"],
            "NETWORK_ERROR",
            1,
            "Check the network and Artificial Analysis service status, then retry.",
        ),
        (
            httpx2.Response(200, text="<html>Maintenance</html>"),
            ["t2v"],
            "INVALID_RESPONSE",
            1,
            "Update artificial-analysis-cli. If the problem persists, report the response shape.",
        ),
    ],
    ids=["forbidden", "rate-limited", "server-error", "timeout", "network", "invalid-response"],
)
def test_failures_have_stable_json_codes_and_remedies(
    failure: httpx2.HTTPError | httpx2.Response,
    arguments: list[str],
    code: str,
    expected_exit_code: int,
    hint: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_request: httpx2.Request) -> httpx2.Response:
        if isinstance(failure, httpx2.HTTPError):
            raise failure
        return failure

    exit_code = main(
        ["list", *arguments, "--format", "json"],
        environ=environment(tmp_path),
        transport=FakeApi(fail),
    )

    captured = capsys.readouterr()
    error = json.loads(captured.err)["error"]
    assert (exit_code, captured.out) == (error["exitCode"], "")
    assert (error["code"], error["exitCode"], error["hint"]) == (code, expected_exit_code, hint)
    assert error["message"]


@pytest.mark.parametrize(
    "body",
    [
        {**page(model("A")), "pagination": {"page": 1, "page_size": 200, "total_pages": "1", "has_more": False}},
        {**page(model("A")), "pagination": {"page": True, "page_size": 200, "total_pages": 1, "has_more": 0}},
        page(model("A", model_creator={"id": "creator"})),
        page(model("A", model_creator="Lab")),
        {**page(), "data": {}},
        {**page(), "data": ""},
    ],
    ids=["pagination-type", "pagination-bool-int", "incomplete-creator", "creator-type", "data-object", "data-string"],
)
@pytest.mark.parametrize("pro", [False, True], ids=["free", "pro"])
def test_malformed_language_models_exit_cleanly_without_output(
    body: JsonObject, pro: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = serving({**body, "tier": "pro" if pro else "free"})

    exit_code = main(["list", "llm", *(["--pro"] if pro else [])], environ=environment(tmp_path), transport=api)

    captured = capsys.readouterr()
    assert (exit_code, captured.out) == (1, "")
    assert "does not match the expected schema" in captured.err


def test_table_ranks_models_and_credits_the_source_apart_from_endpoint_variant_and_account_tier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = serving(
        page(
            model("Model B", elo=1100, ci_95=12.5, model_creator={"id": "b", "name": "Lab B"}),
            model("Model A", elo=1250, ci_95=None, model_creator={"id": "a", "name": "Lab A"}),
            tier="pro",
        )
    )

    exit_code = main(["list", "t2v"], environ=environment(tmp_path), transport=api, now=lambda: FETCHED_AT)

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "#  Creator  Model     Elo  95% CI",
        "-  -------  -------  ----  ------",
        "1  Lab A    Model A  1250       -",
        "2  Lab B    Model B  1100    12.5",
        "",
        "Source: Artificial Analysis (https://artificialanalysis.ai/)",
        "Endpoint variant: free; account tier: pro; fetched 2026-09-28T03:00:00+00:00 from the API",
    ]


def test_tsv_keeps_one_escaped_record_per_line_with_missing_scores_last(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = serving(
        page(
            model("Unscored", aa_wer_index=None),
            model("Tab\tName", aa_wer_index=0.2, model_creator={"id": "x", "name": "Lab\nX"}),
            model("Best", aa_wer_index=0.1),
        )
    )

    exit_code = main(["list", "stt", "--format", "tsv"], environ=environment(tmp_path), transport=api)

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "position\tcreator\tmodel\taa_wer_index\tsource",
        "1\tLab\tBest\t0.1\thttps://artificialanalysis.ai/",
        "2\tLab\\nX\tTab\\tName\t0.2\thttps://artificialanalysis.ai/",
        "3\tLab\tUnscored\t\thttps://artificialanalysis.ai/",
    ]


@pytest.mark.parametrize("output_format", ["table", "tsv", "json"])
def test_output_cannot_execute_terminal_controls_and_json_preserves_the_values(
    output_format: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    name = "日本語\\model\x1b[2J\nforged-row\x9b31m\u202e"
    creator = "Lab\r\t\x07\u2028Name"
    api = serving(page(model(name, model_creator={"id": "lab", "name": creator})))

    exit_code = main(["list", "t2v", "--format", output_format], environ=environment(tmp_path), transport=api)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert not any(control in output for control in ("\x1b", "\x9b", "\u202e", "\r", "\x07", "\u2028"))
    if output_format == "json":
        record = json.loads(output)["collection"]["data"][0]
        assert (record["name"], record["model_creator"]["name"]) == (name, creator)
    else:
        assert "日本語" in output
        assert r"\x1b[2J\nforged-row" in output
        assert len(output.splitlines()) == (6 if output_format == "table" else 2)


def test_api_diagnostics_cannot_execute_terminal_controls(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeApi(lambda _request: httpx2.Response(400, json={"error": "Invalid \x1b[2Jpreset"}))

    exit_code = main(["list", "t2v"], environ=environment(tmp_path), transport=api)

    captured = capsys.readouterr()
    assert (exit_code, captured.out) == (1, "")
    assert "\x1b" not in captured.err
    assert r"Invalid \x1b[2Jpreset" in captured.err


def test_json_keeps_the_api_collection_whole_under_cli_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = model("Model 1", evaluations={**evaluations(60)}, unknown_model_field="kept")
    second = model("Model 2", evaluations={**evaluations(70)})
    pages: dict[str, JsonObject] = {
        "1": {**page(first, tier="pro", number=1, total_pages=2), "unknown_collection_field": "kept"},
        "2": page(second, tier="pro", number=2, total_pages=2),
    }
    api = FakeApi(lambda request: httpx2.Response(200, json=pages[request.url.params["page"]]))

    exit_code = main(
        ["list", "llm", "--pro", "--prompt-type", "medium", "--format", "json", "--top", "1"],
        environ=environment(tmp_path),
        transport=api,
        now=lambda: FETCHED_AT,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "source": {"name": "Artificial Analysis", "url": "https://artificialanalysis.ai/"},
        "family": "llm",
        "endpoint_variant": "pro",
        "options": {"prompt_type": "medium"},
        "fetched_at": "2026-09-28T03:00:00+00:00",
        "cached": False,
        "collection": {
            "tier": "pro",
            "intelligence_index_version": 4.3,
            "unknown_collection_field": "kept",
            "data": [second],
        },
    }


def test_json_preserves_integer_values_and_missing_fields_from_http_and_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [model("A", elo=9007199254740993), model("B", elo=1001)]
    body = {**page(*records), "unknown": {"nested": [9007199254740993, None, False]}}
    api = serving(body)

    for cached in (False, True):
        code = main(["list", "t2v", "--format", "json"], environ=environment(tmp_path), transport=api)
        captured = capsys.readouterr()
        result = json.loads(captured.out)

        assert (code, captured.err, result["cached"]) == (0, "", cached)
        assert result["collection"] == {key: value for key, value in body.items() if key != "pagination"}
        assert type(result["collection"]["data"][0]["elo"]) is int
        assert "categories" not in result["collection"]["data"][0]
    assert len(api.requests) == 1


def evaluations(intelligence: float) -> JsonObject:
    return {
        "artificial_analysis_intelligence_index": intelligence,
        "artificial_analysis_coding_index": None,
        "artificial_analysis_agentic_index": None,
    }


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
@pytest.mark.parametrize("location", ["metric", "unknown-field"])
def test_nonfinite_responses_fail_without_output_or_cache(
    token: str, location: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = model("A", elo="NUMBER") if location == "metric" else model("A", extra={"nested": ["NUMBER"]})
    body = json.dumps(page(record)).replace('"NUMBER"', token)
    api = FakeApi(lambda _: httpx2.Response(200, text=body))
    code = main(["list", "t2v", "--format", "json"], environ=environment(tmp_path), transport=api)
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    assert "not valid JSON" in captured.err
    assert list(tmp_path.rglob("*.json")) == []


@pytest.mark.parametrize(
    ("field", "changed"),
    [("tier", "pro"), ("intelligence_index_version", 5.0)],
)
def test_disagreeing_pages_fail_without_output_further_requests_or_cached_partial_results(
    field: str, changed: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bodies = {
        1: page(model("A"), number=1, total_pages=3),
        2: {**page(model("B"), number=2, total_pages=3), field: changed},
        3: page(model("C"), number=3, total_pages=3),
    }
    api = FakeApi(lambda request: httpx2.Response(200, json=bodies[int(request.url.params["page"])]))
    arguments = ["list", "llm", "--format", "json"]

    code = main(arguments, environ=environment(tmp_path), transport=api)
    failed = capsys.readouterr()

    assert (code, failed.out) == (1, "")
    assert "inconsistent collection metadata" in failed.err
    assert len(api.requests) == 2

    healthy = serving(page(model("Recovered")))
    assert main(arguments, environ=environment(tmp_path), transport=healthy) == 0
    recovered = json.loads(capsys.readouterr().out)
    assert recovered["cached"] is False
    assert recovered["collection"]["data"][0]["name"] == "Recovered"
    assert len(healthy.requests) == 1


def test_list_help_describes_the_family_argument_and_options(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list", "--help"])

    help_text = capsys.readouterr().out
    assert exit_code == 0
    assert "Usage: artificial-analysis list" in help_text
    for fragment in ["t2v-audio", "image-editing", "music-vocal", "--pro", "--prompt-type", "--categories", "--genres"]:
        assert fragment in help_text
