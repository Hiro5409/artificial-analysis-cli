"""Command-line entry point: arguments, environment, exit codes, stdout and stderr."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, assert_never, get_args

import httpx2
from cyclopts import App, CycloptsError, Parameter
from cyclopts.types import PositiveInt

from artificial_analysis_cli.catalog import (
    CategoryFamily,
    CategoryProQuery,
    Family,
    FreeQuery,
    GenreFamily,
    GenreProQuery,
    LanguageModelsProQuery,
    PromptType,
    Query,
    SpeechProQuery,
    cli_name,
)
from artificial_analysis_cli.client import fetch_collection
from artificial_analysis_cli.errors import CliError, ErrorCode
from artificial_analysis_cli.output import OutputFormat, render, terminal_text

API_KEY_VARIABLE = "ARTIFICIAL_ANALYSIS_API_KEY"

app = App(
    name="artificial-analysis",
    help="Query official Artificial Analysis model data.",
    default_parameter=Parameter(negative=""),
    version=version("artificial-analysis-cli"),
    exit_on_error=False,
)


@dataclass(frozen=True)
class _Runtime:
    environ: Mapping[str, str]
    transport: httpx2.BaseTransport | None
    now: Callable[[], datetime]


@app.command
def families() -> None:
    """List the supported model families."""
    for family in Family:
        print(f"{family.alias}\t{family.value}")


@app.command(name="list")
def list_models(
    family: Annotated[Family, Parameter(name_transform=cli_name)],
    /,
    *,
    pro: bool = False,
    prompt_type: PromptType | None = None,
    categories: bool = False,
    genres: bool = False,
    output_format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.TABLE,
    top: PositiveInt | None = None,
    refresh: bool = False,
    runtime: Annotated[_Runtime, Parameter(parse=False)],
) -> None:
    """Fetch a model collection and print it, ranked best first.

    Parameters
    ----------
    family
        Model family alias, as listed by `artificial-analysis families`.
    pro
        Use the Pro endpoint variant. It needs a Pro or Commercial API key.
    prompt_type
        LLM performance benchmark preset. Needs --pro.
    categories
        Include image and video category breakdowns. Needs --pro and --format json.
    genres
        Include music genre breakdowns. Needs --pro and --format json.
    output_format
        Output format.
    top
        Show only the N best-ranked models.
    refresh
        Ignore cached data and spend API quota on a new request.
    """
    query = _query(
        family, pro=pro, prompt_type=prompt_type, categories=categories, genres=genres, output_format=output_format
    )
    api_key = runtime.environ.get(API_KEY_VARIABLE)
    if not api_key:
        raise CliError(
            ErrorCode.AUTH_REQUIRED,
            f"{API_KEY_VARIABLE} is not set.",
            hint=f"Set {API_KEY_VARIABLE}, then retry.",
        )
    if api_key != api_key.strip() or not api_key.isascii() or not api_key.isprintable():
        raise CliError(
            ErrorCode.AUTH_REQUIRED,
            f"{API_KEY_VARIABLE} is not a valid HTTP header value.",
            hint=f"Set {API_KEY_VARIABLE} to the API key without surrounding whitespace.",
        )
    collection = fetch_collection(
        query,
        api_key=api_key,
        cache_dir=_cache_dir(runtime.environ),
        now=runtime.now(),
        refresh=refresh,
        transport=runtime.transport,
    )
    print(render(collection, output_format, top))


def _query(
    family: Family,
    *,
    pro: bool,
    prompt_type: PromptType | None,
    categories: bool,
    genres: bool,
    output_format: OutputFormat,
) -> Query:
    """Turn the arguments into a query, rejecting options that the family's selected endpoint does not accept."""
    category_families: tuple[Family, ...] = get_args(CategoryFamily)
    genre_families: tuple[Family, ...] = get_args(GenreFamily)
    for flag, used, families, needs_json in (
        ("--prompt-type", prompt_type is not None, (Family.LLM,), False),
        ("--categories", categories, category_families, True),
        ("--genres", genres, genre_families, True),
    ):
        if not used:
            continue
        if family not in families:
            raise _invalid_argument(f"{flag} is only available for {', '.join(member.alias for member in families)}.")
        if not pro:
            raise _invalid_argument(f"{flag} requires --pro.")
        if needs_json and output_format is not OutputFormat.JSON:
            raise _invalid_argument(f"{flag} requires --format json.")

    if not pro:
        return FreeQuery(family)
    match family:
        case Family.LLM:
            return LanguageModelsProQuery(prompt_type)
        case Family.T2V | Family.I2V | Family.T2V_AUDIO | Family.I2V_AUDIO | Family.T2I | Family.IMAGE_EDITING:
            return CategoryProQuery(family, include_categories=categories)
        case Family.MUSIC_INSTRUMENTAL | Family.MUSIC_VOCAL:
            return GenreProQuery(family, include_genres=genres)
        case Family.TTS | Family.STS | Family.STT:
            return SpeechProQuery(family)
        case _:
            assert_never(family)


def _cache_dir(environ: Mapping[str, str]) -> Path:
    cache_home = environ.get("XDG_CACHE_HOME")
    return (Path(cache_home) if cache_home else Path.home() / ".cache") / "artificial-analysis-cli"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _invalid_argument(message: str) -> CliError:
    return CliError(
        ErrorCode.INVALID_ARGUMENT,
        message,
        hint="Run artificial-analysis list --help, then retry.",
    )


def _json_output_requested(tokens: Sequence[str]) -> bool:
    requested = False
    for index, token in enumerate(tokens):
        if token == "--format":
            requested = index + 1 < len(tokens) and tokens[index + 1] == "json"
        elif token.startswith("--format="):
            requested = token.removeprefix("--format=") == "json"
    return requested


def _print_error(error: CliError, *, json_output: bool) -> int:
    message = terminal_text(str(error))
    if json_output:
        failure = {"code": error.code.value, "message": message, "exitCode": error.exit_code}
        if error.hint:
            failure["hint"] = terminal_text(error.hint)
        print(json.dumps({"error": failure}, ensure_ascii=True, indent=2), file=sys.stderr)
    else:
        print(f"{error.code}: {message}", file=sys.stderr)
        if error.hint:
            print(f"Hint: {terminal_text(error.hint)}", file=sys.stderr)
    return error.exit_code


def main(
    tokens: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    transport: httpx2.BaseTransport | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> int:
    """Run the command line and return its exit status."""
    arguments = list(tokens) if tokens is not None else sys.argv[1:]
    try:
        command, bound, ignored = app.parse_args(arguments, print_error=False)
        if "runtime" in ignored:
            bound.arguments["runtime"] = _Runtime(environ=environ, transport=transport, now=now)
        command(*bound.args, **bound.kwargs)
    except CycloptsError as error:
        invalid_argument = CliError(
            ErrorCode.INVALID_ARGUMENT,
            str(error),
            hint="Run artificial-analysis --help, then retry.",
        )
        return _print_error(invalid_argument, json_output=_json_output_requested(arguments))
    except CliError as error:
        return _print_error(error, json_output=_json_output_requested(arguments))
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        finally:
            os.close(devnull)
        return 1
    return 0
