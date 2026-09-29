"""Pure selection, ordering and rendering of model collections."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import assert_never

from artificial_analysis_cli.catalog import AccountTier, Query, options

SOURCE_NAME = "Artificial Analysis"
SOURCE_URL = "https://artificialanalysis.ai/"

Number = int | float


class OutputFormat(Enum):
    TABLE = "table"
    JSON = "json"
    TSV = "tsv"


@dataclass(frozen=True)
class Metric:
    """A numeric column. A collection is ranked by its first metric."""

    key: str
    heading: str
    higher_is_better: bool = True


@dataclass(frozen=True)
class ModelRow:
    name: str
    creator: str | None
    metrics: tuple[Number | None, ...]
    """Values aligned with `Collection.columns`."""
    record: Mapping[str, object]
    """The API's model record, including fields this CLI does not interpret."""


@dataclass(frozen=True)
class Collection:
    query: Query
    account_tier: AccountTier
    fetched_at: datetime
    cached: bool
    columns: tuple[Metric, ...]
    rows: tuple[ModelRow, ...]
    fields: Mapping[str, object]
    """The API's collection fields other than `data` and `pagination`."""


def _ranked(collection: Collection, top: int | None) -> tuple[ModelRow, ...]:
    """Rows ordered by the first metric, best first and missing values last, then by model name."""
    metric = collection.columns[0]

    def order(row: ModelRow) -> tuple[bool, Number, str]:
        value = row.metrics[0]
        if value is None:
            return (True, 0, row.name.casefold())
        return (False, -value if metric.higher_is_better else value, row.name.casefold())

    rows = sorted(collection.rows, key=order)
    return tuple(rows if top is None else rows[:top])


def render(collection: Collection, output_format: OutputFormat, top: int | None) -> str:
    rows = _ranked(collection, top)
    match output_format:
        case OutputFormat.TABLE:
            return _table(collection, rows)
        case OutputFormat.TSV:
            return _tsv(collection, rows)
        case OutputFormat.JSON:
            return _json(collection, rows)
        case _:
            assert_never(output_format)


def _table(collection: Collection, rows: tuple[ModelRow, ...]) -> str:
    headings = ["#", "Creator", "Model", *(metric.heading for metric in collection.columns)]
    cells = [
        [
            str(position),
            terminal_text(row.creator or "-"),
            terminal_text(row.name),
            *(_text(value, "-") for value in row.metrics),
        ]
        for position, row in enumerate(rows, start=1)
    ]
    widths = [max(len(line[index]) for line in (headings, *cells)) for index in range(len(headings))]
    numeric = [True, False, False, *(True for _ in collection.columns)]

    def line(values: list[str]) -> str:
        aligned = (
            value.rjust(width) if right else value.ljust(width)
            for value, width, right in zip(values, widths, numeric, strict=True)
        )
        return "  ".join(aligned).rstrip()

    query = collection.query
    details = [
        f"Endpoint variant: {query.variant.value}",
        f"account tier: {collection.account_tier}",
        *(f"{name}: {value}" for name, value in options(query).items()),
        f"fetched {_timestamp(collection.fetched_at)} from {'cache' if collection.cached else 'the API'}",
    ]
    return "\n".join(
        [
            line(headings),
            line(["-" * width for width in widths]),
            *(line(values) for values in cells),
            "",
            f"Source: {SOURCE_NAME} ({SOURCE_URL})",
            "; ".join(details),
        ]
    )


def _tsv(collection: Collection, rows: tuple[ModelRow, ...]) -> str:
    header = ["position", "creator", "model", *(metric.key for metric in collection.columns), "source"]
    lines = [header]
    for position, row in enumerate(rows, start=1):
        values = (_text(value, "") for value in row.metrics)
        lines.append([str(position), row.creator or "", row.name, *values, SOURCE_URL])
    return "\n".join("\t".join(_tsv_cell(value) for value in line) for line in lines)


def _json(collection: Collection, rows: tuple[ModelRow, ...]) -> str:
    query = collection.query
    document = {
        "source": {"name": SOURCE_NAME, "url": SOURCE_URL},
        "family": query.family.alias,
        "endpoint_variant": query.variant.value,
        "options": options(query),
        "fetched_at": _timestamp(collection.fetched_at),
        "cached": collection.cached,
        "collection": {**collection.fields, "data": [row.record for row in rows]},
    }
    # JSON escapes preserve values while keeping C1 controls and bidi formatting out of the terminal.
    return json.dumps(document, ensure_ascii=True, indent=2)


def _text(value: Number | None, missing: str) -> str:
    if value is None:
        return missing
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _tsv_cell(value: str) -> str:
    """Escape a cell so every record stays on one line with a fixed number of fields."""
    return terminal_text(value.replace("\\", "\\\\"))


def terminal_text(value: str) -> str:
    """Display untrusted text on one line without terminal controls or invisible formatting."""
    return "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in value)


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")
