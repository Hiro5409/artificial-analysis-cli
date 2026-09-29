from __future__ import annotations

from pathlib import Path
from runpy import run_path
from subprocess import CompletedProcess
from typing import Protocol, cast
from unittest.mock import patch

from artificial_analysis_cli.catalog import Family


class _SmokeCheck(Protocol):
    def __call__(self, family: Family, *, pro: bool, cache_home: str) -> tuple[str, bool]: ...


def test_unstructured_child_failure_is_reported_without_aborting() -> None:
    script = Path(__file__).parents[1] / "scripts/smoke.py"
    check = cast(_SmokeCheck, run_path(str(script))["check"])
    failure = CompletedProcess(
        args=[],
        returncode=1,
        stdout="",
        stderr="Traceback: child crashed\n  while importing a dependency\n",
    )

    with patch("subprocess.run", return_value=failure):
        result = check(Family.T2V, pro=False, cache_home="unused")

    assert result == (
        "failed: child command exited with status 1: Traceback: child crashed while importing a dependency",
        False,
    )
