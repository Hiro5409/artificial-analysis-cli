"""Check every list endpoint against the live API with the key in ARTIFICIAL_ANALYSIS_API_KEY.

Free endpoint variants must succeed. With --pro the Pro variants are checked too, and a 403 is
reported as unverified because the key's account tier lacks Pro access, not as a failure.
Every run spends API quota: one request per family and variant, plus one per extra language
model page.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from artificial_analysis_cli.catalog import Family

API_KEY_VARIABLE = "ARTIFICIAL_ANALYSIS_API_KEY"


def check(family: Family, *, pro: bool, cache_home: str) -> tuple[str, bool]:
    """A one-line result for the family's endpoint variant, and whether it counts as passing."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "artificial_analysis_cli",
            "list",
            family.alias,
            *(["--pro"] if pro else []),
            "--format",
            "json",
        ],
        env={**os.environ, "XDG_CACHE_HOME": cache_home},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        document = json.loads(result.stdout)
        return f"ok: {len(document['collection']['data'])} models", True
    try:
        document = json.loads(result.stderr)
    except json.JSONDecodeError:
        document = None
    match document:
        case {"error": {"code": str(code), "message": str(error_message)} as error}:
            message = f"{code}: {error_message}"
            if isinstance(hint := error.get("hint"), str):
                message += f" Hint: {hint}"
        case _:
            detail = " ".join(result.stderr.split()) or "no error output"
            return f"failed: child command exited with status {result.returncode}: {detail}", False
    if pro and code == "FORBIDDEN":
        return f"unverified: the key's account tier lacks Pro access ({message})", True
    return f"failed: {message}", False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pro", action="store_true", help="also check the Pro endpoint variants")
    arguments = parser.parse_args()
    if not os.environ.get(API_KEY_VARIABLE):
        print(f"Not run: {API_KEY_VARIABLE} is not set.", file=sys.stderr)
        return 2

    passed = True
    variants = [False, True] if arguments.pro else [False]
    with tempfile.TemporaryDirectory(prefix="artificial-analysis-smoke-") as cache_home:
        for pro in variants:
            for family in Family:
                summary, ok = check(family, pro=pro, cache_home=cache_home)
                passed = passed and ok
                print(f"{'pro' if pro else 'free':<5} {family.alias:<19} {summary}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
