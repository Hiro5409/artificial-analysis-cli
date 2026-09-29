"""Maintain the OpenAPI snapshot and the client generated from it.

fetch     Download the official OpenAPI document into the snapshot (network).
generate  Regenerate the client package from the snapshot (offline).
check     Fail when the checked-in client differs from a fresh generation (offline).
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

OPENAPI_URL = "https://artificialanalysis.ai/api/v2/openapi"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "openapi" / "artificial-analysis.yaml"
GENERATED_PATH = PROJECT_ROOT / "src" / "artificial_analysis_cli" / "_generated"
SELECTED_PATHS = {
    "/api/v2/language/models",
    "/api/v2/language/models/free",
    "/api/v2/media/image-editing/models",
    "/api/v2/media/image-editing/models/free",
    "/api/v2/media/image-to-video-audio/models",
    "/api/v2/media/image-to-video-audio/models/free",
    "/api/v2/media/image-to-video/models",
    "/api/v2/media/image-to-video/models/free",
    "/api/v2/media/music/instrumental/models",
    "/api/v2/media/music/instrumental/models/free",
    "/api/v2/media/music/with-vocals/models",
    "/api/v2/media/music/with-vocals/models/free",
    "/api/v2/media/speech-to-speech/models",
    "/api/v2/media/speech-to-speech/models/free",
    "/api/v2/media/speech-to-text/models",
    "/api/v2/media/speech-to-text/models/free",
    "/api/v2/media/text-to-image/models",
    "/api/v2/media/text-to-image/models/free",
    "/api/v2/media/text-to-speech/models",
    "/api/v2/media/text-to-speech/models/free",
    "/api/v2/media/text-to-video-audio/models",
    "/api/v2/media/text-to-video-audio/models/free",
    "/api/v2/media/text-to-video/models",
    "/api/v2/media/text-to-video/models/free",
}


def _component_references(value: object) -> set[tuple[str, str]]:
    references: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/components/"):
            _, _, section, name = reference.split("/", 3)
            references.add((section, name))
        for nested_value in value.values():
            references.update(_component_references(nested_value))
    elif isinstance(value, list):
        for nested_value in value:
            references.update(_component_references(nested_value))
    return references


def _selected_contract(document: dict[str, Any]) -> dict[str, Any]:
    selected = copy.deepcopy(document)
    selected["paths"] = {path: value for path, value in document["paths"].items() if path in SELECTED_PATHS}
    if selected["paths"].keys() != SELECTED_PATHS:
        missing = sorted(SELECTED_PATHS - selected["paths"].keys())
        raise SystemExit(f"The OpenAPI document is missing selected paths: {', '.join(missing)}")

    references = _component_references(selected["paths"])
    pending = list(references)
    while pending:
        section, name = pending.pop()
        component = document["components"][section][name]
        for reference in _component_references(component):
            if reference not in references:
                references.add(reference)
                pending.append(reference)

    components: dict[str, dict[str, Any]] = {}
    for section, name in sorted(references):
        components.setdefault(section, {})[name] = document["components"][section][name]
    components["securitySchemes"] = document["components"].get("securitySchemes", {})
    selected["components"] = components
    return selected


def _load(spec: Path) -> dict[str, Any]:
    with spec.open(encoding="utf-8") as source:
        return YAML().load(source)


def _generate(spec: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="artificial-analysis-codegen-") as temporary_directory:
        selected_spec = Path(temporary_directory) / "selected-openapi.json"
        contract = _selected_contract(_load(spec))
        selected_spec.write_text(json.dumps(contract, default=str), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "datamodel_code_generator",
                "--input",
                str(selected_spec),
                "--output",
                str(destination / "pydantic_gen.py"),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        _bindings(destination / "pydantic_gen.py", contract)
    subprocess.run(["ruff", "check", "--isolated", "--select", "I,F401", "--fix", str(destination)], check=True)
    subprocess.run(["ruff", "format", "--isolated", "--line-length", "120", str(destination)], check=True)


def _bindings(path: Path, contract: dict[str, Any]) -> None:
    """Derive CLI bindings without changing generated model semantics."""
    source = path.read_text(encoding="utf-8")
    names = {node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    routes = {}
    for route, operation in contract["paths"].items():
        ref = operation["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        name = ref.rsplit("/", 1)[1]
        if name not in names:
            raise ValueError(f"The generator did not produce response model {name}")
        routes[route] = name
    response_names = sorted(set(routes.values()))
    union = " | ".join(response_names)
    imports = ", ".join(response_names)
    table = ",\n".join(f"    {route!r}: {name}" for route, name in sorted(routes.items()))
    prompt_types = contract["components"]["parameters"]["prompt_type"]["schema"]["enum"]
    literals = ", ".join(repr(value) for value in prompt_types)
    path.with_name("__init__.py").write_text(
        "# Generated by scripts/update_client.py from the OpenAPI snapshot; do not edit.\n"
        "from typing import Literal, TypeAlias\n"
        f"from .pydantic_gen import {imports}\n\n"
        f"PromptType: TypeAlias = Literal[{literals}]\n"
        f"Page: TypeAlias = {union}\n"
        f"RESPONSES: dict[str, type[Page]] = {{\n{table}\n}}\n",
        encoding="utf-8",
    )


def _files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def fetch() -> None:
    with tempfile.TemporaryDirectory(prefix="artificial-analysis-openapi-") as temporary_directory:
        downloaded = Path(temporary_directory) / "artificial-analysis.yaml"
        with urllib.request.urlopen(OPENAPI_URL, timeout=30) as response:
            downloaded.write_bytes(response.read())
        _selected_contract(_load(downloaded))
        shutil.move(downloaded, SPEC_PATH)
    print(f"Updated {SPEC_PATH.relative_to(PROJECT_ROOT)}. Run `generate` and review both diffs.")


def generate() -> None:
    with tempfile.TemporaryDirectory(prefix="artificial-analysis-client-") as temporary_directory:
        generated = Path(temporary_directory) / "_generated"
        _generate(SPEC_PATH, generated)
        shutil.rmtree(GENERATED_PATH)
        shutil.move(generated, GENERATED_PATH)


def check() -> int:
    with tempfile.TemporaryDirectory(prefix="artificial-analysis-client-") as temporary_directory:
        generated = Path(temporary_directory) / "_generated"
        _generate(SPEC_PATH, generated)
        expected = _files(generated)
    actual = _files(GENERATED_PATH)
    stale = sorted(path for path in expected.keys() | actual.keys() if expected.get(path) != actual.get(path))
    if not stale:
        return 0
    print("The generated client differs from the OpenAPI snapshot. Run `generate`:", file=sys.stderr)
    for path in stale:
        print(f"  {GENERATED_PATH.relative_to(PROJECT_ROOT) / path}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "generate", "check"))
    match parser.parse_args().command:
        case "fetch":
            fetch()
        case "generate":
            generate()
        case "check":
            return check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
