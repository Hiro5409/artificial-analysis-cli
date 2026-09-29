# List the available development tasks.
list:
    @just --list

# Run source, test, and code generation checks.
check: lint test codegen-check

# Check lint, formatting, types, and dependencies.
lint:
    uv run --locked just --fmt --check
    uv run --locked ruff check
    uv run --locked ruff format --check
    uv run --locked ty check
    uv run --locked deptry src

# Run tests, forwarding additional arguments to pytest.
[positional-arguments]
test *args:
    uv run --locked pytest "$@"

# Apply safe lint fixes and format the source tree.
format:
    uv run --locked ruff check --fix
    uv run --locked ruff format

# Verify that the generated client matches the OpenAPI snapshot.
codegen-check:
    uv run --locked scripts/update_client.py check

# Download the current official OpenAPI document.
openapi-fetch:
    uv run --locked scripts/update_client.py fetch

# Regenerate the client from the checked-in OpenAPI snapshot.
codegen:
    uv run --locked scripts/update_client.py generate

# Check the live API, forwarding options such as --pro.
[positional-arguments]
smoke *args:
    uv run --locked scripts/smoke.py "$@"
