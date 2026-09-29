# artificial-analysis-cli

Typed, read-only command-line access to the model lists of the official [Artificial Analysis Data API](https://artificialanalysis.ai/data-api/docs). It covers the Free and Pro list endpoints for all 12 model families.

## Install

```console
uv tool install artificial-analysis-cli
export ARTIFICIAL_ANALYSIS_API_KEY=...
```

## Use

Discover the current commands, model families, options, and accepted values from the CLI:

```console
artificial-analysis --help
artificial-analysis families
artificial-analysis list --help
```

Examples:

```console
artificial-analysis list t2v --top 10
artificial-analysis list llm --format tsv
artificial-analysis list llm --pro --prompt-type medium_coding
artificial-analysis list t2i --pro --categories --format json
artificial-analysis list music-vocal --pro --genres --format json
```

Successful output goes to stdout as `table`, `json`, or `tsv`. Validated responses are cached for one hour per API key and request; `--refresh` bypasses the cache.

Failures leave stdout empty and write to stderr. Text errors use a stable code and an actionable hint:

```text
AUTH_REQUIRED: ARTIFICIAL_ANALYSIS_API_KEY is not set.
Hint: Set ARTIFICIAL_ANALYSIS_API_KEY, then retry.
```

With `--format json`, errors use the same information in a machine-readable envelope:

```json
{
  "error": {
    "code": "AUTH_REQUIRED",
    "message": "ARTIFICIAL_ANALYSIS_API_KEY is not set.",
    "exitCode": 2,
    "hint": "Set ARTIFICIAL_ANALYSIS_API_KEY, then retry."
  }
}
```

| Exit code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Network, upstream API, or response failure |
| `2` | Invalid command or authentication |
| `3` | Insufficient API access |
| `4` | API rate limit exceeded |

Artificial Analysis data must be credited to [Artificial Analysis](https://artificialanalysis.ai/) wherever it is displayed or shared. See the API's [Terms of Use](https://artificialanalysis.ai/docs/legal/Terms-of-Use.pdf) and [Data Platform Terms](https://artificialanalysiscdn.com/legal/ProDataPlatformTerms.pdf).

## Develop

```console
uv sync --locked
uv run --locked just --list
uv run --locked just check
```
