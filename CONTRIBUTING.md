# Contributing

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for lockfile generation)
- Docker/Podman (for container builds)

## Setup

```bash
git clone https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak.git
cd llama-stack-provider-trustyai-garak

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pre-commit install
```

## Dependency Management

Dependencies are declared in `pyproject.toml` with optional extras:

| Extra | Install Command | What You Get |
|-------|----------------|-------------|
| (none) | `pip install -e .` | Core provider (Llama Stack remote mode) |
| `[inline]` | `pip install -e ".[inline]"` | Core + garak for local scans |
| `[dev]` | `pip install -e ".[dev]"` | Tests + ruff + pre-commit |
| `[server]` | `pip install -e ".[server]"` | Llama Stack server |

**Lockfile**: `requirements.txt` is a pinned lockfile generated from the
[RH AI PyPI index](https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9-test/simple/)
via `uv pip compile`. It's used by downstream production builds for hermetic
dependency pre-fetching. Regenerate with `make lock`.

**Container image**: The dev `Containerfile` installs from standard PyPI with
garak from the midstream git branch. Production images use the AIPCC base image
with the RH AI index. See `Containerfile` for the dev build and
`Dockerfile.konflux` (downstream) for production.

## Development Workflow

### Running Tests

```bash
make test          # pytest with verbose output
make coverage      # pytest with coverage report
```

All tests are unit tests — no cluster, GPU, or network access needed. Garak is
mocked; it does not need to be installed.

### Linting & Formatting

```bash
make lint          # ruff check
make format        # ruff format (auto-fix)
```

Pre-commit hooks run `ruff` automatically on staged files.

### Building the Container Image

```bash
make build         # builds dev Containerfile
```

The dev image installs deps from PyPI and garak from the midstream git branch.
Production images use the AIPCC base image with the RH AI PyPI index.

### Regenerating requirements.txt

When you change `pyproject.toml`, the pre-commit hook automatically regenerates
`requirements.txt` via `uv pip compile`. If you need to do it manually:

```bash
make lock
```

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed module-by-module breakdown.

This repo contains **two independent integrations** that share core logic:

```
src/llama_stack_provider_trustyai_garak/
├── core/           # Shared logic (config, commands, pipeline steps) — used by ALL modes
├── inline/         # Llama Stack Inline mode (local garak)
├── remote/         # Llama Stack Remote KFP mode (KFP pipelines)
├── evalhub/        # Eval-Hub adapter (simple + KFP modes, NO Llama Stack dependency)
└── resources/      # Report templates and chart specs
tests/              # Unit tests (no external deps needed)
```

Only the KFP-based modes (Llama Stack Remote and Eval-Hub KFP) support the
**intents benchmark** — see ARCHITECTURE.md for details.

## Adding a Benchmark Profile

1. Define the profile dict in `base_eval.py` under `_BENCHMARK_PROFILES`
2. Add tests in `tests/test_evalhub_adapter.py`
3. Document in `BENCHMARK_METADATA_REFERENCE.md`

## Code Conventions

- Use type annotations for function signatures
- Config merging uses `deep_merge_dicts` — only leaf values are replaced
- `api_key` fields use `__FROM_ENV__` placeholder for K8s secret injection
- Tests mock garak — never import garak directly in test code

## Submitting Changes

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `make lint test` to validate locally
4. Push and open a PR — CI will run unit tests, security scans, and dep validation
5. Fill out the PR template checklist

## Reporting Bugs

Use the [bug report template](https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak/issues/new?template=bug_report.yml)
on GitHub Issues.
