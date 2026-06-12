# AI Agent Context

## What This Repo Is

This repo provides a **Garak eval-hub adapter** — a `FrameworkAdapter` for the
[eval-hub](https://github.com/opendatahub-io/eval-hub) SDK used by the RHOAI
evaluation platform to orchestrate [Garak](https://github.com/NVIDIA/garak)
LLM red-teaming scans via K8s jobs.

> **Note:** As of v0.5.0, all Llama Stack provider code has been removed.
> This package no longer depends on or supports llama-stack/ogx.

## Two Execution Modes

```
 Eval-Hub
 (eval-hub SDK)
 ┌────────┬────────┐
 │ Simple │  KFP   │
 │ (pod)  │ (pod + │
 │        │  KFP)  │
 └────────┴────────┘
  in-pod     K8s job
  garak      submits to
             KFP, polls
```

| Mode | Code Location | How Garak Runs | Intents Support |
|------|--------------|----------------|-----------------|
| **Simple** | `evalhub/` (simple mode) | Directly in the eval-hub K8s job pod | No |
| **KFP** | `evalhub/` (KFP mode) | K8s job submits to KFP, polls status, pulls artifacts via S3 | **Yes** |

**Intents** uses SDG (synthetic data generation), TAPIntent probes, and
MulticlassJudge detectors to test model behavior against policy taxonomies.
Only KFP mode supports it because it requires the six-step pipeline
(`core/pipeline_steps.py`) running as KFP components.

## Code Layout

```
src/llama_stack_provider_trustyai_garak/
├── core/                    # Shared logic
│   ├── config_resolution.py # Deep-merge user overrides onto benchmark profiles
│   ├── command_builder.py   # Build garak CLI args for OpenAI-compatible endpoints
│   ├── garak_runner.py      # Subprocess runner for garak CLI
│   └── pipeline_steps.py    # Six-step pipeline (validate→taxonomy→SDG→prompts→scan→parse)
│
├── evalhub/                 # Eval-Hub integration (main entry point)
│   ├── garak_adapter.py     # FrameworkAdapter: benchmark resolution, intents overlay, callbacks
│   ├── kfp_adapter.py       # KFP-specific adapter (forces KFP execution mode)
│   ├── kfp_pipeline.py      # Eval-hub KFP pipeline with S3 artifact flow
│   └── s3_utils.py          # S3/Data Connection client
│
├── garak_command_config.py  # Pydantic models for garak YAML config
├── config.py                # Scan profiles and TapIntentConfig
├── intents.py               # Policy taxonomy dataset loading (SDG/intents flows)
├── sdg.py                   # Synthetic data generation via sdg-hub
├── result_utils.py          # Parse garak outputs, TBSA scoring, HTML reports
└── resources/               # Jinja2 templates and Vega chart specs
```

## Key Conventions

- **Config merging**: User overrides are deep-merged onto benchmark profiles via
  `deep_merge_dicts` in `core/config_resolution.py`. Only leaf values are replaced.
- **Intents model overlay**: When `intents_models` is provided, model endpoints
  are applied using `x.get("key") or default` pattern — fills empty slots but
  preserves user-configured values. `api_key` is always forced to `__FROM_ENV__`
  (K8s Secret injection).
- **Benchmark profiles**: Predefined configs live in `config.py` (GarakScanConfig).
  The `intents` profile is the most complex — it includes TAPIntent, MulticlassJudge,
  and SDG configuration.

## Build & Install

```bash
pip install -e .          # Core (eval-hub adapter)
pip install -e ".[sdg]"   # With SDG support
pip install -e ".[dev]"   # Dev (tests + ruff + pre-commit)
```

## Running Tests

```bash
make test      # All tests (no cluster/GPU/network needed)
make coverage  # With coverage report
make lint      # ruff check
```

Tests are 100% unit tests. Garak is mocked — it does not need to be installed.

## Debugging

- `GARAK_SCAN_DIR` — controls where scan artifacts land
- `LOG_LEVEL=DEBUG` — verbose eval-hub adapter logging
- `scan.log` in scan directory — garak subprocess output
- `__FROM_ENV__` in configs — placeholder for K8s Secret api_key injection
