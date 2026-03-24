# AI Agent Context

## What This Repo Is

This repo contains **two independent integrations** for running
[Garak](https://github.com/NVIDIA/garak) LLM red-teaming scans. They share
core logic but serve different orchestration surfaces:

1. **Llama Stack Provider** — An out-of-tree eval provider for the
   [Llama Stack](https://llamastack.github.io/) framework. Exposes garak
   through the Llama Stack `benchmarks.register` / `eval.run_eval` API.

2. **Eval-Hub Adapter** — A `FrameworkAdapter` for the eval-hub SDK.
   Completely independent of Llama Stack. Used by the RHOAI evaluation
   platform to orchestrate garak scans via K8s jobs.

## Four Execution Modes

```
              Llama Stack                          Eval-Hub
         (Llama Stack API)                    (eval-hub SDK)
        ┌────────┬────────┐              ┌────────┬────────┐
        │ Inline │ Remote │              │ Simple │  KFP   │
        │        │  KFP   │              │  (pod) │ (pod + │
        │        │        │              │        │  KFP)  │
        └────────┴────────┘              └────────┴────────┘
          local     KFP                   in-pod     K8s job
          garak   pipelines               garak    submits to
                                                   KFP, polls
```

| Mode | Code Location | How Garak Runs | Intents Support |
|------|--------------|----------------|-----------------|
| **Llama Stack Inline** | `inline/` | Locally in the Llama Stack server process | No |
| **Llama Stack Remote KFP** | `remote/` | As KFP pipeline steps on Kubernetes | **Yes** |
| **Eval-Hub Simple** | `evalhub/` (simple mode) | Directly in the eval-hub K8s job pod | No |
| **Eval-Hub KFP** | `evalhub/` (KFP mode) | K8s job submits to KFP, polls status, pulls artifacts via S3 | **Yes** |

**Intents** is a key upcoming feature — it uses SDG (synthetic data generation),
TAPIntent probes, and MulticlassJudge detectors to test model behavior against
policy taxonomies. Only the two KFP-based modes support it because it requires
the six-step pipeline (`core/pipeline_steps.py`) running as KFP components.

## Code Layout

```
src/llama_stack_provider_trustyai_garak/
├── core/               # Shared logic used by ALL modes
│   ├── config_resolution.py   # Deep-merge user overrides onto benchmark profiles
│   ├── command_builder.py     # Build garak CLI args for OpenAI-compatible endpoints
│   ├── garak_runner.py        # Subprocess runner for garak CLI
│   └── pipeline_steps.py      # Six-step pipeline (validate→taxonomy→SDG→prompts→scan→parse)
│
├── inline/             # Llama Stack Inline mode
│   ├── garak_eval.py          # Async adapter wrapping garak subprocess
│   └── provider.py            # Provider spec with pip dependencies
│
├── remote/             # Llama Stack Remote KFP mode
│   ├── garak_remote_eval.py   # Async adapter managing KFP job lifecycle
│   └── kfp_utils/             # KFP pipeline DAG and @dsl.component steps
│
├── evalhub/            # Eval-Hub integration (NO Llama Stack dependency)
│   ├── garak_adapter.py       # FrameworkAdapter: benchmark resolution, intents overlay, callbacks
│   ├── kfp_adapter.py         # KFP-specific adapter (forces KFP execution mode)
│   ├── kfp_pipeline.py        # Eval-hub KFP pipeline with S3 artifact flow
│   └── s3_utils.py            # S3/Data Connection client
│
├── base_eval.py        # Shared Llama Stack eval lifecycle (NOT used by eval-hub)
├── garak_command_config.py  # Pydantic models for garak YAML config
├── intents.py          # Policy taxonomy dataset loading (SDG/intents flows)
├── sdg.py              # Synthetic data generation via sdg-hub
├── result_utils.py     # Parse garak outputs, TBSA scoring, HTML reports
└── resources/          # Jinja2 templates and Vega chart specs
```

## Key Conventions

- **Config merging**: User overrides are deep-merged onto benchmark profiles via
  `deep_merge_dicts` in `core/config_resolution.py`. Only leaf values are replaced.
- **Intents model overlay**: When `intents_models` is provided, model endpoints
  are applied using `x.get("key") or default` pattern — fills empty slots but
  preserves user-configured values. `api_key` is always forced to `__FROM_ENV__`
  (K8s Secret injection).
- **Benchmark profiles**: Predefined configs live in `base_eval.py` (Llama Stack)
  and `evalhub/garak_adapter.py` (eval-hub). The `intents` profile is the most
  complex — it includes TAPIntent, MulticlassJudge, and SDG configuration.
- **Provider specs**: `inline/provider.py` and `remote/provider.py` define Llama
  Stack provider specs. `pip_packages` is auto-populated from `get_garak_version()`.

## Build & Install

```bash
pip install -e .            # Core (Llama Stack remote mode)
pip install -e ".[inline]"  # With garak for local scans
pip install -e ".[dev]"     # Dev (tests + ruff + pre-commit)
```

## Running Tests

```bash
make test       # All tests (no cluster/GPU/network needed)
make coverage   # With coverage report
make lint       # ruff check
```

Tests are 100% unit tests. Garak is mocked — it does not need to be installed.

## Debugging

- `GARAK_SCAN_DIR` — controls where scan artifacts land
- `LOG_LEVEL=DEBUG` — verbose eval-hub adapter logging
- `scan.log` in scan directory — garak subprocess output
- `__FROM_ENV__` in configs — placeholder for K8s Secret api_key injection
