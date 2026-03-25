# Architecture

## Two Integration Surfaces

This repo serves two **completely independent** orchestration platforms. They
share core garak logic but have different entry points, APIs, and deployment
models.

### 1. Llama Stack Provider

An out-of-tree evaluation provider registered with the Llama Stack framework.
Users interact via the Llama Stack client API (`benchmarks.register`,
`eval.run_eval`). Two execution modes:

| Mode | Module | How It Works |
|------|--------|-------------|
| **Inline** | `inline/` | Garak runs locally inside the Llama Stack server process. Simple, fast, no K8s needed. |
| **Remote KFP** | `remote/` | Llama Stack server submits a KFP pipeline to Kubernetes. Pipeline steps run garak as `@dsl.component`s. Supports intents/SDG. |

Entry points: `inline/garak_eval.py` → `GarakInlineEvalAdapter`,
`remote/garak_remote_eval.py` → `GarakRemoteEvalAdapter`.

Shared base class: `base_eval.py` → `GarakEvalBase` (benchmark resolution,
deep-merge, Files API, scoring).

### 2. Eval-Hub Adapter

A `FrameworkAdapter` for the eval-hub SDK used by the RHOAI evaluation platform.
**Has no dependency on Llama Stack.** Two execution modes:

| Mode | Module | How It Works |
|------|--------|-------------|
| **Simple** | `evalhub/garak_adapter.py` | Garak runs directly inside the eval-hub K8s job pod. No KFP. |
| **KFP** | `evalhub/kfp_adapter.py` | K8s job submits a KFP pipeline, polls for completion, pulls artifacts from S3. Supports intents/SDG. |

Entry points: `evalhub/garak_adapter.py` → `GarakAdapter.main()`,
`evalhub/kfp_adapter.py` → `GarakKFPAdapter` (forces KFP mode).

Can also be invoked as `python -m llama_stack_provider_trustyai_garak.evalhub`.

## Intents Benchmark (Key Feature)

The `intents` benchmark is a major feature this repo. It:

1. Loads a policy taxonomy dataset (CSV/JSON)
2. Runs SDG (synthetic data generation) via sdg-hub to generate test prompts
3. Scans the model with TAPIntent probes (tree-of-attacks with persuasion)
4. Evaluates responses with MulticlassJudge detector

**Only KFP-based modes support intents** because it requires the full six-step
pipeline (`core/pipeline_steps.py`) running as KFP components:

```
validate → taxonomy → SDG → prompts → scan → parse
```

| Mode | Intents Support | Why |
|------|----------------|-----|
| Llama Stack Inline | No | Runs garak directly, no pipeline orchestration |
| Llama Stack Remote KFP | **Yes** | KFP pipeline runs all six steps |
| Eval-Hub Simple | No | In-pod execution, no pipeline orchestration |
| Eval-Hub KFP | **Yes** | KFP pipeline runs all six steps |

### Intents Model Configuration

Intents requires four model endpoints: judge, attacker, evaluator, and target.
These can be configured two ways:

1. **`intents_models` shorthand** — Provide model name/URL per role. The overlay
   fills in garak config using `x.get("key") or default` pattern (respects
   user-configured values, only fills empty slots). `api_key` is always set to
   `__FROM_ENV__` for K8s Secret injection.

2. **`garak_config` only** — Fully configure all model endpoints directly in
   `garak_config.plugins`. No overlay is applied.

## Module Dependency Graph

```
                    ┌──────────────┐
                    │   core/      │
                    │ (shared by   │
                    │  all modes)  │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
     ┌─────▼─────┐  ┌─────▼─────┐  ┌──────▼──────┐
     │  inline/  │  │  remote/  │  │  evalhub/   │
     │           │  │           │  │             │
     │ uses:     │  │ uses:     │  │ uses:       │
     │ base_eval │  │ base_eval │  │ core/ only  │
     │ core/     │  │ core/     │  │ (no Llama   │
     │           │  │ kfp_utils │  │  Stack)     │
     └───────────┘  └───────────┘  └─────────────┘
```

Note: `evalhub/` does NOT import from `base_eval.py` or any Llama Stack types.
It only uses `core/` modules. This is intentional — eval-hub has its own
orchestration and should never depend on Llama Stack.

## Config Resolution Flow

```
Benchmark Profile (predefined in base_eval.py or garak_adapter.py)
        │
        ▼
  deep_merge_dicts(profile, user_overrides)
        │
        ▼
  Effective garak_config
        │
        ▼
  intents_models overlay (only for intents benchmark)
    - model type/name/uri: set if empty, preserve if user-configured
    - api_key: always forced to __FROM_ENV__ (K8s Secret)
    - max_tokens/temperature: setdefault (preserve existing)
        │
        ▼
  Final GarakCommandConfig → CLI args or YAML file
```

## Shared Modules (Top-Level)

| Module | Purpose | Used By |
|--------|---------|---------|
| `garak_command_config.py` | Pydantic models for garak YAML config | All modes |
| `config.py` | Provider config models (inline/remote settings) | Llama Stack only |
| `intents.py` | Policy taxonomy dataset loading | KFP modes (intents) |
| `sdg.py` | Synthetic data generation via sdg-hub | KFP modes (intents) |
| `result_utils.py` | Parse garak outputs, TBSA scoring, HTML reports | All modes |
| `shield_scan.py` | Shield orchestration (input/output guardrails) | Llama Stack only |
| `version_utils.py` | Resolve garak version for provider specs | Llama Stack only |
| `compat.py` | Llama Stack API import compatibility layer | Llama Stack only |
| `constants.py` | KFP image keys, execution mode constants | All modes |

## Dependency Extras

| Extra | Contents | Use Case |
|-------|----------|----------|
| (none) | Core provider + KFP + boto3 | Remote mode / eval-hub |
| `[sdg]` | nest_asyncio + sdg-hub | SDG without garak |
| `[inline]` | `[sdg]` + garak (RH AI index pin) | Local scans |
| `[test]` | pytest + pytest-asyncio | CI test runner |
| `[dev]` | `[test]` + ruff + pre-commit | Local development |
| `[server]` | llama-stack | Running the Llama Stack server |
