# Benchmark Metadata Reference

This document is the reference for fields supported in:

```python
client.alpha.benchmarks.register(..., metadata={...})
```

It covers:

- `garak_config` (detailed command config)
- shield fields (`shield_ids`, `shield_config`)
- runtime controls (`timeout`, remote-only retry/GPU keys)
- deep-merge behavior when updating predefined/existing benchmarks

## 1) Metadata Shape

```python
metadata = {
    "garak_config": {
        "system": {...},
        "run": {...},
        "plugins": {...},
        "reporting": {...},
    },
    "timeout": 1800,
    "shield_ids": ["Prompt-Guard-86M"],  # or use shield_config
    "max_retries": 3,                    # remote mode only
    "use_gpu": False,                    # remote mode only
}
```

If `garak_config` is omitted, provider falls back to default Garak config (effectively broad/default probe selection), which can be very slow.

### 1.1 Build `garak_config` via Python models (optional)

You can construct config using typed models exported by this package:

```python
from llama_stack_provider_trustyai_garak import (
    GarakCommandConfig,
    GarakSystemConfig,
    GarakRunConfig,
    GarakPluginsConfig,
    GarakReportingConfig,
)
```

Example:

```python
garak_cfg = GarakCommandConfig(
    system=GarakSystemConfig(parallel_attempts=20),
    run=GarakRunConfig(generations=2, eval_threshold=0.5),
    plugins=GarakPluginsConfig(probe_spec=["promptinject.HijackHateHumans"]),
    reporting=GarakReportingConfig(taxonomy="owasp"),
)

metadata = {
    "garak_config": garak_cfg.to_dict(),
    "timeout": 900,
}
```

## 2) Top-Level Metadata Keys

| Key | Type | Default | Mode | Notes |
|---|---|---|---|---|
| `garak_config` | `dict` | default `GarakCommandConfig()` | inline + remote | Main Garak command schema. Recommended to always set. |
| `timeout` | `int` (seconds) | provider default (`10800`) | inline + remote | Max scan runtime for a benchmark run. |
| `shield_ids` | `list[str]` | `[]` | inline + remote | Shortcut for input shields only. |
| `shield_config` | `dict` | `{}` | inline + remote | Explicit mapping: `{"input": [...], "output": [...]}`. |
| `max_retries` | `int` | `3` | remote only | KFP pipeline retry count for scan step. |
| `use_gpu` | `bool` | `False` | remote only | Requests GPU scheduling in KFP pipeline. |

Notes:

- If both `shield_ids` and `shield_config` are provided, `shield_ids` takes precedence.
- Unknown top-level keys are passed as provider params but are ignored unless consumed by adapter logic.

## 3) Shield Metadata Rules

### `shield_ids`

```python
"shield_ids": ["Prompt-Guard-86M"]
```

- Must be a list.
- Treated as input shields.
- Easier syntax for common cases.

### `shield_config`

```python
"shield_config": {
    "input": ["Prompt-Guard-86M"],
    "output": ["Llama-Guard-3-8B"]
}
```

- Must be a dictionary.
- Use when you need separate input/output shield chains.

Validation behavior:

- Provider validates shield IDs against Shields API.
- If Shields API is not enabled and shield metadata is present, run fails.

## 4) `garak_config` Detailed Schema

`garak_config` has four primary sections:

- `system`
- `run`
- `plugins`
- `reporting`

### 4.1 `garak_config.system`

| Field | Type | Default | Description |
|---|---|---|---|
| `parallel_attempts` | `bool \| int` | `16` | Parallel prompt attempts where supported. |
| `max_workers` | `int` | `500` | Upper bound for requested worker count. |
| `parallel_requests` | `bool \| int` | `False` | Parallel requests for generators lacking multi-response support. |
| `verbose` | `int` (`0..2`) | `0` | CLI verbosity. |
| `show_z` | `bool` | `False` | Show Z-scores in CLI output. |
| `narrow_output` | `bool` | `False` | Improve output for narrow terminals. |
| `lite` | `bool` | `True` | Lite mode caution output behavior. |
| `enable_experimental` | `bool` | `False` | Enable experimental Garak flags. |

### 4.2 `garak_config.run`

| Field | Type | Default | Description |
|---|---|---|---|
| `generations` | `int` | `1` | Number of generations per prompt. |
| `probe_tags` | `str \| None` | `None` | Tag-based probe selection (e.g. `owasp:llm`). |
| `eval_threshold` | `float` (`0..1`) | `0.5` | Detector threshold for hit/vulnerable decision. |
| `soft_probe_prompt_cap` | `int` | `256` | Preferred prompt cap for autoscaling probes. Lower values reduce prompts per probe and make runs faster (with reduced coverage/comprehensiveness). |
| `target_lang` | `str \| None` | `None` | BCP47 language target. |
| `langproviders` | `list[str] \| None` | `None` | Providers for language conversion. |
| `system_prompt` | `str \| None` | `None` | Default system prompt where applicable. |
| `seed` | `int \| None` | `None` | Reproducibility seed. |
| `deprefix` | `bool` | `True` | Remove prompt prefix echoed by model outputs. |

Performance tuning tip:

- Predefined benchmarks are comprehensive by default.
- To speed up exploratory runs, override `garak_config.run.soft_probe_prompt_cap` with a smaller value.
- For full security assessment/comparability, keep defaults (or use consistent cap across compared runs).

### 4.3 `garak_config.plugins`

| Field | Type | Default | Description |
|---|---|---|---|
| `probe_spec` | `list[str] \| str` | `"all"` | Probe/module/class selection. |
| `detector_spec` | `list[str] \| str \| None` | `None` | Detector override (`None` uses probe defaults). |
| `extended_detectors` | `bool` | `True` | Include extended detector set. |
| `buff_spec` | `list[str] \| str \| None` | `None` | Buff/module selection. |
| `buffs_include_original_prompt` | `bool` | `True` | Keep original prompt when buffing. |
| `buff_max` | `int \| None` | `None` | Cap output count from buffs. |
| `target_type` | `str` | auto-managed | Provider sets this for openai/function mode. |
| `target_name` | `str \| None` | auto-managed | Provider sets this to model or shield orchestrator. |
| `probes` | `dict \| None` | `None` | Probe plugin config tree. |
| `detectors` | `dict \| None` | `None` | Detector plugin config tree. |
| `generators` | `dict \| None` | `None` | Generator plugin config tree. |
| `buffs` | `dict \| None` | `None` | Buff plugin config tree. |
| `harnesses` | `dict \| None` | `None` | Harness plugin config tree. |

Provider behavior worth knowing:

- `probe_spec`, `detector_spec`, `buff_spec` accept string or list, and are normalized before run.
- If shield metadata is present, provider switches generator mode to function-based shield orchestration automatically.
- Otherwise provider uses OpenAI-compatible generator mode.

### 4.4 `garak_config.reporting`

| Field | Type | Default | Description |
|---|---|---|---|
| `taxonomy` | `str \| None` | `None` | Grouping taxonomy (`owasp`, `avid-effect`, `quality`, `cwe`). |
| `show_100_pass_modules` | `bool` | `True` | Include fully passing entries in HTML report details. |
| `show_top_group_score` | `bool` | `True` | Show top-level aggregate in grouped report sections. |
| `group_aggregation_function` | `str` | `"lower_quartile"` | Group aggregation strategy in report. |
| `report_dir` | `str \| None` | auto-managed | Provider-managed output location; usually leave unset. |
| `report_prefix` | `str \| None` | auto-managed | Provider-managed output prefix; usually leave unset. |

Please refer to [Garak configuration docs](https://reference.garak.ai/en/latest/configurable.html#config-files-yaml-and-json) for details about these controls.

## 5) Deep-Merge Behavior (Updating Predefined/Existing Benchmarks)

When registering with `provider_benchmark_id`, metadata is deep-merged:

- base metadata comes from:
  - predefined profile (`trustyai_garak::...`), or
  - existing benchmark metadata
- your new metadata overrides only specified keys

Example:

```python
client.alpha.benchmarks.register(
    benchmark_id="quick_promptinject_tuned",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id=garak_provider_id,
    provider_benchmark_id="trustyai_garak::quick",
    metadata={
        "garak_config": {
            "plugins": {"probe_spec": ["promptinject"]},
            "system": {"parallel_attempts": 20},
        },
        "timeout": 1200,
    },
)
```

## 6) Practical Examples

### Example A: Minimal custom benchmark

```python
metadata = {
    "garak_config": {
        "plugins": {"probe_spec": ["promptinject.HijackHateHumans"]},
        "run": {"generations": 2, "eval_threshold": 0.5},
        "reporting": {"taxonomy": "owasp"},
    },
    "timeout": 900,
}
```

### Example B: Explicit input/output shield mapping

```python
metadata = {
    "garak_config": {
        "plugins": {"probe_spec": ["promptinject.HijackHateHumans"]},
    },
    "shield_config": {
        "input": ["Prompt-Guard-86M"],
        "output": ["Llama-Guard-3-8B"],
    },
    "timeout": 600,
}
```

### Example C: Remote retry/GPU controls

```python
metadata = {
    "garak_config": {
        "run": {"probe_tags": "owasp:llm"},
    },
    "timeout": 7200,
    "max_retries": 2,
    "use_gpu": True,
}
```

### Example D: Faster predefined benchmark variant

```python
metadata = {
    "garak_config": {
        "run": {
            "soft_probe_prompt_cap": 100
        }
    },
    "timeout": 7200,
}

# Register as a tuned variant of a predefined benchmark
client.alpha.benchmarks.register(
    benchmark_id="owasp_fast",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id=garak_provider_id,
    provider_benchmark_id="trustyai_garak::owasp_llm_top10",
    metadata=metadata,
)
```

## 7) Legacy / Compatibility Notes

- Prefer `metadata.garak_config.plugins.probe_spec` over old top-level `metadata.probes`.
- Prefer `metadata.garak_config.run.eval_threshold` for threshold control.
- Keep benchmark metadata focused on benchmark/run concerns.
  KFP control-plane settings such as `experiment_name` belong in provider config (`kubeflow_config.experiment_name`, environment: `KUBEFLOW_EXPERIMENT_NAME`), not benchmark metadata.
