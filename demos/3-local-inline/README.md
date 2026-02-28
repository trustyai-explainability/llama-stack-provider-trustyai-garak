# Deployment 3: Total Inline (Local)

**Everything runs on your laptop** - for development and testing only.

## What Runs Where

| Component | Runs On |
|---|---|
| Llama Stack server | Local machine |
| Garak scans | Local machine |
| Model inference | Your configured model endpoint |

## Prerequisites

- ✅ Python 3.12+
- ✅ **Inline extra installed** (not default!)
- ✅ Model Inference endpoint (OpenAI API compatible)

## Quick Start

### 1) Install with Inline Support

```bash
pip install "llama-stack-provider-trustyai-garak[inline]"
```

### 2) Configure Environment

```bash
# Model endpoint
export VLLM_URL="http://localhost:8000/v1"
export VLLM_API_TOKEN="<token-if-needed>"
```

### 3) Start Server

```bash
llama stack run inline.yaml # (or inline-shield-sample.yaml which is a sample config with local ollama llama-guard as a shield)
```

### 4) Run a Predefined Benchmark

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Quick security scan
benchmark_id = "trustyai_garak::quick"
job = client.alpha.eval.run_eval(
    benchmark_id=benchmark_id,
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "your-model-id",
            "sampling_params": {"max_tokens": 100}
        }
    }
)

status = client.alpha.eval.jobs.status(job_id=job.job_id, benchmark_id=benchmark_id)
if status.status == "completed":
    result = client.alpha.eval.jobs.retrieve(job_id=job.job_id, benchmark_id=benchmark_id)
    print(len(result.generations))
```

### 5) Register Custom Benchmark

```python
garak_provider = next(
    p for p in client.providers.list()
    if p.provider_type.endswith("trustyai_garak")
)
garak_provider_id = garak_provider.provider_id

client.benchmarks.register(
    benchmark_id="custom_inline",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id=garak_provider_id,
    provider_benchmark_id="custom_inline",
    metadata={
        "garak_config": {
            "plugins": {
                "probe_spec": ["promptinject.HijackHateHumans"]
            },
            "system": {
                "parallel_attempts": 12
            }
        },
        "timeout": 900
    }
)
```

### 6) Update Existing/Predefined Benchmarks

Use deep-merge behavior via `provider_benchmark_id`:

```python
client.benchmarks.register(
    benchmark_id="quick_promptinject_tuned",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id=garak_provider_id,
    provider_benchmark_id="trustyai_garak::quick",
    metadata={
        "garak_config": {
            "plugins": {"probe_spec": ["promptinject"]},
            "system": {"parallel_attempts": 20}
        }
    }
)
```

### 7) Read `_overall` and TBSA

```python
result = client.alpha.eval.jobs.retrieve(job_id=job.job_id, benchmark_id=benchmark_id)
overall = result.scores.get("_overall")
if overall:
    print(overall.aggregated_results)
```

`_overall.aggregated_results` include:

- `total_attempts`
- `vulnerable_responses`
- `attack_success_rate`
- `probe_count`
- `tbsa` (Tier-Based Security Aggregate, 1.0 to 5.0, higher is better)

## Canonical Demo

Use `demos/guide.ipynb` for the complete end-to-end story across all features.
