# Deployment 3: Total Inline (Local Development)

**Everything runs on your laptop** - for development and testing only.

## What Runs Where

| Component | Runs On |
|-----------|---------|
| Llama Stack Server | Local (your laptop) |
| Garak Scans | Local (your laptop) |
| Model Inference | Your vLLM endpoint (local or remote) |

## Prerequisites

- ✅ Python 3.12+
- ✅ **Inline extra installed** (not default!)
- ✅ Model endpoint (vLLM or compatible)

## Quick Start

### 1. Install with Inline Support

```bash
# IMPORTANT: Inline is NOT enabled by default
pip install "lama-stack-provider-trustyai-garak[inline]"
```

### 2. Configure Environment

```bash
# Model endpoint
export VLLM_URL="http://localhost:8000/v1"
export INFERENCE_MODEL="your-model-name"
```

### 3. Start Server

```bash
llama stack run configs/total-inline.yaml
```

### 4. Run Security Scan

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Quick security scan
job = client.alpha.eval.run_eval(
    benchmark_id="trustyai_garak::quick",
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "your-model-name",
            "sampling_params": {"max_tokens": 100}
        }
    }
)

# Check status
status = client.alpha.eval.jobs.status(
    job_id=job.job_id,
    benchmark_id="trustyai_garak::quick"
)
print(f"Status: {status.status}")

# Get results when complete
if status.status == "completed":
    results = client.alpha.eval.get_eval_job_result(
        job_id=job.job_id,
        benchmark_id="trustyai_garak::quick"
    )
    print(f"Vulnerable: {len([g for g in results.generations if g['vulnerable']])}/{len(results.generations)}")
```
