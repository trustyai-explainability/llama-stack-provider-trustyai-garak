# TrustyAI Garak (`trustyai_garak`): Out-of-Tree Llama Stack Eval Provider for Garak Red Teaming

## About
This repository implements [Garak](https://github.com/NVIDIA/garak) as a Llama Stack out-of-tree provider for **security testing and red teaming** of Large Language Models with optional **Shield Integration** for enhanced security testing.

## What It Does

- **Automated Security Testing**: Detects prompt injection, jailbreaks, toxicity, and bias vulnerabilities
- **Compliance Scanning**: OWASP LLM Top 10, AVID taxonomy benchmarks
- **Shield Testing**: Compare LLM security with/without guardrails
- **Scalable Deployment**: Local or Kubernetes/Kubeflow execution
- **Comprehensive Reporting**: JSON, HTML, and detailed logs with vulnerability scores (0.0-1.0)

## Installation

```bash
git clone https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak.git
cd llama-stack-provider-trustyai-garak
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# For remote execution: pip install -e ".[remote]"
```

## Quick Start

### 1. Configure Environment

```bash
export VLLM_URL="http://your-model-endpoint/v1"
export INFERENCE_MODEL="your-model-name"
```

### 2. Start Server

```bash
# Basic mode (standard scanning)
llama stack run run.yaml --image-type venv

# Enhanced mode (with shield testing)
llama stack run run-with-safety.yaml --image-type venv

# Remote mode (Kubernetes/KFP)
llama stack run run-remote.yaml --image-type venv
```

Server runs at `http://localhost:8321`

### 3. Run Security Scan

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Quick 5-minute scan
job = client.eval.run_eval(
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
status = client.eval.jobs.status(job_id=job.job_id, benchmark_id="trustyai_garak::quick")
print(f"Status: {status.status}")

# Get results when complete
if status.status == "completed":
    results = client.eval.get_eval_job_result(job_id=job.job_id, benchmark_id="trustyai_garak::quick")
```

## Available Benchmarks

### Compliance Frameworks
| Benchmark ID | Framework | Duration |
|-------------|-----------|----------|
| `trustyai_garak::owasp_llm_top10` | [OWASP LLM Top 10](https://genai.owasp.org/llm-top-10/) | ~8 hours |
| `trustyai_garak::avid_security` | [AVID Security](https://docs.avidml.org/taxonomy/effect-sep-view/security) | ~8 hours |
| `trustyai_garak::avid_ethics` | [AVID Ethics](https://docs.avidml.org/taxonomy/effect-sep-view/ethics) | ~30 minutes |
| `trustyai_garak::avid_performance` | [AVID Performance](https://docs.avidml.org/taxonomy/effect-sep-view/performance) | ~40 minutes |

### Test Profiles
| Benchmark ID | Description | Duration |
|-------------|-------------|----------|
| `trustyai_garak::quick` | Essential security checks (3 probes) | ~5 minutes |
| `trustyai_garak::standard` | Standard attack vectors (5 categories) | ~1 hour |

_Duration estimates based on Qwen2.5 7B via vLLM_

## Advanced Usage

### Other Garak Probes

```python
client.benchmarks.register(
    benchmark_id="custom",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_benchmark_id="custom",
    provider_id="trustyai_garak",
    metadata={
        "probes": ["latentinjection.LatentJailbreak", "snowball.GraphConnectivity"],
        "timeout": 900
    }
)
```

### Shield Testing

```python
# Test with input shield
client.benchmarks.register(
    benchmark_id="with_shield",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_benchmark_id="with_shield",
    provider_id="trustyai_garak",
    metadata={
        "probes": ["promptinject.HijackHateHumans"],
        "shield_ids": ["Prompt-Guard-86M"]  # Input shield only
    }
)

# Test with input/output shields
metadata={
    "probes": ["promptinject.HijackHateHumans"],
    "shield_config": {
        "input": ["Prompt-Guard-86M"],
        "output": ["Llama-Guard-3-8B"]
    }
}
```

### Accessing Reports

```python
# Get report file IDs from job status
scan_report_id = status.metadata["scan.report.jsonl"]
scan_html_id = status.metadata["scan.report.html"]

# Download via Files API
content = client.files.content(scan_report_id)

# Or via HTTP
import requests
report = requests.get(f"http://localhost:8321/v1/openai/v1/files/{scan_html_id}/content")
```

## Remote Execution (Kubernetes/KFP)

### Setup

```bash
# KFP Configuration
export KUBEFLOW_PIPELINES_ENDPOINT="https://your-kfp-endpoint"
export KUBEFLOW_NAMESPACE="your-namespace"
export KUBEFLOW_EXPERIMENT_NAME="trustyai-garak-scans"
export KUBEFLOW_BASE_IMAGE="quay.io/rh-ee-spandraj/trustyai-garak-provider-dsp:cpu" # for gpu - "quay.io/rh-ee-spandraj/trustyai-garak-provider-dsp:gpu"

# S3 Configuration (for artifacts)
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_S3_ENDPOINT="https://your-s3-endpoint"
export AWS_S3_BUCKET="pipeline-artifacts"
export AWS_DEFAULT_REGION="us-east-1"

# Start server
llama stack run run-remote.yaml --image-type venv
```

_Note: If you're running Llama Stack server locally, make sure `BASE_URL` in run-remote*.yaml is accessible from KFP pods (you can use [ngrok](https://ngrok.com/) to create an accessible endpoint for your local Llama stack service)._

### Usage

```python
# Same API, runs as KFP pipeline
job = client.eval.run_eval(benchmark_id="trustyai_garak::owasp_llm_top10", ...)

# Monitor pipeline
status = client.eval.jobs.status(job_id=job.job_id, benchmark_id="trustyai_garak::owasp_llm_top10")
print(f"KFP Run ID: {status.metadata['kfp_run_id']}")
```

## Configuration Reference

### Provider Config (`run.yaml`)

```yaml
providers:
  eval:
    - provider_id: trustyai_garak
      config:
        base_url: ${env.BASE_URL:=http://localhost:8321/v1}
        timeout: ${env.GARAK_TIMEOUT:=10800}
        max_concurrent_jobs: ${env.GARAK_MAX_CONCURRENT_JOBS:=5}
        max_workers: ${env.GARAK_MAX_WORKERS:=5}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_URL` | `http://localhost:8321/v1` | Llama Stack service URL |
| `GARAK_TIMEOUT` | `10800` | Max scan timeout (seconds) |
| `GARAK_MAX_CONCURRENT_JOBS` | `5` | Max concurrent scans |
| `GARAK_MAX_WORKERS` | `5` | Shield scanning parallelism |

## Deployment Modes

| Mode | Config File | Features |
|------|------------|----------|
| **Basic** | `run.yaml` | Standard scanning |
| **Enhanced** | `run-with-safety.yaml` | + Shield integration |
| **Remote** | `run-remote.yaml` | KFP execution |
| **Remote+Safety** | `run-remote-safety.yaml` | + Shield integration |

## Result Interpretation

- **Score Range**: 0.0 (secure) to 1.0 (vulnerable)
- **Threshold**: Scores ≥ 0.5 indicate security concerns
- **Reports**: Available in JSON, HTML, and log formats via Files API

## Examples & Demos

| Notebook | Description |
|----------|-------------|
| [01-getting_started](demos/01-getting_started_with_garak.ipynb) | Basic usage and custom probes |
| [02-scan_with_shields](demos/02-scan_with_shields.ipynb) | Shield integration testing |
| [03-remote_garak](demos/03-remote_garak.ipynb) | KFP remote execution |
