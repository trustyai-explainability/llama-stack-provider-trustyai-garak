# TrustyAI Garak (`trustyai_garak`): Out-of-Tree Llama Stack Eval Provider for Garak Red Teaming

## About
This repository implements [Garak](https://github.com/NVIDIA/garak) as a Llama Stack out-of-tree provider for **security testing and red teaming** of Large Language Models with optional **Shield Integration** for enhanced security testing. Please find the tutorial [here](https://trustyai.org/docs/main/red-teaming-introduction) to get started.

## What It Does

- **Automated Security Testing**: Detects prompt injection, jailbreaks, toxicity, and bias vulnerabilities
- **Compliance Scanning**: OWASP LLM Top 10, AVID taxonomy benchmarks
- **Shield Testing**: Compare LLM security with/without guardrails
- **Scalable Deployment**: Local or Kubernetes/Kubeflow execution
- **Comprehensive Reporting**: JSON, HTML, and detailed logs with vulnerability scores (0.0-1.0)

## Installation

### Production (Remote Execution - Default)

```bash
git clone https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak.git
cd llama-stack-provider-trustyai-garak
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the **remote provider** by default, which executes scans on Kubernetes/Kubeflow Pipelines. This is the recommended mode for production deployments with lightweight dependencies.

### Development (With Inline Execution)

```bash
# Install with inline provider for local development/testing
pip install -e ".[inline]"
```

This adds support for **inline execution** (local scans), which requires heavier dependencies including `garak` and `langchain`.

## Quick Start

### 1. Configure Environment

```bash
# Model serving endpoint
export VLLM_URL="http://your-model-endpoint/v1"
export INFERENCE_MODEL="your-model-name"

# Llama Stack endpoint (for inline: local, for remote: accessible from KFP pods)
export LLAMA_STACK_URL="http://localhost:8321"
```

### 2. Start Server

```bash
# Inline mode - local scanning (requires [inline] extra)
llama stack run run.yaml

# Inline mode with shields (requires [inline] extra)
llama stack run run-with-safety.yaml

# Remote mode - Kubernetes/KFP (default install)
llama stack run run-remote.yaml

# Remote mode with shields (default install)
llama stack run run-remote-safety.yaml
```

Server runs at `http://localhost:8321`

### 3. Run Security Scan

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Quick 5-minute scan
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
status = client.alpha.eval.jobs.status(job_id=job.job_id, benchmark_id="trustyai_garak::quick")
print(f"Status: {status.status}")

# Get results when complete
if status.status == "completed":
    results = client.alpha.eval.get_eval_job_result(job_id=job.job_id, benchmark_id="trustyai_garak::quick")
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
job_id = job.job_id
status = client.alpha.eval.jobs.status(job_id=job_id, benchmark_id="trustyai_garak::quick")

# File IDs are in metadata (for remote: prefixed with job_id)
scan_report_id = status.metadata.get(f"{job_id}_scan.report.jsonl") or status.metadata.get("scan.report.jsonl")
scan_html_id = status.metadata.get(f"{job_id}_scan.report.html") or status.metadata.get("scan.report.html")

# Download via Files API
content = client.files.content(scan_report_id)

# Or via HTTP
import requests
report = requests.get(f"http://localhost:8321/v1/files/{scan_html_id}/content")
```

## Remote Execution (Kubernetes/KFP)

### Setup

```bash
# Llama Stack URL (must be accessible from Kubeflow pods - use ngrok if local)
export KUBEFLOW_LLAMA_STACK_URL="https://your-llama-stack-url.ngrok.io"

# Kubeflow Configuration
export KUBEFLOW_PIPELINES_ENDPOINT="https://your-kfp-endpoint"
export KUBEFLOW_NAMESPACE="your-namespace"
export KUBEFLOW_BASE_IMAGE="quay.io/rh-ee-spandraj/trustyai-lls-garak-provider-dsp:latest"
export KUBEFLOW_PIPELINES_TOKEN=""  # Optional: If not set, uses kubeconfig

# Start server
llama stack run run-remote.yaml
```

**Important Notes:**
- For remote execution, `KUBEFLOW_LLAMA_STACK_URL` must be accessible from KFP pods. If running locally, use [ngrok](https://ngrok.com/)
- Results are stored via the configured Files API provider (S3, LocalFS, GCS, etc.)
- Both server and KFP pods access the same Files API backend automatically

### Usage

```python
# Same API, runs as KFP pipeline
job = client.alpha.eval.run_eval(benchmark_id="trustyai_garak::owasp_llm_top10", ...)

# Monitor pipeline
status = client.alpha.eval.jobs.status(job_id=job.job_id, benchmark_id="trustyai_garak::owasp_llm_top10")
print(f"KFP Run ID: {status.metadata['kfp_run_id']}")
```

## Configuration Reference

### Inline Provider Config (`run.yaml`)

```yaml
providers:
  eval:
    - provider_id: trustyai_garak
      provider_type: inline::trustyai_garak
      config:
        llama_stack_url: ${env.LLAMA_STACK_URL:=http://localhost:8321}
        timeout: ${env.GARAK_TIMEOUT:=10800}
        max_concurrent_jobs: ${env.GARAK_MAX_CONCURRENT_JOBS:=5}
        max_workers: ${env.GARAK_MAX_WORKERS:=5}
```

### Remote Provider Config (`run-remote.yaml`)

```yaml
providers:
  eval:
    - provider_id: trustyai_garak_remote
      provider_type: remote::trustyai_garak
      config:
        llama_stack_url: ${env.KUBEFLOW_LLAMA_STACK_URL}
        timeout: ${env.GARAK_TIMEOUT:=10800}
        kubeflow_config:
          pipelines_endpoint: ${env.KUBEFLOW_PIPELINES_ENDPOINT}
          namespace: ${env.KUBEFLOW_NAMESPACE}
          base_image: ${env.KUBEFLOW_BASE_IMAGE}
          pipelines_api_token: ${env.KUBEFLOW_PIPELINES_TOKEN:=}
  
  # Files provider (S3, LocalFS, or any other backend)
  files:
    - provider_id: s3
      provider_type: remote::s3
      config:
        bucket_name: ${env.S3_BUCKET_NAME}
        region: ${env.AWS_DEFAULT_REGION:=us-east-1}
        # ... S3 configuration
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_STACK_URL` | `http://localhost:8321/v1` | Llama Stack API URL (inline mode) |
| `KUBEFLOW_LLAMA_STACK_URL` | - | Llama Stack URL accessible from KFP pods (remote mode) |
| `GARAK_TIMEOUT` | `10800` | Max scan timeout (seconds) |
| `GARAK_MAX_CONCURRENT_JOBS` | `5` | Max concurrent scans (inline only) |
| `GARAK_MAX_WORKERS` | `5` | Shield scanning parallelism |
| `GARAK_SCAN_DIR` | `/tmp/.cache/llama_stack_garak_scans` | Directory for scan files (must be writable) |
| `XDG_CACHE_HOME` | `/tmp/.cache` | XDG cache directory (auto-configured) |

## Deployment Modes

| Mode | Config File | Provider Type | Dependencies |
|------|------------|---------------|--------------|
| **Inline** | `run.yaml` | `inline::trustyai_garak` | Requires `[inline]` extra |
| **Inline+Safety** | `run-with-safety.yaml` | `inline::trustyai_garak` | Requires `[inline]` extra |
| **Remote** (Default) | `run-remote.yaml` | `remote::trustyai_garak` | Default install |
| **Remote+Safety** | `run-remote-safety.yaml` | `remote::trustyai_garak` | Default install |

## Architecture

### Remote-First Design

The provider uses a **remote-first architecture**:

- **Default Install**: Lightweight, includes only KFP dependencies
- **Remote Provider**: No garak on server, runs scans in Kubernetes pods
- **Files API Integration**: Portable across any Files backend (S3, LocalFS, GCS)
- **Automatic Configuration**: XDG directories auto-configured for writable paths


## Result Interpretation

- **Score Range**: 0.0 (secure) to 1.0 (vulnerable)
- **Threshold**: Scores ≥ 0.5 indicate security concerns
- **Reports**: Available in JSON, HTML, and log formats via Files API

## Installation Options

| Install Command | Providers Available | Use Case |
|----------------|--------------------|-----------| 
| `pip install -e .` | Remote only | Production (default) |
| `pip install -e ".[inline]"` | Remote + Inline | Development/Testing |

## Examples & Demos

| Notebook | Description |
|----------|-------------|
| [01-getting_started](demos/01-getting_started_with_garak.ipynb) | Basic usage and custom probes (inline mode) |
| [02-scan_with_shields](demos/02-scan_with_shields.ipynb) | Shield integration testing (inline mode) |
| [03-remote_garak](demos/03-remote_garak.ipynb) | KFP remote execution |
