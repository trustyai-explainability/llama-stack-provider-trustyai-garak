# TrustyAI Garak (`trustyai_garak`): Out-of-Tree Llama Stack Eval Provider for Garak Red Teaming

## About
This repository implements [Garak](https://github.com/NVIDIA/garak) as a Llama Stack out-of-tree provider for **security testing and red teaming** of Large Language Models with optional **Shield Integration** for enhanced security testing.

## Features
- **Security Vulnerability Detection**: Automated testing for prompt injection, jailbreaks, toxicity, and bias
- **Shield Integration**: Test LLMs with and without Llama Stack shields for comparative security analysis
- **Concurrency Control**: Configurable limits for concurrent scans and shield operations
- **Multiple Attack Categories**: 
  - 🔴 **Prompt Injection**: Tests for input manipulation vulnerabilities
  - 🔴 **Jailbreak**: Attempts to bypass safety guardrails
  - 🔴 **Toxicity**: Detects harmful content generation
  - 🔴 **Bias**: Evaluates discriminatory outputs
- **Flexible Scan Profiles**: Quick, standard, and comprehensive security assessments
- **Custom Probe Support**: Run specific garak security probes
- **Enhanced Reporting**: Multiple garak output formats including HTML reports and detailed logs

## Quick Start

### Prerequisites
- Python 3.12+
- Access to an OpenAI-compatible model endpoint

### Installation
```bash
# Clone the repository
git clone https://github.com/saichandrapandraju/llama-stack-provider-trustyai-garak.git
cd llama-stack-provider-trustyai-garak

# Create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Configuration
Set up your environment variables:
```bash
export VLLM_URL="http://your-model-endpoint/v1"
export INFERENCE_MODEL="your-model-name"

# Optional: Configure scan behavior
export GARAK_TIMEOUT="10800"  # 3 hours default
export GARAK_MAX_CONCURRENT_JOBS="5"  # Max concurrent scans
export GARAK_MAX_WORKERS="5"  # Max workers for shield scanning
```

### Run Security Scans

#### Basic Mode (Standard Garak Scanning)
```bash
# Start the Llama Stack server
llama stack run run.yaml --image-type venv

# The server will be available at http://localhost:8321
```

#### Enhanced Mode (With Shield Integration)
```bash
# Start with safety and shield capabilities
llama stack run run-with-safety.yaml --image-type venv

# Includes safety, shields, and telemetry APIs
```

## Demos
Interactive examples are available in the `demos/` directory:

- **[Getting Started](demos/01-getting_started_with_garak.ipynb)**: Basic usage with predefined scan profiles and user-defined garak probes
- **[Scan Guardrailed System](demos/02-scan_with_shields.ipynb)**: Llama Stack shield integration for scanning guardrailed LLM system
- **[concurrency_limit_test.ipynb](demos/concurrency_limit_test.ipynb)**: Testing concurrent scan limits

## Scan Profiles

| Profile | Duration | Probes | Use Case |
|---------|----------|---------|----------|
| **Quick** | ~5 min | Essential security checks (3 specific probes) | Development testing |
| **Standard** | ~1 hours | Standard attack vectors (5 probe categories) | Pre-deployment validation |
| **Comprehensive** | >5 hours | All available probes related to Bias, Toxicity, Prompt Injection, and Jailbreak attacks (14 probe categories) | Full security audit |

### Scan Profile Details

#### Quick Profile
- `continuation.ContinueSlursReclaimedSlurs` (Bias)
- `promptinject.HijackHateHumans` (Prompt Injection)
- `realtoxicityprompts.RTPProfanity` (Toxicity)

#### Standard Profile
- `continuation` (Bias)
- `dan` (Jailbreak)
- `encoding` (Prompt Injection)
- `promptinject` (Prompt Injection)
- `realtoxicityprompts` (Toxicity)

#### Comprehensive Profile
- `continuation` (Bias)
- `dan` (Jailbreak)
- `donotanswer` (Toxicity)
- `encoding` (Prompt Injection)
- `exploitation` (Prompt Injection)
- `glitch` (Prompt Injection)
- `goodside` (Prompt Injection)
- `grandma` (Toxicity)
- `latentinjection` (Prompt Injection)
- `lmrc` (Toxicity)
- `promptinject` (Prompt Injection)
- `realtoxicityprompts` (Toxicity)
- `suffix` (Jailbreak)
- `tap` (Jailbreak)

## Usage Examples

### Built-in Scan Profiles
```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Register a quick security scan (5 min)
client.benchmarks.register(
    benchmark_id="trustyai_garak::quick",
    dataset_id="trustyai_garak::quick",
    scoring_functions=["string"],
    provider_benchmark_id="quick",
    provider_id="trustyai_garak",
)

# Run evaluation
job = client.eval.run_eval(
    benchmark_id="trustyai_garak::quick",
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "qwen2", # change this to your inference model name
            "provider_id": "trustyai_garak",
            "sampling_params": {},
        }
     },
)
```

### Custom Garak Probes
```python
# Register custom probes
client.benchmarks.register(
    benchmark_id="trustyai_garak::custom",
    dataset_id="trustyai_garak::custom",
    scoring_functions=["string"],
    provider_benchmark_id="custom",
    provider_id="trustyai_garak",
    metadata={
        "probes": ["promptinject.HijackHateHumans", "dan.Dan_11_0"],
        "timeout": 600  # 10 minutes
    }
)
```

### Shield Integration (Enhanced Mode)
```python
# Test with input shields only
client.benchmarks.register(
    benchmark_id="trustyai_garak::with_input_shield",
    dataset_id="trustyai_garak::with_input_shield",
    scoring_functions=["string"],
    provider_benchmark_id="with_input_shield",
    provider_id="trustyai_garak",
    metadata={
        "probes": ["promptinject.HijackHateHumans"],
        "timeout": 600,
        "shield_ids": ["Prompt-Guard-86M"]  # Applied to input only
    }
)

# Test with separate input/output shields
client.benchmarks.register(
    benchmark_id="trustyai_garak::with_io_shields",
    dataset_id="trustyai_garak::with_io_shields",
    scoring_functions=["string"],
    provider_benchmark_id="with_io_shields",
    provider_id="trustyai_garak",
    metadata={
        "probes": ["promptinject.HijackHateHumans"],
        "timeout": 600,
        "shield_config": {
            "input": ["Prompt-Guard-86M"],
            "output": ["Llama-Guard-3-8B"]
        }
    }
)
```

### Job Management
```python
# Check job status
job_status = client.eval.jobs.status(job_id=job.job_id, benchmark_id="trustyai_garak::quick")
print(f"Job status: {job_status.status}")
print(f"Running jobs: {job_status.metadata.get('running_jobs', 'N/A')}")

# Cancel a running job
client.eval.jobs.cancel(job_id=job.job_id, benchmark_id="trustyai_garak::quick")

# Get evaluation results
if job_status.status == "completed":
    results = client.eval.get_eval_job_result(job_id=job.job_id, benchmark_id="trustyai_garak::quick")
```

### Accessing Scan Reports
```python
# Get file metadata
scan_report_id = job_status.metadata["scan_report_file_id"]
scan_log_id = job_status.metadata["scan_log_file_id"]
scan_html_id = job_status.metadata["scan_report_html_file_id"]

# Download files using Files API or direct HTTP calls
import requests
files_url = f"http://localhost:8321/v1/openai/v1/files"
report_content = requests.get(f"{files_url}/{scan_report_id}/content")
```

## Configuration Options

### Provider Configuration (`run.yaml` / `run-with-safety.yaml`)

```yaml
providers:
  eval:
    - provider_id: trustyai_garak
      provider_type: inline::trustyai_garak
      config:
        base_url: ${env.BASE_URL:=http://localhost:8321/v1} # llama-stack service base url
        timeout: ${env.GARAK_TIMEOUT:=10800}  # 3 hours default
        max_concurrent_jobs: ${env.GARAK_MAX_CONCURRENT_JOBS:=5}  # Concurrent scan limit
        max_workers: ${env.GARAK_MAX_WORKERS:=5}  # Shield scanning parallelism
        garak_model_type_openai: "openai.OpenAICompatible"  # For direct LLM scanning
        garak_model_type_function: "function.Single"  # For shield-integrated scanning
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_URL` | `http://localhost:8321/v1` | Llama Stack service base URL |
| `GARAK_TIMEOUT` | `10800` | Maximum timeout for garak scan (seconds) |
| `GARAK_MAX_CONCURRENT_JOBS` | `5` | Maximum concurrent garak scans |
| `GARAK_MAX_WORKERS` | `5` | Maximum workers for shield scanning |


## Results Interpretation

- **Vulnerability Score**: 0.0 (secure) to 1.0 (vulnerable)
- **Threshold**: Scores ≥ 0.5 indicate potential security concerns
- **Shield Comparison**: Compare scores with and without shields to measure effectiveness
- **Detailed Reports**: Available via Llama Stack `files` API in multiple formats:
  - JSON report (`scan_report_file_id`)
  - HTML report (`scan_report_html_file_id`)
  - Detailed logs (`scan_log_file_id`)
  - Hit logs (`scan_hitlog_file_id`)

## Deployment Modes

### Basic Mode (`run.yaml`)
- Standard garak scanning against OpenAI-compatible endpoints
- APIs: `inference`, `eval`, `files`
- Best for: Basic security testing

### Enhanced Mode (`run-with-safety.yaml`)
- Shield-integrated scanning to test Guardrailed systems
- APIs: `inference`, `eval`, `files`, `safety`, `shields`, `telemetry`
- Best for: Advanced security testing with defense evaluation
