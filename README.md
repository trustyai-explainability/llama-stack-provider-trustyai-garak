# TrustyAI Garak (`trustyai_garak`): Out-of-Tree Llama Stack Eval Provider for Garak Red Teaming

## About
This repository implements [Garak](https://github.com/NVIDIA/garak) as a Llama Stack out-of-tree provider for **security testing and red teaming** of Large Language Models with optional **Shield Integration** for enhanced security testing.

## Features
- **Security Vulnerability Detection**: Automated testing for prompt injection, jailbreaks, toxicity, and bias
- **Compliance Framework Support**: Pre-built benchmarks for established standards ([OWASP LLM Top 10](https://genai.owasp.org/llm-top-10/), [AVID taxonomy](https://docs.avidml.org/taxonomy/effect-sep-view))
- **Shield Integration**: Test LLMs with and without Llama Stack shields for comparative security analysis
- **Remote Execution with KFP**: Scale garak scanning via Kubeflow Pipelines
- **Concurrency Control**: Configurable limits for local concurrent scans and shield operations
- **Enhanced Reporting**: Multiple garak output formats including HTML reports and detailed logs

## Quick Start

### Prerequisites
- Python 3.12+
- Access to an OpenAI-compatible model endpoint

### Installation
```bash
# Clone the repository
git clone https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak.git
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
- **[Remote Execution](demos/03-remote_garak.ipynb)**: Running garak scans via Kubeflow Pipelines
- **[concurrency_limit_test.ipynb](demos/concurrency_limit_test.ipynb)**: Testing concurrent scan limits

## Compliance Frameworks

Pre-registered compliance framework benchmarks available immediately:

### Compliance Standards
| Framework | Benchmark ID | Description | Duration |
|-----------|--------------|--------------| --------|
| **[OWASP LLM Top 10](https://genai.owasp.org/llm-top-10/)** | `owasp_llm_top10` | OWASP Top 10 for Large Language Model Applications | ~8 hours |
| **[AVID Security](https://docs.avidml.org/taxonomy/effect-sep-view/security)** | `avid_security` | AI Vulnerability Database - Security vulnerabilities | ~8 hours |
| **[AVID Ethics](https://docs.avidml.org/taxonomy/effect-sep-view/ethics)** | `avid_ethics` | AI Vulnerability Database - Ethical concerns | ~30 minutes |
| **[AVID Performance](https://docs.avidml.org/taxonomy/effect-sep-view/performance)** | `avid_performance` | AI Vulnerability Database - Performance issues | ~40 minutes |

### Scan Profiles for Testing
| Profile | Benchmark ID | Duration | Probes |
|---------|--------------|----------|---------|
| **Quick** | `quick` | ~5 minutes | Essential security checks (3 specific probes) |
| **Standard** | `standard` | ~1 hour | Standard attack vectors (5 probe categories) |

_Note: All the above duration estimates are calculated with a Qwen2.5 7B model deployed via vLLM on Openshift._
## Usage Examples

### Discover Available Benchmarks
```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# List all available benchmarks (auto-registered)
benchmarks = client.benchmarks.list()
for benchmark in benchmarks.data:
    print(f"- {benchmark.identifier}: {benchmark.metadata.get('name', 'No name')}")
```

### Compliance Framework Testing
```python
# Run OWASP LLM Top 10 security assessment
job = client.eval.run_eval(
    benchmark_id="owasp_llm_top10",
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "qwen2", # change this to your inference model name
            "sampling_params": {
                "max_tokens": 100
            },
        }
     },
)

# Run AVID Security assessment
job = client.eval.run_eval(
    benchmark_id="avid_security",
    benchmark_config={
        "eval_candidate": {
            "type": "model", 
            "model": "qwen2",
            "sampling_params": {
                "max_tokens": 100
            },
        }
     },
)
```

### Built-in Scan Profiles for testing
```python
# Quick security scan (5 min)
job = client.eval.run_eval(
    benchmark_id="quick",
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "qwen2", # change this to your inference model name
            "sampling_params": {
                "max_tokens": 100
            },
        }
     },
)
```

### Custom Garak Probes
```python
# Register custom probes
client.benchmarks.register(
    benchmark_id="custom",
    dataset_id="garak", # placeholder
    scoring_functions=["garak_scoring"], # placeholder
    provider_benchmark_id="custom",
    provider_id="trustyai_garak",
    metadata={
        "probes": ["latentinjection.LatentJailbreak", "snowball.GraphConnectivity"],
        "timeout": 900  # 15 minutes
    }
)
```

### Shield Integration (Enhanced Mode)
```python
# Test with input shields only
client.benchmarks.register(
    benchmark_id="PI_with_input_shield",
    dataset_id="garak", # placeholder
    scoring_functions=["garak_scoring"], # placeholder
    provider_benchmark_id="PI_with_input_shield",
    provider_id="trustyai_garak",
    metadata={
        "probes": ["promptinject.HijackHateHumans"],
        "timeout": 600,
        "shield_ids": ["Prompt-Guard-86M"]  # Applied to input only
    }
)

# Test with separate input/output shields
client.benchmarks.register(
    benchmark_id="PI_with_io_shields",
    dataset_id="garak", # placeholder
    scoring_functions=["garak_scoring"], # placeholder
    provider_benchmark_id="PI_with_io_shields",
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
job_status = client.eval.jobs.status(job_id=job.job_id, benchmark_id="quick")
print(f"Job status: {job_status.status}")
print(f"Running jobs: {job_status.metadata.get('running_jobs', 'N/A')}")

# Cancel a running job
client.eval.jobs.cancel(job_id=job.job_id, benchmark_id="quick")

# Get evaluation results
if job_status.status == "completed":
    results = client.eval.get_eval_job_result(job_id=job.job_id, benchmark_id="quick")
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
report_content = requests.get(f"{files_url}/{scan_log_id}/content")

## with llama stack client
# client.files.content(scan_log_id)
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
  - JSON report
  - HTML report
  - Detailed logs
  - Hit logs

## Deployment Modes

### Basic Mode (`run.yaml`)
- Standard garak scanning against OpenAI-compatible endpoints
- APIs: `inference`, `eval`, `files`
- Best for: Basic security testing

### Enhanced Mode (`run-with-safety.yaml`)
- Shield-integrated scanning to test Guardrailed systems
- APIs: `inference`, `eval`, `files`, `safety`, `shields`, `telemetry`
- Best for: Advanced security testing with defense evaluation

### Remote Mode (`run-remote.yaml` / `run-remote-safety.yaml`)
- **Scalable KFP-based execution**: Run garak scans as Kubeflow Pipeline jobs
- **Cloud-native deployment**: Leverages Kubernetes for distributed scanning
- APIs: Same as basic/enhanced mode plus remote job orchestration
- Best for: Production environments, large-scale security assessments

#### Remote Provider Setup
```bash
# Set KFP environment variables
export KUBEFLOW_PIPELINES_ENDPOINT="https://your-kfp-endpoint"
export KUBEFLOW_NAMESPACE="your-namespace"
export KUBEFLOW_EXPERIMENT_NAME="trustyai-garak-scans"
export KUBEFLOW_BASE_IMAGE="quay.io/spandraj/trustyai-garak-provider:latest"

# Configure S3 for artifact storage
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_S3_BUCKET="pipeline-artifacts"
export AWS_S3_ENDPOINT="https://your-s3-endpoint"  # For MinIO

# Start with remote provider
llama stack run run-remote.yaml --image-type venv
```

#### Remote Scanning Example
```python
# Same API, now runs as KFP pipeline
job = client.eval.run_eval(
    benchmark_id="owasp_llm_top10",
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "qwen2",
            "sampling_params": {"max_tokens": 100}
        }
    }
)

# Monitor KFP run status
status = client.eval.jobs.status(job_id=job.job_id, benchmark_id="owasp_llm_top10")
print(f"KFP Run ID: {status.metadata['kfp_run_id']}")
```