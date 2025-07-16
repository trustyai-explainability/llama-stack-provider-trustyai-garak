# TrustyAI Garak (`trustyai_garak`): Out-of-Tree Llama Stack Eval Provider for Garak Red Teaming

## About
This repository implements [Garak](https://github.com/leondz/garak) as a Llama Stack out-of-tree provider for **security testing and red teaming** of Large Language Models. 

## Features
- **Security Vulnerability Detection**: Automated testing for prompt injection, jailbreaks, toxicity, and bias
- **Multiple Attack Categories**: 
  - 🔴 **Prompt Injection**: Tests for input manipulation vulnerabilities
  - 🔴 **Jailbreak**: Attempts to bypass safety guardrails
  - 🔴 **Toxicity**: Detects harmful content generation
  - 🔴 **Bias**: Evaluates discriminatory outputs
- **Flexible Scan Profiles**: Quick, standard, and comprehensive security assessments
- **Custom Probe Support**: Run specific garak security probe

## Quick Start

### Prerequisites
- Python 3.12+
- Access to an OpenAI-compatible model endpoint
- Installation of `uv`
- `llama-stack` CLI tool installed

### Installation
```bash
# Clone the repository
git clone https://github.com/saichandrapandraju/llama-stack-provider-trustyai-garak.git
cd llama-stack-provider-trustyai-garak

# Create & activate venv
uv venv .llamastack-venv
source .llamastack-venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Configuration
Set up your environment variables:
```bash
export VLLM_URL="http://your-model-endpoint/v1"
export INFERENCE_MODEL="your-model-name"
```

### Run Security Scans
```bash
# Start the Llama Stack server
llama stack run run.yaml --image-type venv

# The server will be available at http://localhost:8321
```

## Demo
Check out the [getting started notebook](demos/00-getting_started_with_garak.ipynb) for interactive examples.

## Scan Profiles

| Profile | Duration | Probes | Use Case |
|---------|----------|---------|----------|
| **Quick** | ~5 min | Essential security checks | Development testing |
| **Standard** | ~1 hours | Standard attack vectors | Pre-deployment validation |
| **Comprehensive** | ~5 hours | All available probes | Full security audit |


## Usage Examples

### Built-in Scan Profiles (quick, standard, comprehensive)
```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Register a quick security scan (5 min)
client.benchmarks.register(
    benchmark_id="trustyai_garak::quick",
    dataset_id="trustyai_garak::quick",
    scoring_functions=["string"],
    provider_benchmark_id="string",
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

### Other Garak probes outside built-in scan profile
```python
# Register other valid probes
client.benchmarks.register(
    benchmark_id="trustyai_garak::custom",
    dataset_id="trustyai_garak::custom",
    scoring_functions=["string"],
    provider_benchmark_id="string",
    provider_id="trustyai_garak",
    metadata={
        "probes": ["promptinject.HijackHateHumans", "dan.Dan_11_0"],
        "timeout": 600  # 10 minutes
    }
)
```

## Security Test Categories across pre-built Scan Profiles

- **Prompt Injection**: `promptinject`, `encoding`
- **Jailbreak**: `dan` (Do Anything Now attacks)
- **Toxicity**: `realtoxicityprompts` 
- **Bias**: `continuation` (discriminatory content)

## Configuration

The provider supports various configuration options in `run.yaml`:

```yaml
providers:
  eval:
    - provider_id: trustyai_garak
      provider_type: inline::trustyai_garak
      config:
        base_url: ${env.BASE_URL}
        timeout: ${env.TIMEOUT:10800}  # 3 hours default
```

## Results Interpretation

- **Vulnerability Score**: 0.0 (secure) to 1.0 (vulnerable)
- **Threshold**: Scores ≥ 0.5 indicate potential security concerns
- **Detailed Reports**: Can be accessed via Llama Stack `files` API
