# Deployment 2: Partial Remote (Hybrid)

**Server runs locally, scans run on OpenShift AI KFP** - great for development.

## What Runs Where

| Component | Runs On |
|-----------|---------|
| Llama Stack Server | Your laptop (local) |
| Garak Scans | Data Science Pipelines (KFP) on OpenShift AI |

## Prerequisites

- Python 3.12+ with package installed
- OpenShift AI cluster with Data Science Pipelines
- A VLLM hosted Model. You can easily get one using [LiteMaas](https://litemaas-litemaas.apps.prod.rhoai.rh-aiservices-bu.com/home).

## Quick Start

### 1. Install

```bash
# Remote provider is default, no extra needed
pip install lama-stack-provider-trustyai-garak
```

### 2. Configure Environment

```bash
# Model endpoint
export VLLM_URL="http://your-model-endpoint/v1"
export VLLM_API_TOKEN="<token identifier>"
export INFERENCE_MODEL="your-model-name"

# Llama Stack URL (must be accessible from KFP pods)
# Use ngrok to create an external route (see below section to get this url)
export KUBEFLOW_LLAMA_STACK_URL="https://your-ngrok-url.ngrok.io"

# Kubeflow configuration
export KUBEFLOW_PIPELINES_ENDPOINT="https://your-dsp-pipeline-endpoint" # (`kubectl get routes -A | grep -i pipeline`)
export KUBEFLOW_NAMESPACE="your-namespace"
export KUBEFLOW_BASE_IMAGE="quay.io/rh-ee-spandraj/trustyai-lls-garak-provider-dsp:latest"
```

### 3. Set Up Ngrok (If Running Locally)

```bash
# Install ngrok: https://ngrok.com/download
# Start tunnel
ngrok http 8321

# Copy the HTTPS URL to KUBEFLOW_LLAMA_STACK_URL
export KUBEFLOW_LLAMA_STACK_URL="https://xxxx-xx-xx-xx-xx.ngrok.io"
```

### 4. Start Server

```bash
llama stack run configs/partial-remote.yaml
```

### 5. Run Security Scan with Shields

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Register benchmark with shield testing
client.benchmarks.register(
    benchmark_id="custom_with_shield",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id="trustyai_garak_remote",
    provider_benchmark_id="custom_with_shield",
    metadata={
        "probes": ["promptinject.HijackHateHumans"],
        "shield_ids": ["Prompt-Guard-86M"],  # Test shield effectiveness
        "timeout": 600
    }
)

# Run scan
job = client.alpha.eval.run_eval(
    benchmark_id="custom_with_shield",
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "your-model-name",
            "sampling_params": {"max_tokens": 100}
        }
    }
)

# Monitor KFP pipeline
```

## Troubleshooting

**KFP can't reach server:**
- Ensure ngrok is running
- Check `KUBEFLOW_LLAMA_STACK_URL` uses accessbile URL
- Test: `curl $KUBEFLOW_LLAMA_STACK_URL/v1/files`

**Authentication errors:**
```bash
# Verify oc login
oc whoami
oc project your-namespace
```

## Next Steps

See `demo.ipynb` for complete examples with shield testing.

