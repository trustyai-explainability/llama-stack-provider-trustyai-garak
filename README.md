# TrustyAI Garak: LLM Red Teaming for Eval-Hub

Automated vulnerability scanning and red teaming for Large Language Models using [Garak](https://github.com/NVIDIA/garak). This package provides a `FrameworkAdapter` for the [eval-hub](https://github.com/opendatahub-io/eval-hub) evaluation platform, enabling garak scans to run as Kubernetes Jobs or via Kubeflow Pipelines.

## What It Does

- 🔍 **Vulnerability Assessment**: Red-team LLMs for prompt injection, jailbreaks, toxicity, bias and other vulnerabilities
- 📋 **Compliance**: OWASP LLM Top 10, AVID taxonomy benchmarks
- 🎯 **Intents-based Testing**: Policy taxonomy + SDG + TAPIntent for targeted risk assessment (KFP mode)
- ☁️ **Cloud-Native**: Runs on OpenShift AI / Kubernetes as eval-hub jobs
- 📊 **Detailed Reports**: JSONL, HTML, and AVID-format reports with MLflow integration

## Execution Modes

| Mode | How Garak Runs | Intents Support | Use Case |
|------|---------------|-----------------|----------|
| **Simple** | Directly in the eval-hub K8s Job pod | No | Standard scans |
| **KFP** | K8s Job submits to Kubeflow Pipelines, polls status | Yes | Intents/SDG workflows |

## Installation

```bash
# Core (eval-hub adapter with KFP support)
pip install llama-stack-provider-trustyai-garak

# With SDG support (for intents workflows)
pip install "llama-stack-provider-trustyai-garak[sdg]"

# Development
pip install "llama-stack-provider-trustyai-garak[dev]"
```

## Container Image

```bash
# Build the eval-hub adapter image
docker build -f Containerfile -t trustyai-garak:dev .
```

The container runs as:
```bash
# Simple mode (garak in same pod)
CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]

# KFP mode (garak in a separate KFP pod)
CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub.kfp_adapter"]
```

## Benchmark Profiles

Predefined scan profiles available via `benchmark_id`:

| Profile | Description |
|---------|-------------|
| `quick` | Single DAN probe for fast testing |
| `owasp_llm_top10` | OWASP Top 10 for LLM Applications |
| `avid` | AVID taxonomy — all vulnerabilities |
| `avid_security` | AVID — security vulnerabilities |
| `avid_ethics` | AVID — ethical concerns |
| `avid_performance` | AVID — performance issues |
| `quality` | Violence, profanity, toxicity, hate speech |
| `cwe` | Common Weakness Enumeration |
| `intents` | Intents-based risk assessment (KFP mode only) |

## Job Spec Configuration

The adapter reads a `JobSpec` from a mounted ConfigMap:

```json
{
  "id": "scan-001",
  "provider_id": "garak",
  "benchmark_id": "quick",
  "model": {
    "url": "https://my-model-endpoint.example.com/v1",
    "name": "my-model"
  },
  "parameters": {
    "probes": "dan.Dan_11_0",
    "execution_mode": "simple"
  }
}
```

## Parameters

Key `parameters` in the job spec:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `probes` | Comma-separated probe names | Profile default |
| `probe_tags` | Tag-based probe filtering | — |
| `execution_mode` | `simple` or `kfp` | `simple` |
| `timeout_seconds` | Scan timeout | Profile default |
| `eval_threshold` | Vulnerability threshold (0.0–1.0) | `0.5` |
| `model_type` | Garak generator type | `openai.OpenAICompatible` |
| `garak_config` | Full garak config dict (deep-merged onto profile) | — |

## Results

The adapter reports `EvaluationResult` per probe with:
- `attack_success_rate`: Percentage of successful attacks
- `vulnerable_responses`: Count of vulnerable responses
- `total_attempts`: Total probe attempts

Overall metrics include TBSA (Tier-Based Security Aggregate) when available.

## Development

```bash
make test        # Run all tests
make coverage    # With coverage report
make lint        # ruff check
make format      # ruff format
```

## Support & Documentation

- 📖 **Garak Docs**: https://reference.garak.ai/en/latest/index.html
- 💬 **Issues**: https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak/issues
