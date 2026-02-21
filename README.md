# TrustyAI Garak: LLM Red Teaming for Llama Stack

Automated vulnerability scanning and red teaming for Large Language Models using [Garak](https://github.com/NVIDIA/garak). This project implements garak as an external evaluation provider for [Llama Stack](https://llamastack.github.io/).

## What It Does

- 🔍 **Vulnerability Assessment**: Red Team LLMs for prompt injection, jailbreaks, toxicity, bias and other vulnerabilities
- 📋 **Compliance**: OWASP LLM Top 10, AVID taxonomy benchmarks  
- 🛡️ **Shield Testing**: Measure guardrail effectiveness
- ☁️ **Cloud-Native**: Runs on OpenShift AI / Kubernetes
- 📊 **Detailed Reports**: JSON and HTML reports

## Pick Your Deployment

| Mode | Llama Stack server | Garak scans | Typical use case | Guide |
|---|---|---|---|---|
| Total Remote | OpenShift/Kubernetes | KFP pipelines | Production | [→ Setup](demos/1-openshift-ai/README.md) |
| Partial Remote | Local machine | KFP pipelines | Development | [→ Setup](demos/2-partial-remote/README.md) |
| Total Inline | Local machine | Local machine | Fast local testing | [→ Setup](demos/3-local-inline/README.md) |

- Feature notebook: `demos/guide.ipynb`
- Metadata reference: `BENCHMARK_METADATA_REFERENCE.md`

## Installation

```bash
# For Deployment 1 (Total remote)
## no installation needed! 

# For Deployment 2 (Partial remote)
pip install llama-stack-provider-trustyai-garak

# For Deployment 3 (local scans) - requires extra
pip install "llama-stack-provider-trustyai-garak[inline]"
```

## Quick Workflow

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Discover Garak provider
garak_provider = next(
    p for p in client.providers.list()
    if p.provider_type.endswith("trustyai_garak")
)
garak_provider_id = garak_provider.provider_id

# List predefined benchmarks
benchmarks = client.alpha.benchmarks.list()
print([b.identifier for b in benchmarks if b.identifier.startswith("trustyai_garak::")])

# Run a predefined benchmark
benchmark_id = "trustyai_garak::quick"
job = client.alpha.eval.run_eval(
    benchmark_id=benchmark_id,
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "your-model-id",
            "sampling_params": {"max_tokens": 100},
        }
    },
)

# Poll status
status = client.alpha.eval.jobs.status(job_id=job.job_id, benchmark_id=benchmark_id)
print(status.status)

# Retrieve final result
if status.status == "completed":
    job_result = client.alpha.eval.jobs.retrieve(job_id=job.job_id, benchmark_id=benchmark_id)
```

## Custom Benchmark Schema

Use `metadata.garak_config` for Garak command configuration. Provider-level runtime parameters (for example `timeout`, `shield_ids`) stay at top-level metadata.

```python
client.alpha.benchmarks.register(
    benchmark_id="custom_promptinject",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id=garak_provider_id,
    provider_benchmark_id="custom_promptinject",
    metadata={
        "garak_config": {
            "plugins": {
                "probe_spec": ["promptinject"]
            },
            "reporting": {
                "taxonomy": "owasp"
            }
        },
        "timeout": 900
    }
)
```

## Update and Deep-Merge Behavior

- To create a tuned variant of a predefined (or existing custom) benchmark, set `provider_benchmark_id` to the predefined (or existing custom) benchmark ID and pass overrides in `metadata`.
- Provider metadata is deep-merged, so you can override only the parts you care about.
- Predefined benchmarks are comprehensive by design. For faster exploratory runs, lower `garak_config.run.soft_probe_prompt_cap` to reduce prompts per probe.

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
            "system": {"parallel_attempts": 20}
        },
        "timeout": 1200
    }
)
```

```python
# Faster (less comprehensive) variant of a predefined benchmark
client.alpha.benchmarks.register(
    benchmark_id="owasp_fast",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id=garak_provider_id,
    provider_benchmark_id="trustyai_garak::owasp_llm_top10",
    metadata={
        "garak_config": {
            "run": {"soft_probe_prompt_cap": 100}
        }
    }
)
```

## Shield Testing

Use either `shield_ids` (all treated as input shields) or `shield_config` (explicit input/output mapping).

```python
client.alpha.benchmarks.register(
    benchmark_id="with_shields",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id=garak_provider_id,
    provider_benchmark_id="with_shields",
    metadata={
        "garak_config": {
            "plugins": {"probe_spec": ["promptinject.HijackHateHumans"]}
        },
        "shield_config": {
            "input": ["Prompt-Guard-86M"],
            "output": ["Llama-Guard-3-8B"]
        },
        "timeout": 600
    }
)
```

## Understanding Results (`_overall` and TBSA)

`job_result.scores` contains:

- probe-level entries (for example `promptinject.HijackHateHumans`)
- synthetic `_overall` aggregate entry across all probes

`_overall.aggregated_results` can include:

- `total_attempts`
- `vulnerable_responses`
- `attack_success_rate`
- `probe_count`
- `tbsa` (Tier-Based Security Aggregate, 1.0 to 5.0, higher is better)
- `version_probe_hash`
- `probe_detector_pairs_contributing`

TBSA is derived from probe:detector pass-rate and z-score DEFCON grades with tier-aware aggregation and weighting, to give a more meaningful overall security posture than a plain pass/fail metric.

## Scan Artifacts

Access scan files from job metadata:

- `scan.log`
- `scan.report.jsonl`
- `scan.hitlog.jsonl`
- `scan.avid.jsonl`
- `scan.report.html`

Remote mode stores prefixed keys in metadata (for example `{job_id}_scan.report.html`).

## Notes on Remote Cluster Resources

- Partial remote mode needs KFP resources only.
- Total remote mode needs full stack resources (KFP, LlamaStackDistribution, RBAC, secrets, and Postgres manifests).
- See `lsd_remote/` for full reference manifests.

## Support & Documentation

- 📚 **Tutorial**: https://trustyai.org/docs/main/red-teaming-introduction
- 💬 **Issues**: https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak/issues
- 🦙 **Llama Stack Docs**: https://llamastack.github.io/
- 📖 **Garak Docs**: https://reference.garak.ai/en/latest/index.html
