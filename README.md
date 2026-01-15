# TrustyAI Garak: LLM Red Teaming for Llama Stack

Automated vulnerability scanning and red teaming for Large Language Models using [Garak](https://github.com/NVIDIA/garak). This project implements garak as an external evaluation provider for [Llama Stack](https://llamastack.github.io/).

## What It Does

- 🔍 **Vulnerability Assessment**: Red Team LLMs for prompt injection, jailbreaks, toxicity, bias and other vulnerabilities
- 📋 **Compliance**: OWASP LLM Top 10, AVID taxonomy benchmarks  
- 🛡️ **Shield Testing**: Measure guardrail effectiveness
- ☁️ **Cloud-Native**: Runs on OpenShift AI / Kubernetes
- 📊 **Detailed Reports**: JSON and HTML reports

## Pick Your Deployment

| # | Mode | Server | Scans | Use Case | Guide |
|---|------|--------|-------|----------|-------|
| **1** | **Total Remote** | OpenShift AI | Data Science Pipelines | **Production** | [→ Setup](demos/1-openshift-ai/README.md) |
| **2** | **Partial Remote** | Local laptop | Data Science Pipelines | Development | [→ Setup](demos/2-partial-remote/README.md) |
| **3** | **Total Inline** | Local laptop | Local laptop | Testing only | [→ Setup](demos/3-local-inline/README.md) |


## Installation

```bash
# For Deployment 1 (Total remote)
## no installation needed! 

# For Deployment 2 (Partial remote)
pip install llama-stack-provider-trustyai-garak

# For Deployment 3 (local scans) - requires extra
pip install "llama-stack-provider-trustyai-garak[inline]"
```

## Quick Example

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Run security scan (5 minutes)
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

# Get results
if status.status == "completed":
    results = client.alpha.eval.get_eval_job_result(job_id=job.job_id, benchmark_id="trustyai_garak::quick")
```

## Available Benchmarks

| Benchmark ID | Tests | Duration |
|--------------|-------|----------|
| `trustyai_garak::owasp_llm_top10` | [OWASP Top 10](https://genai.owasp.org/llm-top-10/) | ~2 hrs |
| `trustyai_garak::avid_security` | [AVID Security](https://docs.avidml.org/taxonomy/effect-sep-view/security) | ~2 hrs |
| `trustyai_garak::avid_ethics` | AVID Ethics | ~10 min |
| `trustyai_garak::avid_performance` | AVID Performance | ~10 min |
| `trustyai_garak::quick` | 3 test probes | ~5 min |

Or register custom benchmarks with specific [Garak probes](https://reference.garak.ai/en/latest/probes.html).

## Shield Testing Example

```python
# Test how well guardrails (shields) block attacks
client.benchmarks.register(
    benchmark_id="with_shield",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id="trustyai_garak_remote",  # or trustyai_garak_inline
    provider_benchmark_id="with_shield",
    metadata={
        "probes": ["promptinject.HijackHateHumans"],
        "shield_ids": ["Prompt-Guard-86M"]  # Shield to test
    }
)

job = client.alpha.eval.run_eval(
    benchmark_id="with_shield",
    benchmark_config={"eval_candidate": {"type": "model", "model": "your-model"}}
)
```

Compare results with/without shields to measure effectiveness.

## Understanding Results

### Vulnerability Score
- **0.0** = Secure (model refused attack)
- **0.5** = Threshold (concerning)
- **1.0** = Vulnerable (model was compromised)

### Reports Available
Access via `job.metadata`:
- `scan.log`: Detailed log of this scan.
- `scan.report.jsonl`: Report containing information about each attempt (prompt) of each garak probe.
- `scan.hitlog.jsonl`: Report containing only the information about attempts that the model was found vulnerable to.
- `scan.avid.jsonl`: AVID (AI Vulnerability Database) format of `scan.report.jsonl`. You can find info about AVID [here](https://avidml.org/).
- `scan.report.html`: Visual representation of the scan. In remote mode, this is logged as a html artifact of the pipeline.

```python
# Download HTML report
html_id = job.metadata[f"{job.job_id}_scan.report.html"]
content = client.files.content(html_id)
with open("report.html", "w") as f:
    f.write(content)
```

## Support & Documentation

- 📚 **Tutorial**: https://trustyai.org/docs/main/red-teaming-introduction
- 💬 **Issues**: https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak/issues
- 🦙 **Llama Stack Docs**: https://llamastack.github.io/
- 📖 **Garak Docs**: https://reference.garak.ai/en/latest/index.html
