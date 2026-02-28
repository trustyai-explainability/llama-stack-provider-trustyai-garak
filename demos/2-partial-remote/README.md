# Deployment 2: Partial Remote (Hybrid)

Llama Stack runs locally, while Garak scans run remotely on KFP.

This is usually the best developer workflow:

- quick local iteration on server config
- realistic remote pipeline execution for scans

## What Runs Where

| Component | Runs On |
|---|---|
| Llama Stack server | Your local machine |
| Garak scans | KFP (DSP v2) in cluster |

## Cluster Resource Scope

- Partial remote requires KFP resources only.
- You do not need to deploy full remote Llama Stack Distro/Postgres stack for this mode.
- If your cluster has no DSP yet, use `lsd_remote/kfp-setup/kfp.yaml`.

## Prerequisites

- Python 3.12+
- Local access to run `llama stack`
- OpenShift/Kubernetes cluster with DSP endpoint
- Model endpoint (for example vLLM)

## 1) Install

```bash
pip install llama-stack-provider-trustyai-garak
```
## 2) Expose Local Server URL to KFP

If running locally, expose port 8321 to the cluster (for example ngrok).

```bash
# Install ngrok: https://ngrok.com/download
# Start tunnel
ngrok http 8321

# Copy the HTTPS URL to KUBEFLOW_LLAMA_STACK_URL in the below step
```


## 3) Configure Environment

```bash
# Model endpoint
export VLLM_URL="http://your-model-endpoint/v1"
export VLLM_API_TOKEN="<token>"

# Local Llama Stack URL reachable from KFP pods
# (route/ngrok/public endpoint)
export KUBEFLOW_LLAMA_STACK_URL="https://your-public-url"

# KFP configuration
export KUBEFLOW_PIPELINES_ENDPOINT="https://your-dsp-endpoint" # echo “https://$(oc get routes ds-pipeline-dspa -o jsonpath='{.spec.host}')”
export KUBEFLOW_NAMESPACE="your-namespace"
export KUBEFLOW_GARAK_BASE_IMAGE="quay.io/opendatahub/odh-trustyai-garak-lls-provider-dsp:dev" # quay.io/rhoai/odh-trustyai-garak-lls-provider-dsp-rhel9:rhoai-3.4 (if you have access)
export KUBEFLOW_EXPERIMENT_NAME="trustyai-garak" # optional: group runs under a custom KFP experiment name

# By default this will be loaded from your local machine (login to your cluster beforehand). Or you can manually provide token that has permissions to CRUD KFP
export KUBEFLOW_PIPELINES_TOKEN="<token>"
```

`KUBEFLOW_EXPERIMENT_NAME` provider defaults to `trustyai-garak` if not provided

## 4) Start Local Server

```bash
llama stack run partial-remote.yaml # (or partial-remote-shield.yaml which is a sample config with local ollama llama-guard as a shield)
```

## 5) Register and Run Benchmark 

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Discover provider id
garak_provider = next(
    p for p in client.providers.list()
    if p.provider_type.endswith("trustyai_garak")
)
garak_provider_id = garak_provider.provider_id

client.benchmarks.register(
    benchmark_id="custom_with_shield",
    dataset_id="garak",
    scoring_functions=["garak_scoring"],
    provider_id=garak_provider_id,
    provider_benchmark_id="custom_with_shield",
    metadata={
        "garak_config": {
            "plugins": {
                "probe_spec": ["promptinject.HijackHateHumans"]
            }
        },
        "shield_ids": ["llama-guard"],
        "timeout": 600
    }
)

job = client.alpha.eval.run_eval(
    benchmark_id="custom_with_shield",
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": "your-model-id",
            "sampling_params": {"max_tokens": 100}
        }
    }
)
```

## 6) Check Results

```python
status = client.alpha.eval.jobs.status(job_id=job.job_id, benchmark_id="custom_with_shield")
if status.status == "completed":
    result = client.alpha.eval.jobs.retrieve(job_id=job.job_id, benchmark_id="custom_with_shield")
    overall = result.scores.get("_overall")
    print(overall.aggregated_results if overall else {})
```

## Troubleshooting

### KFP cannot reach local server

- verify route/ngrok URL is correct and active
- verify `KUBEFLOW_LLAMA_STACK_URL` is reachable from cluster
- test with `curl $KUBEFLOW_LLAMA_STACK_URL/v1/files`

### Pipeline auth failures

```bash
oc login --token=<your-token> --server=<your-server>
oc whoami
oc project <your-namespace>
```

Then either:

- grant service account permissions, or
- provide `KUBEFLOW_PIPELINES_TOKEN`

## Canonical Demo

Use `demos/guide.ipynb` for the full walkthrough (predefined listing, custom `garak_config`, benchmark deep-merge updates, shield mapping, `_overall`, and TBSA).

