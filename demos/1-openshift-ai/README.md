# Deployment 1: Total Remote (OpenShift AI, Production Path)

This guide is the complete end-to-end setup for a production-style deployment:

- Llama Stack server runs in-cluster as `LlamaStackDistribution`
- Garak scans run in-cluster via KFP (DSP v2)

## Architecture

| Component | Runs On |
|---|---|
| Llama Stack server | OpenShift AI / ODH cluster |
| Garak scan execution | KFP (Data Science Pipelines) |

## Before You Start

- OpenShift cluster available
- Red Hat OpenShift AI (RHOAI) installed
- Llama Stack Operator enabled (in `DataScienceCluster`, `llamastackoperator` set to `Managed`)
- `oc` CLI access with permissions to create/update namespace resources
- model endpoint details ready (`VLLM_URL`, `INFERENCE_MODEL`, optional API token)

## Manifest Inventory

All manifests used here come from [`lsd_remote/`](../../lsd_remote/).

## 1) Login and Create Namespace

```bash
oc login --token=<token> --server=<server>

export NS=tai-garak-lls
oc create namespace "$NS"
oc project "$NS"
```

## 2) Install Data Science Pipelines (KFP / DSP v2)

```bash
oc apply -f lsd_remote/kfp-setup/kfp.yaml
```

Wait until DSP pods are running, then capture endpoint:

```bash
oc get pods | grep -E "dspa|ds-pipeline"
export KFP_ENDPOINT="https://$(oc get routes ds-pipeline-dspa -o jsonpath='{.spec.host}')"
echo "$KFP_ENDPOINT"
```

## 3) Prepare Manifests for Your Namespace/Environment

Update all hardcoded placeholders (especially namespace `tai-garak-lls`) in:

- [lsd_remote/postgres-complete-deployment.yaml](../../lsd_remote/postgres-complete-deployment.yaml)
- [lsd_remote/llama_stack_distro-setup/lsd-config.yaml](../../lsd_remote/llama_stack_distro-setup/lsd-config.yaml)
- [lsd_remote/llama_stack_distro-setup/lsd-secrets.yaml](../../lsd_remote/llama_stack_distro-setup/lsd-secrets.yaml)
- [lsd_remote/llama_stack_distro-setup/lsd-garak.yaml](../../lsd_remote/llama_stack_distro-setup/lsd-garak.yaml)
- [lsd_remote/llama_stack_distro-setup/lsd-role.yaml](../../lsd_remote/llama_stack_distro-setup/lsd-role.yaml)

### Required values in `lsd-config.yaml`

Set these carefully:

- `POSTGRES_HOST` to `<postgres-service-name>.<namespace>.svc.cluster.local`
- `VLLM_URL`, `INFERENCE_MODEL`, `VLLM_TLS_VERIFY`
- `KUBEFLOW_PIPELINES_ENDPOINT` to `$KFP_ENDPOINT`
- `KUBEFLOW_NAMESPACE` to your namespace
- `KUBEFLOW_GARAK_BASE_IMAGE` is set to 'quay.io/opendatahub/odh-trustyai-garak-lls-provider-dsp:dev' (you can also use 'quay.io/rhoai/odh-trustyai-garak-lls-provider-dsp-rhel9:rhoai-3.4' if you have access)
- `KUBEFLOW_LLAMA_STACK_URL` to `http://<lsd-name>-service.<namespace>.svc.cluster.local:8321`
- optional: add `KUBEFLOW_EXPERIMENT_NAME` (for example `trustyai-garak-prod`) if you want runs grouped under a specific KFP experiment name. Defaults to "trustyai-garak" if not provided

### Required values in `lsd-secrets.yaml`

- set namespace
- set `VLLM_API_TOKEN` when required by your model endpoint

### Required values in `lsd-garak.yaml`

- set namespace
- confirm distribution image
- ensure config/secret refs match names from `lsd-config.yaml` and `lsd-secrets.yaml`

### Required values in `lsd-role.yaml`

- set namespace
- verify role name (`ds-pipeline-dspa`) matches your DSP install
- verify service account name (`<lsd-name>-sa`, default in this repo is `llamastack-garak-distribution-sa`)


## 4) Deploy PostgreSQL

```bash
oc apply -f lsd_remote/postgres-complete-deployment.yaml
```

Verify:

```bash
oc get pods | grep postgres
oc get svc postgres
```

Production note: the provided manifest currently mounts `emptyDir` for PostgreSQL data in the Deployment. Replace it with a PVC-backed mount if you need durable storage across pod restarts.

## 5) Deploy Llama Stack + Garak Configuration

Apply in this order:

```bash
oc apply -f lsd_remote/llama_stack_distro-setup/lsd-config.yaml
oc apply -f lsd_remote/llama_stack_distro-setup/lsd-secrets.yaml
oc apply -f lsd_remote/llama_stack_distro-setup/lsd-garak.yaml
oc apply -f lsd_remote/llama_stack_distro-setup/lsd-role.yaml
```

Why this order:

- config and secrets must exist before the distribution starts
- role binding should be ready before pipeline operations

## 6) Verify End-to-End Health

```bash
oc get pods
oc get llamastackdistribution
oc get svc | grep llamastack
oc get routes -A | grep -i ds-pipeline
```

Inspect logs if needed:

```bash
oc logs deploy/postgres
oc describe llamastackdistribution llamastack-garak-distribution
oc get pods | grep llamastack
```

## 7) Choose Access URL for Client/Notebook

Use one of:

- in-cluster service URL (from a Data Science Workbench in same cluster):  
  `http://<lsd-service>.<namespace>.svc.cluster.local:8321`
- local machine with port-forward:

```bash
oc port-forward svc/llamastack-garak-distribution-service 8321:8321
```
- OpenShift route URL (external access without port-forward):

```bash
# list existing routes
oc get routes | grep -i llamastack

# if needed, expose a route for the service
oc expose svc/llamastack-garak-distribution-service --name=llamastack-garak-route
```

Then set:

- `BASE_URL="http://localhost:8321"` (if port-forwarding), or
- `BASE_URL="<in-cluster-service-url>"`, or
- `BASE_URL="https://<route>"` (use `http://` if your route is configured without TLS)

## 8) Run the Canonical Feature Demo

Open `demos/guide.ipynb` and run it end-to-end.

## Troubleshooting

### KFP run creation fails with authorization errors

- verify `lsd-role.yaml` is applied and points to the correct service account
- if your environment needs token auth, uncomment and populate the optional Kubeflow token secret in `lsd-secrets.yaml` and corresponding env var in `lsd-garak.yaml`

### Llama Stack cannot reach PostgreSQL

- verify `postgres` service exists in same namespace
- verify `POSTGRES_HOST` in `lsd-config.yaml`
- verify postgres secret keys (`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`)

### KFP jobs cannot call Llama Stack URL

- verify `KUBEFLOW_LLAMA_STACK_URL` resolves from inside cluster
- verify service name/port in `lsd-garak.yaml` matches URL configured in `lsd-config.yaml`
