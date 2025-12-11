# Deployment 1: Total Remote (OpenShift AI)

**Everything runs in OpenShift AI** - no local dependencies.

## What Runs Where

| Component | Runs On |
|-----------|---------|
| Llama Stack Server | OpenShift AI (LlamaStackDistribution) |
| Garak Scans | Data Science Pipelines (KFP) |

## Prerequisites
* OpenShift AI or Open Data Hub installed on your OpenShift Cluster
* Data Science Pipeline Server configured
* Llama Stack Operator installed. Follow [this](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html-single/working_with_llama_stack/index#activating-the-llama-stack-operator_rag) section to enable Llama Stack Operator.
* A VLLM hosted Model. You can easily get one using [LiteMaas](https://litemaas-litemaas.apps.prod.rhoai.rh-aiservices-bu.com/home).

## Setup
Create a secret for storing your model's information.
```
export INFERENCE_MODEL="Granite-3.3-8B-Instruct"
export VLLM_URL="https://litellm-litemaas.apps.prod.rhoai.rh-aiservices-bu.com/v1"
export VLLM_API_TOKEN="<token identifier>"
export EMBEDDING_MODEL="placeholder" # we won't use this, this is just a placeholder

oc create secret generic llama-stack-inference-model-secret \
  --from-literal INFERENCE_MODEL="$INFERENCE_MODEL" \
  --from-literal VLLM_URL="$VLLM_URL" \
  --from-literal VLLM_API_TOKEN="$VLLM_API_TOKEN" \
  --from-literal EMBEDDING_MODEL="$EMBEDDING_MODEL
```

## Setup Deployment files
### Configuring the `llama-stack-distribution`
This typically doesn't require any changes for the Garak provider unless you want to modify the `name` and `port` fields as pointed in the [llama-stack-distribution.yaml](deployment/llama-stack-distribution.yaml). Note that if you change the name and/or port, you must also update `KUBEFLOW_LLAMA_STACK_URL` in `kubeflow-garak-config.yaml`.

### Configuring the `kubeflow-garak-config` ConfigMap
Update the [kubeflow-garak-config](deployment/kubeflow-garak-config.yaml) with the following data:
``` bash
KUBEFLOW_LLAMA_STACK_URL: "http://lsd-garak-example-service.model-namespace.svc.cluster.local:8321" # <--- if needed, update this with your service name, namespace and cluster local address
KUBEFLOW_PIPELINES_ENDPOINT: "<your-kfp-endpoint>" # <--- update this with your kubeflow pipelines endpoint (`oc get routes -A | grep -i pipeline`)
KUBEFLOW_NAMESPACE: "model-namespace" # <--- update this with your kubeflow namespace
KUBEFLOW_BASE_IMAGE: "quay.io/rh-ee-spandraj/trustyai-lls-garak-provider-dsp:latest"
```


### Configuring the `pipelines_token` Secret
Unfortunately the Llama Stack distribution service account does not have privilages to create pipeline runs. In order to work around this we provide a user token as a secret to the Llama Stack Distribution.

Create the secret with:
``` bash
# Gather your token with `oc whoami -t`
kubectl create secret generic kubeflow-pipelines-token \
  --from-literal KUBEFLOW_PIPELINES_TOKEN=<your-pipelines-token>
```

## Deploy Llama Stack on OpenShift
You can now deploy the configuration files and the Llama Stack distribution with `oc apply -f deployment/kubeflow-garak-config.yaml` and `oc apply -f deployment/llama-stack-distribution.yaml`

You should now have a Llama Stack server on OpenShift with the remote garak eval provider configured.
You can now follow [openshift-demo.ipynb](./openshift-demo.ipynb) demo but ensure you are running it in a Data Science Workbench and use the `KUBEFLOW_LLAMA_STACK_URL` defined earlier. Alternatively you can run it locally if you create a Route or port-forward.
