# Kubeflow ConfigMap keys and defaults for base image resolution
GARAK_PROVIDER_IMAGE_CONFIGMAP_NAME = "trustyai-service-operator-config"
GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY = "garak-provider-image" # from https://github.com/opendatahub-io/opendatahub-operator/pull/2567
DEFAULT_GARAK_PROVIDER_IMAGE = "quay.io/trustyai/trustyai-garak-lls-provider-dsp:latest"
KUBEFLOW_CANDIDATE_NAMESPACES = ["redhat-ods-applications", "opendatahub"]

# Default values
DEFAULT_TIMEOUT = 600
DEFAULT_MODEL_TYPE = "openai.OpenAICompatible"
DEFAULT_EVAL_THRESHOLD = 0.5

# evalhub adapter
EXECUTION_MODE_SIMPLE = "simple"
EXECUTION_MODE_KFP = "kfp"