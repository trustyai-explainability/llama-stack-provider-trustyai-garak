from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    RemoteProviderSpec
)

def get_provider_spec() -> ProviderSpec:
    return RemoteProviderSpec(
            api=Api.eval,
            provider_type="remote::trustyai_garak",
            config_class="llama_stack_provider_trustyai_garak.config.GarakRemoteConfig",
            api_dependencies=[Api.inference, Api.files, Api.benchmarks, Api.safety, Api.telemetry, Api.shields],
            module="llama_stack_provider_trustyai_garak.remote",
            pip_packages=["garak", "kfp", "kfp-kubernetes", "kfp-server-api", "boto3"],
            adapter_type="trustyai_garak",
            description="TrustyAI’s remote provider for Garak vulnerability scanning on Kubeflow Pipelines.",   
        )