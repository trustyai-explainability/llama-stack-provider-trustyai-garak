from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    AdapterSpec,
    remote_provider_spec
)

def get_provider_spec() -> ProviderSpec:
    return remote_provider_spec(
            api=Api.eval,
            adapter=AdapterSpec(
                adapter_type="trustyai_garak",
                module="llama_stack_provider_trustyai_garak.remote",
                pip_packages=["garak", "kfp", "kfp-kubernetes", "kfp-server-api", "boto3"],
                config_class="llama_stack_provider_trustyai_garak.config.GarakRemoteConfig",
            ),
            api_dependencies=[Api.inference, Api.files, Api.benchmarks, Api.safety, Api.telemetry, Api.shields]
        )