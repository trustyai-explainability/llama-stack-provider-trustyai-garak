from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    AdapterSpec,
    remote_provider_spec,
    InlineProviderSpec
)
from llama_stack.apis import (
    inference, 
    files, 
    safety,
    telemetry,
    shields,
    benchmarks
    )
from typing import List


def get_provider_spec() -> List[ProviderSpec]:
    return [
        remote_provider_spec(
            api=Api.eval,
            adapter=AdapterSpec(
                adapter_type="trustyai_garak",
                module="llama_stack_provider_trustyai_garak",
                pip_packages=["garak", "kfp", "kfp-kubernetes", "kfp-server-api", "boto3"],
                config_class="config.GarakRemoteConfig",
            ),
            api_dependencies=[inference, files, benchmarks],
            optional_api_dependencies=[safety, telemetry, shields]
        ),
        InlineProviderSpec(
            api=Api.eval,
            provider_type="inline::trustyai_garak",
            pip_packages=["garak"],
            config_class="config.GarakEvalProviderConfig",
            module="llama_stack_provider_trustyai_garak",
            api_dependencies=[inference, files, benchmarks],
            optional_api_dependencies=[safety, telemetry, shields]
        )
    ]