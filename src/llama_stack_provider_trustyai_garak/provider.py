from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    InlineProviderSpec
)
from llama_stack.apis import (
    inference, 
    files, 
    safety,
    telemetry
    )

def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
        api=Api.eval,
        provider_type="inline::trustyai_garak",
        pip_packages=["garak"],
        config_class="config.GarakEvalProviderConfig",
        module="llama_stack_provider_trustyai_garak",
        api_dependencies=[inference, files],
        optional_api_dependencies=[safety, telemetry]
    )
