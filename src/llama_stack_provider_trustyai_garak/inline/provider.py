from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    InlineProviderSpec
)

def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
            api=Api.eval,
            provider_type="inline::trustyai_garak",
            pip_packages=["garak==0.12.0"],
            config_class="llama_stack_provider_trustyai_garak.config.GarakEvalProviderConfig",
            module="llama_stack_provider_trustyai_garak.inline",
            api_dependencies=[Api.inference, Api.files, Api.benchmarks],
            optional_api_dependencies=[Api.safety, Api.shields]
        )