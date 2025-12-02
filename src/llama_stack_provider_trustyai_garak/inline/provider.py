from ..compat import ProviderSpec, Api, InlineProviderSpec
from ..version_utils import get_garak_version

def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
            api=Api.eval,
            provider_type="inline::trustyai_garak",
            pip_packages=[get_garak_version()],
            config_class="llama_stack_provider_trustyai_garak.config.GarakInlineConfig",
            module="llama_stack_provider_trustyai_garak.inline",
            api_dependencies=[Api.inference, Api.files, Api.benchmarks],
            optional_api_dependencies=[Api.safety, Api.shields],
            description="TrustyAI's inline llama-stack provider for Garak vulnerability scanning.",
        )