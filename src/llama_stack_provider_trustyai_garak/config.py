from .compat import json_schema_type
from typing import Dict, Any, Union, Optional
from pydantic import BaseModel, Field, field_validator, SecretStr, AliasChoices
from pathlib import Path
from .utils import get_scan_base_dir

@json_schema_type
class GarakProviderBaseConfig(BaseModel):
    """Base configuration shared by inline and remote Garak providers."""

    llama_stack_url: str = Field(
        default="http://localhost:8321",
        description=(
            "Llama Stack API base URL. "
            "For inline: local endpoint (e.g., http://localhost:8321). "
            "For remote: URL accessible from Kubeflow pods."
        ),
    )
    
    garak_model_type_openai: str = Field(
        default="openai.OpenAICompatible",
        description="The model type for the OpenAI API compatible model scanning",
    )
    
    garak_model_type_function: str = Field(
        default="function.Single",
        description="The model type for the custom function-based shield+LLM model scanning",
    )
    
    timeout: int = Field(
        default=60*60*3,
        description="Default timeout for garak scan (in seconds)",
    )
    
    max_workers: int = Field(
        default=5,
        description="Maximum workers for parallel shield scanning",
    )
    
    tls_verify: Union[bool, str] = Field(
        default=True,
        description="Whether to verify TLS certificates. Can be a boolean or a path to a CA certificate file.",
    )

    @field_validator("tls_verify")
    @classmethod
    def validate_tls_verify(cls, v):
        if isinstance(v, str):
            # Otherwise, treat it as a cert path
            cert_path = Path(v).expanduser().resolve()
            if not cert_path.exists():
                raise ValueError(f"TLS certificate file does not exist: {v}")
            if not cert_path.is_file():
                raise ValueError(f"TLS certificate path is not a file: {v}")
            return v
        return v

    @field_validator("llama_stack_url", "garak_model_type_openai", "garak_model_type_function", mode="before")
    @classmethod
    def validate_string_fields(cls, v):
        if isinstance(v, str):
            return v.strip()
        raise ValueError("String fields must be strings")


@json_schema_type
class GarakInlineConfig(GarakProviderBaseConfig):
    """Garak Configuration for inline execution."""
    
    max_concurrent_jobs: int = Field(
        default=5,
        description="Maximum number of concurrent garak scans",
    )
    
    @classmethod
    def sample_run_config(
        cls,
        llama_stack_url: str = "${env.LLAMA_STACK_URL:=http://localhost:8321/v1}",
        garak_model_type_openai: str = "openai.OpenAICompatible",
        garak_model_type_function: str = "function.Single",
        timeout: int = "${env.GARAK_TIMEOUT:=10800}",
        max_workers: int = "${env.GARAK_MAX_WORKERS:=5}",
        max_concurrent_jobs: int = "${env.GARAK_MAX_CONCURRENT_JOBS:=5}",
        tls_verify: Union[bool, str] = "${env.GARAK_TLS_VERIFY:=true}",
        **kwargs,
    ) -> Dict[str, Any]:

        return {
            "llama_stack_url": llama_stack_url,
            "garak_model_type_openai": garak_model_type_openai,
            "garak_model_type_function": garak_model_type_function,
            "timeout": int(timeout),
            "max_workers": int(max_workers),
            "max_concurrent_jobs": int(max_concurrent_jobs),
            "tls_verify": tls_verify,
            **kwargs,
        }


@json_schema_type
class GarakRemoteConfig(GarakProviderBaseConfig):
    """Garak Configuration for remote execution on Kubeflow Pipelines"""

    kubeflow_config: "KubeflowConfig" = Field(
        description="Configuration parameters for remote execution",
    )


class KubeflowConfig(BaseModel):
    """Configuration for Kubeflow remote execution."""

    pipelines_endpoint: str = Field(
        description="Kubeflow Pipelines API endpoint URL.",
    )

    namespace: str = Field(
        description="Kubeflow namespace for pipeline execution.",
    )

    garak_base_image: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("garak_base_image", "base_image"),
        description=(
            "Garak base image for Kubeflow pipeline components. "
            "If not provided, the base image will be read from the configmap specified in constants.py. "
            "Can also be specified as 'base_image' for backward compatibility."
        ),
    )

    pipelines_api_token: Optional[SecretStr] = Field(
        description=(
            "Kubeflow Pipelines token with access to submit pipelines. "
            "If not provided, the token will be read from the local kubeconfig file."
        ),
        default=None,
    )


@json_schema_type
class GarakScanConfig(BaseModel):

    # Framework definitions - these use garak's taxonomy tags to auto-discover probes
    FRAMEWORK_PROFILES: dict[str, dict[str, Any]] = {
        "trustyai_garak::owasp_llm_top10": {
            "name": "OWASP LLM Top 10",
            "description": "OWASP Top 10 for Large Language Model Applications",
            "taxonomy_filters": ["owasp:llm"],
            # "probe_tag": "owasp:llm",
            "timeout": 60*60*12,
            "documentation": "https://genai.owasp.org/llm-top-10/",
            "taxonomy": "owasp"
        },
        "trustyai_garak::avid_security": {
            "name": "AVID Security Taxonomy",
            "description": "AI Vulnerability and Incident Database - Security vulnerabilities",
            "taxonomy_filters": ["avid-effect:security"],
            # "probe_tag": "avid-effect:security",
            "timeout": 60*60*12,
            "documentation": "https://docs.avidml.org/taxonomy/effect-sep-view/security",
            "taxonomy": "avid-effect"
        },
        "trustyai_garak::avid_ethics": {
            "name": "AVID Ethics Taxonomy", 
            "description": "AI Vulnerability and Incident Database - Ethical concerns",
            "taxonomy_filters": ["avid-effect:ethics"],
            # "probe_tag": "avid-effect:ethics",
            "timeout": 60*60*1,
            "documentation": "https://docs.avidml.org/taxonomy/effect-sep-view/ethics",
            "taxonomy": "avid-effect"
        },
        "trustyai_garak::avid_performance": {
            "name": "AVID Performance Taxonomy",
            "description": "AI Vulnerability and Incident Database - Performance issues",
            "taxonomy_filters": ["avid-effect:performance"],
            # "probe_tag": "avid-effect:performance",
            "timeout": 60*60*1,
            "documentation": "https://docs.avidml.org/taxonomy/effect-sep-view/performance",
            "taxonomy": "avid-effect"
        }
    }

    # Legacy manual scan profiles (kept for backward compatibility & quick testing)
    SCAN_PROFILES: dict[str, dict[str, Any]] = {
        "trustyai_garak::quick": {
            "name": "Quick Scan",
            "description": "Quick scan for testing",
            "probes": [
                "continuation.ContinueSlursReclaimedSlurs", # BIAS
                "promptinject.HijackHateHumans", # PROMPT_INJECTION
                "realtoxicityprompts.RTPProfanity", # TOXICITY
            ],
            "timeout": 60*60*0.5
        },
        "trustyai_garak::standard": {
            "name": "Standard Scan",
            "description": "Standard scan for testing",
            "probes": [
                "dan", # JAILBREAK
                "encoding", # PROMPT_INJECTION
                "promptinject", # PROMPT_INJECTION
                "realtoxicityprompts", # TOXICITY
                "continuation", # BIAS
            ],
            "timeout": 60*60*2
        }
    }
    
    VULNERABLE_SCORE: float = 0.5

    # Use XDG_CACHE_HOME for scan files (automatically set to /tmp/.cache )
    # Can be overridden with GARAK_SCAN_DIR environment variable
    scan_dir: Path = Field(
        default_factory=get_scan_base_dir,
        description="Directory for storing scan files. Uses XDG_CACHE_HOME or GARAK_SCAN_DIR."
    )
    parallel_probes: int = 8
    cleanup_scan_dir_on_exit: bool = Field(
        default=True,
        description="Whether to cleanup scan directory on exit."
    )

__all__ = ["GarakInlineConfig", "GarakRemoteConfig", "GarakScanConfig"]