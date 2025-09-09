from llama_stack.schema_utils import json_schema_type
from typing import Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

@json_schema_type
class GarakEvalProviderConfig(BaseModel):
    base_url: str = Field(
        default="http://localhost:8321/v1",
        description="The base URL for the OpenAI API compatible remote model serving endpoint",
    )
    garak_model_type_openai: str = Field(
        default="openai.OpenAICompatible",
        description="The model type for the OpenAI API compatible model scanning",
    )
    garak_model_type_function: str = Field(
        default="function.Single",
        description="The model type for the custom function-based shield+LLMmodel scanning",
    )
    timeout: int = 60*60*3 # default timeout for garak scan
    max_workers: int = 5 # default max workers for shield scanning
    max_concurrent_jobs: int = 5 # max concurrent garak scans
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


    @field_validator("base_url", "garak_model_type_openai", "garak_model_type_function", mode="before")
    @classmethod
    def validate_base_url_garak_model_type(cls, v):
        if isinstance(v, str):
            return v.strip()
        raise ValueError("base_url, garak_model_type_openai and garak_model_type_function must be strings")
    
    @classmethod
    def sample_run_config(
        cls,
        base_url: str = "${env.BASE_URL}",
        garak_model_type_openai: str = "openai.OpenAICompatible",
        garak_model_type_function: str = "function.Single",
        timeout: int = "${env.GARAK_TIMEOUT:=10800}",
        max_workers: int = "${env.GARAK_MAX_WORKERS:=5}",
        max_concurrent_jobs: int = "${env.GARAK_MAX_CONCURRENT_JOBS:=5}",
        tls_verify: Union[bool, str] = "${env.GARAK_TLS_VERIFY:=true}",
        **kwargs,
    ) -> Dict[str, Any]:

        return {
            "base_url": base_url,
            "garak_model_type_openai": garak_model_type_openai,
            "garak_model_type_function": garak_model_type_function,
            "timeout": int(timeout),
            "max_workers": int(max_workers),
            "max_concurrent_jobs": int(max_concurrent_jobs),
            "tls_verify": tls_verify,
            **kwargs,
        }

@json_schema_type
class GarakRemoteConfig(GarakEvalProviderConfig):
    """Configuration for Ragas evaluation provider (remote execution)."""

    kubeflow_config: "KubeflowConfig" = Field(
        description="Additional configuration parameters for remote execution",
    )


class KubeflowConfig(BaseModel):
    """Configuration for Kubeflow remote execution."""

    pipelines_endpoint: str = Field(
        description="Kubeflow Pipelines API endpoint URL (required for remote execution)",
    )

    namespace: str = Field(
        description="Kubeflow namespace for pipeline execution",
    )

    experiment_name: str = Field(
        description="Kubeflow experiment name for pipeline execution",
    )

    base_image: str = Field(
        description="Base image for Kubeflow pipeline components",
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

    base_dir: Path = Path(__file__).parent
    scan_dir: Path = base_dir / "_scan_files"
    parallel_probes: int = 8
    cleanup_scan_dir_on_exit: bool = False

__all__ = ["GarakEvalProviderConfig", "GarakScanConfig"]