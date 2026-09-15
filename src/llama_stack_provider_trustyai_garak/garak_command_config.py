"""
Garak configuration models.

References:
- https://reference.garak.ai/en/latest/configurable.html
- garak.core.yaml in garak repository
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Dict, Any, List, Union


class GarakSystemConfig(BaseModel):
    """
    Garak system configuration.
    Controls system-level behavior like parallelization and output formatting.
    """

    model_config = ConfigDict(extra="forbid")

    parallel_attempts: Union[bool, int] = Field(
        default=False,
        description="For parallelisable generators, how many attempts should be run in parallel? Raising this is a great way of speeding up garak runs for API-based models",
    )
    max_workers: int = Field(
        default=500,
        gt=0,
        description="Cap on how many parallel workers can be requested. When raising this in order to use higher parallelisation, keep an eye on system resources (e.g. ulimit -n 4026 on Linux).",
    )
    parallel_requests: Union[bool, int] = Field(
        default=False,
        description="For generators not supporting multiple responses per prompt: how many requests to send in parallel with the same prompt? (raising parallel_attempts generally yields higher performance, depending on how high generations is set)",
    )
    verbose: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Degree of verbosity (values above 0 are experimental, the report & log are authoritative)",
    )
    show_z: bool = Field(
        default=False,
        description="Display Z-scores and visual indicators on CLI. It's good, but may be too much info until one has seen garak run a couple of times",
    )
    narrow_output: bool = Field(default=False, description="Support output on narrower CLIs")

    lite: bool = Field(default=True, description="Display lite mode caution message")
    enable_experimental: bool = Field(
        default=False,
        description="Enable experimental function CLI flags. Disabled by default. Experimental functions may disrupt your installation and provide unusual/unstable results. Can only be set by editing core config, so a git checkout of garak is recommended for this.",
    )


class GarakRunConfig(BaseModel):
    """
    Garak run configuration.
    Controls runtime behavior like generations, thresholds, and seeds.
    """

    model_config = ConfigDict(extra="forbid")

    generations: int = Field(default=5, gt=0, description="How many times to send each prompt for inference")
    spec: Optional["GarakRunSpec"] = Field(
        default=None,
        description="Unified probe, buff, tag, tier, and intent selection specification",
    )
    harness: Optional[str] = Field(
        default=None,
        description="Optional Garak harness name. Standard profiles use normal dispatch.",
    )
    serve_detectorless_intents: bool = Field(
        default=False,
        description="Whether to serve intents that have no detector",
    )
    eval_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="At what point in the 0..1 range output by detectors does a result count as a successful attack / hit",
    )
    soft_probe_prompt_cap: int = Field(
        default=256,
        gt=0,
        description="For probes that auto-scale their prompt count, the preferred limit of prompts per probe",
    )
    target_lang: Optional[str] = Field(
        default=None,
        description="A single language (as BCP47) that the target application for LLM accepts as prompt and output",
    )
    langproviders: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of configurations representing providers for converting from probe language to lang_spec target languages (BCP47). Each dict can contain 'language', 'model_type', 'model_name', and 'api_key' keys.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="If given and not overriden by the probe itself, probes will pass the specified system prompt when possible for generators that support chat modality.",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    deprefix: bool = Field(
        default=True,
        description="Remove the prompt from the start of the output (some models return the prompt as part of their output)",
    )


class GarakRunSpec(BaseModel):
    """Unified Garak plugin selection specification."""

    model_config = ConfigDict(extra="forbid")

    include: List[Union[str, Dict[str, Union[str, int]]]] = Field(default_factory=list)
    exclude: List[Union[str, Dict[str, Union[str, int]]]] = Field(default_factory=list)

    @field_validator("include", "exclude")
    @classmethod
    def validate_selectors(cls, selectors: List[Union[str, Dict[str, Union[str, int]]]]) -> List:
        allowed_mapping_keys = {"tag", "tier", "intent"}
        for selector in selectors:
            if isinstance(selector, dict) and (len(selector) != 1 or next(iter(selector)) not in allowed_mapping_keys):
                raise ValueError("run.spec selectors must be strings or single-key tag, tier, or intent mappings")
        return selectors


class GarakPluginsConfig(BaseModel):
    """
    Garak plugins configuration.
    Configures probes, detectors, buffs, and other plugin behavior.
    """

    model_config = ConfigDict(extra="forbid")

    detector_spec: Optional[Union[List[str], str]] = Field(
        default="auto",
        description="A list of detectors to use, or 'auto' to use each probe's suggestion.",
    )
    extended_detectors: bool = Field(
        default=True,
        description="Should just the primary detector be used per probe, or should the extended detectors also be run? The former is fast, the latter thorough.",
    )
    buffs_include_original_prompt: bool = Field(
        default=False,
        description="When buffing, should the original pre-buff prompt still be included in those posed to the model?",
    )
    buff_max: Optional[int] = Field(default=None, description="Upper bound on how many items a buff should return")
    target_type: Optional[str] = Field(
        default=None, description="Type of target generator (e.g., 'openai', 'huggingface')"
    )
    target_name: Optional[str] = Field(default=None, description="Specific name of target model")
    probes: Optional[Dict[str, Any]] = Field(default=None, description="Root node for probe plugin configs")
    detectors: Optional[Dict[str, Any]] = Field(default=None, description="Root node for detector plugin configs")
    generators: Optional[Dict[str, Any]] = Field(default=None, description="Root node for generator plugin configs")
    buffs: Optional[Dict[str, Any]] = Field(default=None, description="Root note for buff plugin configs")
    harnesses: Optional[Dict[str, Any]] = Field(default=None, description="Root node for harness plugin configs")


class GarakReportingConfig(BaseModel):
    """
    Garak reporting configuration.
    Controls output format and report generation.
    """

    model_config = ConfigDict(extra="forbid")

    taxonomy: Optional[str] = Field(
        default=None,
        description="Taxonomy to use to group probes when creating HTML report. (options: 'owasp', 'avid-effect', 'risk-cards', 'quality', 'cwe')",
    )
    show_100_pass_modules: bool = Field(
        default=True, description="Should entries scoring 100 still be detailed in the HTML report?"
    )
    show_top_group_score: bool = Field(
        default=True, description="Should the aggregated score be shown as a top-level figure in report concertinas?"
    )
    group_aggregation_function: str = Field(
        default="lower_quartile",
        description="How should scores of probe groups (e.g. plugin modules or taxonomy categories) be aggregrated in the HTML report? Options: 'minimum', 'mean', 'median', 'mean_minus_sd', 'lower_quartile', and 'proportion_passing'. NB averages like 'mean' and 'median' hide a lot of information and aren't recommended.",
    )
    report_dir: Optional[str] = Field(
        default=None,
        description="Directory for storing reports. No need to set this as it will be automatically created by the provider.",
    )
    report_prefix: Optional[str] = Field(
        default=None,
        description="Prefix for output report files. No need to set this as it will be automatically created by the provider.",
    )


class GarakCommandConfig(BaseModel):
    """
    Complete Garak command configuration.

    References:
    - https://reference.garak.ai/en/latest/configurable.html

    Example:
        >>> config = GarakCommandConfig(
        ...     run=GarakRunConfig(
        ...         spec=GarakRunSpec(include=["probes.dan", "probes.encoding"], exclude=[]),
        ...         generations=2,
        ...         seed=42,
        ...         eval_threshold=0.6,
        ...     ),
        ...     system=GarakSystemConfig(
        ...         parallel_attempts=8,
        ...         max_workers=10,
        ...     ),
        ... )
        >>> config.to_dict()
        {"run": {"spec": {"include": ["probes.dan", "probes.encoding"], "exclude": []}, ...}, ...}
    """

    model_config = ConfigDict(extra="allow")

    system: GarakSystemConfig = Field(
        default_factory=GarakSystemConfig, description="System-level configuration (parallelization, verbosity, etc.)"
    )
    run: GarakRunConfig = Field(
        default_factory=GarakRunConfig, description="Runtime configuration (generations, seed, eval_threshold, etc.)"
    )
    plugins: GarakPluginsConfig = Field(
        default_factory=GarakPluginsConfig, description="Plugin configuration (probes, detectors, buffs, harnesses)"
    )
    reporting: GarakReportingConfig = Field(
        default_factory=GarakReportingConfig, description="Reporting configuration (output format, taxonomy, etc.)"
    )

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        Convert to dict suitable for Garak's --config flag.

        Args:
            exclude_none: If True, omit None values from output

        Returns:
            Dictionary with Garak configuration structure
        """
        return self.model_dump(exclude_none=exclude_none)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GarakCommandConfig":
        """
        Create GarakCommandConfig from a dictionary.

        Args:
            config_dict: Dictionary with Garak config structure

        Returns:
            GarakCommandConfig instance
        """
        return cls(**config_dict)


__all__ = [
    "GarakSystemConfig",
    "GarakRunConfig",
    "GarakRunSpec",
    "GarakPluginsConfig",
    "GarakReportingConfig",
    "GarakCommandConfig",
]
