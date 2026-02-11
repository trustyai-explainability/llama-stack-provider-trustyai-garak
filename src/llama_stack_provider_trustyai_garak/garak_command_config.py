"""
Garak configuration models.

References:
- https://reference.garak.ai/en/latest/configurable.html
- garak.core.yaml in garak repository
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List, Union
from .compat import json_schema_type


@json_schema_type
class GarakSystemConfig(BaseModel):
    """
    Garak system configuration.
    Controls system-level behavior like parallelization and output formatting.
    """
    model_config = ConfigDict(extra="forbid")

    parallel_attempts: Union[bool, int] = Field(
        default=16,
        description="For parallelisable generators, how many attempts should be run in parallel? Raising this is a great way of speeding up garak runs for API-based models"
    )
    max_workers: int = Field(
        default=500,
        gt=0,
        description="Cap on how many parallel workers can be requested. When raising this in order to use higher parallelisation, keep an eye on system resources (e.g. ulimit -n 4026 on Linux)."
    )
    parallel_requests: Union[bool, int] = Field(
        default=False,
        description="For generators not supporting multiple responses per prompt: how many requests to send in parallel with the same prompt? (raising parallel_attempts generally yields higher performance, depending on how high generations is set)"
    )
    verbose: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Degree of verbosity (values above 0 are experimental, the report & log are authoritative)"
    )
    show_z: bool = Field(
        default=False,
        description="Display Z-scores and visual indicators on CLI. It's good, but may be too much info until one has seen garak run a couple of times"
    )
    narrow_output: bool = Field(
        default=False,
        description="Support output on narrower CLIs"
    )
    
    lite: bool = Field(
        default=True,
        description="Display lite mode caution message"
    )
    enable_experimental: bool = Field(
        default=False,
        description="Enable experimental function CLI flags. Disabled by default. Experimental functions may disrupt your installation and provide unusual/unstable results. Can only be set by editing core config, so a git checkout of garak is recommended for this."
    )

@json_schema_type
class GarakRunConfig(BaseModel):
    """
    Garak run configuration.
    Controls runtime behavior like generations, thresholds, and seeds.
    """
    model_config = ConfigDict(extra="forbid")

    generations: int = Field(
        default=1,
        gt=0,
        description="How many times to send each prompt for inference"
    )
    probe_tags: Optional[str] = Field(
        default=None,
        description="If given, the probe selection is filtered according to these tags; probes that don't match the tags are not selected (e.g., 'owasp:llm')"
    )
    eval_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="At what point in the 0..1 range output by detectors does a result count as a successful attack / hit"
    )
    soft_probe_prompt_cap: int = Field(
        default=256,
        gt=0,
        description="For probes that auto-scale their prompt count, the preferred limit of prompts per probe"
    )
    target_lang: Optional[str] = Field(
        default=None,
        description="A single language (as BCP47) that the target application for LLM accepts as prompt and output"
    )
    langproviders: Optional[List[str]] = Field(
        default=None,
        description="A list of configurations representing providers for converting from probe language to lang_spec target languages (BCP47)"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="If given and not overriden by the probe itself, probes will pass the specified system prompt when possible for generators that support chat modality."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    deprefix: bool = Field(
        default=True,
        description="Remove the prompt from the start of the output (some models return the prompt as part of their output)"
    )


@json_schema_type
class GarakPluginsConfig(BaseModel):
    """
    Garak plugins configuration.
    Configures probes, detectors, buffs, and other plugin behavior.
    """
    model_config = ConfigDict(extra="forbid")

    probe_spec: Union[List[str], str] = Field(
        default="all",
        description="A list of probe modules or probe classnames (in module.classname) format to be used. If a module is given, only active plugin in that module are chosen. If ['all'] is given, all probes are used. For example: ['dan', 'encoding']"
    )
    detector_spec: Optional[Union[List[str], str]] = Field(
        default=None,
        description="A list of detectors to use, or 'all' for all. Default is to use the probe's suggestion. Specifying detector_spec means the pxd harness will be used."
    )
    extended_detectors: bool = Field(
        default=True,
        description="Should just the primary detector be used per probe, or should the extended detectors also be run? The former is fast, the latter thorough."
    )
    buff_spec: Optional[Union[List[str], str]] = Field(
        default=None,
        description="Comma-separated list of buffs and buff modules to use; same format as probe_spec."
    )
    buffs_include_original_prompt: bool = Field(
        default=True,
        description="When buffing, should the original pre-buff prompt still be included in those posed to the model?"
    )
    buff_max: Optional[int] = Field(
        default=None,
        description="Upper bound on how many items a buff should return"
    )
    target_type: str = Field(
        default="openai.OpenAICompatible",
        description="Type of target generator (e.g., 'openai', 'huggingface')"
    )
    target_name: Optional[str] = Field(
        default=None,
        description="Specific name of target model"
    )
    probes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Root node for probe plugin configs"
    )
    detectors: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Root node for detector plugin configs"
    )
    generators: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Root node for generator plugin configs"
    )
    buffs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Root note for buff plugin configs"
    )
    harnesses: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Root node for harness plugin configs"
    )


@json_schema_type
class GarakReportingConfig(BaseModel):
    """
    Garak reporting configuration.
    Controls output format and report generation.
    """
    model_config = ConfigDict(extra="forbid")

    taxonomy: Optional[str] = Field(
        default=None,
        description="Taxonomy to use to group probes when creating HTML report. (options: 'owasp', 'avid-effect', 'risk-cards', 'quality', 'cwe')"
    )
    show_100_pass_modules: bool = Field(
        default=True,
        description="Should entries scoring 100 still be detailed in the HTML report?"
    )
    show_top_group_score: bool = Field(
        default=True,
        description="Should the aggregated score be shown as a top-level figure in report concertinas?"
    )
    group_aggregation_function: str = Field(
        default="lower_quartile",
        description="How should scores of probe groups (e.g. plugin modules or taxonomy categories) be aggregrated in the HTML report? Options: 'minimum', 'mean', 'median', 'mean_minus_sd', 'lower_quartile', and 'proportion_passing'. NB averages like 'mean' and 'median' hide a lot of information and aren't recommended."
    )
    report_dir: Optional[str] = Field(
        default=None,
        description="Directory for storing reports. No need to set this as it will be automatically created by the provider."
    )
    report_prefix: Optional[str] = Field(
        default=None,
        description="Prefix for output report files. No need to set this as it will be automatically created by the provider."
    )


@json_schema_type
class GarakCommandConfig(BaseModel):
    """
    Complete Garak command configuration.
    
    References:
    - https://reference.garak.ai/en/latest/configurable.html
    
    Example:
        >>> config = GarakCommandConfig(
        ...     plugins=GarakPluginsConfig(
        ...         probe_spec=["dan", "encoding"]
        ...     ),
        ...     run=GarakRunConfig(
        ...         generations=2,
        ...         seed=42,
        ...         eval_threshold=0.6
        ...     ),
        ...     system=GarakSystemConfig(
        ...         parallel_attempts=8,
        ...         max_workers=10
        ...     )
        ... )
        >>> config.to_dict()
        {"plugins": {"probe_spec": ["dan", "encoding"], ...}, ...}
    """
    model_config = ConfigDict(extra="allow")

    system: GarakSystemConfig = Field(
        default_factory=GarakSystemConfig,
        description="System-level configuration (parallelization, verbosity, etc.)"
    )
    run: GarakRunConfig = Field(
        default_factory=GarakRunConfig,
        description="Runtime configuration (generations, seed, eval_threshold, etc.)"
    )
    plugins: GarakPluginsConfig = Field(
        default_factory=GarakPluginsConfig,
        description="Plugin configuration (probes, detectors, buffs, harnesses)"
    )
    reporting: GarakReportingConfig = Field(
        default_factory=GarakReportingConfig,
        description="Reporting configuration (output format, taxonomy, etc.)"
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
    "GarakPluginsConfig",
    "GarakReportingConfig",
    "GarakCommandConfig",
]
