"""Base class for Garak evaluation adapters with common functionality."""

from .compat import (
    Eval,
    BenchmarkConfig,
    EvaluateResponse,
    EvaluateRowsRequest,
    JobResultRequest,
    ProviderSpec,
    BenchmarksProtocolPrivate,
    Api,
    Files,
    Benchmark,
    Benchmarks,
    Job,
    JobStatus,
    Safety,
    Shields,
    RetrieveFileContentRequest,
    GetShieldRequest,
)
from typing import Dict, Optional, Set, List, Any, Union, Tuple
import asyncio
import logging
import uuid
from .config import GarakScanConfig, GarakInlineConfig, GarakRemoteConfig
from .garak_command_config import GarakCommandConfig, GarakRunConfig
from .errors import GarakError, GarakConfigError, GarakValidationError, BenchmarkNotFoundError

logger = logging.getLogger(__name__)


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary with default values
        override: Override dictionary with custom values
        
    Returns:
        Merged dictionary where override values take precedence at all levels
        
    Example:
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"c": 99}, "e": 4}
        result = {"a": {"b": 1, "c": 99}, "d": 3, "e": 4}
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Both are dicts - recursively merge
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # Override takes precedence
            result[key] = value
    
    return result


class GarakEvalBase(Eval, BenchmarksProtocolPrivate):
    """Base class for Garak evaluation adapters containing common functionality."""

    def __init__(self, config: Union[GarakInlineConfig, GarakRemoteConfig], deps: dict[Api, ProviderSpec]):
        super().__init__(config)
        self._config = config
        self.file_api: Optional[Files] = deps.get(Api.files)
        self.benchmarks_api: Optional[Benchmarks] = deps.get(Api.benchmarks)
        self.safety_api: Optional[Safety] = deps.get(Api.safety)
        self.shields_api: Optional[Shields] = deps.get(Api.shields)
        self.scan_config = GarakScanConfig()
        self.benchmarks: Dict[str, Benchmark] = {}  # benchmark_id -> benchmark
        self._benchmarks_lock = asyncio.Lock()
        self.all_probes: Set[str] = set()
        self._verify_ssl = None
        self._initialized: bool = False

        self._jobs: Dict[str, Job] = {}
        self._job_metadata: Dict[str, Dict[str, str]] = {}
    
    def _initialize(self) -> None:
        """Initialize the Garak provider."""
        self._validate_apis()
        self._ensure_garak_installed()
        self.all_probes = self._get_all_probes()

        self._verify_ssl = self._config.tls_verify
        if isinstance(self._verify_ssl, str):
            if self._verify_ssl.lower() in ("true", "1", "yes", "on"):
                self._verify_ssl = True
            elif self._verify_ssl.lower() in ("false", "0", "no", "off"):
                self._verify_ssl = False

    def _validate_apis(self) -> None:
        """Validate that required APIs are provided and optional APIs are consistent."""
        error_msg_template = (
            "{provided_api_name} API is provided but {missing_api_name} API is not provided. "
            "Please provide both APIs."
        )
        if self.shields_api and not self.safety_api:
            raise GarakConfigError(
                error_msg_template.format(
                    provided_api_name="Shields", missing_api_name="Safety"
                )
            )
        elif not self.shields_api and self.safety_api:
            raise GarakConfigError(
                error_msg_template.format(
                    provided_api_name="Safety", missing_api_name="Shields"
                )
            )

    def _ensure_garak_installed(self) -> None:
        """Ensure garak is installed."""
        try:
            import garak  # noqa: F401
        except ImportError:
            raise GarakError(
                "Garak is not installed. Please install it with: pip install garak"
            )

    def _get_all_probes(self) -> Set[str]:
        """Get all available Garak probes.
        
        Returns:
            Set of probe names (both full names and module names)
        """
        # Unfortunately, garak doesn't have a public API to list all probes
        # so we need to enumerate all probes manually from private API
        try:
            from garak._plugins import enumerate_plugins
        except ImportError:
            raise GarakError(
                "Unable to import garak's enumerate_plugins. The internal API may have changed."
            )

        probes_names = enumerate_plugins(category="probes", skip_base_classes=True)
        plugin_names = [p.replace("probes.", "") for p, _ in probes_names]
        module_names = set([m.split(".")[0] for m in plugin_names])
        plugin_names += list(module_names)
        return set(plugin_names)

    async def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """Get a benchmark by its id.
        
        Args:
            benchmark_id: The benchmark identifier
            
        Returns:
            Benchmark object if found, None otherwise
        """
        async with self._benchmarks_lock:
            return self.benchmarks.get(benchmark_id)

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark by checking if it's a pre-defined scan profile or compliance framework.
        
        Args:
            benchmark: The benchmark to register
        """
        pre_defined_profiles = {
            **self.scan_config.SCAN_PROFILES,
            **self.scan_config.FRAMEWORK_PROFILES
        }
        # override pre-defined profiles or exising benchmarks
        if benchmark.provider_benchmark_id:
            if benchmark.provider_benchmark_id in pre_defined_profiles:
                logger.info(
                    f"Deep merging the provider benchmark id '{benchmark.provider_benchmark_id}' with the benchmark metadata."
                )
                base_metadata = pre_defined_profiles.get(benchmark.provider_benchmark_id, {})
                benchmark.metadata = deep_merge_dicts(base_metadata, benchmark.metadata or {})
            elif benchmark.provider_benchmark_id in self.benchmarks:
                existing_benchmark = self.benchmarks.get(benchmark.provider_benchmark_id)
                if existing_benchmark:
                    logger.info(
                        f"Deep merging the existing benchmark '{existing_benchmark.identifier}' with the benchmark metadata."
                    )
                    base_metadata = existing_benchmark.metadata or {}
                    benchmark.metadata = deep_merge_dicts(base_metadata, benchmark.metadata or {})

        if not benchmark.metadata:
            logger.info(
                f"Benchmark '{benchmark.identifier}' has no metadata. Using defaults from profile."
            )
            benchmark.metadata = pre_defined_profiles.get(benchmark.identifier, {})


        async with self._benchmarks_lock:
            self.benchmarks[benchmark.identifier] = benchmark
    
    async def unregister_benchmark(
        self,
        benchmark_id: str,
    ) -> None:
        """Unregister a benchmark.
        
        Args:
            benchmark_id: The benchmark identifier
        """
        async with self._benchmarks_lock:
            if benchmark_id in self.benchmarks:
                del self.benchmarks[benchmark_id]
                logger.info(f"Unregistered benchmark: {benchmark_id}")
            else:
                logger.warning(f"Benchmark {benchmark_id} not found in provider's internal state")

    def _get_job_id(self, prefix: str = "garak-job-") -> str:
        """Generate a unique job ID.
        
        Args:
            prefix: Prefix for the job ID
            
        Returns:
            Unique job ID string
        """
        return f"{prefix}{str(uuid.uuid4())}"

    def _normalize_list_arg(self, arg: Union[str, List[str]]) -> Optional[str]:
        """Normalize a list argument to a comma-separated string.
        
        Args:
            arg: String or list of strings
            
        Returns:
            Comma-separated string
        """
        if not arg:
            return None
        return arg if isinstance(arg, str) else ",".join(arg)
    
    def _parse_benchmark_metadata(
        self, 
        benchmark_metadata: Dict[str, Any]
    ) -> Tuple[GarakCommandConfig, Dict[str, Any]]:
        """
        Parse benchmark metadata into Garak config and provider params.
        
        Args:
            benchmark_metadata: Raw metadata dict from benchmark
            
        Returns:
            Tuple of (garak_config, remaining_metadata)
            - garak_config: GarakCommandConfig
            - remaining_metadata: Provider-specific params (shields, timeout, etc.)
        """
        if not benchmark_metadata:
            benchmark_metadata = {}

        provider_params = {
                k: v for k, v in benchmark_metadata.items()
                if k != "garak_config"
            }
        
        garak_config = None

        if "garak_config" in benchmark_metadata:
            try:
                if isinstance(benchmark_metadata["garak_config"], GarakCommandConfig):
                    garak_config = benchmark_metadata["garak_config"]
                else:
                    garak_config = GarakCommandConfig.from_dict(benchmark_metadata["garak_config"])
            except Exception as e:
                raise GarakValidationError(
                    f"Invalid garak_config in benchmark metadata: {e}"
                ) from e
            
        else:
            logger.warning("No garak_config found in the benchmark metadata. Using default (all) probes - this could be slow.")
            garak_config = GarakCommandConfig()
        return garak_config, provider_params

    async def _get_generator_options(self, benchmark_config: "BenchmarkConfig", provider_params: dict, garak_config: "GarakCommandConfig") -> dict:
        """Get the generator options based on the availability of shields.
        
        Args:
            benchmark_config: Configuration for the evaluation task
            benchmark_metadata: Metadata from the benchmark
            
        Returns:
            Generator options dictionary
        """
        if bool(provider_params.get("shield_ids", []) or provider_params.get("shield_config", {})):
            return await self._get_function_based_generator_options(benchmark_config, provider_params)
        else:
            return await self._get_openai_compatible_generator_options(benchmark_config, garak_config)

    async def _get_openai_compatible_generator_options(
        self, benchmark_config: "BenchmarkConfig", garak_config: "GarakCommandConfig"
    ) -> dict:
        """Get the generator options for the OpenAI compatible generator.
        
        Args:
            benchmark_config: Configuration for the evaluation task
            garak_config: Garak configuration
            
        Returns:
            Generator options for OpenAI compatible model
        """
        import os
        
        if garak_config and garak_config.plugins and garak_config.plugins.generators:
            return garak_config.plugins.generators
        
        llama_stack_url: str = self._get_llama_stack_url()

        generator_options = {
            "openai": {
                "OpenAICompatible": {
                    "uri": llama_stack_url,
                    "model": benchmark_config.eval_candidate.model,
                    "api_key": os.getenv("OPENAICOMPATIBLE_API_KEY", "DUMMY"),
                    "suppressed_params": ["n"]
                }
            }
        }
        
        # Add extra params
        valid_params: dict = self._get_sampling_params(benchmark_config)
        
        if valid_params:
            generator_options["openai"]["OpenAICompatible"].update(valid_params)
        return generator_options
    
    def _get_sampling_params(self, benchmark_config: "BenchmarkConfig") -> dict:
        """Extract and validate sampling parameters from benchmark config.
        
        Args:
            benchmark_config: Configuration containing sampling parameters
            
        Returns:
            Dictionary of valid sampling parameters
            
        Raises:
            GarakValidationError: If benchmark_config or sampling_params are invalid
        """
        from .compat import SamplingParams, SamplingStrategy, TopPSamplingStrategy, TopKSamplingStrategy

        if not benchmark_config:
            raise GarakValidationError("benchmark_config cannot be None")
        if not hasattr(benchmark_config, 'eval_candidate'):
            raise GarakValidationError("benchmark_config must have eval_candidate attribute")
        if not hasattr(benchmark_config.eval_candidate, 'sampling_params'):
            logger.warning("benchmark_config.eval_candidate has no sampling_params, using defaults")
            return {}

        sampling_params: SamplingParams = benchmark_config.eval_candidate.sampling_params
        if not sampling_params:
            logger.warning("sampling_params is None, using defaults")
            return {}
        
        valid_params: dict = {}

        if hasattr(sampling_params, 'strategy') and sampling_params.strategy:
            strategy: SamplingStrategy = sampling_params.strategy
            # valid_params["strategy"] = strategy.type
            if isinstance(strategy, TopPSamplingStrategy):
                if strategy.top_p is not None:
                    valid_params['top_p'] = strategy.top_p
                if strategy.temperature is not None:
                    valid_params['temperature'] = strategy.temperature
            elif isinstance(strategy, TopKSamplingStrategy):
                if strategy.top_k is not None:
                    valid_params['top_k'] = strategy.top_k

        if sampling_params.max_tokens is not None:
            valid_params['max_tokens'] = sampling_params.max_tokens
        # if sampling_params.repetition_penalty is not None:
        #     valid_params['repetition_penalty'] = sampling_params.repetition_penalty
        if sampling_params.stop is not None:
            valid_params['stop'] = sampling_params.stop
        
        return valid_params
    
    def _get_llama_stack_url(self) -> str:
        """Get the normalized Llama Stack URL with /v1 suffix.
        
        Returns:
            Llama Stack URL with /v1 suffix
            
        Raises:
            GarakConfigError: If llama_stack_url is invalid
        """
        import re

        if not hasattr(self._config, 'llama_stack_url') or not self._config.llama_stack_url:
            raise GarakConfigError("llama_stack_url is not configured")
        
        llama_stack_url: str = self._config.llama_stack_url.strip()
        
        if not llama_stack_url:
            raise GarakConfigError("llama_stack_url cannot be empty")
        
        if not llama_stack_url.startswith(("http://", "https://")):
            raise GarakConfigError(
                f"llama_stack_url must start with http:// or https://, got: {llama_stack_url}"
            )
        
        llama_stack_url = llama_stack_url.rstrip("/")
        
        # check if URL ends with /v{number} and if not, add v1
        if not re.match(r"^.*\/v\d+$", llama_stack_url):
            llama_stack_url = f"{llama_stack_url}/v1"
        return llama_stack_url

    async def _get_function_based_generator_options(
        self, benchmark_config: "BenchmarkConfig", provider_params: dict
    ) -> dict:
        """Get the generator options for the custom function-based generator.
        
        Args:
            benchmark_config: Configuration for the evaluation task
            provider_params: Provider-specific parameters
            
        Returns:
            Generator options for function-based model with shield support
            
        Raises:
            GarakValidationError: If benchmark_config is invalid
            GarakConfigError: If llama_stack_url is not configured
        """
        from llama_stack_provider_trustyai_garak import shield_scan
        
        if not benchmark_config:
            raise GarakValidationError("benchmark_config cannot be None")
        if not hasattr(benchmark_config, 'eval_candidate') or not benchmark_config.eval_candidate:
            raise GarakValidationError("benchmark_config must have eval_candidate")
        if not hasattr(benchmark_config.eval_candidate, 'model') or not benchmark_config.eval_candidate.model:
            raise GarakValidationError("benchmark_config.eval_candidate must have model")
        
        if not hasattr(self._config, 'llama_stack_url') or not self._config.llama_stack_url:
            raise GarakConfigError("llama_stack_url is not configured")
        
        llama_stack_url: str = self._config.llama_stack_url.strip().rstrip("/")
        
        if not llama_stack_url:
            raise GarakConfigError("llama_stack_url cannot be empty")
        
        # Map the shields to the input and output of LLM
        llm_io_shield_mapping: dict = {
            "input": [],
            "output": []
        }

        if not provider_params:
            logger.warning("No provider parameters found.")
        else:
            # get the shield_ids and/or shield_config from the benchmark metadata
            shield_ids: List[str] = provider_params.get("shield_ids", [])
            shield_config: dict = provider_params.get("shield_config", {})
            if not shield_ids and not shield_config:
                logger.warning("No shield_ids or shield_config found in the provider parameters")
            elif shield_ids:
                if not isinstance(shield_ids, list):
                    raise GarakValidationError("shield_ids must be a list")
                if shield_config:
                    logger.warning(
                        "Both shield_ids and shield_config found in the benchmark metadata. "
                        "Using shield_ids only."
                    )
                llm_io_shield_mapping["input"] = shield_ids
            elif shield_config:
                if not isinstance(shield_config, dict):
                    raise GarakValidationError("shield_config must be a dictionary")
                if not shield_config.get("input") and not shield_config.get("output"):
                    logger.warning("No input or output found in the shield_config.")
                llm_io_shield_mapping["input"] = shield_config.get("input", [])
                llm_io_shield_mapping["output"] = shield_config.get("output", [])
            
            await self._check_shield_availability(llm_io_shield_mapping)

        generator_options = {
            "function": {
                "Single": {
                    "name": f"{shield_scan.__name__}#simple_shield_orchestrator",
                    "kwargs": {
                        "model": benchmark_config.eval_candidate.model,
                        "base_url": llama_stack_url,
                        "llm_io_shield_mapping": llm_io_shield_mapping,
                        "max_workers": self._config.max_workers,
                        # "skip_llm": benchmark_metadata.get("skip_llm", False)
                    }
                }
            }
        }
        valid_params: dict = self._get_sampling_params(benchmark_config)
        if valid_params:
            generator_options["function"]["Single"]["kwargs"]["sampling_params"] = valid_params
        return generator_options
    
    async def _check_shield_availability(self, llm_io_shield_mapping: dict) -> None:
        """Check the availability of shields and raise an error if any shield is not available.
        
        Args:
            llm_io_shield_mapping: Mapping of input/output shields
            
        Raises:
            GarakConfigError: If shields API is not available
            GarakValidationError: If any specified shield is not available
        """
        if not self.shields_api:
            raise GarakConfigError("Shields API is not available. Please enable shields API.")
        
        error_msg: str = "{type} shield '{shield_id}' is not available. Please provide a valid shield_id in the benchmark metadata."
        
        for shield_id in llm_io_shield_mapping["input"]:
            if not await self.shields_api.get_shield(GetShieldRequest(identifier=shield_id)):
                raise GarakValidationError(error_msg.format(type="Input", shield_id=shield_id))
            
        for shield_id in llm_io_shield_mapping["output"]:
            if not await self.shields_api.get_shield(GetShieldRequest(identifier=shield_id)):
                raise GarakValidationError(error_msg.format(type="Output", shield_id=shield_id))

    async def _build_command(
        self,
        benchmark_config: "BenchmarkConfig",
        garak_config: "GarakCommandConfig",
        provider_params: Dict[str, Any],
        scan_report_prefix: Optional[str] = None
    ) -> dict:
        """Build the garak command to run the scan.
        
        Args:
            benchmark_config: Configuration for the evaluation task
            garak_config: Garak command configuration
            provider_params: Provider-specific parameters
            scan_report_prefix: Optional prefix for the scan report (used in inline mode)
            
        Returns:
            Dictionary of command arguments for garak
            
        Raises:
            GarakValidationError: If configuration is invalid
        """
        from llama_stack_provider_trustyai_garak import shield_scan
        
        garak_config.plugins.generators = await self._get_generator_options(benchmark_config, provider_params, garak_config)
        if not benchmark_config.eval_candidate.type == "model":
            raise GarakValidationError("Eval candidate type must be 'model'")

        if bool(provider_params.get("shield_ids", []) or provider_params.get("shield_config", {})):
            garak_config.plugins.target_type = self._config.garak_model_type_function
            garak_config.plugins.target_name = f"{shield_scan.__name__}#simple_shield_orchestrator"
        else:
            garak_config.plugins.target_type = self._config.garak_model_type_openai
            garak_config.plugins.target_name = benchmark_config.eval_candidate.model
        
        if scan_report_prefix:
            garak_config.reporting.report_prefix = scan_report_prefix.strip()
        
        garak_config.plugins.probe_spec = self._normalize_list_arg(garak_config.plugins.probe_spec)
        garak_config.plugins.detector_spec = self._normalize_list_arg(garak_config.plugins.detector_spec)
        garak_config.plugins.buff_spec = self._normalize_list_arg(garak_config.plugins.buff_spec)
        
        cmd_config: dict = garak_config.to_dict()
        
        # Extract probes from garak_config.plugins.probe_spec
        if garak_config.plugins and garak_config.plugins.probe_spec and not garak_config.run.probe_tags:
            probes = garak_config.plugins.probe_spec
        
            # Validate probes if in inline mode (when all_probes is populated)
            # Remote mode skips validation - Garak validates in the container
            if probes and probes != "all" and self.all_probes:
                for probe in probes.split(","):
                    if probe not in self.all_probes:
                        raise GarakValidationError(
                                f"Probe '{probe}' not found in garak. "
                                "Please provide valid garak probe name. "
                        )
        
        return cmd_config

    async def _validate_run_eval_request(
        self, benchmark_id: str, benchmark_config: "BenchmarkConfig"
    ) -> Tuple[GarakCommandConfig, Dict[str, Any]]:
        """Validate run_eval request and return benchmark and metadata.
        
        Args:
            benchmark_id: The benchmark identifier
            benchmark_config: Configuration for the evaluation task
            
        Returns:
            Tuple of (garak_command_config, provider_params)
            
        Raises:
            GarakValidationError: If validation fails
        """
        
        if not isinstance(benchmark_config, BenchmarkConfig):
            raise GarakValidationError("Required benchmark_config to be of type BenchmarkConfig")
        
        # Validate that the benchmark exists
        benchmark = await self.get_benchmark(benchmark_id)
        if not benchmark:
            available_benchmarks = list(self.benchmarks.keys())
            raise BenchmarkNotFoundError(
                f"Benchmark '{benchmark_id}' not found. "
                f"Available benchmarks: {', '.join(available_benchmarks[:10])}"
                f"{'...' if len(available_benchmarks) > 10 else ''}"
            )
        
        benchmark_metadata: dict = getattr(benchmark, "metadata", {})
        
        garak_config, provider_params = self._parse_benchmark_metadata(benchmark_metadata)

        return garak_config, provider_params

    async def job_result(self, request: JobResultRequest, prefix: str = "") -> EvaluateResponse:
        """Get the result of a job (common implementation).
        
        Args:
            request: Job result request containing benchmark_id and job_id
            prefix: Optional prefix for scan reports
        Returns:
            EvaluateResponse with results or empty response
            
        Raises:
            BenchmarkNotFoundError: If benchmark not found
        """
        import json
        
        benchmark_id = request.benchmark_id
        job_id = request.job_id
        
        stored_benchmark = await self.get_benchmark(benchmark_id)
        if not stored_benchmark:
            raise BenchmarkNotFoundError(f"Benchmark {benchmark_id} not found")

        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return EvaluateResponse(generations=[], scores={})
        
        if job.status in [JobStatus.scheduled, JobStatus.in_progress]:
            logger.warning(f"Job {job_id} is not completed")
            return EvaluateResponse(generations=[], scores={})
        
        elif job.status == JobStatus.failed:
            logger.warning(f"Job {job_id} failed")
            return EvaluateResponse(generations=[], scores={})
        
        elif job.status == JobStatus.cancelled:
            logger.warning(f"Job {job_id} was cancelled")
            return EvaluateResponse(generations=[], scores={})
        
        elif job.status == JobStatus.completed:
            if self._job_metadata[job_id].get(f"{prefix}scan_result.json"):
                scan_result_file_id: str = self._job_metadata[job_id].get(f"{prefix}scan_result.json", "")
                scan_result = await self.file_api.openai_retrieve_file_content(RetrieveFileContentRequest(file_id=scan_result_file_id))
                return EvaluateResponse(**json.loads(scan_result.body.decode("utf-8")))
            else:
                logger.error(f"No {prefix}scan_result.json file found for job {job_id}")
                return EvaluateResponse(generations=[], scores={})
        
        else:
            logger.warning(f"Job {job_id} has an unknown status: {job.status}")
            return EvaluateResponse(generations=[], scores={})

    async def evaluate_rows(
        self,
        request: EvaluateRowsRequest,
    ) -> EvaluateResponse:
        """Evaluate rows (not implemented for Garak).
        
        Args:
            request: Evaluate rows request containing benchmark_id, input_rows, scoring_functions, and benchmark_config
            
        Raises:
            NotImplementedError: This method is not implemented for Garak
        """
        raise NotImplementedError("evaluate_rows is not implemented for Garak provider")

    def _convert_datetime_to_str(self, datetime_obj) -> str:
        """Convert a datetime object to ISO format string.
        
        Args:
            datetime_obj: Datetime object to convert
            
        Returns:
            ISO format string representation
        """
        from datetime import datetime
        return datetime_obj.isoformat() if isinstance(datetime_obj, datetime) else str(datetime_obj)

