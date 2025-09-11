from llama_stack.apis.eval import Eval, EvaluateResponse, BenchmarkConfig
from llama_stack.apis.inference import SamplingParams, SamplingStrategy, TopPSamplingStrategy, TopKSamplingStrategy
from llama_stack.providers.datatypes import ProviderSpec, BenchmarksProtocolPrivate
from llama_stack.apis.datatypes import Api
from llama_stack.apis.files import Files
from llama_stack.apis.benchmarks import Benchmark, Benchmarks
from llama_stack.apis.common.job_types import Job, JobStatus
from typing import List, Dict, Optional, Any, Union, Set
import os
import logging
import json
import re
from ..config import GarakRemoteConfig, GarakScanConfig
from datetime import datetime
from llama_stack_provider_trustyai_garak import shield_scan
from llama_stack.apis.safety import Safety
from llama_stack.apis.shields import Shields
from ..errors import GarakError, GarakConfigError, GarakValidationError, BenchmarkNotFoundError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
JOB_ID_PREFIX = "garak-job-"

class GarakRemoteEvalAdapter(Eval, BenchmarksProtocolPrivate):
    def __init__(self, config: GarakRemoteConfig, deps:dict[Api, ProviderSpec]):
        super().__init__(config)
        self._config: GarakRemoteConfig = config
        self.file_api: Files = deps[Api.files]
        self.benchmarks_api: Benchmarks = deps[Api.benchmarks]
        self.safety_api: Optional[Safety] = deps.get(Api.safety, None)
        self.shields_api: Optional[Shields] = deps.get(Api.shields, None)
        self.scan_config = GarakScanConfig()
        self.benchmarks: Dict[str, Benchmark] = {} # benchmark_id -> benchmark
        self.all_probes: Set[str] = set()
        self._initialized: bool = False
        self._s3_bucket: str = os.getenv('AWS_S3_BUCKET', 'pipeline-artifacts')
        self.s3_client = None
        self.kfp_client = None

        # Job management
        self._jobs: Dict[str, Job] = {}
        self._job_metadata: Dict[str, Dict[str, str]] = {}

    async def initialize(self) -> None:
        """Initialize the Garak provider"""
        logger.info("Initializing Garak provider")
        
        self._ensure_garak_installed()
        self.all_probes = self._get_all_probes()
        error_msg_template = "{provided_api_name} API is provided but {missing_api_name} API is not provided. Please provide both APIs."
        if self.shields_api and not self.safety_api:
            raise GarakConfigError(error_msg_template.format(provided_api_name="Shields", missing_api_name="Safety"))
        elif not self.shields_api and self.safety_api:
            raise GarakConfigError(error_msg_template.format(provided_api_name="Safety", missing_api_name="Shields"))
        
        self._verify_ssl = self._config.tls_verify
        if isinstance(self._verify_ssl, str):
            if self._verify_ssl.lower() in ("true", "1", "yes", "on"):
                self._verify_ssl = True
            elif self._verify_ssl.lower() in ("false", "0", "no", "off"):
                self._verify_ssl = False

        # # TODO: Do we REALLY need S3 client? Is there a better way to get the mapping from pod?
        # self._create_s3_client()
        
        # self._create_kfp_client()

        self._initialized = True
    
    def _ensure_s3_client(self):
        if not self.s3_client:
            self._create_s3_client()
    
    def _ensure_kfp_client(self):
        if not self.kfp_client:
            self._create_kfp_client()
    
    def _create_s3_client(self):
        try:
            import boto3
            self.s3_client = boto3.client('s3',
                                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                                        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
                                        endpoint_url=os.getenv('AWS_S3_ENDPOINT'),  # if using MinIO
                                        verify=self._verify_ssl,
                                    )
        except ImportError as e:
            raise GarakError(
                "Boto3 is not installed. Install with: pip install boto3"
            ) from e
        except Exception as e:
            raise GarakError(
                f"Unable to connect to S3."
            ) from e

    def _create_kfp_client(self):
        try:
            from kfp import Client
            from kfp_server_api.exceptions import ApiException

            ssl_cert = None
            if isinstance(self._verify_ssl, str):
                ssl_cert = self._verify_ssl
                verify_ssl = True
            else:
                verify_ssl = self._verify_ssl

            # TODO: Is this the best way to get the token?
            token = self._get_token()
            self.kfp_client = Client(
                host=self._config.kubeflow_config.pipelines_endpoint,
                existing_token=token,
                verify_ssl=verify_ssl,
                ssl_ca_cert=ssl_cert
            )
        except ImportError:
            raise GarakError(
                "Kubeflow Pipelines SDK not available. Install with: pip install -e .[remote]"
            )
        except ApiException as e:
            raise GarakError(
                "Unable to connect to Kubeflow Pipelines. Please check if you are logged in to the correct cluster. "
                "If you are logged in, please check if the token & pipeline endpoint is valid. "
                "If you are not logged in, please run `oc login` command first."
            ) from e
        except Exception as e:
            raise GarakError(
                f"Unable to connect to Kubeflow Pipelines."
            ) from e
    
    def _get_token(self) -> str:
        try:
            from kubernetes.client.configuration import Configuration
            from kubernetes.config.kube_config import load_kube_config
            from kubernetes.config.config_exception import ConfigException
            from kubernetes.client.exceptions import ApiException

            config = Configuration()

            load_kube_config(client_configuration=config)
            token = config.api_key["authorization"].split(" ")[-1]
            return token
        except (KeyError, ConfigException) as e:
            raise ApiException(
                401, "Unauthorized, try running command like `oc login` first"
            ) from e
        except ImportError as e:
            raise GarakError(
                "Kubernetes client is not installed. Install with: pip install kubernetes"
            ) from e

    def _resolve_framework_to_probes(self, framework_id: str) -> List[str]:
        """Resolve a framework ID to a list of probes using taxonomy filters
        
        Args:
            framework_id: The framework identifier (e.g., 'owasp_llm_top10')
            
        Returns:
            List of probe names that match the framework's taxonomy filters
        """
        framework_info = self.scan_config.FRAMEWORK_PROFILES.get(framework_id)
        if not framework_info:
            raise GarakValidationError(f"Unknown framework: {framework_id}")
        
        taxonomy_filters = framework_info.get("taxonomy_filters", [])
        if not taxonomy_filters:
            logger.warning(f"No taxonomy filters defined for framework {framework_id}")
            return []
        
        # Import garak's config parsing functionality (unfortunately not public API)
        try:
            from garak._config import parse_plugin_spec

        except ImportError:
            raise GarakError(
                "Unable to import garak's parse_plugin_spec. The internal API may have changed."
            )
        
        resolved_probes = []
        
        # For each taxonomy filter, get matching probes
        for tag_filter in taxonomy_filters:
            try:
                # Use garak's built-in tag filtering
                probes, _ = parse_plugin_spec("all", "probes", probe_tag_filter=tag_filter)
                resolved_probes.extend(probes)
                
            except Exception as e:
                logger.error(f"Error resolving probes for tag filter '{tag_filter}': {e}")
                continue
        
        # Remove duplicates and garak prefix
        unique_probes = list(set([p.replace("probes.", "") for p in resolved_probes]))
        unique_probes.sort()
        
        logger.info(f"Framework '{framework_id}' resolved to {len(unique_probes)} probes: {unique_probes[:5]}{'...' if len(unique_probes) > 5 else ''}")
        
        return unique_probes
        
    def _ensure_garak_installed(self):
        """Ensure garak is installed"""
        try:
            import garak
        except ImportError:
            raise GarakError(
                "Garak is not installed. Please install it with: "
                "pip install garak"
            )
    
    def _get_all_probes(self) -> Set[str]:
        ## unfortunately, garak don't have a public API to list all probes
        ## so we need to enumerate all probes manually from private API
        try:
            from garak._plugins import enumerate_plugins
        except ImportError:
            raise GarakError(
                "Unable to import garak's enumerate_plugins. The internal API may have changed."
            )
        probes_names = enumerate_plugins(category="probes", skip_base_classes=True)
        plugin_names = [p.replace(f"probes.", "") for p, _ in probes_names]
        module_names = set([m.split(".")[0] for m in plugin_names])
        plugin_names += module_names
        return set(plugin_names)

    async def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """Get a benchmark by its id"""
        # return self.benchmarks.get(benchmark_id)
        return await self.benchmarks_api.get_benchmark(benchmark_id)
    
    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark by checking if it's a pre-defined scan profile or compliance framework profile"""
        
        if benchmark.identifier in self.scan_config.SCAN_PROFILES | self.scan_config.FRAMEWORK_PROFILES:
            logger.info(f"Benchmark '{benchmark.identifier}' is a pre-defined scan profile or compliance framework profile. It is not recommended to register it as a custom benchmark.")

        if not benchmark.metadata:
            logger.info(f"Benchmark '{benchmark.identifier}' is pre-defined but has no metadata provided. Using default metadata.")
            benchmark.metadata = self.scan_config.SCAN_PROFILES.get(benchmark.identifier, {}) or self.scan_config.FRAMEWORK_PROFILES.get(benchmark.identifier, {})
        
        if not benchmark.metadata.get("probes", None):
            if benchmark.identifier in self.scan_config.SCAN_PROFILES:
                logger.info(f"Benchmark '{benchmark.identifier}' is a pre-defined legacy scan profile but has no probes provided. Using default probes.")
                benchmark.metadata["probes"] = self.scan_config.SCAN_PROFILES[benchmark.identifier]["probes"]
            elif benchmark.identifier in self.scan_config.FRAMEWORK_PROFILES:
                logger.info(f"Benchmark '{benchmark.identifier}' is a pre-defined compliance framework profile but has no probes provided. Using default probes.")
                benchmark.metadata["probes"] = self._resolve_framework_to_probes(benchmark.identifier)

        self.benchmarks[benchmark.identifier] = benchmark
    
    def _get_job_id(self) -> str:
        """Generate a unique job ID.

        Returns:
            Unique job ID string
        """
        import uuid

        return f"{JOB_ID_PREFIX}{str(uuid.uuid4())}"
    
    async def run_eval(self, benchmark_id: str, benchmark_config: BenchmarkConfig) -> Dict[str, Union[str, Dict[str, str]]]:
        """Run an evaluation for a specific benchmark and configuration.

        Args:
            benchmark_id: The benchmark id
            benchmark_config: Configuration for the evaluation task
        """
        if not self._initialized:
            await self.initialize()
        
        if not isinstance(benchmark_config, BenchmarkConfig):
            raise GarakValidationError("Required benchmark_config to be of type BenchmarkConfig")
        
        # Validate that the benchmark exists
        benchmark = await self.get_benchmark(benchmark_id)
        if not benchmark:
            available_benchmarks = list(self.benchmarks.keys())
            raise BenchmarkNotFoundError(f"Benchmark '{benchmark_id}' not found. "
                           f"Available benchmarks: {', '.join(available_benchmarks[:10])}{'...' if len(available_benchmarks) > 10 else ''}")
        
        job_id = self._get_job_id()
        job = Job(
            job_id=job_id,
            status=JobStatus.scheduled
        )
        self._jobs[job_id] = job
        
        stored_benchmark = await self.get_benchmark(benchmark_id)
        benchmark_metadata: dict = getattr(stored_benchmark, "metadata", {})

        try:
            if not benchmark_metadata.get("probes", None):
                raise GarakValidationError("No probes found for benchmark. Please specify probes list in the benchmark metadata.")
                
            scan_profile_config:dict = {
                "probes": benchmark_metadata["probes"],
                "timeout": benchmark_metadata.get("timeout", self._config.timeout)
            }

            cmd: List[str] = await self._build_command(benchmark_config, benchmark_id, scan_profile_config)

            from .kfp_utils.pipeline import garak_scan_pipeline
            
            self._ensure_kfp_client()
            
            run = self.kfp_client.create_run_from_pipeline_func(
                garak_scan_pipeline,
                arguments={
                    "command": cmd,
                    "llama_stack_url": self._config.base_url.rstrip("/").removesuffix("/v1"),
                    "job_id": job_id,
                    "eval_threshold": float(benchmark_metadata.get("eval_threshold", self.scan_config.VULNERABLE_SCORE)),
                    "timeout_seconds": int(scan_profile_config.get("timeout", self._config.timeout)),
                    "max_retries": int(benchmark_metadata.get("max_retries", 3)),
                    "use_gpu": benchmark_metadata.get("use_gpu", False),
                    "verify_ssl": str(self._verify_ssl),
                },
                run_name=f"garak-{benchmark_id.split('::')[-1]}-{job_id.removeprefix(JOB_ID_PREFIX)}",
                namespace=self._config.kubeflow_config.namespace,
                experiment_name=self._config.kubeflow_config.experiment_name
            )
            
            self._job_metadata[job_id] = {
                "created_at": self._convert_datetime_to_str(run.run_info.created_at), 
                "kfp_run_id": run.run_id}
                
            return {"job_id": job_id, "status": job.status, "metadata": self._job_metadata.get(job_id, {})}
        except Exception as e:
            logger.error(f"Error running eval: {e}")
            job.status = JobStatus.failed
            self._job_metadata[job.job_id]["error"] = str(e)
        
    async def _build_command(self, benchmark_config: BenchmarkConfig, benchmark_id: str, scan_profile_config: dict) -> List[str]:
        """Build the command to run the scan.

        Args:
            benchmark_config: Configuration for the evaluation task
            scan_report_prefix: Prefix for the scan report
            scan_log_file: Path to the scan log file
            scan_profile_config: Configuration for the scan profile
        """
        stored_benchmark = await self.get_benchmark(benchmark_id)
        if not stored_benchmark:
            raise BenchmarkNotFoundError(f"Benchmark {benchmark_id} not found")

        benchmark_metadata: dict = getattr(stored_benchmark, "metadata", {})

        generator_options:dict = await self._get_generator_options(benchmark_config, benchmark_metadata)
        if not benchmark_config.eval_candidate.type == "model":
            raise GarakValidationError("Eval candidate type must be 'model'")

        if bool(benchmark_metadata.get("shield_ids", []) or benchmark_metadata.get("shield_config", {})):
            model_type: str = self._config.garak_model_type_function
            model_name: str = f"{shield_scan.__name__}#simple_shield_orchestrator" 
        else:
            model_type: str = self._config.garak_model_type_openai
            model_name: str = benchmark_config.eval_candidate.model
            
        cmd: List[str] = ["garak",
                          "--model_type", model_type,
                          "--model_name", model_name,
                          "--generator_options", json.dumps(generator_options),
                        #   "--report_prefix", scan_report_prefix.strip(),
                          ]
        
        cmd.extend(["--parallel_attempts", str(benchmark_metadata.get("parallel_attempts", self.scan_config.parallel_probes))])
        cmd.extend(["--generations", str(benchmark_metadata.get("generations", 1))])

        if "seed" in benchmark_metadata:
            cmd.extend(["--seed", str(benchmark_metadata["seed"])])

        if "deprefix" in benchmark_metadata:
            cmd.extend(["--deprefix", benchmark_metadata["deprefix"]])

        if "eval_threshold" in benchmark_metadata:
            cmd.extend(["--eval_threshold", str(benchmark_metadata["eval_threshold"])])
        
        if "probe_tags" in benchmark_metadata:
            cmd.extend(["--probe_tags", self._normalize_list_arg(benchmark_metadata["probe_tags"])])
        
        if "probe_options" in benchmark_metadata:
            cmd.extend(["--probe_options", json.dumps(benchmark_metadata["probe_options"])])
        
        if "detectors" in benchmark_metadata:
            cmd.extend(["--detectors", self._normalize_list_arg(benchmark_metadata["detectors"])])
        
        if "extended_detectors" in benchmark_metadata:
            cmd.extend(["--extended_detectors", self._normalize_list_arg(benchmark_metadata["extended_detectors"])])
        
        if "detector_options" in benchmark_metadata:
            cmd.extend(["--detector_options", json.dumps(benchmark_metadata["detector_options"])])
        
        if "buffs" in benchmark_metadata:
            cmd.extend(["--buffs", self._normalize_list_arg(benchmark_metadata["buffs"])])
        
        if "buff_options" in benchmark_metadata:
            cmd.extend(["--buff_options", json.dumps(benchmark_metadata["buff_options"])])
        
        if "harness_options" in benchmark_metadata:
            cmd.extend(["--harness_options", json.dumps(benchmark_metadata["harness_options"])])
        
        if "taxonomy" in benchmark_metadata:
            cmd.extend(["--taxonomy", benchmark_metadata["taxonomy"]])
        
        if "generate_autodan" in benchmark_metadata:
            cmd.extend(["--generate_autodan", benchmark_metadata["generate_autodan"]])
    
        # Add probes
        probes = scan_profile_config["probes"]
        if isinstance(probes, str):
            if "," in probes:
                probes = probes.split(",")
            else:
                probes = [probes]
        
        if probes != ["all"]:
            for probe in probes:
                if probe not in self.all_probes:
                    raise GarakValidationError(f"Probe '{probe}' not found in garak. "
                                     "Please provide valid garak probe name. "
                                     "Or you can just use predefined scan profiles ('quick', 'standard') as benchmark_id.")
            cmd.extend(["--probes", ",".join(probes)])
        return cmd
    
    def _normalize_list_arg(self, arg: str | list[str]) -> str:
        """Normalize the list argument to a string."""
        if isinstance(arg, str):
            return arg
        else:
            return ",".join(arg)

    async def _get_generator_options(self, benchmark_config: BenchmarkConfig, benchmark_metadata: dict) -> dict:
        """Get the generator options based on the availability of the shields."""
        if bool(benchmark_metadata.get("shield_ids", []) or benchmark_metadata.get("shield_config", {})):
            return await self._get_function_based_generator_options(benchmark_config, benchmark_metadata)
        else:
            return await self._get_openai_compatible_generator_options(benchmark_config, benchmark_metadata)

    async def _get_openai_compatible_generator_options(self, benchmark_config: BenchmarkConfig, benchmark_metadata: dict) -> dict:
        """Get the generator options for the OpenAI compatible generator."""

        if 'generator_options' in benchmark_metadata:
            return benchmark_metadata['generator_options']
        
        base_url: str = self._config.base_url.rstrip("/")
        if not base_url.endswith("openai/v1"):
            if base_url.endswith("/v1"):
                base_url = f"{base_url}/openai/v1"
            else:
                base_url = f"{base_url}/v1/openai/v1"


        generator_options = {
                    "openai": {
                        "OpenAICompatible": {
                            "uri": base_url,
                            "model": benchmark_config.eval_candidate.model,
                            "api_key": os.getenv("OPENAICOMPATIBLE_API_KEY", "DUMMY"),
                            "suppressed_params": ["n"]
                        }
                    }
                }
        # Add extra params
        sampling_params:SamplingParams = benchmark_config.eval_candidate.sampling_params
        valid_params:dict = {}

        # TODO: use model_dump() here..?
        strategy: SamplingStrategy = sampling_params.strategy
        valid_params["strategy"] = strategy.type
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
        if sampling_params.repetition_penalty is not None:
            valid_params['repetition_penalty'] = sampling_params.repetition_penalty
        if sampling_params.stop is not None:
            valid_params['stop'] = sampling_params.stop
        
        if valid_params:
            generator_options["openai"]["OpenAICompatible"].update(valid_params)
        return generator_options
    
    async def _get_function_based_generator_options(self, benchmark_config: BenchmarkConfig, benchmark_metadata: dict) -> dict:
        """Get the generator options for the custom function-based generator."""
        
        base_url: str = self._config.base_url.rstrip("/")
        
        # check if base_url ends with /v{number} and if not, add v1
        if not re.match(r"^.*\/v\d+$", base_url):
            base_url = f"{base_url}/v1"
        
        # Map the shields to the input and output of LLM
        llm_io_shield_mapping: dict = {
            "input": [],
            "output": []
        }

        if not benchmark_metadata:
            logger.warning(f"No benchmark metadata found.")
        else:
            ## get the shield_ids and/or shield_config from the benchmark metadata
            shield_ids: List[str] = benchmark_metadata.get("shield_ids", [])
            shield_config: dict = benchmark_metadata.get("shield_config", {})
            if not shield_ids and not shield_config:
                logger.warning("No shield_ids or shield_config found in the benchmark metadata")
            elif shield_ids:
                if shield_config:
                    logger.warning("Both shield_ids and shield_config found in the benchmark metadata. "
                                "Using shield_ids only.")
                llm_io_shield_mapping["input"] = shield_ids
            elif shield_config:
                if not shield_config.get("input", None) and not shield_config.get("output", None):
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
                                "base_url": base_url,
                                "llm_io_shield_mapping": llm_io_shield_mapping,
                                "max_workers": self._config.max_workers
                            }
                        }
                    }
                }
        return generator_options
    
    async def _check_shield_availability(self, llm_io_shield_mapping: dict):
        """Check the availability of the shields. and raise an error if any shield is not available."""
        if not self.shields_api:
            raise GarakConfigError("Shields API is not available. Please enable shields API.")
        
        error_msg: str = "{type} shield '{shield_id}' is not available. Please provide a valid shield_id in the benchmark metadata."
        
        for shield_id in llm_io_shield_mapping["input"]:
            if not await self.shields_api.get_shield(shield_id):
                raise GarakValidationError(error_msg.format(type="Input", shield_id=shield_id))
            
        for shield_id in llm_io_shield_mapping["output"]:
            if not await self.shields_api.get_shield(shield_id):
                raise GarakValidationError(error_msg.format(type="Output", shield_id=shield_id))

    def _map_kfp_run_state_to_job_status(self, run_state) -> JobStatus:
        """Map the KFP run state to the job status."""
        from kfp_server_api.models import V2beta1RuntimeState

        if run_state in [V2beta1RuntimeState.RUNTIME_STATE_UNSPECIFIED, V2beta1RuntimeState.PENDING]:
            return JobStatus.scheduled
        elif run_state in [V2beta1RuntimeState.RUNNING, V2beta1RuntimeState.CANCELING, V2beta1RuntimeState.PAUSED]:
            return JobStatus.in_progress
        elif run_state in [V2beta1RuntimeState.SUCCEEDED, V2beta1RuntimeState.SKIPPED]:
            return JobStatus.completed
        elif run_state == V2beta1RuntimeState.FAILED:
            return JobStatus.failed
        elif run_state == V2beta1RuntimeState.CANCELED:
            return JobStatus.cancelled
        else:
            logger.warning(f"KFP run has an unknown status: {run_state}, mapping to scheduled")
            return JobStatus.scheduled
    
    def _convert_datetime_to_str(self, datetime_obj: datetime) -> str:
        """Convert a datetime object to string."""
        return datetime_obj.isoformat() if isinstance(datetime_obj, datetime) else str(datetime_obj)

    async def job_status(self, benchmark_id: str, job_id: str) -> Dict[str, Union[str, Dict[str, str]]]:
        """Get the status of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        from kfp_server_api.models import V2beta1Run
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return {"status": "not_found", "job_id": job_id}
        
        metadata: dict = self._job_metadata.get(job_id, {})

        if "kfp_run_id" not in metadata:
            logger.warning(f"Job {job_id} has no kfp run id")
            return {"status": "not_found", "job_id": job_id}
        
        try:
            self._ensure_kfp_client()
            run: V2beta1Run = self.kfp_client.get_run(metadata["kfp_run_id"])
            job.status = self._map_kfp_run_state_to_job_status(run.state)
            if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
                self._job_metadata[job_id]['finished_at'] = self._convert_datetime_to_str(run.finished_at)

                if job.status == JobStatus.completed:
                    # Read from S3
                    try:
                        self._ensure_s3_client()

                        response = self.s3_client.get_object(
                            Bucket=self._s3_bucket,
                            Key=f'{job_id}.json'
                        )
                        file_id_mapping: dict = json.loads(response['Body'].read())
                        
                        # Store the file IDs in job metadata
                        # logger.info(f"====== File mapping: {file_id_mapping} ======")
                        for key, value in file_id_mapping.items():
                            self._job_metadata[job_id][key] = value
                        
                        # # Clean up - delete the file from S3
                        # self.s3_client.delete_object(
                        #     Bucket=self._s3_bucket,
                        #     Key=f'{job_id}.json'
                        # )
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from S3 object {job_id}.json: {e}")
                    except Exception as e:
                        logger.warning(f"Could not retrieve outputs from S3 for job {job_id}: {e}")
        except Exception as e:
            logger.error(f"Error getting KFP run {metadata['kfp_run_id']}: {e}")
            return {"status": "not_found", "job_id": job_id}
        
        return {"job_id": job_id, "status": job.status, "metadata": self._job_metadata.get(job_id, {})}
    
    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        job_status = await self.job_status(benchmark_id, job_id)
        stored_benchmark = await self.get_benchmark(benchmark_id)
        if not stored_benchmark:
            raise BenchmarkNotFoundError(f"Benchmark {benchmark_id} not found")

        if job_status["status"] == "not_found":
            logger.warning(f"Job {job_id} not found")
            return EvaluateResponse(generations=[], scores={})
        
        if job_status["status"] in [JobStatus.scheduled, JobStatus.in_progress]:
            logger.warning(f"Job {job_id} is not completed")
            return EvaluateResponse(generations=[], scores={})
        
        elif job_status["status"] == JobStatus.failed:
            logger.warning(f"Job {job_id} failed")
            return EvaluateResponse(generations=[], scores={})
        
        elif job_status["status"] == JobStatus.cancelled:
            logger.warning(f"Job {job_id} was cancelled")
            return EvaluateResponse(generations=[], scores={})
        
        elif job_status["status"] == JobStatus.completed:
            if self._job_metadata[job_id].get("scan_result.json", None):
                scan_result_file_id: str = self._job_metadata[job_id].get("scan_result.json", "")
                results = await self.file_api.openai_retrieve_file_content(scan_result_file_id)
                return EvaluateResponse(**json.loads(results.body.decode("utf-8")))

            else:
                logger.error(f"Error parsing scan results for job {job_id}")
                return EvaluateResponse(generations=[], scores={})

        
        else:
            logger.warning(f"Job {job_id} has an unknown status: {job_status['status']}")
            return EvaluateResponse(generations=[], scores={})
    
    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job and kill the process.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        # check/update the current status of the job
        current_status = await self.job_status(benchmark_id, job_id)
        if current_status["status"] == "not_found":
            return
        elif current_status["status"] in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
            logger.warning(f"Job {job_id} is not running. Can't cancel.")
            return
        else:
            try:
                self._ensure_kfp_client()
                self.kfp_client.terminate_run(self._job_metadata[job_id]["kfp_run_id"])
            except Exception as e:
                logger.error(f"Error cancelling KFP run {self._job_metadata[job_id]['kfp_run_id']}: {e}")
    
    async def evaluate_rows(self, benchmark_id: str, 
                            input_rows: list[dict[str, Any]], 
                            scoring_functions: list[str], 
                            benchmark_config: BenchmarkConfig) -> EvaluateResponse:
        raise NotImplementedError("Not implemented")
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down Garak provider")
        # Kill all running jobs
        for job_id, job in list(self._jobs.items()):
            if job.status in [JobStatus.in_progress, JobStatus.scheduled]:
                await self.job_cancel("placeholder", job_id)
        
        # # Clear all running tasks, jobs and job metadata
        self._jobs.clear()
        self._job_metadata.clear()

        # Close the shield scanning HTTP client
        shield_scan.simple_shield_orchestrator.close()
        