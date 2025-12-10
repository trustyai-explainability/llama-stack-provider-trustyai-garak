from ..compat import (
    EvaluateResponse, 
    BenchmarkConfig,
    Benchmark,
    ProviderSpec, 
    Api, 
    Job, 
    JobStatus,
    OpenAIFilePurpose
)
from typing import List, Dict, Union
import os
import logging
import json
import asyncio
from ..config import GarakRemoteConfig
from ..base_eval import GarakEvalBase
from llama_stack_provider_trustyai_garak import shield_scan
from ..errors import GarakError, GarakConfigError, GarakValidationError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
JOB_ID_PREFIX = "garak-job-"


class GarakRemoteEvalAdapter(GarakEvalBase):
    """Remote Garak evaluation adapter for running scans on Kubeflow Pipelines."""

    def __init__(self, config: GarakRemoteConfig, deps: dict[Api, ProviderSpec]):
        super().__init__(config, deps)
        self._config: GarakRemoteConfig = config
        
        self.kfp_client = None
        self._jobs_lock = asyncio.Lock()  # Will be initialized in async initialize()

    def _ensure_garak_installed(self) -> None:
        """Override: Skip garak check for remote provider - it runs in container."""
        logger.debug("Skipping garak installation check for remote provider (runs in container)")
        pass

    def _get_all_probes(self) -> set[str]:
        """Override: Skip probe enumeration for remote provider - validation happens in pod.
        
        Returns empty set to allow any probe names in benchmark metadata.
        Remote validation will occur when the scan runs in the Kubernetes pod.
        """
        logger.debug("Skipping probe enumeration for remote provider (validated in container)")
        return set()  # Allow any probes; validation happens in the pod

    def _resolve_framework_to_probes(self, framework_id: str) -> List[str]:
        """Override: Skip resolution for remote provider - use probe_tags instead.
        
        For remote execution, we can't resolve frameworks on the server (no garak installed).
        Instead, we return ['all'] and set probe_tags in the metadata, letting garak
        resolve the taxonomy filters at runtime in the KFP pod.
        
        Args:
            framework_id: The framework identifier (e.g., 'trustyai_garak::owasp_llm_top10')
            
        Returns:
            List containing 'all' - actual filtering happens via probe_tags
            
        Raises:
            GarakValidationError: If framework is unknown
        """
        from ..errors import GarakValidationError
        
        framework_info = self.scan_config.FRAMEWORK_PROFILES.get(framework_id)
        if not framework_info:
            raise GarakValidationError(f"Unknown framework: {framework_id}")
        
        taxonomy_filters = framework_info.get('taxonomy_filters', [])
        
        logger.debug(
            f"Remote mode: Framework '{framework_id}' will use probe_tags for filtering. "
            f"Taxonomy filters: {taxonomy_filters}"
        )
        
        # Return 'all' - the actual filtering will happen via --probe_tags in the command
        # The benchmark registration will store taxonomy_filters as probe_tags in metadata
        return ['all']

    async def register_benchmark(self, benchmark: "Benchmark") -> None:
        """Override: Handle frameworks specially for remote execution.
        
        For framework-based benchmarks, store taxonomy_filters as probe_tags
        so garak can resolve them at runtime in the KFP pod.
        """
        # Call parent to handle basic registration
        await super().register_benchmark(benchmark)
        
        # For frameworks, add probe_tags to metadata for runtime resolution
        if benchmark.identifier in self.scan_config.FRAMEWORK_PROFILES:
            framework_info = self.scan_config.FRAMEWORK_PROFILES[benchmark.identifier]
            taxonomy_filters = framework_info.get('taxonomy_filters', [])
            
            if taxonomy_filters and not benchmark.metadata.get('probe_tags'):
                # Store taxonomy filters as probe_tags for garak's --probe_tags flag
                benchmark.metadata['probe_tags'] = taxonomy_filters
                logger.info(
                    f"Framework '{benchmark.identifier}': Set probe_tags={taxonomy_filters} "
                    f"for runtime resolution in KFP pod"
                )
        
        # Update the stored benchmark
        self.benchmarks[benchmark.identifier] = benchmark

    async def initialize(self) -> None:
        """Initialize the remote Garak provider."""
        self._initialize()

        self._initialized = True
        logger.info("Initialized Garak remote provider.")
    
    def _ensure_kfp_client(self):
        if not self.kfp_client:
            self._create_kfp_client()

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

            # Use token from config if provided, otherwise get from kubeconfig
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
        """Get authentication token from environment variable or kubernetes config."""
        if self._config.kubeflow_config.pipelines_api_token:
            logger.info("Using KUBEFLOW_PIPELINES_TOKEN from config")
            return self._config.kubeflow_config.pipelines_api_token.get_secret_value()
        else:
            logger.info("Using authentication token from kubernetes config")
        try:
            from .kfp_utils.utils import _load_kube_config
            from kubernetes.client.exceptions import ApiException
            from kubernetes.config.config_exception import ConfigException

            config = _load_kube_config()
            token = str(config.api_key["authorization"].split(" ")[-1])
        except (KeyError, ConfigException) as e:
            raise ApiException(
                401, "Unauthorized, try running command like `oc login` first"
            ) from e
        except ImportError as e:
            raise GarakError(
                "Kubernetes client is not installed. Install with: pip install kubernetes"
            ) from e
        except Exception as e:
            raise GarakError(
                f"Unable to get authentication token from kubernetes config: {e}. Please run `oc login` and try again."
            ) from e
        return token
    
    async def run_eval(self, benchmark_id: str, benchmark_config: BenchmarkConfig) -> Dict[str, Union[str, Dict[str, str]]]:
        """Run an evaluation for a specific benchmark and configuration.

        Args:
            benchmark_id: The benchmark id
            benchmark_config: Configuration for the evaluation task
            
        Raises:
            GarakValidationError: If benchmark_id or benchmark_config are invalid
            GarakConfigError: If configuration is invalid
            GarakError: If KFP pipeline creation fails
        """
        if not self._initialized:
            await self.initialize()
        
        if not benchmark_id or not isinstance(benchmark_id, str):
            raise GarakValidationError("benchmark_id must be a non-empty string")
        if not benchmark_config:
            raise GarakValidationError("benchmark_config cannot be None")
        
        _, benchmark_metadata = await self._validate_run_eval_request(benchmark_id, benchmark_config)
        
        job_id = self._get_job_id(prefix=JOB_ID_PREFIX)
        job = Job(
            job_id=job_id,
            status=JobStatus.scheduled
        )
        
        async with self._jobs_lock:
            self._jobs[job_id] = job
            self._job_metadata[job_id] = {}  # Initialize metadata dict

        try:
            scan_profile_config: dict = {
                "probes": benchmark_metadata["probes"],
                "timeout": benchmark_metadata.get("timeout", self._config.timeout)
            }

            cmd: List[str] = await self._build_command(benchmark_config, benchmark_id, scan_profile_config)

            from .kfp_utils.pipeline import garak_scan_pipeline
            
            self._ensure_kfp_client()
            
            # Validate config before creating pipeline
            if not self._config.kubeflow_config.namespace:
                raise GarakConfigError("kubeflow_config.namespace is not configured")
            if not self._config.llama_stack_url:
                raise GarakConfigError("llama_stack_url is not configured")
            
            experiment_name = f"trustyai-garak-{self._config.kubeflow_config.namespace}"
            
            llama_stack_url = self._config.llama_stack_url.strip().rstrip("/")
            if not llama_stack_url:
                raise GarakConfigError("llama_stack_url cannot be empty after normalization")
            
            run = self.kfp_client.create_run_from_pipeline_func(
                garak_scan_pipeline,
                arguments={
                    "command": cmd,
                    "llama_stack_url": llama_stack_url,
                    "job_id": job_id,
                    "eval_threshold": float(benchmark_metadata.get("eval_threshold", self.scan_config.VULNERABLE_SCORE)),
                    "timeout_seconds": int(scan_profile_config.get("timeout", self._config.timeout)),
                    "max_retries": int(benchmark_metadata.get("max_retries", 3)),
                    "use_gpu": benchmark_metadata.get("use_gpu", False),
                    "verify_ssl": str(self._verify_ssl),
                },
                run_name=f"garak-{benchmark_id.split('::')[-1]}-{job_id.removeprefix(JOB_ID_PREFIX)}",
                namespace=self._config.kubeflow_config.namespace,
                experiment_name=experiment_name
            )
            
            async with self._jobs_lock:
                self._job_metadata[job_id] = {
                    "created_at": self._convert_datetime_to_str(run.run_info.created_at), 
                    "kfp_run_id": run.run_id}
                
            return {"job_id": job_id, "status": job.status, "metadata": self._job_metadata.get(job_id, {})}
        except Exception as e:
            logger.error(f"Error running eval for {benchmark_id}: {e}")
            async with self._jobs_lock:
                job.status = JobStatus.failed
                self._job_metadata[job.job_id]["error"] = str(e)
            raise e

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
    
    async def job_status(self, benchmark_id: str, job_id: str) -> Dict[str, Union[str, Dict[str, str]]]:
        """Get the status of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        from kfp_server_api.models import V2beta1Run
        
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return {"status": "not_found", "job_id": job_id}
            
            metadata: dict = self._job_metadata.get(job_id, {})

            if "kfp_run_id" not in metadata:
                logger.warning(f"Job {job_id} has no kfp run id")
                return {"status": "not_found", "job_id": job_id}
            
            kfp_run_id = metadata["kfp_run_id"]
        
        try:
            self._ensure_kfp_client()
            run: V2beta1Run = self.kfp_client.get_run(kfp_run_id)
            new_status = self._map_kfp_run_state_to_job_status(run.state)
            
            async with self._jobs_lock:
                job.status = new_status
                if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
                    self._job_metadata[job_id]['finished_at'] = self._convert_datetime_to_str(run.finished_at)

                    if job.status == JobStatus.completed:
                        # Retrieve file_id_mapping from Files API using predictable filename
                        try:
                            # The KFP pod uploads mapping with filename: {job_id}_mapping.json
                            mapping_filename = f"{job_id}_mapping.json"
                            
                            logger.debug(f"Searching for mapping file: {mapping_filename}")
                            
                            if 'mapping_file_id' not in self._job_metadata[job_id]:
                                # List files and find the mapping file by name
                                files_list = await self.file_api.openai_list_files(purpose=OpenAIFilePurpose.BATCH)
                                
                                for file_obj in files_list.data:
                                    if hasattr(file_obj, 'filename') and file_obj.filename == mapping_filename:
                                        self._job_metadata[job_id]['mapping_file_id'] = file_obj.id
                                        logger.debug(f"Found mapping file: {mapping_filename} (ID: {file_obj.id})")
                                        break
                            
                            if mapping_file_id := self._job_metadata[job_id].get('mapping_file_id'):
                                # Retrieve the mapping file via Files API
                                mapping_content = await self.file_api.openai_retrieve_file_content(mapping_file_id)
                                if mapping_content:
                                    file_id_mapping: dict = json.loads(mapping_content.body.decode("utf-8"))
                                    
                                    if not isinstance(file_id_mapping, dict):
                                        raise ValueError(f"Invalid file mapping format: {type(file_id_mapping)}")
                                    
                                    # Store the file IDs in job metadata
                                    for key, value in file_id_mapping.items():
                                        self._job_metadata[job_id][key] = value
                                    
                                    logger.debug(f"Successfully retrieved {len(file_id_mapping)} file IDs from Files API")
                                else:
                                    logger.warning(f"Empty mapping file content for file ID: {mapping_file_id}")
                            else:
                                logger.warning(
                                    f"Could not find mapping file '{mapping_filename}' in Files API. "
                                    f"This might be expected if the pipeline is still running or failed."
                                )
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON from mapping file: {e}")
                        except Exception as e:
                            logger.warning(f"Could not retrieve mapping from Files API for job {job_id}: {e}")
                
                return_metadata = self._job_metadata.get(job_id, {})
                return_status = job.status
        except Exception as e:
            logger.error(f"Error getting KFP run {kfp_run_id}: {e}")
            return {"status": "not_found", "job_id": job_id}
        
        return {"job_id": job_id, "status": return_status, "metadata": return_metadata}
    
    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job (remote-specific: updates job state from KFP first).

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        # Update job status from KFP before getting results
        await self.job_status(benchmark_id, job_id)
        
        return await super().job_result(benchmark_id, job_id, prefix=f"{job_id}_")
    
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
            async with self._jobs_lock:
                kfp_run_id = self._job_metadata[job_id].get("kfp_run_id")
            
            if kfp_run_id:
                try:
                    self._ensure_kfp_client()
                    self.kfp_client.terminate_run(kfp_run_id)
                except Exception as e:
                    logger.error(f"Error cancelling KFP run {kfp_run_id}: {e}")
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down Garak provider")
        
        # Get snapshot of jobs to cancel
        async with self._jobs_lock:
            jobs_to_cancel = [(job_id, job) for job_id, job in self._jobs.items() 
                             if job.status in [JobStatus.in_progress, JobStatus.scheduled]]
        
        # Kill all running jobs
        for job_id, job in jobs_to_cancel:
            await self.job_cancel("placeholder", job_id)
        
        # # Clear all running tasks, jobs and job metadata
        async with self._jobs_lock:
            self._jobs.clear()
            self._job_metadata.clear()

        # Close the shield scanning HTTP client
        shield_scan.simple_shield_orchestrator.close()
        