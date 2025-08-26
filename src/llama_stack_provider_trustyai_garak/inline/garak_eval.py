from llama_stack.apis.eval import Eval, EvaluateResponse, BenchmarkConfig
from llama_stack.apis.inference import SamplingParams, SamplingStrategy, TopPSamplingStrategy, TopKSamplingStrategy
from llama_stack.providers.datatypes import ProviderSpec, BenchmarksProtocolPrivate
from llama_stack.apis.datatypes import Api
from llama_stack.apis.files import Files, OpenAIFilePurpose, OpenAIFileObject
from fastapi import UploadFile, Response
from llama_stack.apis.benchmarks import Benchmark, Benchmarks
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.scoring import ScoringResult
from typing import List, Dict, Optional, Any, Union, Set
import os
import logging
import json
import re
from pathlib import Path
from ..config import GarakEvalProviderConfig, GarakScanConfig
from datetime import datetime
import asyncio
import signal
import shutil
from llama_stack_provider_trustyai_garak import shield_scan
from llama_stack.apis.safety import Safety
from llama_stack.apis.shields import Shields
from ..errors import GarakError, GarakConfigError, GarakValidationError, BenchmarkNotFoundError

logger = logging.getLogger(__name__)

class GarakEvalAdapter(Eval, BenchmarksProtocolPrivate):
    def __init__(self, config: GarakEvalProviderConfig, deps:dict[Api, ProviderSpec]):
        super().__init__(config)
        self._config: GarakEvalProviderConfig = config
        self.file_api: Files = deps[Api.files]
        self.benchmarks_api: Benchmarks = deps[Api.benchmarks]
        self.safety_api: Optional[Safety] = deps.get(Api.safety, None)
        self.shields_api: Optional[Shields] = deps.get(Api.shields, None)
        self.scan_config = GarakScanConfig()
        self.benchmarks: Dict[str, Benchmark] = {} # benchmark_id -> benchmark
        self.all_probes: Set[str] = set()
        self._initialized: bool = False

        # Job management
        self._jobs: Dict[str, Job] = {}
        self._job_metadata: Dict[str, Dict[str, str]] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {} # job_id -> running task
        ## simple concurrency control for using semaphore
        self._job_semaphore: Optional[asyncio.Semaphore] = None

    async def initialize(self) -> None:
        """Initialize the Garak provider"""
        logger.info("Initializing Garak provider")

        self.scan_config.scan_dir.mkdir(exist_ok=True, parents=True)
        
        self._ensure_garak_installed()
        self.all_probes = self._get_all_probes()
        error_msg_template = "{provided_api_name} API is provided but {missing_api_name} API is not provided. Please provide both APIs."
        if self.shields_api and not self.safety_api:
            raise GarakConfigError(error_msg_template.format(provided_api_name="Shields", missing_api_name="Safety"))
        elif not self.shields_api and self.safety_api:
            raise GarakConfigError(error_msg_template.format(provided_api_name="Safety", missing_api_name="Shields"))

        self._job_semaphore = asyncio.Semaphore(self._config.max_concurrent_jobs)
        logger.info(f"Initialized Garak provider with max concurrent jobs: {self._config.max_concurrent_jobs}")

        self._initialized = True

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
        from garak._config import parse_plugin_spec
        
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
        from garak._plugins import enumerate_plugins
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

        return f"garak-job-{str(uuid.uuid4())}"
    
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
        self._job_metadata[job_id] = {"created_at": datetime.now().isoformat()}

        self._running_tasks[job_id] = asyncio.create_task(self._run_scan_with_semaphore(job, benchmark_id, benchmark_config), name=f"garak-job-{job_id}")
        return {"job_id": job_id, "status": job.status, "metadata": self._job_metadata.get(job_id, {})}
    
    async def _run_scan_with_semaphore(self, job: Job, benchmark_id: str, benchmark_config: BenchmarkConfig):
        """Wrapper to run the scan with semaphore"""
        async with self._job_semaphore:
            logger.info(f"Starting job {job.job_id} (Slots used: {self._config.max_concurrent_jobs - self._job_semaphore._value}/{self._config.max_concurrent_jobs})")
            await self._run_scan(job, benchmark_id, benchmark_config)
        
    async def _run_scan(self, job: Job, benchmark_id: str, benchmark_config: BenchmarkConfig):
        """Run the scan with the given command.

        Args:
            job: The job object
            benchmark_id: The benchmark id
            benchmark_config: The benchmark configuration
        """
        
        stored_benchmark = await self.get_benchmark(benchmark_id)
        benchmark_metadata: dict = getattr(stored_benchmark, "metadata", {})

        job.status = JobStatus.in_progress
        self._job_metadata[job.job_id]["started_at"] = datetime.now().isoformat()

        job_scan_dir: Path = self.scan_config.scan_dir / job.job_id
        job_scan_dir.mkdir(exist_ok=True, parents=True)

        scan_log_file: Path = job_scan_dir / "scan.log"
        scan_log_file.touch(exist_ok=True)
        scan_report_prefix: Path = job_scan_dir / "scan"
        
        try:
            if not benchmark_metadata.get("probes", None):
                raise GarakValidationError("No probes found for benchmark. Please specify probes list in the benchmark metadata.")
                
            scan_profile_config:dict = {
                "probes": benchmark_metadata["probes"],
                "timeout": benchmark_metadata.get("timeout", self._config.timeout)
            }

            cmd: List[str] = await self._build_command(benchmark_config, benchmark_id, str(scan_report_prefix), scan_profile_config)
            logger.info(f"Running scan with command: {' '.join(cmd)}")

            env = os.environ.copy()
            env["GARAK_LOG_FILE"] = str(scan_log_file)
            env["GARAK_TLS_VERIFY"] = str(self._config.tls_verify)

            process = await asyncio.create_subprocess_exec(*cmd, 
                                                           stdout=asyncio.subprocess.PIPE, 
                                                           stderr=asyncio.subprocess.PIPE, 
                                                           env=env)
            
            self._job_metadata[job.job_id]["process_id"] = str(process.pid)
            timeout: int = scan_profile_config.get("timeout", self._config.timeout)
            
            _, stderr = await asyncio.wait_for(process.communicate(), 
                                                    timeout=timeout)

            if process.returncode == 0:
                # Upload scan files to file storage
                upload_scan_report: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".report.jsonl"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_report:
                    self._job_metadata[job.job_id]["scan.report.jsonl"] = upload_scan_report.id

                upload_scan_log: OpenAIFileObject = await self._upload_file(
                    file=scan_log_file, 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_log:
                    self._job_metadata[job.job_id]["scan.log"] = upload_scan_log.id

                upload_scan_hitlog: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".hitlog.jsonl"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_hitlog:
                    self._job_metadata[job.job_id]["scan.hitlog.jsonl"] = upload_scan_hitlog.id

                upload_scan_report_html: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".report.html"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_report_html:
                    self._job_metadata[job.job_id]["scan.report.html"] = upload_scan_report_html.id
                
                # parse results
                scan_report_file_id: str = self._job_metadata[job.job_id].get("scan.report.jsonl", "")
                if scan_report_file_id:
                    scan_result = await self._parse_scan_results(scan_report_file_id, job.job_id, benchmark_metadata)

                    # save file and upload results to llama stack
                    scan_result_file = Path(job_scan_dir) / "scan_result.json"
                    with open(scan_result_file, 'w') as f:
                        json.dump(scan_result.model_dump(), f)
                    
                    upload_scan_result: OpenAIFileObject = await self._upload_file(
                        file=scan_result_file, 
                        purpose=OpenAIFilePurpose.ASSISTANTS)
                    if upload_scan_result:
                        self._job_metadata[job.job_id]["scan_result.json"] = upload_scan_result.id

                job.status = JobStatus.completed
                # cleanup the tmp job dir
                shutil.rmtree(job_scan_dir, ignore_errors=True)

            else:
                job.status = JobStatus.failed
                self._job_metadata[job.job_id]["error"] = f"Scan failed with return code {process.returncode} - {stderr.decode('utf-8')}"
        except asyncio.TimeoutError:
            job.status = JobStatus.failed
            self._job_metadata[job.job_id]["error"] = f"Scan timed out after {timeout} seconds."
        except Exception as e:
            job.status = JobStatus.failed
            self._job_metadata[job.job_id]["error"] = str(e)
        finally:
            self._job_metadata[job.job_id]["completed_at"] = datetime.now().isoformat()
            if 'process' in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            self._running_tasks.pop(job.job_id, None)

    async def _upload_file(self, file: Path, purpose: OpenAIFilePurpose) -> Optional[OpenAIFileObject]:
        """Upload a file to the file storage and return the file object.

        Args:
            file: The file to upload
            purpose: The purpose of the file
        """
        if file.exists():
            with open(file, "rb") as f:
                upload_file: OpenAIFileObject = await self.file_api.openai_upload_file(
                    # file: The File object (not file name) to be uploaded.
                    file=UploadFile(file=f, filename=file.name), 
                    purpose=purpose
                )
                return upload_file
        else:
            logger.warning(f"File {file} does not exist")
            return None
        
    async def _build_command(self, benchmark_config: BenchmarkConfig, benchmark_id: str, scan_report_prefix: str, scan_profile_config: dict) -> List[str]:
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
                          "--report_prefix", scan_report_prefix.strip(),
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

    async def _parse_scan_results(self, report_file_id: str, job_id: str, benchmark_metadata: dict) -> EvaluateResponse:
        """Parse the scan results from the report file.
        
        Args:
            report_file_id: The id of the report file
            job_id: The id of the job
        """
        report_file_content: Response = await self.file_api.openai_retrieve_file_content(report_file_id)
        if not report_file_content:
            logger.warning(f"Report file not found for job {job_id}")
            return EvaluateResponse(generations=[], scores={})
        
        try:
            generations: List[Dict[str, Any]] = []
            # Each key is a garak probe/attack_type and value is a ScoringResult object containing scores from respective garak detectors
            scores: Dict[str, ScoringResult] = {} 

            score_rows: Dict[str, List[Dict[str, float]]] = {}
            aggregated_results_by_probe: Dict[str, Dict[str, List[float]]] = {}
            report_file_content_str: str = report_file_content.body.decode("utf-8")
            eval_threshold: float = float(benchmark_metadata.get("eval_threshold", self.scan_config.VULNERABLE_SCORE))

            for line in report_file_content_str.split("\n"):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    generation: Dict[str, Any] = {}

                    # Handle attempt entries
                    if (entry.get("entry_type") == "attempt" and entry.get("status") == 2):
                        probe_name: str = entry.get("probe_classname", "unknown")
                        probe_category: str = probe_name.split('.')[0]

                        detector_results: dict = entry.get("detector_results", {})
                        detector_maxscore_map: Dict[str, float] = {}
                        is_vulnerable: bool = False
                        
                        for detector, _scores in detector_results.items():
                            _scores = _scores if isinstance(_scores, list) else [_scores]
                            if any(score >= eval_threshold for score in _scores):
                                is_vulnerable = True
                            detector_maxscore_map[detector] = max(_scores) if _scores else 0
                        outputs: list = entry.get("outputs", [])

                        generation["probe"] = probe_name
                        # generation["attack_type"] = self.scan_config.PROBE_TO_ATTACK.get(probe_category, AttackType.CUSTOM)
                        generation["vulnerable"] = is_vulnerable
                        generation["prompt"] = entry.get("prompt", "")
                        generation["response"] = outputs[0] if len(outputs) > 0 and outputs[0] else ""
                        generations.append(generation)

                        if probe_name not in score_rows:
                            score_rows[probe_name] = []
                        score_rows[probe_name].append(detector_maxscore_map)

                        if probe_name not in aggregated_results_by_probe:
                            aggregated_results_by_probe[probe_name] = {}
                        for detector, score in detector_maxscore_map.items():
                            if detector not in aggregated_results_by_probe[probe_name]:
                                aggregated_results_by_probe[probe_name][detector] = []
                            aggregated_results_by_probe[probe_name][detector].append(score)
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON line in report file for job {job_id}: {line} - {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error parsing line in report file for job {job_id}: {line} - {e}")
                    continue

            # Calculate the mean of the scores for each probe
            aggregated_results_mean: Dict[str, Dict[str, float]] = {}
            for probe_name, results in aggregated_results_by_probe.items():
                aggregated_results_mean[probe_name] = {}
                for detector, _scores in results.items():
                    detector_mean_score: float = round(sum(_scores) / len(_scores), 3) if _scores else 0
                    aggregated_results_mean[probe_name][f"{detector}_mean"] = detector_mean_score

            if len(aggregated_results_mean.keys()) != len(score_rows.keys()):
                raise GarakValidationError(f"Number of probes in aggregated results ({len(aggregated_results_mean.keys())}) "
                                    f"does not match number of probes in score rows ({len(score_rows.keys())})")
            
            all_probes: List[str] = list(aggregated_results_mean.keys())
            for probe_name in all_probes:
                scores[probe_name] = ScoringResult(
                    score_rows=score_rows[probe_name],
                    aggregated_results=aggregated_results_mean[probe_name]
                    )

            return EvaluateResponse(generations=generations, scores=scores)    

        except Exception as e:
            logger.error(f"Error parsing scan results for job {job_id}: {e}")
            return EvaluateResponse(generations=[], scores={})

    async def job_status(self, benchmark_id: str, job_id: str) -> Dict[str, Union[str, Dict[str, str]]]:
        """Get the status of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return {"status": "not_found", "job_id": job_id}
        
        metadata: dict = self._job_metadata.get(job_id, {}).copy()

        if self._job_semaphore:
            metadata["running_jobs"] = str(self._config.max_concurrent_jobs - self._job_semaphore._value)
            metadata["max_concurrent_jobs"] = str(self._config.max_concurrent_jobs)

        return {"job_id": job_id, "status": job.status, "metadata": metadata}
    
    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        job = self._jobs.get(job_id)
        stored_benchmark = await self.get_benchmark(benchmark_id)
        if not stored_benchmark:
            raise BenchmarkNotFoundError(f"Benchmark {benchmark_id} not found")

        benchmark_metadata: dict = getattr(stored_benchmark, "metadata", {})

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
            if self._job_metadata[job_id].get("scan_result.json", None):
                scan_result_file_id: str = self._job_metadata[job_id].get("scan_result.json", "")
                scan_result: Response = await self.file_api.openai_retrieve_file_content(scan_result_file_id)
                return EvaluateResponse(**json.loads(scan_result.body.decode("utf-8")))
            elif not self._job_metadata[job_id].get("scan.report.jsonl", None):
                logger.error(f"No scan.report.jsonl/scan_result.json file found for job {job_id}")
                return EvaluateResponse(generations=[], scores={})
            else:
                logger.error(f"Results not found for job {job_id}")
                return EvaluateResponse(generations=[], scores={})
        
        else:
            logger.warning(f"Job {job_id} has an unknown status: {job.status}")
            return EvaluateResponse(generations=[], scores={})
    
    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job and kill the process.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return
        
        if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
            logger.warning(f"Job {job_id} is not running")

        elif job.status in [JobStatus.in_progress, JobStatus.scheduled]:
            process_id: str = self._job_metadata[job_id].get("process_id", None)
            if process_id:
                process_id: int = int(process_id)
                logger.info(f"Killing process {process_id} for job {job_id}")
                try:
                    # TODO: Check if the process is graceful shutdown and if not, kill it with SIGKILL
                    os.kill(process_id, signal.SIGTERM)
                except ProcessLookupError:
                    logger.warning(f"Process {process_id} not found for job {job_id}")
                except Exception as e:
                    logger.error(f"Error killing process {process_id} for job {job_id}: {e}")
            job.status = JobStatus.cancelled
            self._jobs[job_id] = job
            self._job_metadata[job_id]["cancelled_at"] = datetime.now().isoformat()
            self._job_metadata[job_id]["error"] = "Job cancelled"
        else:
            logger.warning(f"Job {job_id} has an unknown status: {job.status}")
    
    async def evaluate_rows(self, benchmark_id: str, 
                            input_rows: list[dict[str, Any]], 
                            scoring_functions: list[str], 
                            benchmark_config: BenchmarkConfig) -> EvaluateResponse:
        raise NotImplementedError("Not implemented")
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down Garak provider")
        # Cancel all running asyncio tasks
        for job_id, task in self._running_tasks.items():
            if not task.done():
                logger.info(f"Cancelling running task {task.get_name()} for job {job_id}")
                task.cancel()
        
        # Wait for tasks to be cancelled (with timeout)
        if self._running_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._running_tasks.values(), return_exceptions=True),
                    timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks didn't cancel within timeout")
        
        # Kill all running jobs
        for job_id, job in self._jobs.items():
            if job.status in [JobStatus.in_progress, JobStatus.scheduled]:
                await self.job_cancel("placeholder", job_id)
        
        # Clear all running tasks, jobs and job metadata
        self._running_tasks.clear()
        self._jobs.clear()
        self._job_metadata.clear()

        # Close the shield scanning HTTP client
        shield_scan.simple_shield_orchestrator.close()
        
        # Cleanup the scan directory
        shutil.rmtree(self.scan_config.scan_dir, ignore_errors=True)