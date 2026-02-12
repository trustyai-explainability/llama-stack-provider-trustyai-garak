"""Compatibility helpers for importing llama_stack APIs across versions.

The llama_stack APIs were moved under a separate `llama_stack_api` package
upstream. Prefer the new package layout and fall back to the legacy one so
this provider can run with both.
"""

from __future__ import annotations

try:
    # try to import from llama_stack_api package
    from llama_stack_api import (
        Api,
        ProviderSpec,
        BenchmarksProtocolPrivate,
        RemoteProviderSpec,
        InlineProviderSpec,
        Job,
        JobStatus,
        json_schema_type,
        Benchmark,
        Benchmarks,
        Eval,
        BenchmarkConfig,
        EvaluateResponse,
        RunEvalRequest,
        EvaluateRowsRequest,
        JobStatusRequest,
        JobCancelRequest,
        JobResultRequest,
        Files,
        OpenAIFilePurpose,
        OpenAIFileObject,
        ListFilesRequest,
        RetrieveFileContentRequest,
        UploadFileRequest,
        Safety,
        RunShieldResponse,
        GetShieldRequest,
        ViolationLevel,
        Shields,
        OpenAIChatCompletion,
        SamplingParams,
        SamplingStrategy,
        TopPSamplingStrategy,
        TopKSamplingStrategy,
        ScoringResult,
    )
except ModuleNotFoundError:  # fallback to legacy llama_stack layout
    # common datatypes
    from llama_stack.apis.datatypes import Api
    from llama_stack.providers.datatypes import (
        ProviderSpec, 
        BenchmarksProtocolPrivate,
        RemoteProviderSpec,
        InlineProviderSpec,
    )
    
    from llama_stack.apis.common.job_types import (
        Job, 
        JobStatus
    )
    from llama_stack.schema_utils import json_schema_type

    # evals
    from llama_stack.apis.benchmarks import (
        Benchmark, Benchmarks,
    )
    from llama_stack.apis.eval import (
        Eval, 
        BenchmarkConfig, 
        EvaluateResponse,
        RunEvalRequest,
        EvaluateRowsRequest,
        JobStatusRequest,
        JobCancelRequest,
        JobResultRequest,
    )

    # files
    from llama_stack.apis.files import (
        Files,
        OpenAIFilePurpose,
        OpenAIFileObject,
        ListFilesRequest,
        RetrieveFileContentRequest,
        UploadFileRequest,
    )

    # safety
    from llama_stack.apis.safety import (
        Safety,
        RunShieldResponse, 
        ViolationLevel,
        GetShieldRequest,
    )

    # shields
    from llama_stack.apis.shields import Shields

    # inference
    from llama_stack.apis.inference import (
        OpenAIChatCompletion,
        SamplingParams,
        SamplingStrategy,
        TopPSamplingStrategy,
        TopKSamplingStrategy
    )

    # scoring
    from llama_stack.apis.scoring import ScoringResult
finally:
    # Patch Job model to allow extra fields (e.g., metadata)
    # This enables additional context in Job responses
    # The client-side Job model already has extra='allow', so this ensures
    # the server-side model doesn't strip out extra fields during serialization
    if not Job.model_config.get('extra'):
        Job.model_config['extra'] = 'allow'
        # Rebuild the model to apply the config change
        Job.model_rebuild(force=True)


__all__ = [
    # common
    "Api",
    "ProviderSpec",
    "BenchmarksProtocolPrivate",
    "RemoteProviderSpec",
    "InlineProviderSpec",
    "Job",
    "JobStatus",
    "json_schema_type",
    # evals
    "Benchmark",
    "Benchmarks",
    "Eval",
    "BenchmarkConfig",
    "EvaluateResponse",
    "RunEvalRequest",
    "EvaluateRowsRequest",
    "JobStatusRequest",
    "JobCancelRequest",
    "JobResultRequest",
    # files
    "Files",
    "OpenAIFilePurpose",
    "OpenAIFileObject",
    "ListFilesRequest",
    "RetrieveFileContentRequest",
    "UploadFileRequest",
    # safety
    "Safety",
    "RunShieldResponse",
    "ViolationLevel",
    "GetShieldRequest",
    # shields
    "Shields",
    # inference
    "OpenAIChatCompletion",
    "SamplingParams",
    "SamplingStrategy",
    "TopPSamplingStrategy",
    "TopKSamplingStrategy",
    # scoring
    "ScoringResult",
]