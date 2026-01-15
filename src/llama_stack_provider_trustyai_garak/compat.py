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
        GetBenchmarkRequest,
        Eval,
        BenchmarkConfig,
        EvaluateResponse,
        Files,
        OpenAIFilePurpose,
        OpenAIFileObject,
        ListFilesRequest,
        RetrieveFileContentRequest,
        UploadFileRequest,
        Safety,
        RunShieldResponse,
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
        Benchmark, Benchmarks, GetBenchmarkRequest,
    )
    from llama_stack.apis.eval import (
        Eval, BenchmarkConfig, EvaluateResponse
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
        ViolationLevel
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
    "GetBenchmarkRequest",
    "Eval",
    "BenchmarkConfig",
    "EvaluateResponse",
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