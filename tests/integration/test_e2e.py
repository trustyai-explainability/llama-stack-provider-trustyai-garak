"""
End-to-end integration test for llama-stack-provider-trustyai-garak.

This test:
1. Connects to a running llama-stack server (deployed on OpenShift)
2. Registers a minimal benchmark (dan.Dan_11_0 - single prompt)
3. Runs the eval
4. Polls for completion
5. Validates and reports results
"""

import os
import sys
import time

import requests
from llama_stack_client import LlamaStackClient


# Configuration
BASE_URL = os.environ.get("LLAMA_STACK_URL", "http://localhost:8321")
MODEL_NAME = os.environ.get("TEST_MODEL_NAME", "vllm/Granite-3.3-8B-Instruct")
PROVIDER_ID = os.environ.get("TEST_PROVIDER_ID", "trustyai_garak")
MAX_WAIT_TIME = int(os.environ.get("TEST_MAX_WAIT_TIME", "600"))  # 10 minutes default
POLL_INTERVAL = int(os.environ.get("TEST_POLL_INTERVAL", "20"))  # 20 seconds


def get_summary_metrics(aggregated_scores: dict) -> dict:
    """Calculate summary metrics from aggregated scores."""
    summary_metrics = {
        "total_attempts": 0,
        "vulnerable_responses": 0,
        "attack_success_rate": 0,
    }
    for aggregated_results in aggregated_scores.values():
        if isinstance(aggregated_results, dict):
            summary_metrics["total_attempts"] += aggregated_results.get(
                "total_attempts", 0
            )
            summary_metrics["vulnerable_responses"] += aggregated_results.get(
                "vulnerable_responses", 0
            )

    if summary_metrics["total_attempts"] > 0:
        summary_metrics["attack_success_rate"] = round(
            (
                summary_metrics["vulnerable_responses"]
                / summary_metrics["total_attempts"]
                * 100
            ),
            2,
        )

    return summary_metrics


def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Wait for the llama-stack server to be ready."""
    print(f"Waiting for server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/v1/health", timeout=5)
            if resp.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def run_e2e_test():
    """Run the end-to-end integration test."""
    print("=" * 70)
    print("  E2E Integration Test - llama-stack-provider-trustyai-garak")
    print("=" * 70)
    print(f"  Server URL: {BASE_URL}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Provider: {PROVIDER_ID}")
    print("=" * 70)

    # Wait for llama-stack server
    if not wait_for_server(BASE_URL, timeout=120):
        print("❌ ERROR: llama-stack server did not become ready!")
        sys.exit(1)

    # Create client
    client = LlamaStackClient(base_url=BASE_URL)

    # Step 1: List providers to verify connectivity
    print("\n📡 Step 1: Verify connectivity")
    print("-" * 40)
    try:
        providers = client.providers.list()
        print(f"✅ Connected! Found {len(providers)} providers")

        # Find the garak provider
        garak_provider = None
        for p in providers:
            if "garak" in p.provider_type.lower():
                garak_provider = p
                print(f"   Found garak provider: {p.provider_id} ({p.provider_type})")
                break

        if not garak_provider:
            print("❌ ERROR: Could not find garak provider!")
            sys.exit(1)

    except Exception as e:
        print(f"❌ ERROR: Failed to connect to server: {e}")
        sys.exit(1)

    # Step 2: List existing benchmarks
    print("\n📋 Step 2: List benchmarks")
    print("-" * 40)
    benchmarks = client.benchmarks.list()
    print(f"✅ Found {len(benchmarks)} pre-registered benchmarks")
    for b in benchmarks[:5]:  # Show first 5
        print(f"   - {b.identifier}")
    if len(benchmarks) > 5:
        print(f"   ... and {len(benchmarks) - 5} more")

    # Step 3: Register minimal benchmark with dan.Dan_11_0 (1 prompt)
    print("\n📝 Step 3: Register minimal test benchmark")
    print("-" * 40)
    test_benchmark_id = "ci_test::minimal_dan"

    try:
        client.benchmarks.register(
            benchmark_id=test_benchmark_id,
            dataset_id="garak",
            scoring_functions=["garak_scoring"],
            provider_benchmark_id=test_benchmark_id,
            provider_id=PROVIDER_ID,
            metadata={
                "probes": ["dan.Dan_11_0"],
                "timeout": 300,  # 5 minute timeout for this tiny test
            },
        )
        print(f"✅ Registered benchmark: {test_benchmark_id}")
        print("   Probe: dan.Dan_11_0 (1 prompt)")
    except Exception as e:
        print(f"⚠️  Benchmark registration note: {e}")
        print("   (May already exist, continuing...)")

    # Step 4: Run eval
    print("\n🚀 Step 4: Run eval")
    print("-" * 40)
    try:
        job = client.eval.run_eval(
            benchmark_id=test_benchmark_id,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": MODEL_NAME,
                    "sampling_params": {"max_tokens": 100},
                }
            },
        )
        print(f"✅ Started job: {job.job_id}")
        if hasattr(job, "metadata") and job.metadata:
            if "kfp_run_id" in job.metadata:
                print(f"   KFP Run ID: {job.metadata['kfp_run_id']}")
    except Exception as e:
        print(f"❌ ERROR: Failed to start eval job: {e}")
        sys.exit(1)

    # Step 5: Poll for completion
    print("\n⏳ Step 5: Waiting for job completion")
    print("-" * 40)
    print(f"   Max wait time: {MAX_WAIT_TIME}s, Poll interval: {POLL_INTERVAL}s")
    start_time = time.time()
    final_status = None
    last_status = None

    while time.time() - start_time < MAX_WAIT_TIME:
        try:
            job_status = client.eval.jobs.status(
                job_id=job.job_id, benchmark_id=test_benchmark_id
            )

            if job_status.status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"   [{elapsed}s] Status: {job_status.status}")
                last_status = job_status.status

            if job_status.status in ["completed", "failed", "cancelled"]:
                final_status = job_status.status
                break

        except Exception as e:
            print(f"   ⚠️  Error checking status: {e}")

        time.sleep(POLL_INTERVAL)

    elapsed_total = int(time.time() - start_time)

    if final_status is None:
        print(f"❌ ERROR: Job timed out after {MAX_WAIT_TIME} seconds!")
        sys.exit(1)

    if final_status == "failed":
        print(f"❌ ERROR: Job failed after {elapsed_total}s!")
        sys.exit(1)

    if final_status == "cancelled":
        print(f"❌ ERROR: Job was cancelled after {elapsed_total}s!")
        sys.exit(1)

    print(f"✅ Job completed in {elapsed_total}s")

    # Step 6: Retrieve results
    print("\n📊 Step 6: Retrieve results")
    print("-" * 40)
    try:
        result = client.eval.jobs.retrieve(
            job_id=job.job_id, benchmark_id=test_benchmark_id
        )

        if result.scores:
            print("✅ Results received!")

            # Extract aggregated scores
            aggregated_scores = {}
            for key, value in result.scores.items():
                if hasattr(value, "aggregated_results"):
                    aggregated_scores[key] = value.aggregated_results
                    print(f"\n   Probe: {key}")
                    if isinstance(value.aggregated_results, dict):
                        for metric, val in value.aggregated_results.items():
                            print(f"      {metric}: {val}")

            # Calculate summary metrics
            if aggregated_scores:
                summary = get_summary_metrics(aggregated_scores)
                print("\n" + "=" * 40)
                print("   SUMMARY METRICS")
                print("=" * 40)
                print(f"   Total Attempts: {summary['total_attempts']}")
                print(f"   Vulnerable Responses: {summary['vulnerable_responses']}")
                print(f"   Attack Success Rate: {summary['attack_success_rate']}%")

        else:
            print("⚠️  WARNING: No scores in result")

    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve results: {e}")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 70)
    print("  ✅ E2E INTEGRATION TEST PASSED!")
    print("=" * 70)
    print(f"  Server: {BASE_URL}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Job ID: {job.job_id}")
    print(f"  Duration: {elapsed_total}s")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_test()
