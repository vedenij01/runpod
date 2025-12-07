import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional

import runpod

# NOTE: 'requests' imported inside functions to avoid import order issues with CUDA

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pow.compute.gpu_group import create_gpu_groups, GpuGroup, NotEnoughGPUResources
from pow.compute.autobs_v2 import get_batch_size_for_gpu_group
from pow.compute.worker import ParallelWorkerManager
from pow.compute.model_init import ModelWrapper
from pow.models.utils import Params

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum job duration: 7 minutes
MAX_JOB_DURATION = 7 * 60

# Warmup polling interval
WARMUP_POLL_INTERVAL = 5  # seconds
WARMUP_MAX_DURATION = 10 * 60  # 10 minutes max warmup time

# Global flag to track if CUDA is broken - prevents repeated failed attempts
_cuda_broken = False


def fetch_warmup_params(callback_url: str, job_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch generation params from MLNode callback URL.
    Returns params dict if available, None if still waiting.
    """
    import requests
    try:
        url = f"{callback_url}/warmup/params"
        response = requests.get(
            url,
            params={"job_id": job_id},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ready":
            return data.get("params")
        return None

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch warmup params: {e}")
        return None


def send_batch_to_callback(callback_url: str, batch: Dict[str, Any], node_id: int) -> bool:
    """
    Send a generated batch to the callback URL.
    Used in warmup mode where DelegationController is not polling.
    """
    import requests
    try:
        url = f"{callback_url}/generated"
        payload = {
            "public_key": batch.get("public_key", ""),
            "block_hash": batch.get("block_hash", ""),
            "block_height": batch.get("block_height", 0),
            "nonces": batch.get("nonces", []),
            "dist": batch.get("dist", []),
            "node_id": node_id,
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to send batch to callback: {e}")
        return False


def notify_generation_complete(callback_url: str, job_id: str, stats: Dict[str, Any]) -> bool:
    """
    Notify MLNode that generation is complete so it can stop the warmup job.
    """
    import requests
    try:
        url = f"{callback_url}/warmup/complete"
        payload = {
            "job_id": job_id,
            "total_batches": stats.get("total_batches", 0),
            "total_computed": stats.get("total_computed", 0),
            "total_valid": stats.get("total_valid", 0),
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Notified MLNode of generation complete: {stats}")
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to notify generation complete: {e}")
        return False


def warmup_handler(event: Dict[str, Any], callback_url: str, job_id: str):
    """
    Warmup mode handler - reserves worker and waits for params.

    IMPORTANT: Does NOT initialize CUDA early - that causes 5min slowdown!
    Just polls for params, then runs normal generation (which initializes CUDA).

    1. Poll callback_url for generation params
    2. When params received, run normal generation
    """
    logger.info(f"WARMUP MODE: job_id={job_id}, callback_url={callback_url}")
    logger.info("Worker reserved, waiting for generation params (no CUDA init)")

    yield {
        "status": "warmup_ready",
        "job_id": job_id,
    }

    # Poll for params
    warmup_start = time.time()
    poll_count = 0

    while True:
        elapsed = time.time() - warmup_start

        # Check warmup timeout
        if elapsed > WARMUP_MAX_DURATION:
            logger.info(f"WARMUP TIMEOUT: {elapsed:.0f}s exceeded {WARMUP_MAX_DURATION}s limit")
            yield {"status": "warmup_timeout", "elapsed": int(elapsed)}
            return

        poll_count += 1
        if poll_count % 12 == 1:  # Log every minute (12 * 5s = 60s)
            logger.info(f"Warmup poll #{poll_count}: waiting for params... ({int(elapsed)}s)")

        params = fetch_warmup_params(callback_url, job_id)

        if params:
            logger.info(f"Warmup got params after {elapsed:.0f}s, starting generation")
            yield {"status": "warmup_params_received", "elapsed": int(elapsed)}

            # Create event with received params and run generation
            # NOTE: Don't use HTTP callback (send_to_callback=False) because:
            # 1. Runpod cannot access internal Docker URLs (http://api:9100)
            # 2. DelegationController polls our /stream for results instead
            generation_event = {"input": params}
            yield from generation_handler(
                generation_event,
                send_to_callback=False,  # DelegationController polls us via /stream
                callback_url="",
                node_id=0,
                warmup_job_id=job_id,
                warmup_callback_url=callback_url,
            )
            return

        # Yield heartbeat
        yield {
            "status": "warmup_waiting",
            "poll_count": poll_count,
            "elapsed": int(elapsed),
        }

        time.sleep(WARMUP_POLL_INTERVAL)


def generation_handler(
    event: Dict[str, Any],
    send_to_callback: bool = False,
    callback_url: str = "",
    node_id: int = 0,
    warmup_job_id: str = "",
    warmup_callback_url: str = "",
):
    """
    Parallel streaming nonce generator using multiple GPU groups.

    Each GPU group runs as an independent worker process,
    processing different nonce ranges in parallel.

    Stops when:
    1. Client calls POST /cancel/{job_id}
    2. Timeout after 7 minutes (MAX_JOB_DURATION)

    Input from client (ALL REQUIRED):
    {
        "block_hash": str,
        "block_height": int,
        "public_key": str,
        "r_target": float,
        "batch_size": int,  # This is now total batch size, will be distributed
        "start_nonce": int,
        "params": dict,
    }

    Args:
        event: Runpod event with input data
        send_to_callback: If True, send results via HTTP to callback_url
        callback_url: URL to send results to (used in warmup mode)
        node_id: Node ID to include in batches sent to callback
        warmup_job_id: Job ID for warmup mode (to notify completion)
        warmup_callback_url: Callback URL for warmup completion notification

    Yields:
    {
        "nonces": [...],
        "dist": [...],
        "batch_number": int,
        "worker_id": int,
        "elapsed_seconds": int,
        ...
    }
    """
    global _cuda_broken

    # Fast exit if CUDA already known to be broken - don't waste time retrying
    if _cuda_broken:
        logger.error("CUDA already marked as broken - killing worker immediately")
        yield {"error": "Worker CUDA is broken", "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)  # Hard exit - cannot be caught

    aggregated_batch_count = 0
    total_computed = 0
    total_valid = 0

    try:
        input_data = event.get("input", {})

        # Get ALL parameters from client - NO DEFAULTS
        block_hash = input_data["block_hash"]
        block_height = input_data["block_height"]
        public_key = input_data["public_key"]
        r_target = input_data["r_target"]
        client_batch_size = input_data["batch_size"]
        start_nonce = input_data["start_nonce"]
        params_dict = input_data["params"]

        params = Params(**params_dict)

        # Auto-detect GPUs and create groups
        import torch
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPUs")

        # Create GPU groups based on VRAM requirements
        gpu_groups = create_gpu_groups(params=params)
        n_workers = len(gpu_groups)

        logger.info(f"Created {n_workers} GPU groups for parallel processing:")
        for i, group in enumerate(gpu_groups):
            logger.info(f"  Worker {i}: {group} (VRAM: {group.get_total_vram_gb():.1f}GB)")

        # Calculate batch size per worker
        # Use auto-calculated batch size based on GPU memory
        batch_sizes = []
        for group in gpu_groups:
            bs = get_batch_size_for_gpu_group(group, params)
            batch_sizes.append(bs)
            logger.info(f"  Worker batch size for {group.devices}: {bs}")

        # Use minimum batch size across all groups to ensure consistency
        batch_size_per_worker = min(batch_sizes)
        total_batch_size = batch_size_per_worker * n_workers

        logger.info(f"START: block={block_height}, workers={n_workers}, "
                   f"batch_per_worker={batch_size_per_worker}, total_batch={total_batch_size}, "
                   f"start={start_nonce}")

        # Convert GPU groups to device string lists
        gpu_group_devices = [group.get_device_strings() for group in gpu_groups]

        # Build base model ONCE on GPU 0 (this is the slow part)
        logger.info("Building base model on GPU 0...")
        base_model_start = time.time()
        base_model_data = ModelWrapper.build_base_model(
            hash_=block_hash,
            params=params,
            max_seq_len=params.seq_len,
        )
        logger.info(f"Base model built in {time.time() - base_model_start:.1f}s")

        # Create and start parallel worker manager with pre-built model
        manager = ParallelWorkerManager(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            r_target=r_target,
            batch_size_per_worker=batch_size_per_worker,
            gpu_groups=gpu_group_devices,
            start_nonce=start_nonce,
            max_duration=MAX_JOB_DURATION,
            base_model_data=base_model_data,  # Pass pre-built model to all workers
        )

        manager.start()

        # Wait for all workers to initialize models
        if not manager.wait_for_ready(timeout=180):
            logger.error("Workers failed to initialize within timeout")
            yield {
                "error": "Worker initialization timeout",
                "error_type": "TimeoutError",
            }
            manager.stop()
            return

        logger.info("All workers ready, starting streaming")

        start_time = time.time()
        last_result_time = start_time

        # Streaming results from all workers
        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > MAX_JOB_DURATION:
                logger.info(f"TIMEOUT: {elapsed:.0f}s exceeded {MAX_JOB_DURATION}s limit")
                break

            # Check if all workers have stopped
            if not manager.is_alive():
                logger.info("All workers have stopped")
                break

            # Get results from workers
            results = manager.get_results(timeout=0.5)

            if not results:
                # No results available, check for stall
                if time.time() - last_result_time > 60:
                    logger.warning("No results for 60s, workers may be stuck")
                continue

            last_result_time = time.time()

            # Yield each result
            for result in results:
                if "error" in result:
                    logger.error(f"Worker {result.get('worker_id')} error: {result['error']}")
                    yield result
                    continue

                aggregated_batch_count += 1
                total_computed += result.get("batch_computed", 0)
                total_valid += result.get("batch_valid", 0)

                # Add aggregated stats and fix batch_number for deduplication
                result["aggregated_batch_number"] = aggregated_batch_count
                result["aggregated_total_computed"] = total_computed
                result["aggregated_total_valid"] = total_valid
                result["n_workers"] = n_workers
                # Override batch_number with globally unique value
                # (delegation_controller deduplicates by batch_number)
                result["batch_number"] = aggregated_batch_count

                # Send to callback URL if in warmup mode
                if send_to_callback and callback_url:
                    send_batch_to_callback(callback_url, result, node_id)

                logger.info(f"Batch #{aggregated_batch_count} from worker {result['worker_id']}: "
                           f"{result.get('batch_valid', 0)} valid, elapsed={int(elapsed)}s")

                yield result

        logger.info(f"STOPPED: {aggregated_batch_count} batches, {total_computed} computed, {total_valid} valid")
        manager.stop()

        # Notify MLNode that generation is complete (for warmup mode auto-stop)
        if warmup_job_id and warmup_callback_url:
            notify_generation_complete(warmup_callback_url, warmup_job_id, {
                "total_batches": aggregated_batch_count,
                "total_computed": total_computed,
                "total_valid": total_valid,
            })

        # Free GPU 0 memory - delete base model data after all workers done
        logger.info("Freeing GPU 0 memory...")
        del base_model_data
        torch.cuda.empty_cache()

    except GeneratorExit:
        logger.info(f"CANCELLED: {aggregated_batch_count} batches, {total_computed} computed, {total_valid} valid")
        try:
            manager.stop()
        except:
            pass
        try:
            del base_model_data
            torch.cuda.empty_cache()
        except:
            pass
    except NotEnoughGPUResources as e:
        # CUDA/GPU initialization failed - mark as broken and kill worker
        _cuda_broken = True
        logger.error(f"GPU INIT FAILED: {str(e)}")
        logger.error("CUDA broken - killing worker with os._exit(1)")
        yield {
            "error": str(e),
            "error_type": "NotEnoughGPUResources",
            "fatal": True,
        }
        # os._exit(1) cannot be caught - forces immediate process termination
        os._exit(1)

    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        yield {
            "error": str(e),
            "error_type": type(e).__name__,
        }
        try:
            manager.stop()
        except:
            pass
        try:
            del base_model_data
            torch.cuda.empty_cache()
        except:
            pass


def handler(event: Dict[str, Any]):
    """
    Main handler - dispatches to warmup or generation mode.

    Warmup mode input:
    {
        "warmup_mode": True,
        "callback_url": "http://mlnode:9090"
    }

    Generation mode input (normal):
    {
        "block_hash": str,
        "block_height": int,
        ...
    }
    """
    input_data = event.get("input", {})
    warmup_mode = input_data.get("warmup_mode", False)

    if warmup_mode:
        callback_url = input_data.get("callback_url", "")
        job_id = event.get("id", "unknown")

        if not callback_url:
            yield {"error": "callback_url is required for warmup mode", "error_type": "ValueError"}
            return

        yield from warmup_handler(event, callback_url, job_id)
    else:
        yield from generation_handler(event)


# Start serverless handler with streaming support
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
