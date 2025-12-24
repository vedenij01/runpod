import time
import queue
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import Future

from pow.data import ProofBatch
from pow.compute.compute import Compute
from pow.models.utils import Params
from pow.random import get_target
from common.logger import create_logger

logger = create_logger(__name__)


class NonceIterator:
    """
    Deterministic nonce iterator that distributes nonces across workers.
    Each worker gets non-overlapping nonce ranges.
    """
    def __init__(self, worker_id: int, n_workers: int, start_nonce: int = 0):
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.start_nonce = start_nonce
        self._current_x = 0
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self) -> int:
        with self._lock:
            # Nonce = start + worker_id + current_x * n_workers
            # This ensures non-overlapping nonces across workers
            value = self.start_nonce + self.worker_id + self._current_x * self.n_workers
            self._current_x += 1
            return value


class RangeNonceIterator:
    """
    Nonce iterator for a specific range.
    Distributes nonces among local GPUs within the assigned range.

    Used in pooled mode where each RunPod worker gets a fixed range
    (e.g., 0-9999) from the orchestrator.

    Example with range 0-9999 and 4 GPUs:
        GPU 0: 0, 4, 8, 12, ...
        GPU 1: 1, 5, 9, 13, ...
        GPU 2: 2, 6, 10, 14, ...
        GPU 3: 3, 7, 11, 15, ...
    """
    def __init__(
        self,
        local_gpu_id: int,
        n_local_gpus: int,
        range_start: int,
        range_end: int,
    ):
        """
        Initialize range-based nonce iterator.

        Args:
            local_gpu_id: GPU index within this worker (0, 1, 2, ...)
            n_local_gpus: Total GPUs in this worker
            range_start: Start of nonce range (inclusive)
            range_end: End of nonce range (inclusive)
        """
        self.local_gpu_id = local_gpu_id
        self.n_local_gpus = n_local_gpus
        self.range_start = range_start
        self.range_end = range_end
        self._current_x = 0
        self._lock = threading.Lock()
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self) -> int:
        with self._lock:
            if self._exhausted:
                raise StopIteration

            # nonce = range_start + local_gpu_id + current_x * n_local_gpus
            value = self.range_start + self.local_gpu_id + self._current_x * self.n_local_gpus

            if value > self.range_end:
                self._exhausted = True
                raise StopIteration

            self._current_x += 1
            return value

    def reset(self):
        """Reset iterator to start of range (used when switching public_key)."""
        with self._lock:
            self._current_x = 0
            self._exhausted = False

    def remaining(self) -> int:
        """Get approximate number of remaining nonces."""
        with self._lock:
            if self._exhausted:
                return 0
            current = self.range_start + self.local_gpu_id + self._current_x * self.n_local_gpus
            return max(0, (self.range_end - current) // self.n_local_gpus + 1)

    def get_next_batch(self, batch_size: int) -> List[int]:
        """Get next batch of nonces, handling exhaustion gracefully."""
        nonces = []
        for _ in range(batch_size):
            try:
                nonces.append(next(self))
            except StopIteration:
                break
        return nonces


def worker_thread(
    worker_id: int,
    params: Params,
    block_hash: str,
    block_height: int,
    public_key: str,
    r_target: float,
    batch_size: int,
    devices: List[str],
    start_nonce: int,
    n_workers: int,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    ready_event: threading.Event,
    max_duration: float = 420.0,  # 7 minutes default
    base_model_data: dict = None,  # Pre-built model state_dict for fast loading
):
    """
    Worker thread that runs on a specific GPU group.
    Generates nonces and puts results into result_queue.

    If base_model_data is provided, uses fast path to load model from state_dict.
    Otherwise, builds model from scratch (slow path).
    """
    import torch

    try:
        logger.info(f"[Worker {worker_id}] Starting on devices {devices}, batch_size={batch_size}")

        # Initialize compute for this worker's GPU(s)
        if base_model_data is not None:
            logger.info(f"[Worker {worker_id}] Creating Compute from pre-built model...")
        else:
            logger.info(f"[Worker {worker_id}] Creating Compute (building model from scratch)...")

        compute = Compute(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            r_target=r_target,
            devices=devices,
            node_id=worker_id,
            base_model_data=base_model_data,
        )

        target = get_target(block_hash, params.vocab_size)

        # Signal that worker is ready
        ready_event.set()
        logger.info(f"[Worker {worker_id}] Model loaded, ready to generate")

        # Create nonce iterator for this worker
        nonce_iter = NonceIterator(
            worker_id=worker_id,
            n_workers=n_workers,
            start_nonce=start_nonce
        )

        start_time = time.time()
        batch_count = 0
        total_computed = 0
        total_valid = 0

        # Pre-generate first batch of nonces
        next_nonces = [next(nonce_iter) for _ in range(batch_size)]

        while not stop_event.is_set():
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                logger.info(f"[Worker {worker_id}] Timeout after {elapsed:.0f}s")
                break

            # Current batch
            nonces = next_nonces
            # Pre-fetch next batch
            next_nonces = [next(nonce_iter) for _ in range(batch_size)]

            # Process batch with pre-fetching
            future: Future = compute(
                nonces=nonces,
                public_key=public_key,
                target=target,
                next_nonces=next_nonces,
            )

            # Get result
            proof_batch = future.result()
            filtered_batch = proof_batch.sub_batch(r_target)

            batch_count += 1
            total_computed += len(proof_batch)
            total_valid += len(filtered_batch)

            # Put result in queue
            if filtered_batch.nonces:
                # Convert dist to list if it's a numpy array
                dist_list = filtered_batch.dist
                if hasattr(dist_list, 'tolist'):
                    dist_list = dist_list.tolist()
                elif not isinstance(dist_list, list):
                    dist_list = list(dist_list)

                result = {
                    "worker_id": worker_id,
                    "public_key": filtered_batch.public_key,
                    "block_hash": filtered_batch.block_hash,
                    "block_height": filtered_batch.block_height,
                    "nonces": filtered_batch.nonces,
                    "dist": dist_list,
                    "node_id": filtered_batch.node_id,
                    "batch_number": batch_count,
                    "batch_computed": len(proof_batch),
                    "batch_valid": len(filtered_batch),
                    "total_computed": total_computed,
                    "total_valid": total_valid,
                    "elapsed_seconds": int(elapsed),
                    "next_nonce": max(nonces) + n_workers if nonces else 0,
                }
                try:
                    result_queue.put(result, timeout=5)
                except queue.Full:
                    logger.warning(f"[Worker {worker_id}] Queue full, skipping batch")

            if batch_count % 10 == 0:
                logger.info(f"[Worker {worker_id}] Batch #{batch_count}: {len(filtered_batch)} valid, elapsed={int(elapsed)}s")

        logger.info(f"[Worker {worker_id}] Stopped: {batch_count} batches, {total_computed} computed, {total_valid} valid")

        # Cleanup
        compute.shutdown()

    except Exception as e:
        logger.error(f"[Worker {worker_id}] Error: {e}", exc_info=True)
        try:
            result_queue.put({
                "worker_id": worker_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }, timeout=5)
        except:
            pass
    finally:
        ready_event.set()  # Ensure we signal even on error


class ParallelWorkerManager:
    """
    Manages multiple worker threads for parallel nonce generation.
    Uses threading instead of multiprocessing to avoid spawn issues.
    """
    def __init__(
        self,
        params: Params,
        block_hash: str,
        block_height: int,
        public_key: str,
        r_target: float,
        batch_size_per_worker: int,
        gpu_groups: List[List[str]],  # List of device lists per worker
        start_nonce: int = 0,
        max_duration: float = 420.0,
        base_model_data: dict = None,  # Pre-built model for fast loading
    ):
        self.params = params
        self.block_hash = block_hash
        self.block_height = block_height
        self.public_key = public_key
        self.r_target = r_target
        self.batch_size_per_worker = batch_size_per_worker
        self.gpu_groups = gpu_groups
        self.start_nonce = start_nonce
        self.max_duration = max_duration
        self.n_workers = len(gpu_groups)
        self.base_model_data = base_model_data

        self.result_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.workers: List[threading.Thread] = []
        self.ready_events: List[threading.Event] = []

    def start(self):
        """Start all worker threads"""
        for worker_id, devices in enumerate(self.gpu_groups):
            ready_event = threading.Event()
            self.ready_events.append(ready_event)

            worker = threading.Thread(
                target=worker_thread,
                args=(
                    worker_id,
                    self.params,
                    self.block_hash,
                    self.block_height,
                    self.public_key,
                    self.r_target,
                    self.batch_size_per_worker,
                    devices,
                    self.start_nonce,
                    self.n_workers,
                    self.result_queue,
                    self.stop_event,
                    ready_event,
                    self.max_duration,
                    self.base_model_data,  # Pass pre-built model
                ),
                daemon=True,
                name=f"Worker-{worker_id}",
            )
            self.workers.append(worker)
            worker.start()
            logger.info(f"Started worker {worker_id} on devices {devices}")

    def wait_for_ready(self, timeout: float = 300.0) -> bool:
        """Wait for all workers to be ready (models loaded)"""
        start = time.time()
        for i, event in enumerate(self.ready_events):
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                logger.warning(f"Timeout waiting for worker {i}")
                return False
            if not event.wait(timeout=remaining):
                logger.warning(f"Worker {i} not ready within timeout")
                return False
        logger.info(f"All {self.n_workers} workers ready in {time.time() - start:.1f}s")
        return True

    def get_results(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Get all available results from workers"""
        results = []
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        return results

    def stop(self):
        """Stop all workers gracefully"""
        logger.info("Stopping all workers...")
        self.stop_event.set()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not stop in time")

        logger.info("All workers stopped")

    def is_alive(self) -> bool:
        """Check if any worker is still running"""
        return any(w.is_alive() for w in self.workers)


# ============================================================================
# Pooled Mode - For orchestrator-managed workers
# ============================================================================

def pooled_worker_thread(
    worker_id: int,
    params: Params,
    block_hash: str,
    block_height: int,
    r_target: float,
    batch_size: int,
    devices: List[str],
    range_start: int,
    range_end: int,
    n_local_gpus: int,
    result_queue: queue.Queue,
    command_queue: queue.Queue,
    stop_event: threading.Event,
    ready_event: threading.Event,
    max_duration: float = 600.0,
    base_model_data: dict = None,
):
    """
    Worker thread for pooled mode with orchestrator.

    Supports:
    - Dynamic public_key switching via command_queue
    - Range-based nonce iteration
    - Results tagged with public_key for routing

    Commands via command_queue:
    - {"type": "set_public_key", "public_key": "..."}
    - {"type": "switch_public_key", "public_key": "..."} - also resets nonces
    """
    import torch

    try:
        logger.info(f"[PooledWorker {worker_id}] Starting on devices {devices}, range={range_start}-{range_end}")

        # Initialize compute (model only depends on block_hash, not public_key)
        compute = Compute(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            public_key="",  # Will be set via command
            r_target=r_target,
            devices=devices,
            node_id=worker_id,
            base_model_data=base_model_data,
        )

        target = get_target(block_hash, params.vocab_size)

        # Create range-based nonce iterator
        nonce_iter = RangeNonceIterator(
            local_gpu_id=worker_id,
            n_local_gpus=n_local_gpus,
            range_start=range_start,
            range_end=range_end,
        )

        # Signal ready
        ready_event.set()
        logger.info(f"[PooledWorker {worker_id}] Model loaded, waiting for public_key")

        current_public_key: Optional[str] = None
        start_time = time.time()
        batch_count = 0
        total_computed = 0
        total_valid = 0

        while not stop_event.is_set():
            # Check for commands (non-blocking)
            try:
                command = command_queue.get_nowait()
                cmd_type = command.get("type")

                if cmd_type == "set_public_key":
                    current_public_key = command["public_key"]
                    compute.update_public_key(current_public_key)
                    logger.info(f"[PooledWorker {worker_id}] Set public_key: {current_public_key[:16]}...")

                elif cmd_type == "switch_public_key":
                    current_public_key = command["public_key"]
                    compute.update_public_key(current_public_key)
                    nonce_iter.reset()  # Reset to start of range
                    batch_count = 0
                    total_computed = 0
                    total_valid = 0
                    logger.info(f"[PooledWorker {worker_id}] Switched to public_key: {current_public_key[:16]}..., nonces reset")

                elif cmd_type == "stop":
                    logger.info(f"[PooledWorker {worker_id}] Received stop command")
                    break

                continue  # Process command, then continue loop

            except queue.Empty:
                pass

            # Skip if no public_key set yet
            if current_public_key is None:
                time.sleep(0.1)
                continue

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                logger.info(f"[PooledWorker {worker_id}] Timeout after {elapsed:.0f}s")
                break

            # Check if range exhausted
            if nonce_iter.remaining() == 0:
                logger.info(f"[PooledWorker {worker_id}] Range exhausted, waiting for new range or public_key")
                time.sleep(0.5)
                continue

            # Get batch of nonces
            nonces = nonce_iter.get_next_batch(batch_size)
            if not nonces:
                time.sleep(0.1)
                continue

            # Pre-fetch next batch for overlap
            next_nonces = nonce_iter.get_next_batch(batch_size)

            # Process batch
            future: Future = compute(
                nonces=nonces,
                public_key=current_public_key,
                target=target,
                next_nonces=next_nonces if next_nonces else None,
            )

            # Wait for result while checking for commands (reduces switch latency)
            pending_switch = None
            while not future.done():
                try:
                    command = command_queue.get_nowait()
                    cmd_type = command.get("type")

                    if cmd_type == "switch_public_key":
                        # Queue the switch for after current batch completes
                        pending_switch = command
                        logger.info(f"[PooledWorker {worker_id}] Switch queued, will apply after current batch")
                    elif cmd_type == "set_public_key":
                        pending_switch = command
                    elif cmd_type == "stop":
                        logger.info(f"[PooledWorker {worker_id}] Stop received during batch, will exit after")
                        stop_event.set()
                except queue.Empty:
                    pass
                time.sleep(0.01)  # 10ms - don't busy-wait

            # Get result (now ready)
            proof_batch = future.result()
            filtered_batch = proof_batch.sub_batch(r_target)

            batch_count += 1
            total_computed += len(proof_batch)
            total_valid += len(filtered_batch)

            # Always put result with public_key (even if empty, for stats)
            dist_list = filtered_batch.dist
            if hasattr(dist_list, 'tolist'):
                dist_list = dist_list.tolist()
            elif not isinstance(dist_list, list):
                dist_list = list(dist_list)

            result = {
                "worker_id": worker_id,
                "public_key": current_public_key,  # IMPORTANT: tagged with public_key
                "block_hash": filtered_batch.block_hash,
                "block_height": filtered_batch.block_height,
                "nonces": filtered_batch.nonces if filtered_batch.nonces else [],
                "dist": dist_list,
                "node_id": worker_id,
                "batch_number": batch_count,
                "batch_computed": len(proof_batch),
                "batch_valid": len(filtered_batch),
                "total_computed": total_computed,
                "total_valid": total_valid,
                "elapsed_seconds": int(elapsed),
            }

            try:
                result_queue.put(result, timeout=1)
            except queue.Full:
                logger.warning(f"[PooledWorker {worker_id}] Queue full, dropping batch")

            # Apply pending switch AFTER sending result with current public_key
            if pending_switch:
                cmd_type = pending_switch.get("type")
                new_public_key = pending_switch["public_key"]
                current_public_key = new_public_key
                compute.update_public_key(current_public_key)

                if cmd_type == "switch_public_key":
                    nonce_iter.reset()
                    batch_count = 0
                    total_computed = 0
                    total_valid = 0
                    logger.info(f"[PooledWorker {worker_id}] Switched to public_key: {current_public_key[:16]}..., nonces reset")
                else:
                    logger.info(f"[PooledWorker {worker_id}] Set public_key: {current_public_key[:16]}...")

            if batch_count % 10 == 0:
                logger.info(
                    f"[PooledWorker {worker_id}] Batch #{batch_count}: "
                    f"{len(filtered_batch)} valid, remaining={nonce_iter.remaining()}"
                )

        logger.info(
            f"[PooledWorker {worker_id}] Stopped: {batch_count} batches, "
            f"{total_computed} computed, {total_valid} valid"
        )
        compute.shutdown()

    except Exception as e:
        logger.error(f"[PooledWorker {worker_id}] Error: {e}", exc_info=True)
        try:
            result_queue.put({
                "worker_id": worker_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }, timeout=5)
        except:
            pass
    finally:
        ready_event.set()


class PooledWorkerManager:
    """
    Worker manager for pooled mode with orchestrator.

    Key differences from ParallelWorkerManager:
    - Supports dynamic public_key switching
    - Uses range-based nonce iteration
    - Can flush pending results before switching
    - Results are tagged with public_key for routing
    """

    def __init__(
        self,
        params: Params,
        block_hash: str,
        block_height: int,
        r_target: float,
        batch_size_per_worker: int,
        gpu_groups: List[List[str]],
        range_start: int,
        range_end: int,
        max_duration: float = 600.0,
        base_model_data: dict = None,
    ):
        self.params = params
        self.block_hash = block_hash
        self.block_height = block_height
        self.r_target = r_target
        self.batch_size_per_worker = batch_size_per_worker
        self.gpu_groups = gpu_groups
        self.range_start = range_start
        self.range_end = range_end
        self.max_duration = max_duration
        self.n_workers = len(gpu_groups)
        self.base_model_data = base_model_data

        self.result_queue = queue.Queue(maxsize=1000)
        self.command_queues: List[queue.Queue] = []
        self.stop_event = threading.Event()
        self.workers: List[threading.Thread] = []
        self.ready_events: List[threading.Event] = []

        self.current_public_key: Optional[str] = None

    def start(self):
        """Start all worker threads."""
        for worker_id, devices in enumerate(self.gpu_groups):
            ready_event = threading.Event()
            command_queue = queue.Queue()

            self.ready_events.append(ready_event)
            self.command_queues.append(command_queue)

            worker = threading.Thread(
                target=pooled_worker_thread,
                args=(
                    worker_id,
                    self.params,
                    self.block_hash,
                    self.block_height,
                    self.r_target,
                    self.batch_size_per_worker,
                    devices,
                    self.range_start,
                    self.range_end,
                    self.n_workers,
                    self.result_queue,
                    command_queue,
                    self.stop_event,
                    ready_event,
                    self.max_duration,
                    self.base_model_data,
                ),
                daemon=True,
                name=f"PooledWorker-{worker_id}",
            )
            self.workers.append(worker)
            worker.start()
            logger.info(f"Started pooled worker {worker_id} on devices {devices}")

    def wait_for_ready(self, timeout: float = 300.0) -> bool:
        """Wait for all workers to load models."""
        start = time.time()
        for i, event in enumerate(self.ready_events):
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                logger.warning(f"Timeout waiting for pooled worker {i}")
                return False
            if not event.wait(timeout=remaining):
                logger.warning(f"Pooled worker {i} not ready within timeout")
                return False
        logger.info(f"All {self.n_workers} pooled workers ready in {time.time() - start:.1f}s")
        return True

    def set_public_key(self, public_key: str):
        """Set public_key for all workers (initial assignment)."""
        self.current_public_key = public_key
        for cmd_queue in self.command_queues:
            cmd_queue.put({
                "type": "set_public_key",
                "public_key": public_key,
            })
        logger.info(f"Set public_key for all workers: {public_key[:16]}...")

    def switch_public_key(self, public_key: str):
        """Switch all workers to new public_key and reset nonces."""
        self.current_public_key = public_key
        for cmd_queue in self.command_queues:
            cmd_queue.put({
                "type": "switch_public_key",
                "public_key": public_key,
            })
        logger.info(f"Switched all workers to public_key: {public_key[:16]}...")

    def get_results(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Get all available results from workers."""
        results = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = self.result_queue.get(timeout=0.01)
                results.append(result)
            except queue.Empty:
                break
        return results

    def get_all_pending_results(self) -> List[Dict[str, Any]]:
        """Get ALL results from queue (for flushing before switch)."""
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results

    def stop(self):
        """Stop all workers gracefully."""
        logger.info("Stopping all pooled workers...")
        self.stop_event.set()

        # Send stop command to all workers
        for cmd_queue in self.command_queues:
            try:
                cmd_queue.put({"type": "stop"}, timeout=1)
            except queue.Full:
                pass

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Pooled worker {worker.name} did not stop in time")

        logger.info("All pooled workers stopped")

    def is_alive(self) -> bool:
        """Check if any worker is still running."""
        return any(w.is_alive() for w in self.workers)
