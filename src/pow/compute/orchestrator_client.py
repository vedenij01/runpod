"""
Orchestrator Client - HTTP client for communication with the orchestrator server.

This module handles all communication between RunPod workers and the central
orchestrator, including registration, job polling, result submission, and heartbeats.
"""

import time
import uuid
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import requests

from common.logger import create_logger

logger = create_logger(__name__)

# Configuration constants
POLL_INTERVAL = 0.5          # 500ms - how often to poll for commands
RESULT_SEND_TIMEOUT = 5      # seconds - timeout for sending results
MAX_RETRY_ATTEMPTS = 5       # max retries before giving up
RETRY_BACKOFF_BASE = 1.0     # seconds - base for exponential backoff (1, 2, 4, 8, 16)
MAX_PENDING_RESULTS = 1000   # max results to buffer locally
HEARTBEAT_INTERVAL = 10      # seconds - how often to send heartbeat
REQUEST_TIMEOUT = 10         # seconds - general request timeout


@dataclass
class WorkerConfig:
    """Configuration received from orchestrator."""
    block_hash: str = ""
    block_height: int = 0
    r_target: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    public_key: str = ""
    nonce_range_start: int = 0
    nonce_range_end: int = 0


class OrchestratorClient:
    """
    HTTP client for orchestrator communication.

    Handles:
    - Worker registration with GPU info
    - Polling for block_hash and job commands
    - Sending batch results (with retry and buffering)
    - Heartbeat to maintain connection
    """

    def __init__(self, orchestrator_url: str, worker_id: str = None):
        """
        Initialize orchestrator client.

        Args:
            orchestrator_url: Base URL of orchestrator (e.g., "https://orchestrator.example.com")
            worker_id: Unique worker ID. If None, generates UUID.
        """
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.worker_id = worker_id or str(uuid.uuid4())

        # State
        self.registered = False
        self.gpu_count = 0
        self.current_config = WorkerConfig()

        # Result buffering for retry
        self.pending_results: List[Dict[str, Any]] = []
        self._pending_lock = threading.Lock()

        # Heartbeat thread
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Session for connection pooling
        self._session = requests.Session()

        logger.info(f"OrchestratorClient initialized: worker_id={self.worker_id}, url={self.orchestrator_url}")

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Dict = None,
        params: Dict = None,
        retry: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., "/api/workers/connect")
            json_data: JSON body for POST requests
            params: Query parameters for GET requests
            retry: Whether to retry on failure

        Returns:
            Response JSON or None on failure
        """
        url = f"{self.orchestrator_url}{endpoint}"
        attempts = MAX_RETRY_ATTEMPTS if retry else 1

        for attempt in range(attempts):
            try:
                if method.upper() == "GET":
                    response = self._session.get(
                        url,
                        params=params,
                        timeout=REQUEST_TIMEOUT
                    )
                elif method.upper() == "POST":
                    response = self._session.post(
                        url,
                        json=json_data,
                        timeout=REQUEST_TIMEOUT
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)

                if attempt < attempts - 1:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{attempts}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {attempts} attempts: {e}")
                    return None

        return None

    def register(self, gpu_count: int, gpu_info: List[Dict] = None) -> Dict[str, Any]:
        """
        Register worker with orchestrator.

        Args:
            gpu_count: Number of GPUs available
            gpu_info: Optional detailed GPU information

        Returns:
            Response dict with status and optional config
            {"status": "wait"} - waiting for block_hash
            {"status": "ok", "block_hash": "...", ...} - ready to load model
        """
        self.gpu_count = gpu_count

        response = self._request(
            "POST",
            "/api/workers/connect",
            json_data={
                "worker_id": self.worker_id,
                "gpu_count": gpu_count,
                "gpu_info": gpu_info or [],
            }
        )

        if response:
            self.registered = True
            logger.info(f"Registered with orchestrator: {response}")

            # Start heartbeat thread
            self._start_heartbeat()

            # Parse config if provided
            if response.get("block_hash"):
                self._update_config(response)

            return response

        return {"status": "error", "message": "Failed to register"}

    def poll_config(self) -> Optional[Dict[str, Any]]:
        """
        Poll orchestrator for configuration (block_hash, job commands).

        Returns:
            Config dict or None if no updates
            {"type": "config", "block_hash": "...", "params": {...}}
            {"type": "compute", "public_key": "...", "nonce_range_start": 0, ...}
            {"type": "switch_job", "public_key": "..."}
            {"type": "shutdown"}
        """
        response = self._request(
            "GET",
            f"/api/workers/{self.worker_id}/config",
            retry=False  # Don't retry polling - just try again next interval
        )

        if response:
            self._update_config(response)

        return response

    def notify_model_loaded(self) -> Dict[str, Any]:
        """
        Notify orchestrator that model is loaded and worker is ready.

        Returns:
            Job configuration with public_key and nonce range
        """
        response = self._request(
            "POST",
            f"/api/workers/{self.worker_id}/ready",
            json_data={
                "gpu_count": self.gpu_count,
            }
        )

        if response:
            logger.info(f"Notified ready, received job config: {response}")
            self._update_config(response)
            return response

        return {"status": "error", "message": "Failed to notify ready"}

    def send_result(self, result: Dict[str, Any]) -> bool:
        """
        Send batch result to orchestrator.

        If sending fails, buffers result locally for retry.

        Args:
            result: Batch result dict (must include public_key)

        Returns:
            True if sent successfully, False if buffered
        """
        # Try to send any pending results first
        self._flush_pending()

        # Try to send current result
        response = self._request(
            "POST",
            f"/api/workers/{self.worker_id}/results",
            json_data=result,
            retry=False  # Don't block on retry, buffer instead
        )

        if response and response.get("status") == "ok":
            return True

        # Buffer for retry
        self._buffer_result(result)
        return False

    def send_results_batch(self, results: List[Dict[str, Any]]) -> int:
        """
        Send multiple results in one request (more efficient).

        Args:
            results: List of batch results

        Returns:
            Number of results successfully sent
        """
        if not results:
            return 0

        response = self._request(
            "POST",
            f"/api/workers/{self.worker_id}/results/batch",
            json_data={"results": results},
            retry=False
        )

        if response and response.get("status") == "ok":
            return len(results)

        # Buffer all for retry
        for result in results:
            self._buffer_result(result)

        return 0

    def heartbeat(self) -> bool:
        """
        Send heartbeat to orchestrator.

        Returns:
            True if successful
        """
        response = self._request(
            "POST",
            f"/api/workers/{self.worker_id}/heartbeat",
            json_data={
                "pending_results": len(self.pending_results),
            },
            retry=False
        )

        return response is not None

    def notify_shutdown(self, stats: Dict[str, Any] = None) -> bool:
        """
        Notify orchestrator that worker is shutting down.

        Args:
            stats: Optional final statistics

        Returns:
            True if notification sent
        """
        # Flush any pending results first
        self._flush_pending()

        response = self._request(
            "POST",
            f"/api/workers/{self.worker_id}/shutdown",
            json_data={
                "stats": stats or {},
            },
            retry=True  # Important to notify shutdown
        )

        self._stop_heartbeat()

        return response is not None

    def _update_config(self, response: Dict[str, Any]):
        """Update current config from response."""
        if "block_hash" in response:
            self.current_config.block_hash = response["block_hash"]
        if "block_height" in response:
            self.current_config.block_height = response["block_height"]
        if "r_target" in response:
            self.current_config.r_target = response["r_target"]
        if "params" in response:
            self.current_config.params = response["params"]
        if "public_key" in response:
            self.current_config.public_key = response["public_key"]
        if "nonce_range_start" in response:
            self.current_config.nonce_range_start = response["nonce_range_start"]
        if "nonce_range_end" in response:
            self.current_config.nonce_range_end = response["nonce_range_end"]

    def _buffer_result(self, result: Dict[str, Any]):
        """Buffer result for later retry."""
        with self._pending_lock:
            if len(self.pending_results) < MAX_PENDING_RESULTS:
                self.pending_results.append(result)
                logger.debug(f"Buffered result, pending count: {len(self.pending_results)}")
            else:
                logger.warning(f"Pending buffer full ({MAX_PENDING_RESULTS}), dropping oldest result")
                self.pending_results.pop(0)
                self.pending_results.append(result)

    def _flush_pending(self) -> int:
        """Try to send all pending results."""
        sent_count = 0

        with self._pending_lock:
            if not self.pending_results:
                return 0

            # Try to send in batch
            results_to_send = self.pending_results.copy()

        response = self._request(
            "POST",
            f"/api/workers/{self.worker_id}/results/batch",
            json_data={"results": results_to_send},
            retry=False
        )

        if response and response.get("status") == "ok":
            with self._pending_lock:
                # Remove sent results
                self.pending_results = self.pending_results[len(results_to_send):]
            sent_count = len(results_to_send)
            logger.info(f"Flushed {sent_count} pending results")

        return sent_count

    def get_pending_count(self) -> int:
        """Get number of pending results."""
        with self._pending_lock:
            return len(self.pending_results)

    def _start_heartbeat(self):
        """Start heartbeat background thread."""
        if self._heartbeat_thread is not None:
            return

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"Heartbeat-{self.worker_id[:8]}"
        )
        self._heartbeat_thread.start()
        logger.debug("Heartbeat thread started")

    def _stop_heartbeat(self):
        """Stop heartbeat background thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
        logger.debug("Heartbeat thread stopped")

    def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while not self._heartbeat_stop.wait(timeout=HEARTBEAT_INTERVAL):
            try:
                self.heartbeat()
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    def close(self):
        """Close client and cleanup resources."""
        self._stop_heartbeat()
        self._session.close()
        logger.info(f"OrchestratorClient closed: worker_id={self.worker_id}")
