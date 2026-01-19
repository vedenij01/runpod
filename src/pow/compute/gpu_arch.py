"""
GPU architecture detection and optimization configuration.

Supports:
- Blackwell: B200 (SM100)
- Hopper: H100, H200 (SM90)
- Ampere: A100 (SM80)
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

import torch

from common.logger import create_logger


logger = create_logger(__name__)


class GPUArchitecture(Enum):
    """Supported GPU architectures."""
    BLACKWELL = "blackwell"  # SM100 (B200)
    HOPPER = "hopper"        # SM90 (H100, H200)
    AMPERE = "ampere"        # SM80 (A100)
    UNKNOWN = "unknown"


@dataclass
class GPUCapabilities:
    """GPU capabilities detected at runtime."""
    architecture: GPUArchitecture
    compute_capability: tuple  # (major, minor)
    device_name: str
    total_memory_gb: float
    supports_fp8: bool
    supports_fp6: bool
    supports_fp4: bool
    supports_bfloat16: bool
    supports_flash_attention: bool
    recommended_dtype: torch.dtype
    max_batch_multiplier: float  # vs baseline H100


def get_gpu_architecture(device_id: int = 0) -> GPUCapabilities:
    """
    Detect GPU architecture and return optimization capabilities.

    Args:
        device_id: CUDA device index

    Returns:
        GPUCapabilities with detected features and recommended settings

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    props = torch.cuda.get_device_properties(device_id)
    cc = (props.major, props.minor)
    device_name = props.name
    total_memory_gb = props.total_memory / (1024**3)

    # Determine architecture from compute capability
    if cc[0] >= 10:
        # Blackwell: SM100 (B200), SM103 (B300), SM120 (RTX PRO 6000)
        arch = GPUArchitecture.BLACKWELL
        supports_fp8 = True
        supports_fp6 = True   # Blackwell 5th gen tensor cores
        supports_fp4 = True   # Blackwell 5th gen tensor cores
        supports_bfloat16 = True
        supports_flash_attention = True
        recommended_dtype = torch.bfloat16

        # Detect workstation vs datacenter GPU
        is_workstation = "RTX" in device_name.upper() or cc[1] >= 20  # SM120+ = workstation

        if is_workstation:
            # RTX PRO 6000 - lower memory bandwidth than datacenter GPUs
            max_batch_multiplier = 0.7
        else:
            # B200 and datacenter GPUs
            max_batch_multiplier = 0.8

    elif cc[0] == 9:
        # Hopper: SM90 (H100, H200)
        arch = GPUArchitecture.HOPPER
        supports_fp8 = True   # H100 has FP8 but less efficient than Blackwell
        supports_fp6 = False
        supports_fp4 = False
        supports_bfloat16 = True
        supports_flash_attention = True
        recommended_dtype = torch.bfloat16

        # Same batch multiplier for all Hopper GPUs (H100, H200)
        max_batch_multiplier = 0.8

    elif cc[0] == 8:
        # Ampere: SM80 (A100)
        arch = GPUArchitecture.AMPERE
        supports_fp8 = False
        supports_fp6 = False
        supports_fp4 = False
        supports_bfloat16 = True
        supports_flash_attention = True
        recommended_dtype = torch.bfloat16
        max_batch_multiplier = 0.8  # A100 slightly slower

    else:
        # Unknown/older architecture
        arch = GPUArchitecture.UNKNOWN
        supports_fp8 = False
        supports_fp6 = False
        supports_fp4 = False
        supports_bfloat16 = torch.cuda.is_bf16_supported()
        supports_flash_attention = cc >= (7, 5)  # Volta and above
        recommended_dtype = torch.float16 if not supports_bfloat16 else torch.bfloat16
        max_batch_multiplier = 0.5

    capabilities = GPUCapabilities(
        architecture=arch,
        compute_capability=cc,
        device_name=device_name,
        total_memory_gb=total_memory_gb,
        supports_fp8=supports_fp8,
        supports_fp6=supports_fp6,
        supports_fp4=supports_fp4,
        supports_bfloat16=supports_bfloat16,
        supports_flash_attention=supports_flash_attention,
        recommended_dtype=recommended_dtype,
        max_batch_multiplier=max_batch_multiplier,
    )

    logger.info(f"Detected GPU: {device_name} (SM{cc[0]}{cc[1]}) - {arch.value}")
    logger.info(f"  Memory: {total_memory_gb:.1f}GB, FP8: {supports_fp8}, BF16: {supports_bfloat16}")

    return capabilities


def get_optimal_dtype(device_id: int = 0, prefer_fp8: bool = False) -> torch.dtype:
    """
    Get optimal dtype for the detected GPU.

    Args:
        device_id: GPU device ID
        prefer_fp8: If True and supported, prefer FP8 path (returns bfloat16 for model,
                    FP8 handled separately in compute)

    Returns:
        Optimal torch.dtype for this GPU
    """
    caps = get_gpu_architecture(device_id)
    return caps.recommended_dtype


def is_blackwell_gpu(device_id: int = 0) -> bool:
    """
    Quick check if running on Blackwell GPU.

    Args:
        device_id: GPU device ID

    Returns:
        True if Blackwell architecture (SM100+)
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(device_id)
    return props.major >= 10


def is_hopper_gpu(device_id: int = 0) -> bool:
    """
    Quick check if running on Hopper GPU.

    Args:
        device_id: GPU device ID

    Returns:
        True if Hopper architecture (SM90)
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(device_id)
    return props.major == 9


def get_safe_configuration(device_id: int = 0) -> Dict[str, Any]:
    """
    Get safe/conservative configuration for the detected GPU.
    Use this when encountering issues with optimized settings.

    Args:
        device_id: GPU device ID

    Returns:
        Dictionary with conservative settings
    """
    return {
        "dtype": torch.float16,  # Most compatible
        "use_flash_attention": False,
        "use_torch_compile": False,
        "use_fp8": False,
        "batch_size_multiplier": 0.8,
        "target_memory_usage": 0.85,
    }


def should_use_fallback_mode() -> bool:
    """
    Check environment variable for fallback mode.
    Set BENCHMARK_FALLBACK_MODE=1 to use conservative settings.

    Returns:
        True if fallback mode is enabled
    """
    return os.environ.get("BENCHMARK_FALLBACK_MODE", "0") == "1"


def get_architecture_config(device_id: int = 0) -> Dict[str, Any]:
    """
    Get architecture-specific configuration for optimization.

    Args:
        device_id: GPU device ID

    Returns:
        Dictionary with architecture-optimized settings
    """
    if should_use_fallback_mode():
        return get_safe_configuration(device_id)

    caps = get_gpu_architecture(device_id)

    if caps.architecture == GPUArchitecture.BLACKWELL:
        # Use same proven settings as Hopper, just enable FP8
        # Coefficients should be tuned with real benchmarks, not guessed
        return {
            "dtype": torch.bfloat16,
            "use_flash_attention": True,
            "use_torch_compile": True,
            "use_fp8": True,
            "batch_size_multiplier": caps.max_batch_multiplier,
            "target_memory_usage": 0.90,  # Same as Hopper
            "activation_overhead_factor": 8.0,  # Same as Hopper
        }
    elif caps.architecture == GPUArchitecture.HOPPER:
        return {
            "dtype": torch.bfloat16,
            "use_flash_attention": True,
            "use_torch_compile": True,
            "use_fp8": False,  # FP8 less efficient on Hopper
            "batch_size_multiplier": caps.max_batch_multiplier,
            "target_memory_usage": 0.90,
            "activation_overhead_factor": 8.0,
        }
    elif caps.architecture == GPUArchitecture.AMPERE:
        return {
            "dtype": torch.bfloat16,
            "use_flash_attention": True,
            "use_torch_compile": True,
            "use_fp8": False,
            "batch_size_multiplier": caps.max_batch_multiplier,
            "target_memory_usage": 0.90,
            "activation_overhead_factor": 8.0,
        }
    else:
        return get_safe_configuration(device_id)
