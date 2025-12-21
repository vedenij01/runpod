"""
FP8 utilities for Blackwell GPUs.

FP8 provides ~2x speedup on Blackwell's 5th gen tensor cores.
This module provides safe FP8 operations with fallback mechanisms.

Supported formats:
- E4M3 (4 exponent bits, 3 mantissa bits) - for activations/weights
- E5M2 (5 exponent bits, 2 mantissa bits) - for gradients (not used in inference)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from common.logger import create_logger


logger = create_logger(__name__)

# Check FP8 availability in PyTorch
HAS_FP8 = hasattr(torch, 'float8_e4m3fn') and hasattr(torch, 'float8_e5m2')


@dataclass
class FP8Config:
    """Configuration for FP8 operations."""
    enabled: bool = False
    scale_factor: float = 1.0
    amax_history_len: int = 16
    use_dynamic_scaling: bool = True


class FP8ScaleTracker:
    """
    Tracks scaling factors for FP8 conversion.
    Uses dynamic scaling to prevent overflow/underflow.
    """

    def __init__(self, history_len: int = 16):
        self.history_len = history_len
        self.amax_history = []
        self.scale = 1.0

    def update(self, tensor: torch.Tensor) -> float:
        """
        Update scale based on tensor's absolute maximum.

        Args:
            tensor: Input tensor to analyze

        Returns:
            Updated scale factor
        """
        with torch.no_grad():
            amax = tensor.abs().max().item()

        self.amax_history.append(amax)
        if len(self.amax_history) > self.history_len:
            self.amax_history.pop(0)

        # Use max of history for stability
        max_amax = max(self.amax_history) if self.amax_history else 1.0

        # FP8 E4M3 has max value of 448
        fp8_max = 448.0
        self.scale = fp8_max / (max_amax + 1e-12)

        return self.scale

    def reset(self):
        """Reset tracking history."""
        self.amax_history = []
        self.scale = 1.0


def is_fp8_available() -> bool:
    """
    Check if FP8 is available in current PyTorch installation.

    Returns:
        True if FP8 dtypes are available
    """
    return HAS_FP8


def is_fp8_beneficial(device_id: int = 0) -> bool:
    """
    Check if FP8 would be beneficial on this GPU.
    Only Blackwell GPUs get significant benefit from FP8.

    Args:
        device_id: GPU device ID

    Returns:
        True if FP8 would provide speedup
    """
    if not HAS_FP8:
        return False

    if not torch.cuda.is_available():
        return False

    props = torch.cuda.get_device_properties(device_id)
    # Only Blackwell (SM100+) has efficient 5th gen FP8 tensor cores
    return props.major >= 10


def convert_to_fp8(
    tensor: torch.Tensor,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """
    Convert tensor to FP8 E4M3 format with scaling.

    Args:
        tensor: Input tensor (float16/bfloat16/float32)
        scale: Scale factor to apply before conversion

    Returns:
        Tuple of (FP8 tensor, inverse scale for dequantization)

    Raises:
        RuntimeError: If FP8 is not available
    """
    if not HAS_FP8:
        raise RuntimeError("FP8 not available in this PyTorch version")

    # Scale and convert
    scaled = tensor * scale
    fp8_tensor = scaled.to(torch.float8_e4m3fn)

    return fp8_tensor, 1.0 / scale


def convert_from_fp8(
    fp8_tensor: torch.Tensor,
    inv_scale: float,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Convert FP8 tensor back to higher precision.

    Args:
        fp8_tensor: FP8 tensor
        inv_scale: Inverse scale factor from conversion
        target_dtype: Target dtype for output

    Returns:
        Dequantized tensor in target dtype
    """
    return fp8_tensor.to(target_dtype) * inv_scale


def maybe_convert_to_fp8(
    tensor: torch.Tensor,
    enabled: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[float]]:
    """
    Optionally convert tensor to FP8 for Blackwell optimization.

    Args:
        tensor: Input tensor
        enabled: Whether FP8 conversion is enabled
        scale: Optional scale factor (auto-computed if None)

    Returns:
        Tuple of (tensor, inverse_scale). If not converted, returns
        (original_tensor, None).
    """
    if not enabled or not HAS_FP8:
        return tensor, None

    try:
        if scale is None:
            # Auto-compute scale based on tensor range
            with torch.no_grad():
                amax = tensor.abs().max().item()
            scale = 448.0 / (amax + 1e-12)  # FP8 E4M3 max = 448

        fp8_tensor, inv_scale = convert_to_fp8(tensor, scale)
        return fp8_tensor, inv_scale

    except Exception as e:
        logger.warning(f"FP8 conversion failed: {e}, using original dtype")
        return tensor, None


class FP8LinearWrapper(nn.Module):
    """
    Wrapper for linear layers with FP8 weight storage.
    Weights are stored in FP8, converted to compute dtype during forward.
    This saves memory and enables FP8 tensor core operations.
    """

    def __init__(
        self,
        linear: nn.Linear,
        fp8_config: FP8Config,
    ):
        super().__init__()
        self.fp8_config = fp8_config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None

        if fp8_config.enabled and HAS_FP8:
            # Convert weights to FP8
            with torch.no_grad():
                weight = linear.weight.data
                amax = weight.abs().max().item()
                self.weight_scale = 448.0 / (amax + 1e-12)
                self.register_buffer(
                    'weight_fp8',
                    (weight * self.weight_scale).to(torch.float8_e4m3fn)
                )
                self.weight_inv_scale = 1.0 / self.weight_scale

            if self.has_bias:
                self.register_buffer('bias', linear.bias.data.clone())
            else:
                self.bias = None

            self.use_fp8 = True
            logger.debug(f"Converted linear {self.in_features}x{self.out_features} to FP8")
        else:
            # Keep original weights
            self.register_buffer('weight', linear.weight.data.clone())
            if self.has_bias:
                self.register_buffer('bias', linear.bias.data.clone())
            else:
                self.bias = None
            self.use_fp8 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp8:
            # Dequantize weights to compute dtype
            weight = self.weight_fp8.to(x.dtype) * self.weight_inv_scale
        else:
            weight = self.weight.to(x.dtype)

        return nn.functional.linear(x, weight, self.bias)


def convert_model_to_fp8(
    model: nn.Module,
    fp8_config: FP8Config,
) -> nn.Module:
    """
    Convert model's linear layers to use FP8 weight storage.

    Args:
        model: PyTorch model
        fp8_config: FP8 configuration

    Returns:
        Model with FP8 linear layers (if enabled)
    """
    if not fp8_config.enabled or not HAS_FP8:
        logger.info("FP8 disabled or not available, returning original model")
        return model

    converted_count = 0

    def replace_linear(module: nn.Module, name: str, parent: nn.Module):
        nonlocal converted_count
        if isinstance(module, nn.Linear):
            setattr(parent, name, FP8LinearWrapper(module, fp8_config))
            converted_count += 1

    # Recursively replace linear layers
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            replace_linear(child, name, model)
        else:
            convert_model_to_fp8(child, fp8_config)

    if converted_count > 0:
        logger.info(f"Converted {converted_count} linear layers to FP8")

    return model


def get_fp8_memory_savings(model: nn.Module) -> dict:
    """
    Estimate memory savings from FP8 conversion.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with memory statistics
    """
    total_params = 0
    linear_params = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_params += module.weight.numel()
            if module.bias is not None:
                linear_params += module.bias.numel()

    for param in model.parameters():
        total_params += param.numel()

    # FP8 uses 1 byte vs 2 bytes for float16/bfloat16
    fp16_memory_mb = total_params * 2 / (1024 * 1024)
    fp8_linear_memory_mb = linear_params * 1 / (1024 * 1024)
    other_memory_mb = (total_params - linear_params) * 2 / (1024 * 1024)
    fp8_total_mb = fp8_linear_memory_mb + other_memory_mb

    return {
        "total_params": total_params,
        "linear_params": linear_params,
        "fp16_memory_mb": fp16_memory_mb,
        "fp8_memory_mb": fp8_total_mb,
        "savings_mb": fp16_memory_mb - fp8_total_mb,
        "savings_percent": (1 - fp8_total_mb / fp16_memory_mb) * 100 if fp16_memory_mb > 0 else 0,
    }
