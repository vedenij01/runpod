from pow.compute.autobs import get_batch_size_from_memory
from pow.compute.gpu_group import GpuGroup
from pow.models.utils import Params, PARAMS_V1, PARAMS_V2
from common.logger import create_logger
import math

logger = create_logger(__name__)

# Lazy import to avoid circular dependency
_gpu_arch_module = None


def _get_architecture_config(device_id: int = 0):
    """Lazy import of gpu_arch module to get architecture config."""
    global _gpu_arch_module
    if _gpu_arch_module is None:
        from pow.compute import gpu_arch
        _gpu_arch_module = gpu_arch
    return _gpu_arch_module.get_architecture_config(device_id)


def _params_match_model(params: Params, reference: Params) -> bool:
    """
    Compare params by core model architecture, ignoring optimization flags like use_fp8.
    This ensures batch size estimation works correctly when FP8 is enabled.
    """
    return (
        params.dim == reference.dim and
        params.n_layers == reference.n_layers and
        params.n_heads == reference.n_heads and
        params.n_kv_heads == reference.n_kv_heads and
        params.vocab_size == reference.vocab_size and
        params.ffn_dim_multiplier == reference.ffn_dim_multiplier and
        params.multiple_of == reference.multiple_of and
        params.seq_len == reference.seq_len
    )


def get_batch_size_for_gpu_group(gpu_group: GpuGroup, params: Params, target_memory_usage: float = 0.9) -> int:
    # Compare by model architecture, not exact equality (ignore use_fp8 flag)
    if _params_match_model(params, PARAMS_V1):
        return get_batch_size_from_memory(
            target_memory_usage=target_memory_usage,
            device_id=gpu_group.primary_device
        )

    if _params_match_model(params, PARAMS_V2):
        return estimate_batch_size(gpu_group, params, target_memory_usage)

    return 100


def estimate_batch_size(gpu_group: GpuGroup, params: Params, target_memory_usage: float = 0.9) -> int:
    """
    Estimate optimal batch size based on GPU memory and architecture.

    For Blackwell GPUs (B200), uses optimized parameters:
    - Higher target memory usage (0.95 vs 0.90) due to TMEM
    - Lower activation overhead factor (6.0 vs 8.0) for 5th gen tensor cores

    Args:
        gpu_group: GPU group to estimate batch size for
        params: Model parameters
        target_memory_usage: Target fraction of free memory to use

    Returns:
        Estimated batch size that fits in GPU memory
    """
    # --- 1. Define Constants and Assumptions ---
    BYTES_PER_ELEMENT = 2  # float16/bfloat16

    # Get architecture-specific settings
    try:
        arch_config = _get_architecture_config(gpu_group.primary_device)
        ACTIVATION_OVERHEAD_FACTOR = arch_config.get("activation_overhead_factor", 8.0)
        target_memory_usage = arch_config.get("target_memory_usage", target_memory_usage)
        batch_multiplier = arch_config.get("batch_size_multiplier", 1.0)
        logger.debug(
            f"Using architecture config: activation_factor={ACTIVATION_OVERHEAD_FACTOR}, "
            f"target_mem={target_memory_usage}, batch_mult={batch_multiplier}"
        )
    except Exception as e:
        logger.debug(f"Could not get architecture config: {e}, using defaults")
        ACTIVATION_OVERHEAD_FACTOR = 8.0
        batch_multiplier = 1.0

    SAFETY_MARGIN = 0.90
    num_gpus = gpu_group.group_size

    if num_gpus == 0:
        return 1

    # --- 2. Calculate Static Memory Usage (Model Weights) ---
    # This calculation remains the same, as it's for the whole model.
    ffn_hidden_dim = params.multiple_of * math.ceil(
        (params.ffn_dim_multiplier * (2/3 * 4 * params.dim)) / params.multiple_of
    )
    attention_params = params.n_layers * (4 * params.dim**2)
    ffn_params = params.n_layers * (
        params.dim * ffn_hidden_dim
        + ffn_hidden_dim * params.dim
        + params.dim * ffn_hidden_dim
    )
    output_params = params.vocab_size * params.dim
    total_params = attention_params + ffn_params + output_params
    total_model_weights_mb = (total_params * BYTES_PER_ELEMENT) / (1024**2)

    # Assume accelerate balances the weights evenly across all GPUs.
    weights_per_gpu_mb = total_model_weights_mb / num_gpus

    # --- 3. Find the Bottleneck GPU ---
    # Get the free memory for each device individually.
    free_vram_per_device_mb = gpu_group.get_free_vram_mb_per_device()

    memory_for_activations_per_gpu = {}
    for device_id, free_mb in free_vram_per_device_mb.items():
        # For each GPU, calculate usable memory and subtract its share of the weights.
        usable_free_mb = free_mb * target_memory_usage * SAFETY_MARGIN
        memory_for_activations_per_gpu[device_id] = usable_free_mb - weights_per_gpu_mb

    # The true available memory is limited by the GPU with the LEAST space for activations.
    if not memory_for_activations_per_gpu:
        return 1

    bottleneck_memory_mb = min(memory_for_activations_per_gpu.values())

    if bottleneck_memory_mb <= 0:
        logger.warning(
            f"The most constrained GPU has no memory left after loading model weights. "
            f"Estimated weights per GPU: {weights_per_gpu_mb:.2f} MB. "
            f"Check `nvidia-smi` for other running processes."
        )
        return 1

    # --- 4. Calculate Dynamic Memory per Batch Item ---
    # This part is the same, as it represents the peak load that will hit the bottleneck GPU.
    kv_cache_bytes_per_item = 2 * params.n_layers * params.seq_len * params.dim * BYTES_PER_ELEMENT
    activations_bytes_per_item = ACTIVATION_OVERHEAD_FACTOR * params.seq_len * params.dim * BYTES_PER_ELEMENT
    attention_scores_bytes_per_item = params.n_heads * (params.seq_len**2) * BYTES_PER_ELEMENT

    memory_per_batch_item_mb = (
        kv_cache_bytes_per_item +
        activations_bytes_per_item +
        attention_scores_bytes_per_item
    ) / (1024**2)

    if memory_per_batch_item_mb < 1e-6:
        return 1

    # --- 5. Determine Final Batch Size based on the Bottleneck ---
    estimated_bs = math.floor(bottleneck_memory_mb / memory_per_batch_item_mb)

    # Apply architecture-specific batch size multiplier
    # Blackwell GPUs can handle larger batches more efficiently
    final_bs = int(estimated_bs * batch_multiplier)

    return max(1, final_bs)
