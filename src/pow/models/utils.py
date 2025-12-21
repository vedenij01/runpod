from dataclasses import dataclass

import torch

from common.logger import create_logger


logger = create_logger(__name__)


@dataclass
class Params:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 16
    vocab_size: int = 8192
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = False

    seq_len: int = 16

    # FP8 support for Blackwell GPUs (B200)
    use_fp8: bool = False


PARAMS_V1 = Params(
    dim=1024,
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,
    vocab_size=8196, 
    ffn_dim_multiplier=10.0,
    multiple_of=2048,
    norm_eps=1e-05,
    rope_theta=10000.0,
    use_scaled_rope=False,
    seq_len=128
)

PARAMS_V2 = Params(
    dim=1792,
    n_layers=64,
    n_heads=64,
    n_kv_heads=64,
    vocab_size=8196,
    ffn_dim_multiplier=10.0,
    multiple_of=4*2048,
    norm_eps=1e-5,
    rope_theta=10000.0,
    use_scaled_rope=False,
    seq_len=256,
)


def count_params(
    model: torch.nn.Module,
    print_summary: bool = True
) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    if print_summary:
        logger.info(f"Total number of parameters: {total_params / 1e9:.2f}B")
    return total_params


def set_default_dtype(
    device: str,
    dtype: torch.dtype = torch.float16,
):
    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, using CPU instead")
        return

    if dtype == torch.bfloat16:
        if torch.cuda.is_bf16_supported():
            logger.info("Model is using bfloat16")
            torch.set_default_dtype(torch.bfloat16)
        else:
            logger.warning(
                "bfloat16 is not supported on this device, falling back to float16"
            )
            torch.set_default_dtype(torch.float16)
    elif dtype == torch.float16:
        logger.info("Model is using float16")
        torch.set_default_dtype(torch.float16)
    elif dtype == torch.float32:
        logger.info("Model is using float32")
        torch.set_default_dtype(torch.float32)
    else:
        logger.warning(f"Unsupported dtype {dtype}, falling back to float16")
        torch.set_default_dtype(torch.float16)


def get_params_with_fp8(params: Params, enable_fp8: bool = False) -> Params:
    """
    Return params with FP8 setting based on GPU architecture.

    For Blackwell GPUs (B200), enables FP8 for ~2x performance improvement.

    Args:
        params: Base model parameters
        enable_fp8: Whether to enable FP8 (typically set based on GPU detection)

    Returns:
        Params with use_fp8 field set appropriately
    """
    if enable_fp8 and not params.use_fp8:
        # Create new params with FP8 enabled
        return Params(
            dim=params.dim,
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            n_kv_heads=params.n_kv_heads,
            vocab_size=params.vocab_size,
            ffn_dim_multiplier=params.ffn_dim_multiplier,
            multiple_of=params.multiple_of,
            norm_eps=params.norm_eps,
            rope_theta=params.rope_theta,
            use_scaled_rope=params.use_scaled_rope,
            seq_len=params.seq_len,
            use_fp8=True,
        )
    return params
