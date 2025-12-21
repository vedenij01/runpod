import time
import os
from typing import Any, List

import torch

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from pow.compute.autobs import get_total_GPU_memory
from pow.compute.utils import TimeStats
from pow.models.llama31 import ModelArgs, Transformer
from pow.models.utils import Params, count_params, set_default_dtype
from pow.random_pool_optimized import initialize_model_with_pool
from common.logger import create_logger


logger = create_logger(__name__)


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        devices: List[str],
        output_device: int = None,
        stats: TimeStats = None,
    ):
        super().__init__()
        self.output_device = output_device
        self.stats = stats
        self.module = module

    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        with torch.no_grad():
            with self.stats.time_infer():
                # Get device and dtype from model weights
                device = self.module.layers[0].attention.wq.weight.device
                dtype = self.module.layers[0].attention.wq.weight.dtype
                # Convert inputs to match model's device and dtype
                inputs = inputs.to(device=device, dtype=dtype)
                return self.module(inputs, **kwargs)

    @staticmethod
    def build_base_model(
        hash_: str,
        params: Params = Params(),
        seed: int = 42,
        max_seq_len: int = 1024,
        max_batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> dict:
        """Build model on CPU, load to GPU 0, return GPU state_dict for fast GPU→GPU cloning."""
        torch.manual_seed(seed)
        start_time = time.time()

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            flash=False,
            **(params.__dict__),
        )

        logger.info("Creating base model on CPU...")
        with torch.device("meta"):
            model = Transformer(model_args)
        model.to_empty(device="cpu")
        logger.info(f"Model structure created in {time.time() - start_time:.2f}s")

        model.eval()
        model.requires_grad_(False)

        if dtype == torch.float16:
            model = model.half()
            logger.info("Model converted to float16")
        elif dtype == torch.bfloat16:
            model = model.bfloat16()
        elif dtype == torch.float32:
            model = model.float()

        from pow.random_pool_optimized import initialize_model_with_pool
        initialize_model_with_pool(model, str(hash_), dtype=dtype, pool_fraction=0.05)
        model.recompute_freqs_cis()

        cpu_init_time = time.time() - start_time
        logger.info(f"Base model initialized on CPU in {cpu_init_time:.2f}s | {count_params(model)} params")

        # Load model to GPU 0
        gpu0_start = time.time()
        logger.info("Loading model to GPU 0...")
        model = model.to("cuda:0")
        torch.cuda.synchronize(0)
        logger.info(f"Model loaded to GPU 0 in {time.time() - gpu0_start:.2f}s")

        # Get state_dict from GPU 0 (tensors are on GPU 0)
        gpu_state_dict = model.state_dict()

        total_time = time.time() - start_time
        logger.info(f"Base model ready on GPU 0 in {total_time:.2f}s")

        return {
            "gpu_state_dict": gpu_state_dict,  # Tensors on GPU 0
            "model": model,  # Keep model for Worker 0 to use directly
            "model_args": model_args,
            "dtype": dtype,
        }

    @staticmethod
    def build_from_gpu_state_dict(
        base_model_data: dict,
        stats: TimeStats,
        target_device: str,
    ) -> "ModelWrapper":
        """Clone model from GPU 0 to target GPU using fast GPU→GPU copy."""
        start_time = time.time()

        target_device = torch.device(target_device)
        target_idx = target_device.index

        model_args = base_model_data["model_args"]
        gpu_state_dict = base_model_data["gpu_state_dict"]
        dtype = base_model_data["dtype"]
        base_model = base_model_data.get("model")  # Pre-loaded model on GPU 0

        # Check if target is same as source (GPU 0)
        source_device = next(iter(gpu_state_dict.values())).device
        is_same_device = (target_device == source_device)

        if is_same_device and base_model is not None:
            # Worker 0: Use existing model directly (no copy, no new allocation)
            logger.info(f"Using existing model on {target_device} (no copy needed)...")
            model = base_model
            set_default_dtype(device=target_device, dtype=dtype)
            logger.info(f"Model ready on {target_device} in {time.time() - start_time:.2f}s")
            return ModelWrapper(model, devices=[target_device], stats=stats)

        # Workers 1-N: Clone from GPU 0
        logger.info(f"Cloning model to {target_device} via GPU→GPU copy...")

        # Copy state_dict tensors from GPU 0 to target GPU
        copy_start = time.time()
        target_state_dict = {}
        for name, tensor in gpu_state_dict.items():
            # GPU→GPU copy (uses NVLink if available)
            target_state_dict[name] = tensor.to(target_device, non_blocking=True)

        # Synchronize to ensure all copies complete
        torch.cuda.synchronize(target_idx)
        logger.info(f"State dict copied to {target_device} in {time.time() - copy_start:.2f}s")

        # Create model directly on target device
        load_start = time.time()
        with torch.device(target_device):
            model = Transformer(model_args)

        # Load state_dict
        model.load_state_dict(target_state_dict)
        model.eval()
        model.requires_grad_(False)

        # Free target_state_dict after loading
        del target_state_dict
        torch.cuda.empty_cache()

        # Recompute freqs_cis on target device
        model.recompute_freqs_cis()

        set_default_dtype(device=target_device, dtype=dtype)
        logger.info(f"Model structure loaded in {time.time() - load_start:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"Model cloned to {target_device} in {total_time:.2f}s")

        return ModelWrapper(model, devices=[target_device], stats=stats)

    @staticmethod
    def build(
        hash_: str,
        stats: TimeStats,
        params: Params = Params(),
        seed: int = 42,
        max_seq_len: int = 1024,
        max_batch_size: int = 1,
        devices: List[str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "ModelWrapper":
        with stats.time_model_load():
            devices = [torch.device(device) for device in devices]
            primary_device = devices[0]

            torch.manual_seed(seed)
            start_time = time.time()

            model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                flash=False,
                **(params.__dict__),
            )

            logger.info("Creating model...")
            with torch.device("meta"):
                model = Transformer(model_args)
            model.to_empty(device="cpu")
            logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")

            model.eval()
            model.requires_grad_(False)
            
            # Convert model to specified dtype before moving to GPUs
            if dtype == torch.float16:
                model = model.half()
                logger.info("Model converted to float16")
            elif dtype == torch.bfloat16:
                model = model.bfloat16()
                logger.info("Model converted to bfloat16")
            elif dtype == torch.float32:
                model = model.float()
                logger.info("Model converted to float32")

            initialize_model_with_pool(model, str(hash_), dtype=dtype, pool_fraction=0.05)
            # Recompute freqs_cis after model is on CPU and properly initialized
            model.recompute_freqs_cis()

            init_time = time.time() - start_time
            logger.info(f"Model initialized in {init_time:.2f}s | {count_params(model)} params")

            try:
                max_memory = {}
                for device in devices:
                    device_id = device.index
                    max_memory[device_id] = f"{get_total_GPU_memory(device_id)}MB"
                max_memory = get_balanced_memory(model, max_memory=max_memory)
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=["TransformerBlock"],
                    dtype=dtype
                )
                logger.info(f"Inferred device map: {device_map}")
                model = dispatch_model(model, device_map=device_map)
                logger.info("Multi-GPU distribution successful")
            except Exception as e:
                logger.error(f"Multi-GPU distribution failed: {e}")
                logger.error("Falling back to single GPU")
                raise e
            
            model.eval()
            model.requires_grad_(False)

            set_default_dtype(device=primary_device, dtype=dtype)
            
            logger.info("Wrapping model in ModelWrapper")
            model_wrapper = ModelWrapper(model, devices=devices, stats=stats)
            logger.info(f"ModelWrapper created in {stats.model_load_time:.2f}s")

            return model_wrapper
