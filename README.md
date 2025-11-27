# LLaMA GPU Benchmark - Runpod Serverless

[![Runpod](https://api.runpod.io/badge/vedenij/runpod)](https://console.runpod.io/hub/vedenij/runpod)

GPU performance benchmark using LLaMA model inference. This service runs deterministic transformer computations to measure GPU throughput and validate hardware performance.

## Overview

This is a Runpod serverless implementation that:
- Runs LLaMA-based transformer inference workloads
- Measures GPU compute throughput across multiple samples
- Filters results by distance threshold
- Returns performance metrics and valid samples

## Architecture

```
handler.py          # Runpod serverless entry point
src/
  models/           # LLaMA model architecture
    llama31.py      # Transformer implementation
    utils.py        # Model utilities
  compute/          # Compute engine
    compute.py      # Simplified synchronous compute
    model_init.py   # Model initialization with multi-GPU
    utils.py        # Stats and utilities
  random.py         # Deterministic random generation
  random_pool_optimized.py  # Fast weight initialization
  data.py           # Batch data structures
  common/           # Common utilities
    logger.py       # Logging setup
```

## API Contract

The endpoint supports **streaming mode** for continuous benchmark generation.

### Streaming Mode

Generates samples continuously in batches until timeout or cancellation.

**Input Format:**
```json
{
  "input": {
    "block_hash": "string (hex)",
    "block_height": 123,
    "public_key": "string",
    "r_target": 2.0,
    "batch_size": 16,
    "start_nonce": 0,
    "params": {
      "dim": 4096,
      "n_layers": 32,
      "n_heads": 32,
      "n_kv_heads": 8,
      "vocab_size": 128256,
      "seq_len": 1024,
      "ffn_dim_multiplier": 1.3,
      "multiple_of": 1024,
      "norm_eps": 1e-05,
      "rope_theta": 500000.0
    }
  }
}
```

**Output Format (stream of batches):**
```json
{
  "public_key": "string",
  "block_hash": "string",
  "block_height": 123,
  "nonces": [5, 8, 12],
  "dist": [1.2, 1.5, 1.8],
  "node_id": 0,
  "batch_number": 1,
  "batch_computed": 16,
  "batch_valid": 3,
  "total_computed": 16,
  "total_valid": 3,
  "next_nonce": 16
}
```

### Error Format

```json
{
  "error": "error message",
  "error_type": "ExceptionType"
}
```

## Key Features

### Model Caching
- Model is loaded once per block_hash and reused across requests
- Significant performance improvement for warm requests
- Different block_hash triggers model reinitialization

### Multi-GPU Support
- Automatic detection of all available GPUs
- Model distribution via Accelerate library
- Automatic device mapping and load balancing

## Deployment

### Build Docker Image

```bash
docker build -t llama-gpu-benchmark .
```

### Test Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run handler with test input
python test_local.py
```

### Deploy to Runpod

1. Build and push Docker image to registry:
```bash
docker tag llama-gpu-benchmark:latest your-registry/llama-gpu-benchmark:latest
docker push your-registry/llama-gpu-benchmark:latest
```

2. Create serverless endpoint on Runpod:
- Go to Runpod Serverless
- Create new endpoint
- Use your Docker image
- Configure GPU requirements (24GB+ VRAM recommended)
- Set timeout to 600s

3. Test the endpoint:
```bash
curl -X POST https://your-endpoint.runpod.ai/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @test_input.json
```

## Performance Considerations

### Model Initialization
- First request (cold start): ~30-60 seconds for model initialization
- Subsequent requests (warm): ~1-5 seconds depending on batch size
- Different block_hash requires reinitialization

### Batch Size
- Recommended: 16-256 samples per batch
- Larger batches improve GPU utilization
- Balance between latency and throughput

### Memory Requirements
- Model size varies by configuration
- Recommended: GPUs with 24GB+ VRAM (A6000, A100, H100, etc.)

## Configuration

### Environment Variables

- `LOG_LEVEL`: Logging level (default: INFO)
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocator settings

### Model Parameters

Default parameters match LLaMA 3.1 8B architecture:
- dim: 4096
- n_layers: 32
- n_heads: 32
- vocab_size: 128256

Can be overridden per request via `params` field.

## Troubleshooting

### Out of Memory
- Reduce batch size
- Check PYTORCH_CUDA_ALLOC_CONF settings

### Slow Performance
- Check if model is being reinitialized (different block_hash)
- Increase batch size for better GPU utilization
- Verify GPU is being used (check logs)

### Model Initialization Errors
- Verify GPU has sufficient VRAM (24GB+ recommended)
- Check CUDA compatibility
