# Kimi-K2.5 on ICE

Deployment of [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5), Moonshot AI's 1 trillion parameter MoE model, on the ICE GPU cluster using KT-Kernel for CPU-GPU heterogeneous inference.

## Model Overview

| Spec | Value |
|------|-------|
| Total Parameters | 1 trillion |
| Activated Parameters | 32B (MoE) |
| Experts | 384 total, 8 selected per token |
| Context Length | 256K tokens |
| Model Size | ~600GB (INT4 quantized) |

## Hardware Requirements

KT-Kernel enables running this massive model on modest GPU hardware by offloading MoE experts to CPU memory.

**Minimum:**
- 2× RTX 4090 (48GB VRAM)
- 600GB system RAM
- CPU with AVX512F support

**Our ICE Configuration:**
- 2× H100 NVL (188GB VRAM)
- 1.5TB system RAM
- AMD EPYC 9654 (384 cores, full AVX512)

## Files

```
kimi-k2.5-pvc.yaml       # 700GB persistent storage for model
kimi-k2.5-download.yaml  # Pod to download model from HuggingFace
kimi-k2.5-serve.yaml     # Inference server (2× H100 + KT-Kernel)
```

## Deployment

### 1. Create Storage and Download Model

```bash
# Create PVC
kubectl apply -f kimi-k2.5-pvc.yaml

# Start download (~600GB, takes a while)
kubectl apply -f kimi-k2.5-download.yaml

# Monitor progress
kubectl exec kimi-download -n aic -- du -sh /models/kimi-k2.5
kubectl logs -f kimi-download -n aic
```

### 2. Start Inference Server

```bash
# Stop download pod when complete
kubectl delete pod kimi-download -n aic

# Start server (takes 2-3 min to load)
kubectl apply -f kimi-k2.5-serve.yaml

# Check logs
kubectl logs -f kimi-serve -n aic
```

### 3. Access the API

```bash
# Port forward
kubectl port-forward kimi-serve -n aic 31245:31245

# Test inference
curl -s http://localhost:31245/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kimi-K2.5",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     H100 Node (ICE)                         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │   H100 #1   │  │   H100 #2   │   GPU: Active experts    │
│  │   94GB      │  │   94GB      │   + KV cache + attention │
│  └─────────────┘  └─────────────┘                          │
│         │                │                                  │
│         └───────┬────────┘                                  │
│                 │ NVLink                                    │
│         ┌───────┴────────┐                                  │
│         │   KT-Kernel    │   CPU-GPU orchestration         │
│         └───────┬────────┘                                  │
│                 │                                           │
│  ┌──────────────┴──────────────┐                           │
│  │     AMD EPYC 9654           │   CPU: Inactive experts   │
│  │     384 cores, 1.5TB RAM    │   (INT4, AVX512)          │
│  └─────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--tensor-parallel-size 2` | 2 | Distribute across 2 H100s |
| `--kt-num-gpu-experts 60` | 60 | Experts kept on GPU (of 384) |
| `--kt-cpuinfer 128` | 128 | CPU threads for expert inference |
| `--kt-method RAWINT4` | RAWINT4 | INT4 quantization (AMD compatible) |
| `--max-total-tokens 50000` | 50K | Max context window |

## References

- [Kimi-K2.5 on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2.5)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
- [KT-Kernel SGLang Integration](https://github.com/kvcache-ai/sglang)
- [ICE Cluster Documentation](../icemgmt/icemgmt.md)
