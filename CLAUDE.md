# Kimi-K2.5 Deployment on ICE

> Kubernetes deployment of Moonshot AI's Kimi-K2.5 (1T parameter MoE) using KT-Kernel CPU-GPU heterogeneous inference.

## Project Context

This project deploys Kimi-K2.5 on the ICE GPU cluster (RISE). The model is too large for pure GPU inference on available hardware, so we use KT-Kernel to offload MoE experts to CPU memory while keeping active computations on GPU.

## Cluster Details

- **Namespace**: `aic`
- **Target nodes**: H100 nodes (`accelerator: nvidia-h100`)
- **Storage**: Ceph RBD (`rook-ceph-rbd`)

### H100 Node Specs (verified)
- **RAM**: 1.5TB (1,512 GB)
- **CPU**: 2× AMD EPYC 9654 (384 cores total)
- **GPU**: 8× H100 NVL (94GB each)
- **AVX512**: Full support (avx512f, avx512bw, avx512vl, etc.)

## Files

| File | Purpose |
|------|---------|
| `kimi-k2.5-pvc.yaml` | 700GB PersistentVolumeClaim for model weights |
| `kimi-k2.5-download.yaml` | Download pod (runs on 2080 Ti node) |
| `kimi-k2.5-serve.yaml` | Inference server (2× H100 + KT-Kernel) |

## Common Operations

### Check Download Progress
```bash
kubectl exec kimi-download -n aic -- du -sh /models/kimi-k2.5
```

### Check Pod Status
```bash
kubectl get pods -n aic -l app=kimi-download
kubectl get pods -n aic -l app=kimi-serve
```

### View Logs
```bash
kubectl logs -f kimi-download -n aic
kubectl logs -f kimi-serve -n aic
```

### Delete Pods
```bash
kubectl delete pod kimi-download -n aic
kubectl delete pod kimi-serve -n aic
```

### Port Forward for Local Access
```bash
kubectl port-forward kimi-serve -n aic 31245:31245
```

### Test Inference
```bash
curl -s http://localhost:31245/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Kimi-K2.5", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Deployment Workflow

1. **Storage**: `kubectl apply -f kimi-k2.5-pvc.yaml`
2. **Download**: `kubectl apply -f kimi-k2.5-download.yaml` (wait for ~600GB)
3. **Cleanup**: `kubectl delete pod kimi-download -n aic`
4. **Serve**: `kubectl apply -f kimi-k2.5-serve.yaml`

## Key Configuration Choices

### Why 2× H100 (not 8×)?
KT-Kernel offloads most experts to CPU, so we don't need all 8 GPUs. 2× H100 provides 188GB VRAM for active experts + KV cache, while the 1.5TB RAM handles the rest.

### Why RAWINT4 (not AMXINT4)?
The H100 nodes have AMD EPYC CPUs, which lack Intel AMX. RAWINT4 uses AVX512 which AMD fully supports.

### Resource Allocation
- **CPU**: 128 cores requested (of 384 available)
- **Memory**: 800GB requested (of 1.5TB available)
- **GPU**: 2× H100 (188GB VRAM)
- **Shared memory**: 128GB for NCCL/tensor ops

## Troubleshooting

### Pod stuck in Pending
```bash
kubectl describe pod <pod-name> -n aic | tail -20
```
Check for resource constraints or node selector issues.

### Download fails
The download pod uses `snapshot_download` from huggingface_hub. Check logs for rate limiting or network issues. The download is resumable.

### Inference server won't start
KT-Kernel compilation can fail. Check logs for build errors. May need to adjust CUDA/cuDNN versions.

## Dependencies

- **KTransformers**: `kvcache-ai/ktransformers` (branch: `kimi_k2.5`)
- **SGLang**: `kvcache-ai/sglang` (branch: `kimi_k2.5`)
- **Model**: `moonshotai/Kimi-K2.5` (~600GB INT4 weights)

## Current Status

**Status: BLOCKED - Awaiting quota increase**

### What's Done
- [x] Model downloaded to PVC `kimi-k2.5-model` (555GB)
- [x] KT-Kernel deployment tested and working (loads up to layer 35/60)
- [x] All YAML manifests created and tested

### What's Blocking
- Namespace `aic` has **500Gi memory quota**
- KT-Kernel needs **~600GB RAM** to load all MoE experts
- Server crashes at layer 35/60 due to OOM

### Next Steps (after quota increase to 700Gi+)
```bash
# 1. Update serve yaml to use more memory
# Edit kimi-k2.5-serve.yaml: change memory from 480Gi to 650Gi

# 2. Start the server
kubectl apply -f kimi-k2.5-serve.yaml

# 3. Wait for loading (~5-10 min)
kubectl logs -f kimi-serve -n aic

# 4. Port forward when ready
kubectl port-forward kimi-serve -n aic 31245:31245

# 5. Test
curl -s http://localhost:31245/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Kimi-K2.5", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Resources Currently Allocated
- **PVC**: `kimi-k2.5-model` - 700Gi (billing ~0.21 SEK/hr if HDD)
- **Pods**: None running

## Related Projects

- `../icemgmt` - ICE cluster management tools
