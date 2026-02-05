# Kimi-K2.5 Deployment on ICE

> Kubernetes deployment of Moonshot AI's Kimi-K2.5 (1T parameter MoE) using KT-Kernel CPU-GPU heterogeneous inference.

## Current Status: LIVE

- **Public URL**: https://aic-kimi.icedc.se (Open WebUI)
- **API Endpoint**: `kimi-serve.aic.svc.cluster.local:31245` (cluster internal)
- **Performance**: ~16 tokens/second
- **Context Window**: 256K tokens
- **Multimodal**: Image input supported

## Project Context

This project deploys Kimi-K2.5 on the ICE GPU cluster (RISE). The model is too large for pure GPU inference on available hardware, so we use KT-Kernel to offload MoE experts to CPU memory while keeping active computations on GPU.

## Cluster Details

- **Namespace**: `aic`
- **Quota**: 800Gi memory (increased from 500Gi)
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
| `kimi-k2.5-download.yaml` | Download pod (already completed) |
| `kimi-k2.5-serve.yaml` | Inference server (2× H100 + KT-Kernel) |
| `kimi-k2.5-service.yaml` | ClusterIP service for internal access |
| `open-webui.yaml` | Open WebUI + Ingress for public access |
| `test_kimi.py` | Comprehensive test suite |

## Start the Full Stack

```bash
# 1. Start inference server (takes ~10 min to load model)
kubectl apply -f kimi-k2.5-serve.yaml

# 2. Wait for model to load
kubectl logs -f kimi-serve -n aic
# Look for: "The server is fired up and ready to roll!"

# 3. Start service and Open WebUI
kubectl apply -f kimi-k2.5-service.yaml
kubectl apply -f open-webui.yaml

# 4. Verify
kubectl get pods -n aic -l app=kimi-serve
kubectl get pods -n aic -l app=open-webui
```

Access at: https://aic-kimi.icedc.se

## Stop the Full Stack (Stop Billing)

```bash
# Stop Open WebUI (keeps data in PVC)
kubectl delete -f open-webui.yaml

# Stop inference server (THIS IS THE EXPENSIVE PART - 650Gi RAM + 2 GPUs)
kubectl delete pod kimi-serve -n aic

# Optional: delete service
kubectl delete -f kimi-k2.5-service.yaml
```

**Note**: The PVCs (`kimi-k2.5-model`, `open-webui-data`) continue billing even when pods are stopped. Delete them only if you want to remove all data.

## Check Status

```bash
# All resources
kubectl get pods,svc,ingress -n aic | grep -E "kimi|open-webui"

# Resource usage
kubectl top pod kimi-serve -n aic

# Logs
kubectl logs -f kimi-serve -n aic
kubectl logs -f -l app=open-webui -n aic
```

## Local Development Access

If Open WebUI is not running, use port-forward for direct API access:

```bash
kubectl port-forward kimi-serve -n aic 31245:31245
```

Then test:
```bash
curl -s http://localhost:31245/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Kimi-K2.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

## Resource Allocation

### Inference Server (`kimi-serve`)
- **CPU**: 64 cores requested
- **Memory**: 650Gi (for KT-Kernel expert offloading)
- **GPU**: 2× H100 NVL (188GB VRAM)
- **Shared memory**: 128Gi for NCCL/tensor ops

### Open WebUI
- **CPU**: 2 cores
- **Memory**: 2Gi
- **Storage**: 10Gi PVC

### Persistent Storage
- `kimi-k2.5-model`: 700Gi (model weights, ~555GB used)
- `open-webui-data`: 10Gi (user data, chat history)

## Performance Benchmarks

From `test_kimi.py` results:

| Metric | Value |
|--------|-------|
| Generation speed | 15.57 tok/s avg (16-17 steady state) |
| Context window | 262,144 tokens (256K) |
| Large context test | 10K tokens ✓ |
| Coding tests | 3/3 passed |
| Factual accuracy | 5/5 correct |
| Image understanding | Supported ✓ |
| Reasoning tests | 2/2 passed |

## Key Configuration Choices

### Why 2× H100 (not 8×)?
KT-Kernel offloads most experts to CPU, so we don't need all 8 GPUs. 2× H100 provides 188GB VRAM for active experts + KV cache, while the 650GB RAM handles the rest.

### Why RAWINT4 (not AMXINT4)?
The H100 nodes have AMD EPYC CPUs, which lack Intel AMX. RAWINT4 uses AVX512 which AMD fully supports.

### Why Open WebUI?
Provides a ChatGPT-like interface with user accounts, chat history, and easy access via public URL with auto-TLS.

## Namespace Quota Management

The namespace quota is managed via Rancher annotations. To change it:

```bash
# Check current quota
kubectl get resourcequota -n aic

# Patch to new value (e.g., 800Gi)
kubectl patch namespace aic --type='json' \
  -p='[{"op": "replace", "path": "/metadata/annotations/field.cattle.io~1resourceQuota", "value": "{\"limit\":{\"persistentVolumeClaims\":\"25\",\"requestsCpu\":\"256000m\",\"requestsMemory\":\"819200Mi\",\"requestsStorage\":\"10240000Mi\",\"limitsMemory\":\"819200Mi\"}}"}]'
```

## Troubleshooting

### Pod stuck in Pending
```bash
kubectl describe pod <pod-name> -n aic | tail -20
```
Check for resource constraints or node selector issues.

### OOM during model loading
Increase memory quota (see above) and update `kimi-k2.5-serve.yaml` memory requests.

### Inference server won't start
KT-Kernel compilation can fail. Check logs for build errors. May need to adjust CUDA/cuDNN versions.

### Open WebUI can't connect to model
Verify the service exists and pod is running:
```bash
kubectl get svc kimi-serve -n aic
kubectl get pods -l app=kimi-serve -n aic
```

### TLS certificate issues
```bash
kubectl get certificate -n aic
kubectl describe certificate open-webui-tls -n aic
```
Uses ClusterIssuer `letsencrypt` for auto-TLS.

## Dependencies

- **KTransformers**: `kvcache-ai/ktransformers` (branch: `kimi_k2.5`)
- **SGLang**: `kvcache-ai/sglang` (branch: `kimi_k2.5`)
- **Model**: `moonshotai/Kimi-K2.5` (~555GB INT4 weights)
- **Open WebUI**: `ghcr.io/open-webui/open-webui:main`

## Related Projects

- `../icemgmt` - ICE cluster management tools
