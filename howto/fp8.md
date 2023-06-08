# FP8 mixed precision

[NVIDIA's Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower memory utilization in both training and inference

To use it, you'll need access to an H100 machine. If you don't have it, you can use CoreWeave's cloud.

## Access CoreWeave's H100 machines

If you haven't accessed CoreWeave's Kubernetes, plese follow [these instruction](https://docs.coreweave.com/coreweave-kubernetes/getting-started) to set it up.

Create this `manifest.yaml` file. Feel free to adapt it to your requirements:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: h100
spec:
  containers:
  - name: h100
    image: "ghcr.io/coreweave/ml-containers/torch:ceeb8c2-nccl-cuda12.0.1-nccl2.18.1-1-torch2.0.1-vision0.15.2-audio2.0.2"
    command:  ["tail", "-f", "/dev/null"]  # keep it running indefinitely
    resources:
      limits:
        memory: 960Gi
        nvidia.com/gpu: 8
      requests:
        cpu: 110
        memory: 960Gi
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu.nvidia.com/class
            operator: In
            values:
            - H100_NVLINK_80GB
          - key: topology.kubernetes.io/region
            operator: In
            values:
            - LAS1
```

Use `kubectl` to launch and interact with your pod:

```shell
kubectl apply -f manifest.yaml  # launch h100 pod
kubectl exec -it pod/h100 -- /bin/bash  # SSH into it
kubectl delete pod/h100  # Delete the pod once you are done
```

You can find a list of useful commands [here](https://docs.coreweave.com/coreweave-kubernetes/useful-commands). The `pod_id` is `pod/h100`.

## Setup your environment

TransformerEngine requires some specific installation steps:

```shell
pip install git+https://github.com/Lightning-AI/lightning.git@carmocca/transformer-engine
# you'll want CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/nightly/cu121 --pre 'torch>=2.1.0dev'
# needs to be installed separately until https://github.com/HazyResearch/flash-attention/issues/246 is resolved
pip install flash-attn --no-build-isolation
NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@main
```

## WIP

for inference, training:
    Baseline speed on H100
    TE linear layers speed
    TE linear layers + autocast speed (+Fabric)
    TE full layers + autocast speed (+Fabric)
