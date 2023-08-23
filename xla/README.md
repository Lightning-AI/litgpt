# TPU support

This project uses [`Fabric`](https://lightning.ai/docs/fabric/stable) under the hood, which itself supports TPUs (via [PyTorch XLA](https://github.com/pytorch/xla)).

The following commands will allow you to set up a Google Cloud instance with a [TPU v4](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) VM:

```shell
gcloud compute tpus tpu-vm create lit-gpt --version=tpu-vm-v4-base --accelerator-type=v4-8 --zone=us-central2-b
gcloud compute tpus tpu-vm ssh lit-gpt --zone=us-central2-b
```

<details>
<summary>Multihost caveats</summary>
  
TPU v4-8 uses a single host. SSH'ing into the machine and running commands manually will only work when using a single host (1 slice in the TPU pod).
In multi-host environments, as in larger TPU pod slices, it's necessary to launch all commands on all hosts simultaneously to avoid hangs.
For local development, we suggest uploading a zip with all your current changes and executing that inside the VM from your personal computer:

```shell
# zip the local directory. exclude large directories from the zip. you might want to keep them
zip -r local_changes.zip . -x  ".git/*" "checkpoints/*" "data/*" "out/*"
# copy the .zip to the TPU VM
gcloud compute tpus tpu-vm scp --worker=all local_changes.zip "lit-gpt:~"
# unzip on each host
gcloud compute tpus tpu-vm ssh lit-gpt --worker=all --command="cd ~; unzip -q -o local_changes.zip"

# example of typical workflow
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="cd ~; bash install_dependencies.sh"
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="cd ~; bash prepare_checkpoints.sh"
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="cd ~; bash run_desired_script.sh"

# this will allow you to kill all python processes on all workers
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="pkill -e python"
```

The rest of this tutorial will assume that it's being run in a single host for simplicity.

</details>

Now that you are in the machine, let's clone the repository and install the dependencies

```shell
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
pip install -r requirements.txt
```

Install Optimized BLAS

```shell
sudo apt update
sudo apt install libopenblas-dev
```

Since Lit-GPT requires a torch version newer than torch 2.0.0, we need to manually install nightly builds of torch and torch_xla:

```shell
pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl
pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
```

By default, computations will run using the new (and experimental) PjRT runtime. Still, it's recommended that you set the following environment variables

```shell
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
export PT_XLA_DEBUG=0
export USE_TORCH=ON
export PJRT_DEVICE=TPU
```

> [!NOTE]\
> You can find an extensive guide on how to get set-up and all the available options [here](https://cloud.google.com/tpu/docs/v4-users-guide).

Since you created a new machine, you'll probably need to download the weights.
You could scp them into the machine with `gcloud compute tpus tpu-vm scp` or you can follow the steps described in our [downloading guide](download_stablelm.md).

It is also a good idea to set up a persistent disk from which you can load checkpoints.
You can do so by following [this guide](https://cloud.google.com/tpu/docs/setup-persistent-disk#setting_up_a_tpu_vm_and_a_persistent_disk).
Multihost VM setups do not support read-write disks, so persistent disks cannot be used to save checkpoints in that case.
Persistent disks can still be useful in read-only mode to load pretrained weights before finetuning or inference.
In the multihost setting, since FSDP will save checkpoint shards per host and consolidate them into a single checkpoint, it's recommended that the consolidated checkpoints are uploaded to a Google Cloud bucket for safekeeping.
Another alternative would be to `scp` these checkpoints outside the TPU VM periodically. These alternatives are not implemented in our scripts.


## Inference

We have bespoke versions of our regular recipes to run with XLA in this directory. For example, generation:

```shell
python3 xla/generate/base.py --prompt "Hello, my name is" --num_samples 3
```

This command will take ~17s for the first generation time as XLA needs to compile the graph.
You'll notice that afterwards, generation times drop to ~2s.

## Finetuning

You can get started fine-tuning Falcon 7B with adapter by running the following:

```shell
python3 xla/finetune/adapter.py --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true
```

And generation with the adapter finetuned model weights can be done with:

```shell
python3 xla/generate/adapter.py --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --adapter_path out/adapter/alpaca/lit_model_adapter_finetuned.pth --devices 4
```

> **Warning**
> When you are done, remember to delete your instance
>
> ```shell
> gcloud compute tpus tpu-vm delete lit-gpt --zone=us-central2-b
> ```

## Computational performance

We were able to reach 49.57% MFU with Falcon 7B on a v4-32 (micro batch size 7) and 39.67% MFU with Falcon 40B on a v4-512 (micro batch size 3) using the [adapter finetuning script](finetune/adapter.py) and XLA's FSDP implementation at a fixed 1034 max sequence length.

Since the TPU VM hosts is limited in system memory (RAM) compared to device memory (HBM), we enabled specific techniques to limit peak RAM usage when loading the model and pretrained weights before sharding as well as when saving sharded checkpoints.
A v4 chip has 32 GiB HBM, so at 4 devices per host that's 4*32=128 GiB HBM. In comparison, each host has 188 GiB RAM but that's shared across the devices.
This means that any RAM allocation over 188/4=47 GiB would already max out the host's RAM memory.
A ~24B parameter model on CPU (with half precision) would be the largest possible model under this setup without the techniques used in our scripts.
