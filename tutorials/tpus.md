# TPU support

This project uses `lightning.Fabric` under the hood, which itself supports TPUs (via [PyTorch XLA](https://github.com/pytorch/xla)).

The following commands will allow you to set up a `Google Cloud` instance with a [TPU v4](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) VM:

```shell
gcloud compute tpus tpu-vm create lit-gpt --version=tpu-vm-v4-base --accelerator-type=v4-8 --zone=us-central2-b
gcloud compute tpus tpu-vm ssh lit-gpt --zone=us-central2-b
```

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
export PJRT_DEVICE=TPU
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
```

> **Note**
> You can find an extensive guide on how to get set-up and all the available options [here](https://cloud.google.com/tpu/docs/v4-users-guide).

Since you created a new machine, you'll probably need to download the weights. You could scp them into the machine with `gcloud compute tpus tpu-vm scp` or you can follow the steps described in our [downloading guide](download_stablelm.md).

## Inference

Generation works out-of-the-box with TPUs:

```shell
python3 generate/base.py --prompt "Hello, my name is" --num_samples 3
```

This command will take take ~17s for the first generation time as XLA needs to compile the graph.
You'll notice that afterwards, generation times drop to ~2s.

## Finetuning

Coming soon.

> **Warning**
> When you are done, remember to delete your instance
>
> ```shell
> gcloud compute tpus tpu-vm delete lit-gpt --zone=us-central2-b
> ```
