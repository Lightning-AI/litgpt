# TPU support

Lit-LLaMA used `lightning.Fabric` under the hood, which itself supports TPUs (via [PyTorch XLA](https://github.com/pytorch/xla)).

The following commands will allow you to set up a `Google Cloud` instance with a [TPU v4](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) VM:

```shell
gcloud compute tpus tpu-vm create lit-llama --version=tpu-vm-v4-pt-2.0 --accelerator-type=v4-8 --zone=us-central2-b
gcloud compute tpus tpu-vm ssh lit-llama --zone=us-central2-b
```

Now that you are in the machine, let's clone the repository and install the dependencies

```shell
git clone https://github.com/Lightning-AI/lit-llama
cd lit-llama
pip install -r requirements.txt
```

By default, computations will run using the new (and experimental) PjRT runtime. Still, it's recommended that you set the following environment variables

```shell
export PJRT_DEVICE=TPU
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
```

> **Note**
> You can find an extensive guide on how to get set-up and all the available options [here](https://cloud.google.com/tpu/docs/v4-users-guide).

Since you created a new machine, you'll probably need to download the weights. You could scp them into the machine with `gcloud compute tpus tpu-vm scp` or you can follow the steps described in our [downloading guide](download_weights.md).

## Inference

Generation works out-of-the-box with TPUs:

```shell
python3 generate.py --prompt "Hello, my name is" --num_samples 2
```

This command will take a long time as XLA needs to compile the graph (~13 min) before running the model.
In fact, you'll notice that the second sample takes considerable less time (~12 sec).

## Finetuning

Coming soon.

> **Warning**
> When you are done, remember to delete your instance 
> ```shell
> gcloud compute tpus tpu-vm delete lit-llama --zone=us-central2-b
> ```