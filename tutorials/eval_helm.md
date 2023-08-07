# Evaluating LLMs with HELM

> ![NOTE]\
> acknowledgement to NeurIPS Challenge Organizers for the instructions shown below

## Installing HELM

Install HELM with:

```sh
pip install crfm-helm
```

## Configure HELM

You can configure which datasets to run HELM (Holistic Evaluation of Language Models) on by editing a `run_specs.conf`.

Here's how you can create a simple `run_spec.conf` for local testing:

```sh
echo 'entries: [{description: "mmlu:model=neurips/local,subject=college_computer_science", priority: 4}]' > run_specs.conf
```

To run your model on a large set of datasets, take a look at [the example](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs_lite.conf) for inspiration.

## Run and Analyze Your Results

After creating `run_spec.conf`, you can run a quick local test with:

```sh
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1000
```

and then analyze the results with:

```sh
helm-summarize --suite v1
helm-server
```

This will analyze results and then launch a server on your local host, if you're working on a remote machine you might need to setup port forwarding. If everything worked correctly you should see a page that looks like [this](https://user-images.githubusercontent.com/3282513/249620854-080f4d77-c5fd-4ea4-afa4-cf6a9dceb8c9.png)
