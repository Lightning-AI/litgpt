# Deploy LitGPT on Nebius Serverless Endpoints

This tutorial packages [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) and serves it through a managed HTTPS URL on [Nebius Serverless Endpoints](https://docs.nebius.com/serverless/endpoints/manage). It uses LitGPT's existing OpenAI-compatible chat API and one GPU. The small, ungated model makes this a deployment smoke test, not a throughput benchmark.

## Prerequisites

- A Nebius project with permission to create Serverless Endpoints and sufficient GPU, Compute, disk and networking quotas. Follow the [endpoint prerequisites](https://docs.nebius.com/serverless/quickstart/endpoints#prerequisites).
- The [Nebius CLI](https://docs.nebius.com/cli/install), configured for that project. The commands below use CLI **0.12.279**.
- Docker with Buildx, Bash, `curl`, and `jq` on your workstation.
- A container registry repository that you can push to and that Nebius can pull from. The example uses a **public** image containing only the ungated model and runtime. For a private registry, configure a [registry secret](https://docs.nebius.com/serverless/endpoints/manage) and add `--registry-secret` when creating the endpoint.

Running endpoints incur charges even without requests. Check [Serverless pricing and quotas](https://docs.nebius.com/serverless/pricing-quotas) and [current prices](https://nebius.com/prices) for your selected platform and region before creation. Plan to delete this smoke-test endpoint as soon as the requests below finish; stopping also stops endpoint compute and storage billing.

## 1. Build a reproducible image

Run these commands from the LitGPT repository root. The [Dockerfile](nebius-serverless/Dockerfile) pins Python by image digest and uses a [Linux x86_64 dependency lock](nebius-serverless/requirements.txt), including LitGPT 0.5.13, LitServe 0.2.19 and PyTorch 2.7.1 (CUDA 12.6). The [download script](nebius-serverless/download_model.py) pins the model revision and converts it to LitGPT format during the build, which does not need a GPU.

```bash
# Replace with a repository you own; authenticate to that registry with docker login.
export IMAGE_TAG='docker.io/YOUR_ACCOUNT/litgpt-smollm2:0.5.13'

docker buildx build --platform linux/amd64 \
  --tag "$IMAGE_TAG" --push \
  --metadata-file /tmp/litgpt-image-metadata.json \
  tutorials/nebius-serverless

# Deploy the exact image built above, even if the tag is changed later.
IMAGE_DIGEST=$(jq -er '."containerimage.digest"' /tmp/litgpt-image-metadata.json)
export IMAGE_REF="${IMAGE_TAG%:*}@${IMAGE_DIGEST}"
```

Docker caches the dependency and model layers. The converted checkpoint lives at `/app/checkpoints/HuggingFaceTB/SmolLM2-135M-Instruct` inside the image, so each replica has the same weights without a runtime Hugging Face download, token, or external volume. `HF_HUB_OFFLINE=1` prevents runtime Hub access. Rebuilding after changing the model or download script invalidates the model layer; ordinary endpoint restarts reuse the image. Keep adequate local disk space for CUDA dependencies and the build cache.

The runtime listens on `0.0.0.0:8000`. `--openai_spec true` enables `/v1/chat/completions`; clients select streaming per request with `"stream": true`, so a second server or `--stream true` is unnecessary.

For an optional local GPU check on a Linux machine with the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all --name litgpt-smoke \
  -p 127.0.0.1:8000:8000 "$IMAGE_REF"
```

In another terminal, `curl --fail http://127.0.0.1:8000/health` should eventually return `ok`. Stop the local container with `docker stop litgpt-smoke`.

## 2. Select resources and create an authenticated endpoint

Use the same Bash session for the rest of the tutorial:

```bash
set -o pipefail
nebius version
nebius profile current
PROJECT_ID=$(nebius config get parent-id)
export PROJECT_ID
export ENDPOINT_NAME="litgpt-smoke-$(date -u +%Y%m%d%H%M%S)"

# Choose a platform/preset pair actually offered in your project's region.
nebius compute platform list --parent-id "$PROJECT_ID" --format json --all \
  | jq '.items[] | {platform: .metadata.name, presets: [.spec.presets[].name]}'

# Example only: use these values if they appear together in the catalog above.
export PLATFORM='gpu-h100-sxm'
export PRESET='1gpu-16vcpu-200gb'
```

A catalog entry does not guarantee available capacity or sufficient quota. Choose a single-GPU preset and review its price. For this example, the container requires an NVIDIA GPU supporting BF16 and a driver compatible with CUDA 12.6.

```bash
# Check for an existing endpoint before creating one. An API/auth error is not
# evidence that the name is unused; resolve it before continuing.
nebius ai endpoint list --parent-id "$PROJECT_ID" --format json \
  | jq --arg name "$ENDPOINT_NAME" \
      '[.items[]? | select(.metadata.name == $name) | {id: .metadata.id, state: .status.state}]'

create_endpoint() {
  nebius ai endpoint create \
    --parent-id "$PROJECT_ID" --name "$ENDPOINT_NAME" \
    --image "$IMAGE_REF" --platform "$PLATFORM" --preset "$PRESET" \
    --on-demand --disk-size 50Gi --container-port 8000/http --auth token \
    --format json "$@"
}

# Validate configuration without allocating a GPU.
create_endpoint --dry-run > /dev/null

# Run once, after checking price/quota. The CLI generates the endpoint token.
# Suppress raw output because endpoint responses can contain that token.
create_endpoint --async > /dev/null
```

If creation fails or times out, inspect the endpoint by name before retrying: a request may have reached the service even if the client lost the response. The managed HTTPS URL does not need `--public` or a raw public IP.

Nebius checks the bearer token at the managed endpoint. LitGPT's `--access_token` is for downloading gated models, **not** for authenticating clients. This image does not set LitServe's separate `LIT_SERVER_API_KEY` (`X-API-Key`) mechanism. Keep all remote traffic on the managed, authenticated URL.

## 3. Wait for model readiness

Save the ID as soon as the endpoint exists, so you can clean up even if startup fails:

```bash
ENDPOINT_ID=$(nebius ai endpoint get-by-name \
  --parent-id "$PROJECT_ID" --name "$ENDPOINT_NAME" --format json \
  | jq -er '.metadata.id')
export ENDPOINT_ID

nebius ai endpoint get --id "$ENDPOINT_ID" --format json \
  | jq '{id: .metadata.id, state: .status.state, urls: .status.public_endpoints}'
```

If the asynchronous create has not registered the name yet, retry the lookup briefly. If it remains absent, investigate the create error rather than submitting a second create request. Keep `PROJECT_ID` and `ENDPOINT_NAME` until cleanup has succeeded. A platform state alone does not establish model readiness: LitServe's `/health` returns HTTP 200 only when the model workers are ready (otherwise HTTP 503).

The following function waits about ten minutes, with bounded CLI and HTTP requests. It keeps the generated token out of terminal output. Do not enable shell tracing (`set -x`).

```bash
wait_for_model() {
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    ENDPOINT_URL=$(nebius ai endpoint get --id "$ENDPOINT_ID" --format json \
      --no-browser --auth-timeout 10s --timeout 10s --per-retry-timeout 5s --retries 1 \
      | jq -r '.status.public_endpoints[0] // empty') || return 1
    ENDPOINT_TOKEN=$(nebius ai endpoint get --id "$ENDPOINT_ID" --format json \
      --no-browser --auth-timeout 10s --timeout 10s --per-retry-timeout 5s --retries 1 \
      | jq -er '.spec.auth_token') || return 1
    if [[ -n "$ENDPOINT_URL" ]] && curl --fail --silent --show-error \
      --connect-timeout 5 --max-time 10 \
      -H "Authorization: Bearer $ENDPOINT_TOKEN" \
      "${ENDPOINT_URL%/}/health"; then
      printf '\n'
      return 0
    fi
    sleep 5
  done
  echo 'Model did not become ready; inspect logs and stop or delete the endpoint.' >&2
  return 1
}
wait_for_model
```

Continue only after this succeeds. For troubleshooting, retrieve a bounded log snapshot and safe status fields:

```bash
nebius ai endpoint logs "$ENDPOINT_ID" --tail 100
nebius ai endpoint get --id "$ENDPOINT_ID" --format json \
  | jq '{state: .status.state, urls: .status.public_endpoints}'
```

Image-pull failures usually require checking the image digest, registry visibility or registry secret. Model initialization errors require inspecting the container logs; `--accelerator cuda` deliberately fails if CUDA is unavailable. A readiness timeout leaves the endpoint allocated: stop or delete it using step 5 while investigating. Do not print or share raw `endpoint get` output, which includes the generated token.

## 4. Check authentication, generate, and stream

First check that the gateway rejects requests without a token. Expect HTTP 401 or 403; a 2xx response means authentication is not configured as intended, and a 5xx response indicates the endpoint is not ready to test.

```bash
curl --silent --output /dev/null --write-out '%{http_code}\n' \
  --max-time 15 "${ENDPOINT_URL%/}/health"
```

Generate a short completion and save the response locally:

```bash
curl --fail-with-body --silent --show-error --max-time 120 \
  "${ENDPOINT_URL%/}/v1/chat/completions" \
  -H "Authorization: Bearer $ENDPOINT_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"SmolLM2-135M-Instruct","messages":[{"role":"user","content":"Name one planet."}],"max_completion_tokens":32,"stream":false}' \
  > completion.json
jq -er '.choices[0].message.content' completion.json
```

For streaming, use the same server and add `"stream": true`. `curl --no-buffer` displays the server-sent events as they arrive; successful completion ends with `data: [DONE]`.

```bash
curl --fail-with-body --silent --show-error --no-buffer --max-time 120 \
  "${ENDPOINT_URL%/}/v1/chat/completions" \
  -H "Authorization: Bearer $ENDPOINT_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"SmolLM2-135M-Instruct","messages":[{"role":"user","content":"Name one planet."}],"max_completion_tokens":32,"stream":true}' \
  | tee completion.sse
```

The outputs are `completion.json` and `completion.sse` on your workstation. There are no training artifacts to retrieve from this endpoint; the model checkpoint is already in the image. A client timeout does not stop the endpoint. Inspect errors and then clean up, including after interrupted streaming requests.

## 5. Stop or delete the endpoint

Delete the smoke-test endpoint even if a previous step failed:

```bash
nebius ai endpoint delete --id "$ENDPOINT_ID"
unset ENDPOINT_TOKEN
```

If ID capture failed, recover it with the `get-by-name` command in step 3 before deleting. Confirm the deletion succeeded; if the CLI reports an error, check the resource state and resolve it rather than assuming billing stopped.

To keep the configuration for a later session, use `nebius ai endpoint stop --id "$ENDPOINT_ID"` instead, then verify its stopped state with the filtered `get` command. Resume it with `nebius ai endpoint start --id "$ENDPOINT_ID"` and repeat the readiness check before querying. Stopped endpoints do not incur endpoint compute or storage charges; separately provisioned registry storage or other resources can still incur charges.

The CLI version used here has no endpoint update command. To change the image or configuration, create a new endpoint, verify it, and delete the old one. Remove the uploaded image from your registry when you no longer need it, and delete the local completion files if they are no longer needed.

## Updating the pinned runtime

The recipe is deliberately versioned independently of your checkout. To change dependencies, edit [requirements.in](nebius-serverless/requirements.in), regenerate the lock, rebuild the image, and repeat the readiness, generation, streaming and authentication checks:

```bash
uv pip compile tutorials/nebius-serverless/requirements.in \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --no-annotate --output-file tutorials/nebius-serverless/requirements.txt
```
