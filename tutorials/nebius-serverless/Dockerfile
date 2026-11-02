FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --only-binary=:all: -r requirements.txt && python -m pip check
COPY download_model.py .
RUN python download_model.py

# The model is part of the image; serving must not download a different revision.
ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["litgpt", "serve", "/app/checkpoints/HuggingFaceTB/SmolLM2-135M-Instruct", "--openai_spec", "true", "--accelerator", "cuda", "--devices", "1", "--precision", "bf16-true", "--port", "8000", "--max_new_tokens", "64", "--timeout", "120"]
