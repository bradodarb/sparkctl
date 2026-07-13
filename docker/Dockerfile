# Thin derived image: NGC vLLM 26.06 (vLLM 0.22.1, CUDA 13, sm_121) + Ray for multi-node serving.
# NVIDIA dropped Ray from the 26.06 image, but vLLM's `--distributed-executor-backend ray` still
# needs it for cross-node TP/PP. Ray 2.54.0 is the version NGC shipped in 26.02 (known-good on this
# aarch64 + py3.12 + CUDA 13 stack), so we pin it for reproducibility.
# Built on each node by `sparkctl build`; the cluster default image points here (cluster.yaml).
ARG BASE=nvcr.io/nvidia/vllm:26.06-py3
FROM ${BASE}
RUN pip install --no-cache-dir "ray[default]==2.54.0" \
    && ray --version
