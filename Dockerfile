FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/Miniforge3.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/Miniforge3.sh -b -p /opt/conda && \
    rm /tmp/Miniforge3.sh

ENV PATH=/opt/conda/bin:$PATH

COPY environment.yaml /tmp/environment.yaml

RUN conda --version && \
    conda config --set always_yes yes --set changeps1 no && \
    conda env create -f /tmp/environment.yaml -v && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=terravis
ENV PATH=/opt/conda/envs/terravis/bin:/opt/conda/bin:$PATH

# flash-attn --no-build-isolation compiles against the torch already installed, so cu130 torch goes first.
# --force-reinstall --no-deps: replaces the PyPI torch the env pulled in, even at the same version.
RUN pip install --force-reinstall --no-deps torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu130

RUN pip install flash-attn==2.8.3 --no-build-isolation

# vllm 0.26.0 pins torch==2.11.0; an unpinned vllm replaces the cu130 torch and breaks the flash-attn build.
RUN pip install vllm==0.26.0 --extra-index-url https://download.pytorch.org/whl/cu130

# laion_aesthetic_score only (AGPL-3.0, kept out of the base install).
RUN pip install aesthetic-predictor-v2-5==2024.12.18.1
