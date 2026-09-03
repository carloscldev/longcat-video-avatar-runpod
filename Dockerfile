FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# torch pinned to what LongCat-Video's requirements.txt expects, from the
# cu124 wheel index so it matches the cuda12.4 base image.
RUN python -m pip install --upgrade pip && \
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt requirements_avatar.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -r requirements_avatar.txt && \
    pip install runpod

# Built after torch is already in place -- flash-attn's setup.py needs a
# working torch import to compile against, and it's the slow step so it
# gets its own layer, cached separately from the rest of the deps above.
# --no-build-isolation runs setup.py in this environment rather than an
# isolated one, so it needs setuptools/pkg_resources and ninja here itself.
# Unpinned setuptools has dropped pkg_resources entirely on recent releases
# (flash-attn's setup.py, via wheel/bdist_wheel.py, still imports it), so
# this pins to the last line that ships it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "setuptools<81" wheel ninja && \
    MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation

# The upstream repo, for its pipeline/module code -- not the weights,
# those come from RunPod's HF model cache at run time.
RUN git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video.git /app/longcat_repo && \
    cp -r /app/longcat_repo/longcat_video /app/longcat_video && \
    rm -rf /app/longcat_repo

COPY handler.py .

ENTRYPOINT []
CMD ["sh", "-c", "python3 -u handler.py; code=$?; echo \"HANDLER_EXITED code=$code\"; sleep 120; exit $code"]
