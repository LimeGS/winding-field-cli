ARG BASE_IMAGE=ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
FROM ghcr.io/astral-sh/uv:0.9.17@sha256:5cb6b54d2bc3fe2eb9a8483db958a0b9eebf9edff68adedb369df8e7b98711a2 AS uv
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS certificates
FROM ${BASE_IMAGE} AS build
ARG TARGETARCH
ARG TORCH_VARIANT=cpu
LABEL org.opencontainers.image.source="https://github.com/LimeGS/winding-field-cli"
COPY --from=uv /uv /usr/local/bin/uv
# Ubuntu's minimal base needs trust roots before its first HTTPS snapshot request.
COPY --from=certificates /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_NO_CACHE=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_ARGS=-DVC_READER_ENABLE_CUDA=OFF \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    HOME=/home/worker XDG_CACHE_HOME=/cache TMPDIR=/scratch \
    PATH=/opt/winding-field/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY tools/ubuntu-packages.txt /opt/runtime-provenance/tools/ubuntu-packages.txt
RUN mkdir -p /scratch && test "$TARGETARCH" = amd64 && \
    case "$TORCH_VARIANT" in cpu|cu128) ;; *) exit 1 ;; esac && \
    rm -f /etc/apt/sources.list.d/cuda*.list && \
    sed -i -E 's|http://[^ ]+/ubuntu/?|https://snapshot.ubuntu.com/ubuntu/20261101T000000Z/|g' /etc/apt/sources.list.d/ubuntu.sources && \
    printf 'Acquire::Retries "5";\nAcquire::https::Pipeline-Depth "0";\n' > /etc/apt/apt.conf.d/80snapshot && \
    apt-get update -o APT::Update::Error-Mode=any && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates $(cat /opt/runtime-provenance/tools/ubuntu-packages.txt) \
      libglx0 libopengl0 libglib2.0-0 libfreetype6 libfontconfig1 squashfs-tools && \
    mkdir -p /opt/runtime-provenance /scratch && \
    dpkg-query -W > /opt/runtime-provenance/os-packages.txt && \
    rm -rf /var/lib/apt/lists/*

COPY docker/*-cpu.txt docker/*-cu128.txt docker/build-constraints.txt /opt/runtime-provenance/docker/
COPY tools/install-runtime.sh /opt/runtime-provenance/tools/install-runtime.sh
RUN bash /opt/runtime-provenance/tools/install-runtime.sh "$TORCH_VARIANT" /opt/winding-field /opt/lasagna dependencies
COPY provenance /opt/runtime-provenance/sources
RUN python -c 'import json,urllib.request; p=json.load(open("/opt/runtime-provenance/sources/flatboi.json")); urllib.request.urlretrieve(p["url"],"/scratch/vc3d.AppImage")' && \
    python -c 'import hashlib,json; p=json.load(open("/opt/runtime-provenance/sources/flatboi.json")); assert hashlib.file_digest(open("/scratch/vc3d.AppImage","rb"),"sha256").hexdigest()==p["sha256"]' && \
    unsquashfs -no-progress -d /opt/vc3d -o 944632 /scratch/vc3d.AppImage && \
    rm /scratch/vc3d.AppImage

WORKDIR /build
COPY pyproject.toml README.md LICENSE THIRD_PARTY.md ./
COPY licenses ./licenses
COPY provenance ./provenance
COPY src ./src
RUN bash /opt/runtime-provenance/tools/install-runtime.sh "$TORCH_VARIANT" /opt/winding-field /opt/lasagna application /scratch/wheels && \
    sha256sum /scratch/wheels/*.whl > /opt/runtime-provenance/cli-wheel.sha256 && \
    uv pip freeze --python /opt/winding-field/bin/python > /opt/runtime-provenance/cli-installed.txt && \
    uv pip freeze --python /opt/lasagna/bin/python > /opt/runtime-provenance/lasagna-installed.txt && \
    rm -rf /build /scratch/wheels

WORKDIR /outputs
COPY docker/*.py docker/*.in /opt/runtime-provenance/docker/
RUN mkdir -p /opt/accelerators && \
    python -m winding_field.tifxyz_build --backend cpu --output /opt/accelerators/cpu && \
    if [ "$TORCH_VARIANT" = cu128 ]; then \
      python -m winding_field.tifxyz_build --backend cuda --host-compiler /usr/bin/g++ --output /opt/accelerators/cuda; \
    fi && \
    useradd --uid 10001 --create-home worker && \
    chmod 755 /home/worker && \
    mkdir -p /home/worker/.cache/winding-field/models /data /cache /scratch /outputs && \
    ln -s /opt/lasagna /home/worker/.cache/winding-field/lasagna-venv && \
    ln -s /cache /home/worker/.cache/winding-field/ct && \
    ln -s /cache /home/worker/.cache/winding-field/patch-ct && \
    chmod -R a+rX /opt/accelerators && \
    chown -R worker:worker /home/worker /cache /scratch /outputs

FROM build AS assembly
ARG TORCH_VARIANT=cpu
# Only merged assembly files are copied, not the builder's development layers.
RUN rm -rf /opt/nvidia && \
    python /opt/runtime-provenance/docker/deduplicate-libraries.py /opt \
      > /opt/runtime-provenance/library-sharing.json && \
    mkdir -p /cuda-runtime && \
    if [ "$TORCH_VARIANT" = cu128 ]; then \
      mkdir -p /cuda-runtime/cuda-12.8/bin /cuda-runtime/cuda-12.8/targets/x86_64-linux/lib && \
      for name in nvcc nvcc.profile nvlink ptxas cudafe++ fatbinary bin2c; do \
        cp -a "/usr/local/cuda-12.8/bin/$name" /cuda-runtime/cuda-12.8/bin/ || exit 1; \
      done && \
      cp -a /usr/local/cuda-12.8/nvvm /cuda-runtime/cuda-12.8/ && \
      cp -a /usr/local/cuda-12.8/targets/x86_64-linux/include /cuda-runtime/cuda-12.8/targets/x86_64-linux/ && \
      cp -a /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart* \
        /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudadevrt.a \
        /usr/local/cuda-12.8/targets/x86_64-linux/lib/libculibos.a \
        /usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs \
        /cuda-runtime/cuda-12.8/targets/x86_64-linux/lib/ && \
      ln -s targets/x86_64-linux/include /cuda-runtime/cuda-12.8/include && \
      ln -s targets/x86_64-linux/lib /cuda-runtime/cuda-12.8/lib64 && \
      ln -s cuda-12.8 /cuda-runtime/cuda; \
    fi

FROM ${RUNTIME_IMAGE}
ARG TORCH_VARIANT=cpu
ARG NVIDIA_REQUIRE_CUDA=""
LABEL org.opencontainers.image.source="https://github.com/LimeGS/winding-field-cli"
COPY --from=certificates /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
# G++ and Ninja are required by explicit accelerator builds and Lasagna JIT.
RUN sed -i -E 's|http://[^ ]+/ubuntu/?|https://snapshot.ubuntu.com/ubuntu/20261101T000000Z/|g' /etc/apt/sources.list.d/ubuntu.sources && \
    printf 'Acquire::Retries "5";\nAcquire::https::Pipeline-Depth "0";\n' > /etc/apt/apt.conf.d/80snapshot && \
    apt-get update -o APT::Update::Error-Mode=any && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates g++ ninja-build libcurl4t64 libsqlite3-0 libblosc1 libzstd1 \
      liblz4-1 zlib1g libtiff6 libpng16-16t64 libjpeg-turbo8 libgomp1 libnuma1 \
      libglx0 libopengl0 libglib2.0-0t64 libfreetype6 libfontconfig1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --uid 10001 --create-home worker && \
    mkdir -p /data /cache /scratch /outputs /home/worker/.cache/winding-field/models && \
    ln -s /opt/lasagna /home/worker/.cache/winding-field/lasagna-venv && \
    ln -s /cache /home/worker/.cache/winding-field/ct && \
    ln -s /cache /home/worker/.cache/winding-field/patch-ct && \
    chown -R worker:worker /home/worker /cache /scratch /outputs
COPY --from=uv /uv /usr/local/bin/uv
# One copy preserves hardlinks across the two isolated Python environments.
COPY --from=assembly /opt /opt
COPY --from=assembly /cuda-runtime/ /usr/local/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_NO_CACHE=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    HOME=/home/worker XDG_CACHE_HOME=/cache TMPDIR=/scratch \
    PATH=/opt/winding-field/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA=$NVIDIA_REQUIRE_CUDA \
    TORCHINDUCTOR_CACHE_DIR=/scratch/torchinductor
RUN dpkg-query -W > /opt/runtime-provenance/os-packages.txt && \
    python /opt/runtime-provenance/docker/verify-runtime.py --variant "$TORCH_VARIANT" \
      > /opt/runtime-provenance/runtime.json
WORKDIR /outputs
USER 10001:10001
ENTRYPOINT ["winding-field"]
CMD ["--help"]
