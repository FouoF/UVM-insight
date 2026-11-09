ARG BUILD_IMAGE=ubuntu:20.04
ARG GO_IMAGE=golang:1.23-bookworm
ARG RUNTIME_IMAGE=gcr.io/distroless/static-debian12

# The collector is loaded into application processes, so it is built against
# an old glibc (Ubuntu 20.04, 2.31) to load in most CUDA images. CUPTI headers
# and libraries come from the NVIDIA PyPI wheels, which avoids pulling a full
# CUDA devel image.
FROM ${BUILD_IMAGE} AS collector
ARG CUPTI_VERSION=12.9.79
ARG CUDART_VERSION=12.9.79
ARG NVCC_VERSION=12.9.86
# Optional: an Ubuntu mirror host such as mirrors.aliyun.com, and a proxy
# used only for the PyPI downloads.
ARG APT_MIRROR=
ARG FETCH_PROXY=
ENV DEBIAN_FRONTEND=noninteractive
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i -E "s#http://(archive|security).ubuntu.com#http://$APT_MIRROR#g" /etc/apt/sources.list; \
    fi \
  && apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl g++ python3 unzip \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /wheels
RUN set -e; \
  fetch() { \
    url=$(curl -fsSL ${FETCH_PROXY:+-x "$FETCH_PROXY"} "https://pypi.org/pypi/$1/$2/json" | python3 -c \
      'import json,sys; print(next(u["url"] for u in json.load(sys.stdin)["urls"] if "x86_64" in u["filename"]))'); \
    curl -fsSL --retry 3 ${FETCH_PROXY:+-x "$FETCH_PROXY"} -o "$1.whl" "$url"; \
    unzip -q -o "$1.whl" 'nvidia/*'; \
  }; \
  fetch nvidia-cuda-cupti-cu12 "$CUPTI_VERSION"; \
  fetch nvidia-cuda-runtime-cu12 "$CUDART_VERSION"; \
  fetch nvidia-cuda-nvcc-cu12 "$NVCC_VERSION"; \
  ln -sf libcupti.so.12 nvidia/cuda_cupti/lib/libcupti.so
WORKDIR /src
COPY collector/src/ collector/src/
RUN g++ -std=c++17 -O2 -Wall -shared -fPIC -fvisibility=hidden \
    -Icollector/src \
    -I/wheels/nvidia/cuda_cupti/include \
    -I/wheels/nvidia/cuda_runtime/include \
    -I/wheels/nvidia/cuda_nvcc/include \
    collector/src/encode.cpp collector/src/injection.cpp \
    -L/wheels/nvidia/cuda_cupti/lib -lcupti -ldl -pthread \
    -Wl,-rpath,'$ORIGIN' -o libuvminsight.so \
  && mkdir -p /out/lib \
  && cp libuvminsight.so /out/lib/ \
  && cp -a /wheels/nvidia/cuda_cupti/lib/*.so* /out/lib/ \
  && rm /out/lib/libcupti.so

FROM ${GO_IMAGE} AS agent
WORKDIR /src/agent
COPY agent/go.mod agent/go.sum ./
RUN go mod download
COPY agent/ ./
ENV CGO_ENABLED=0
RUN go build -trimpath -ldflags "-s -w" -o /out/uvm-insight-agent ./cmd/uvm-insight-agent

FROM ${RUNTIME_IMAGE}
COPY --from=agent /out/uvm-insight-agent /usr/local/bin/uvm-insight-agent
COPY --from=collector /out/lib/ /opt/uvm-insight/lib/
EXPOSE 9400
ENTRYPOINT ["/usr/local/bin/uvm-insight-agent"]
CMD ["--listen", "0.0.0.0:9400", "--socket", "/run/uvm-insight/agent.sock", "--ui=false"]
