# syntax=docker/dockerfile:1

FROM node:22-bookworm AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM golang:1.23-bookworm AS agent
WORKDIR /src/agent
COPY agent/go.mod agent/go.sum ./
RUN go mod download
COPY agent/ ./
COPY --from=web /src/web/dist/ ./pkg/webui/dist/
ENV CGO_ENABLED=0
RUN go build -o /out/uvm-insight-agent ./cmd/uvm-insight-agent

FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS collector
RUN apt-get update \
  && apt-get install -y --no-install-recommends cmake g++ make \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY collector/ collector/
COPY examples/ examples/
RUN cmake -S collector -B collector/build && cmake --build collector/build -j$(nproc) \
  && cmake -S examples -B examples/build && cmake --build examples/build -j$(nproc)
RUN mkdir -p /out/lib /out/examples \
  && cp -a /usr/local/cuda/extras/CUPTI/lib64/. /out/lib/ \
  && cp collector/build/libuvminsight.so /out/lib/ \
  && cp examples/build/prefetch examples/build/alternate \
        examples/build/oversubscribe examples/build/thrash /out/examples/

FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
COPY --from=agent /out/uvm-insight-agent /usr/local/bin/uvm-insight-agent
COPY --from=collector /out/lib/ /opt/uvm-insight/lib/
COPY --from=collector /out/examples/ /opt/uvm-insight/examples/
ENV LD_LIBRARY_PATH=/opt/uvm-insight/lib
EXPOSE 9400
ENTRYPOINT ["/usr/local/bin/uvm-insight-agent"]
CMD ["--listen", "0.0.0.0:9400", "--socket", "/tmp/uvm-insight/agent.sock"]
