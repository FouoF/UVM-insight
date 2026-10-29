#!/bin/sh
# Run a CUDA process with the UVM Insight collector attached.
# Usage: scripts/trace.sh ./examples/build/prefetch 256 4

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
export CUDA_INJECTION64_PATH="${CUDA_INJECTION64_PATH:-$ROOT/collector/build/libuvminsight.so}"
export UVMI_SOCKET="${UVMI_SOCKET:-/tmp/uvm-insight/agent.sock}"
if [ -d /usr/local/cuda/extras/CUPTI/lib64 ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$@"
