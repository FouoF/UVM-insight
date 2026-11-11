# UVM Insight

观察 NVIDIA 统一内存（Unified Memory）在主机内存和显存之间怎么搬。页面上能看到迁移方向、带宽、缺页、抖动，以及是哪一段地址在反复搬家。采集库不改应用代码，agent 把同一份数据同时做成 Prometheus 指标和实时页面，后面也可以接到别的组件上。

## 页面上有什么

- **数据流动**：左边是主机内存，右边是每张 GPU。粒子从哪边流向哪边，线有多粗，就是这一秒的带宽。颜色是迁移原因：一致性、推测预取、用户预取（`cudaMemPrefetchAsync`）、驱逐、访问计数。虚线是显式 `cudaMemcpy`，用来和统一内存对比。
- **带宽**：迁入显存、迁回主机、卡间迁移，背景里的浅色块是 kernel 的执行区间。
- **缺页 / 抖动 / 节流**。
- **地址热度**：2MB 一档，看哪一段地址在反复迁移。
- **事件表**：最近的迁移明细。

指标在 `http://127.0.0.1:9400/metrics`。

## 架构

```text
CUDA 应用
  └─ libuvminsight.so     CUPTI Activity，CUDA_INJECTION64_PATH 注入
        │  Unix socket，4 字节大端长度 + protobuf（proto/uvm_event.proto）
        ▼
uvm-insight-agent         聚合、/metrics、/api、WebSocket、页面
```

协议定义在 `proto/uvm_event.proto`。注入库里手写编码，不链接 libprotobuf，避免和被观察的进程抢符号。agent 按同一份字段号解码。

## 没有 GPU 时先看页面

```fish
make demo
```

浏览器打开 http://127.0.0.1:9400 。`--mock` 会放两个演示进程（`train.py`、`infer.py`），数据是生成的，用来看页面和指标，不是真实的 CUPTI 事件。

## 在 GPU 机器上采集

需要 Linux、Pascal 及更新的 GPU、CUDA 12 的应用。

采集库链接的是 `libcupti.so.12`，用的是进程里实际加载的那一份。PyTorch 这类自带 CUPTI 的镜像会在 `import torch` 时按绝对路径先加载自己的 `libcupti.so.12`，采集库会直接复用它，进程里始终只有一份 CUPTI；应用没带 CUPTI 时，才用采集库旁边那份 12.9。CUDA 12.0 到 12.9 的 CUPTI 里，采集库读到的记录字段（统一内存计数器 `Counter2`/`Counter3`、`Memory3`/`Memory4`、`Memcpy5`/`Memcpy6`、`Kernel9`）偏移都一样，迁移原因在两版计数器记录里都是 `flags`。所以不需要 `LD_PRELOAD`，也不需要替换镜像里的 CUPTI。进程实际用的 CUPTI API 版本会出现在 `uvm_process_info{cupti_api}` 里（12.1 是 18，12.9 是 28）。

```fish
# 终端 1
make agent
./bin/uvm-insight-agent --listen 127.0.0.1:9400 --socket /tmp/uvm-insight/agent.sock

# 终端 2，在有 CUDA 的机器上
cmake -S collector -B collector/build
cmake --build collector/build -j
cmake -S examples -B examples/build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build examples/build -j
./scripts/trace.sh ./examples/build/prefetch 256 8
```

`scripts/trace.sh` 会设置：

- `CUDA_INJECTION64_PATH` 指向 `libuvminsight.so`
- `LD_LIBRARY_PATH` 带上 CUPTI
- `UVMI_SOCKET`，要和 agent 的 `--socket` 一致

示例程序：

| 程序 | 在看什么 |
|---|---|
| `prefetch [MB] [轮数]` | `cudaMemPrefetchAsync`，原因是用户预取 |
| `alternate [MB] [轮数]` | CPU / GPU 轮流写同一块内存，原因多是一致性 |
| `oversubscribe [MB]` | 按页遍历。参数 `0` 时分配「显存总量 + 512MB」，用来把页驱逐回主机 |
| `thrash [MB] [轮数]` | 小范围来回踩，争取让驱动报出抖动和节流 |

采集库还认这些环境变量：

| 变量 | 默认 | 作用 |
|---|---|---|
| `UVMI_SOCKET` | `/tmp/uvm-insight/agent.sock` | agent 的 Unix socket |
| `UVMI_FLUSH_MS` | `200` | 刷新间隔（ms），最小 10。开了事件时也是向 CUPTI 要一批的间隔 |
| `UVMI_APP_NAME` | 进程名 | 页面和指标里的应用名 |
| `UVMI_EVENTS` | `1` | 换入换出事件管线（CUPTI UM counter / fault 等）。`0` 时不开迁移类 CUPTI |
| `UVMI_ENABLE_MEMCPY` | `1` | 同时记录显式拷贝（需 `UVMI_EVENTS=1`） |
| `UVMI_ENABLE_KERNEL` | `1` | 记录 kernel 区间（需 `UVMI_EVENTS=1`）。用的是 `CONCURRENT_KERNEL` |
| `UVMI_QUIET` | `0` | 关掉注入库打到 stderr 的日志 |
| `UVMI_RESIDENCY` | `auto` | 驻留管线：`off` / `auto` / `driver`（`cuMemGetLocationInfo`）/ `delta`（CUPTI HtoD/DtoH 推测）。与事件开关独立；`delta` 需要 `UVMI_EVENTS=1`；`auto` 优先 driver，否则在事件开启时用 delta |
| `UVMI_LOC_SUMMARY_MIB` | `64` | driver 模式下 location 汇总粒度（MiB） |
| `UVMI_LOC_SAMPLE_MIB` | `2` | driver 模式下区间内采样步长（MiB）；`0` 表示驱动默认 |

两套功能可单独开关，例如只要驻留真值：`UVMI_EVENTS=0 UVMI_RESIDENCY=driver`；只要换入换出：`UVMI_RESIDENCY=off`。

systemd 单元在 `deploy/uvm-insight-agent.service`，socket 放在 `/run/uvm-insight/agent.sock`。Prometheus 抓取示例在 `deploy/prometheus.yml`。

agent 参数：`--socket`、`--listen`（默认 `127.0.0.1:9400`）、`--retention`（默认 `10m`）、`--ui`（默认 `true`，`false` 时只有 `/metrics` 和 `/healthz`）、`--node-name`（默认取 `NODE_NAME`）、`--mock`、`--web-dir`。要给别的机器抓指标时，把监听改成 `--listen :9400`。`uvm-insight-agent install --src DIR --dst DIR` 把采集库装到主机目录，每个文件先写临时名再改名，正在运行、已经映射了旧库的进程不受影响。

## Kubernetes 部署

Helm chart 在 `deploy/helm/uvm-insight`，按 exporter 的方式部署：

- **DaemonSet**，只调度到 `nvidia.com/gpu.present=true` 的节点。initContainer 把采集库装到主机的 `/var/lib/uvm-insight/lib`，exporter 在 `/var/lib/uvm-insight/agent.sock` 收事件，在 9400 端口出 `/metrics`。
- **Mutating Webhook**（默认开启）：给匹配的新建 Pod 自动加上 `CUDA_INJECTION64_PATH`、`UVMI_SOCKET`、hostPath 挂载和 `SYS_ADMIN`。DaemonSet 本身不做注入；webhook 改的是即将创建的 Pod。
- **hostPID**：agent 通过 `SO_PEERCRED` 拿到主机 pid，`pid` 标签和 `nvidia-smi` 一致，同一节点上的多个 Pod 也不会撞号。
- **Pod 归属**：采集库从 `/proc/self/mountinfo` 里 kubelet 挂载的路径读出 Pod UID 和容器名，exporter 用 ServiceAccount 列出本节点的 Pod，换成 `namespace`、`pod`。RBAC 只有 `pods` 的 `get`、`list`。
- **Service** 是 headless 的，带 `prometheus.io/scrape` 注解。`serviceMonitor.enabled=true` 时生成 ServiceMonitor，`honorLabels: true`，让指标里工作负载的 `namespace`、`pod`、`container` 不被 exporter 自己的 target 标签改名成 `exported_*`。

```fish
helm upgrade --install uvm-insight deploy/helm/uvm-insight -n uvm-insight --create-namespace \
  --set image.tag=v0.2.0-dev \
  --set serviceMonitor.enabled=true --set serviceMonitor.labels.release=prometheus
```

`serviceMonitor.labels` 要和 Prometheus 的 `serviceMonitorSelector` 对上，kube-prometheus-stack 一般是 `release: <helm release 名>`。

Webhook 默认会注入这些 Pod：

- 容器申请了 `nvidia.com/gpu`（可用 `webhook.gpuResources` 改）
- 或者带有 `nvidia.com/use-gpuuuid`（HAMi / Kantaloupe）
- 或者显式标注 `uvm-insight.dynamia.ai/inject=true`

不注入：`uvm-insight.dynamia.ai/inject=false`，以及 `kube-system` 和 chart 自己的命名空间。`failurePolicy` 默认是 `Ignore`，webhook 挂了也不会卡住建 Pod。已经在跑的 Pod 不会被改，需要重建一次。

关掉自动注入：`--set webhook.enabled=false`，再按 NOTES 里的片段手动挂卷、设环境变量。

## 指标

每条序列都带 `namespace`、`pod`、`container`（不在 Kubernetes 里时为空）、`pid`、`app`。和 GPU 相关的序列还有 `gpu`（进程内的设备序号，HAMi 下每个容器都是 `gpu0`）和 `gpu_uuid`（和 dcgm-exporter、`nvidia-smi` 对得上）。

| 指标 | 含义 |
|---|---|
| `uvm_process_info{cupti_api,driver_version}` | 正在上报的 CUDA 进程，值恒为 1。`cupti_api` 是进程里实际加载的 CUPTI API 版本 |
| `uvm_migrated_bytes_total{direction,cause}` | 统一内存迁移的字节。`direction` 是 `htod` / `dtoh` / `dtod`，`cause` 是迁移原因 |
| `uvm_migration_duration_seconds` | 迁移耗时直方图 |
| `uvm_page_faults_total{side}` | 缺页。`side` 是 `cpu` 或 `gpu` |
| `uvm_thrashing_events_total` | 抖动 |
| `uvm_throttling_events_total` | 节流 |
| `uvm_remote_map_events_total` | remote map |
| `uvm_explicit_copy_bytes_total` | 显式拷贝的字节 |
| `uvm_managed_allocated_bytes` | 进程 `/dev/nvidia-uvm` 映射的虚拟大小，也就是托管内存的总量 |
| `uvm_gpu_resident_bytes` | 托管内存里在显存上的部分（`managed = gpu + host + unknown`）。仅当驻留管线开启时上报 |
| `uvm_host_resident_bytes` | 托管内存里在主机上的部分（含 `HOST_NUMA`）。仅当驻留管线开启时上报 |
| `uvm_residency_source` | 驻留来源。`source` 为 `none` / `driver` / `delta` |
| `uvm_collector_dropped_events_total` | 采集端来不及送出而丢掉的事件 |

几条常用查询：

```promql
# 每个 Pod 的迁入/迁回带宽
sum by (namespace, pod, direction) (rate(uvm_migrated_bytes_total[1m]))

# 驱逐引起的迁回占比，超分时这个值高说明显存不够用
sum by (pod) (rate(uvm_migrated_bytes_total{direction="dtoh",cause="eviction"}[5m]))
  / sum by (pod) (rate(uvm_migrated_bytes_total{direction="dtoh"}[5m]))

# 每张卡上各 Pod 的估算驻留
sum by (gpu_uuid, pod) (uvm_gpu_resident_bytes)
```

HTTP：

- `GET /api/processes`
- `GET /api/snapshot?pid=`
- `GET /api/heatmap?pid=`
- `WS /api/stream?pid=`，大约每 500ms 推一次

`pid` 省略或写成 `all` 时是当前还活着的进程合在一起。

## 镜像

本机是 arm64 时，镜像按 linux/amd64 构建：

```fish
make image
```

镜像名默认是 `dynamia-ai-registry.cn-hangzhou.cr.aliyuncs.com/artifacts/dynamia-ai/uvm-insight:dev`，`make image TAG=v0.2.0-dev` 换标签，`make push` 推送。采集库在 Ubuntu 20.04（glibc 2.31）上编译，CUPTI 12.9 的头文件和库取自 NVIDIA 发布在 PyPI 上的 wheel，运行镜像是 distroless。容器里：

- `/usr/local/bin/uvm-insight-agent`
- `/opt/uvm-insight/lib/libuvminsight.so`，旁边是 CUPTI 12.9 的 `libcupti.so.12` 和 `libnvperf_*`

示例程序要用 nvcc 编译，不在镜像里，用 `make examples` 在 GPU 机器上构建。

## 限制

CUPTI 的 activity 有开销。`UVMI_FLUSH_MS` 调大，事件会更晚到页面上，对应用的打扰会小一些。kernel 记录用的是 `CONCURRENT_KERNEL`；如果改成 `CUPTI_ACTIVITY_KIND_KERNEL`，CUPTI 会把 kernel 启动串行化，应用会明显变慢。

统一内存计数器在 WSL、MPS，以及一部分 vGPU 上会不可用或者只有一部分计数器有数。注入库遇到这种情况会在 stderr 写一行，然后继续记内存分配和 kernel。

**注意：** 驱动如果把 profiling 限制成管理员（`NVreg_RestrictProfilingToAdminUsers=1`，`/proc/driver/nvidia/params` 里是 `RmProfilingAdminOnly: 1`），普通用户的 CUPTI 会被拒绝。两种办法：给接入的容器加 `SYS_ADMIN` capability，这个权限很大；或者在内核模块参数里改成 `NVreg_RestrictProfilingToAdminUsers=0` 并重新加载驱动，这会放宽节点上谁可以做 GPU profiling。

采集库只支持 CUDA 12（`libcupti.so.12`）。CUDA 13 的应用会加载 `libcupti.so.13`，和 `.so.12` 是两个库，需要另外用 CUDA 13 的 CUPTI 编一份采集库，按应用的 CUDA 大版本选择 `CUDA_INJECTION64_PATH`。

**注意：** agent 把 socket 和它所在的目录设成了 `0777`，这样别的用户的 CUDA 进程也能连上来。任何本机进程都可以往这个 socket 里写事件。多人共用的机器上，把 `--socket` 指到一个权限更紧的目录。

fork 之后子进程里的采集线程不会继续工作。采集是挂在调用 CUDA 的那个进程上的。

CUPTI 的时间戳不是墙上时钟。agent 用每个连接里第一条事件把 CUPTI 时间和本机时间对齐，kernel 色块和迁移曲线因此能画在同一条时间轴上。对齐只在进程刚连上时做一次。
