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

需要 Linux、Pascal 及更新的 GPU、CUDA 12。迁移原因（页面上的颜色）需要 CUDA 12.9 及以上的 CUPTI：这一版才把 `CUpti_ActivityUnifiedMemoryCounter3` 作为统一内存计数器的记录，`flags` 里才是迁移原因。更早的工具包仍然上报字节数和缺页，原因会是「未标记」。

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
| `UVMI_FLUSH_MS` | `200` | 向 CUPTI 要一批事件的间隔，最小 10 |
| `UVMI_APP_NAME` | 进程名 | 页面和指标里的应用名 |
| `UVMI_ENABLE_MEMCPY` | `1` | 同时记录显式拷贝 |
| `UVMI_ENABLE_KERNEL` | `1` | 记录 kernel 区间。用的是 `CONCURRENT_KERNEL`，不会把 kernel 串行化 |
| `UVMI_QUIET` | `0` | 关掉注入库打到 stderr 的日志 |

systemd 单元在 `deploy/uvm-insight-agent.service`，socket 放在 `/run/uvm-insight/agent.sock`。Prometheus 抓取示例在 `deploy/prometheus.yml`。

agent 参数：`--socket`、`--listen`（默认 `127.0.0.1:9400`）、`--retention`（默认 `10m`）、`--mock`、`--web-dir`。要给别的机器抓指标时，把监听改成 `--listen :9400`。

## 指标

| 指标 | 含义 |
|---|---|
| `uvm_migrated_bytes_total{pid,app,gpu,direction,cause}` | 统一内存迁移的字节。`direction` 是 `htod` / `dtoh` / `dtod` |
| `uvm_migration_duration_seconds` | 迁移耗时直方图 |
| `uvm_page_faults_total{pid,app,gpu,side}` | 缺页。`side` 是 `cpu` 或 `gpu` |
| `uvm_thrashing_events_total` | 抖动 |
| `uvm_throttling_events_total` | 节流 |
| `uvm_remote_map_events_total` | remote map |
| `uvm_explicit_copy_bytes_total` | 显式拷贝的字节 |
| `uvm_managed_allocated_bytes` | 当前 `cudaMallocManaged` 还没释放的字节 |
| `uvm_collector_dropped_events_total` | 采集端来不及送出而丢掉的事件 |

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

镜像名默认是 `dynamia-ai-registry.cn-hangzhou.cr.aliyuncs.com/artifacts/dynamia-ai/uvm-insight:dev`。基础镜像是 `nvidia/cuda:12.9.1`。容器里：

- `/usr/local/bin/uvm-insight-agent`
- `/opt/uvm-insight/lib/libuvminsight.so`（旁边有 CUPTI）
- `/opt/uvm-insight/examples/`

跑应用的容器需要同一个 `libuvminsight.so`，并且能连上 agent 的 socket。同一台机器上把 socket 目录挂进去即可。

## 限制

CUPTI 的 activity 有开销。`UVMI_FLUSH_MS` 调大，事件会更晚到页面上，对应用的打扰会小一些。kernel 记录用的是 `CONCURRENT_KERNEL`；如果改成 `CUPTI_ACTIVITY_KIND_KERNEL`，CUPTI 会把 kernel 启动串行化，应用会明显变慢。

统一内存计数器在 WSL、MPS，以及一部分 vGPU 上会不可用或者只有一部分计数器有数。注入库遇到这种情况会在 stderr 写一行，然后继续记内存分配和 kernel。

**注意：** 驱动如果把 profiling 限制成管理员（`NVreg_RestrictProfilingToAdminUsers=1`），普通用户的 CUPTI 会被拒绝。需要在内核模块参数里改成 `NVreg_RestrictProfilingToAdminUsers=0` 并重新加载驱动。这会放宽谁可以做 GPU profiling。

**注意：** agent 把 socket 和它所在的目录设成了 `0777`，这样别的用户的 CUDA 进程也能连上来。任何本机进程都可以往这个 socket 里写事件。多人共用的机器上，把 `--socket` 指到一个权限更紧的目录。

fork 之后子进程里的采集线程不会继续工作。采集是挂在调用 CUDA 的那个进程上的。

CUPTI 的时间戳不是墙上时钟。agent 用每个连接里第一条事件把 CUPTI 时间和本机时间对齐，kernel 色块和迁移曲线因此能画在同一条时间轴上。对齐只在进程刚连上时做一次。
