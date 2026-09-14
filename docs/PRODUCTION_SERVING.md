# 推荐 API 生产运行手册

## 容器化启动（推荐给面试演示）

先按 `docs/MOVIELENS_GUIDE.md` 准备宿主机上的 `data/` 与 `checkpoints/`，然后：

```bash
docker compose up --build -d
curl http://localhost:8000/ready
curl http://localhost:8000/openapi.json
```

镜像不内置数据集和权重；Compose 以只读卷挂载二者，并把 SQLite 反馈库和
worker 指标放入持久卷。容器使用 CPU Torch、非 root 用户和固定摘要的 Python 基础镜像。
默认只绑定宿主机 `127.0.0.1:8000`；打开该地址即可使用主网页。
新客户端使用 `POST /v1/recommendations`、
`/v1/events/impression` 和 `/v1/events/feedback`；旧路径继续兼容。
`/openapi.json` 可在模型未加载时读取，因此部署流水线可以独立校验接口合约。

提交前运行单元测试与 `docker compose config --quiet`。具备真实权重的发布环境
运行 `scripts/production_smoke.py --base-url http://127.0.0.1:8000 --tag <unique-tag>`，并用
`scripts/load_test_api.py` 记录目标机器上的 p95/p99、成功 RPS、429 分类与 RSS；
历史本机数字只能作为基线，不能冒充部署环境 SLO。

外部模式直接检查这个容器，不另起本地 API，也不停止被测服务。
本轮构建、完整交互、重试、重建持久化和压测证据见 [交付验证](APPLICATION_DELIVERY.md)。

```bash
curl -X POST http://localhost:8000/v1/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"user_id":1,"k":10}'
```

推荐请求会生成并持久化 `recommendation_id` 后才返回，因此正式接口采用有副作用
语义清晰的 POST。旧 `GET /recommend` 仅为已有 Demo 兼容保留。

空结果返回 `recommendation_id=null`，不会创建空批次。v1 接口拒绝布尔值、小数和
数字字符串冒充整数。曝光接口接收实际展示的 `movie_ids`；反馈必须有匹配曝光，
并提供客户端生成的 `event_id`。相同事件重试返回 200 和 `duplicate=true`，冲突内容返回 400。

完整的固定用户演示：

```powershell
.venv/Scripts/python.exe scripts/demo_recommendation_flow.py --base-url http://127.0.0.1:8000 --tag my-first-flow
```

脚本写入 `demo-*` 演示事件；压测写入 `load-*` 事件。报告中明确标记为脚本流量。
在独立测试实例/数据卷中执行，不要把这些事件与真人偏好混合做效果评估。

Compose 支持 `RECSYS_PORT`、`RECSYS_WORKERS`、`RECSYS_MAX_IN_FLIGHT`、
`RECSYS_RATE_LIMIT` 和 `RECSYS_RATE_BURST`。默认分别是 8000、2、8、80、8。
`RECSYS_API_KEY` 启用可选鉴权；演示、压测和冒烟脚本读取同名环境变量，不在报告中保存密钥。

## 推荐启动配置

以下参数来自本机（Windows、12 逻辑核、ML-1M）真实压测，适合作为起点，
不是跨机器通用常数：

```powershell
.venv/Scripts/python.exe serve.py `
  --host 0.0.0.0 `
  --port 8000 `
  --preload `
  --ranking-policy retrieval `
  --workers 2 `
  --max-in-flight 8 `
  --rate-limit 80 `
  --rate-burst 8 `
  --metrics-interval 0.5 `
  --graceful-timeout 30
```

- `--max-in-flight`：每 worker 同时执行的业务请求数，超出返回
  `429 / code=overloaded`。
- `--rate-limit`：每 worker 平均令牌数/秒，`0` 表示关闭。两 worker、
  每个 80，实例总配置速率约 160 RPS。
- `--rate-burst`：每 worker 可瞬时消耗的令牌数；超出返回
  `429 / code=rate_limited`。
- `--workers`：每个 worker 都会加载一份模型；本机实测 1/2/4 worker
  约占 0.89/1.78/3.55 GiB RSS。
- `--ranking-policy retrieval`：默认且已晋升的双塔 Top-K。只有做失败复现或
  候选排序器离线验证时才显式使用 `deepfm`；可同时传 `--ranker-checkpoint`。

### 权重供应链校验

服务通过 `src/checkpoint_io.py` 加载权重，默认强制 `weights_only=True`；
已登记权重的 size+SHA-256 启动校验是显式开关，适合部署参考产物时启用：

```powershell
serve.py --preload --verify-checkpoint-hashes
# 或环境变量 RECSYS_VERIFY_CHECKPOINTS=1
```

开启后，若权重在 `artifacts/release_manifest.json` 中登记且不匹配，则**启动失败**。
从零训练复现时不要开启该开关，因为自训权重尚未登记进 manifest。

对外绑定 `0.0.0.0` 时，必须由防火墙或反向代理限制管理指标端点；服务支持可选
`X-API-Key`，但不实现 TLS，生产环境仍应放在入口网关之后。

### 当前默认路径延迟参考

2026-08-16 复测（默认双塔 Top-K、`--concurrency 1`、200 请求）：

- WSL/Linux CPU、SQLite 位于本地 `/tmp`：HTTP mean **5.42 ms**，p95 **9.67 ms**；
- 推荐核心约 **0.19 ms**；
- 旧 Windows DeepFM 路径 4.6 ms 只作历史参考，不代表当前默认服务。

## 探针与指标

| 路径 | 语义 | 是否触发模型加载 |
|---|---|---:|
| `GET /live` | 进程与事件循环存活 | 否 |
| `GET /ready` | 当前 worker 模型已加载、产物完整 | 否 |
| `GET /health` | 向后兼容的综合检查 | 冷启动时会 |
| `GET /metrics` | 当前 worker 的 JSON 指标 | 否 |
| `GET /metrics/aggregate` | 同实例所有新鲜 worker 的聚合 JSON | 否 |
| `GET /metrics/prometheus` | 聚合后的 Prometheus 文本 | 否 |

聚合指标包括：

- 10/60 秒滚动 QPS、状态码、错误分类和路由延迟；
- 模型加载成功/失败与已就绪 worker 数；
- 准入容量、in-flight、shed 总数；
- 令牌桶配置、放行与拒绝总数；
- worker CPU、RSS、事件循环 lag；
- 每个 worker PID、更新时间与模型状态。

worker 每隔 `--metrics-interval` 秒将有界快照原子写入
`runtime/metrics/<host-port>/`。聚合只读取 5 秒内的新鲜文件，并自动清理过期
worker/临时文件。优雅停机时 worker 会主动注销。

这套文件注册表解决的是**同一台主机、同一个实例目录**的多进程聚合。跨主机部署仍应
让 Prometheus 分别抓取各实例，再由监控后端聚合；不要把运行时目录放在高延迟网络盘上。

## Prometheus

示例抓取配置：

```yaml
scrape_configs:
  - job_name: recsys-api
    metrics_path: /metrics/prometheus
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

项目提供可直接引用的告警起点：
`monitoring/prometheus_rules.yml`。阈值必须按目标机器和真实 SLO 调整。

## 429 客户端约定

所有 429 都带：

- `Retry-After`：至少等待的秒数；
- `X-Error-Class: rate_limited`：入口速率超限；
- `X-Error-Class: overloaded`：并发槽位已满；
- `X-Request-ID`：关联日志。

客户端应采用带抖动的指数退避，并设置总重试预算；不要收到 429 后立即无界重试。
曝光与反馈可以按原内容重试，反馈必须沿用原 `event_id`。创建推荐的 POST 每次
生成新批次；旧 GET 也有登记副作用，不能因为使用 GET 就假设它天然幂等。

## 优雅停机

使用 Ctrl+C、服务管理器的正常终止信号或编排平台的终止信号。Uvicorn 最多等待
`--graceful-timeout` 秒完成在途请求，随后每个 worker 删除自己的注册文件。
不要直接结束整个进程树，除非正常停机已超时。

## 验收

完整真实模型冒烟：

```powershell
.venv/Scripts/python.exe scripts/production_smoke.py --base-url http://127.0.0.1:8000 --tag <unique-tag>
```

该命令验证：

1. 两 worker 均预加载并通过 live/ready；
2. 低速流量全部成功；
3. 突发流量产生可分类的 429；
4. JSON 聚合计数与客户端请求数一致；
5. Prometheus、CPU/RSS、事件循环 lag 均存在；
6. 每个 worker 都处理到业务流量；
7. 成功推荐数与实际反馈库中的批次增量一致。

省略 `--base-url` 时，脚本另起一个隔离的本地服务，结束后还检查崩溃遗留注册文件回收；
指定外部服务时不检查其关闭清理。
正常 lifespan 停机与“最后一次后台写入晚于注销”的竞态由
`tests/test_serve_lifecycle.py` 单独验证。

## 已知边界

- 指标和限流都是单主机方案；跨主机限流需要入口网关或共享后端。
- 聚合延迟分位数基于每 worker 最近的有界样本，不代表无限历史。
- 进程内 Counter 会在 worker 重启时归零；Prometheus 的 `increase()` 能处理常见重启。
- 本轮容器结果包含 POST JSON 与 Windows 到 Docker 的本地转发；不包含 TLS、反向代理、公网流量或多租户影响。

## 反馈数据

推荐服务会同步持久化成功返回的推荐及完整候选池；如果 SQLite 写入失败，
`/v1/recommendations` 返回 503，不会宣称已经登记成功。默认数据库为
`runtime/feedback/events.sqlite3`，多 worker 通过 WAL 共享。仅自动启动本地服务的
冒烟模式使用按 tag 隔离的数据库；外部冒烟和压测写入目标服务的数据卷，需要使用独立测试实例。

上线时必须显式设置稳定的 `--model-version`、`--ranking-policy` 和持久盘上的
`--feedback-db`。默认 `--exploration-rate 0`；探索会改变用户看到的物品，
只有在业务护栏和实验授权齐备时才应开启。事件协议、导出、OPE 解释边界见
`docs/FEEDBACK_LOOP.md`。

本地真人采集应使用 `feedback_app.py`。匿名体验者代号不得包含姓名、邮箱等
个人信息。采集端默认 10% 探索仅适用于获准的本地实验；公开部署仍须补充认证、
隐私告知、删除/保留策略和实验分流治理。
