# 推荐 API 生产运行手册

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
推荐只重试幂等的 GET 请求。

## 优雅停机

使用 Ctrl+C、服务管理器的正常终止信号或编排平台的终止信号。Uvicorn 最多等待
`--graceful-timeout` 秒完成在途请求，随后每个 worker 删除自己的注册文件。
不要直接结束整个进程树，除非正常停机已超时。

## 验收

完整真实模型冒烟：

```powershell
.venv/Scripts/python.exe scripts/production_smoke.py --tag <unique-tag>
```

该命令验证：

1. 两 worker 均预加载并通过 live/ready；
2. 低速流量全部成功；
3. 突发流量产生可分类的 429；
4. JSON 聚合计数与客户端请求数一致；
5. Prometheus、CPU/RSS、事件循环 lag 均存在；
6. 每个 worker 都处理到业务流量；
7. 强制崩溃后的过期注册文件可自动回收。

正常 lifespan 停机与“最后一次后台写入晚于注销”的竞态由
`tests/test_serve_lifecycle.py` 单独验证。

## 已知边界

- 指标和限流都是单主机方案；跨主机限流需要入口网关或共享后端。
- 聚合延迟分位数基于每 worker 最近的有界样本，不代表无限历史。
- 进程内 Counter 会在 worker 重启时归零；Prometheus 的 `increase()` 能处理常见重启。
- 本机结果不包含 TLS、反向代理、真实网络、请求体或多租户影响。

## 反馈数据

推荐服务会同步持久化成功返回的推荐及完整候选池；如果 SQLite 写入失败，
`/recommend` 返回 503，不会产生无法追踪的推荐。默认数据库为
`runtime/feedback/events.sqlite3`，多 worker 通过 WAL 共享。压测和生产冒烟
脚本使用按 tag 隔离的数据库，避免把合成流量混入默认反馈。

上线时必须显式设置稳定的 `--model-version`、`--ranking-policy` 和持久盘上的
`--feedback-db`。默认 `--exploration-rate 0`；探索会改变用户看到的物品，
只有在业务护栏和实验授权齐备时才应开启。事件协议、导出、OPE 解释边界见
`docs/FEEDBACK_LOOP.md`。

本地真人采集应使用 `feedback_app.py`。匿名体验者代号不得包含姓名、邮箱等
个人信息。采集端默认 10% 探索仅适用于获准的本地实验；公开部署仍须补充认证、
隐私告知、删除/保留策略和实验分流治理。
