# -*- coding: utf-8 -*-
"""推荐系统 REST API（Starlette + uvicorn，零新增依赖）。

复用 `src/serving.py` 的 `Recommender`，把默认双塔 Top-K 与显式实验精排
暴露成 HTTP 服务。模型只在进程启动时加载一次（常驻内存）。

启动：
    .venv/Scripts/python.exe serve.py                 # 默认 127.0.0.1:8000
    .venv/Scripts/python.exe -m uvicorn serve:app     # 等价（uvicorn CLI）

过载保护（admission control）：
    单进程事件循环 + 线程池并发处理重请求；同时在途请求数达到上限时，
    多出来的请求立即以 429 + Retry-After 拒绝（而非无限排队），把已放行
    请求的尾延迟压住。上限由 `--max-in-flight`（或环境变量
    RECSYS_MAX_IN_FLIGHT）控制；/health 与 /metrics 永远豁免，便于过载
    时仍可观测。多 worker 横向扩展由 `--workers` 控制。

接口：
    GET /health                      → 产物/模型加载状态 + 准入闸门快照
    GET /live, /ready                → 存活/就绪探针（不触发模型加载）
    GET /metrics                     → 当前 worker 的 JSON 指标
    GET /metrics/aggregate           → 同实例跨 worker 聚合 JSON
    GET /metrics/prometheus          → Prometheus 文本指标
    GET /users?limit=50              → 可选用户 ID 列表
    GET /users/{uid}                 → 用户画像 + 历史高分电影
    GET /recommend?user_id=1&k=10    → Top-k 推荐（召回 50 → 精排）
    POST /events/impression          → 确认推荐结果实际曝光
    POST /events/feedback            → 记录点击/喜欢/跳过等反馈
    GET /events/stats                → 反馈闭环数据画像
"""
import asyncio
import hmac
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, FileResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from serving import (Recommender, UnknownUserError,  # noqa: E402
                     ArtifactsMissingError, check_artifacts)
from observability import (ObservabilityMiddleware,  # noqa: E402
                           AdmissionControlMiddleware, RateLimitMiddleware,
                           RuntimeMetrics, ServiceMetrics, process_rss_bytes)
from metrics_registry import (FileMetricsRegistry,  # noqa: E402
                              render_prometheus)
from feedback import (FeedbackStore, FeedbackValidationError,  # noqa: E402
                      select_slate)

# 每个 worker 最多同时处理的在途请求数（超出即 429 卸载）。多 worker 通过
# 重新 import 本模块拿到进程级配置，所以从环境变量读取，CLI 只是代设环境变量。
_MAX_IN_FLIGHT = max(1, int(os.environ.get("RECSYS_MAX_IN_FLIGHT", "8")))
_RATE_LIMIT_RPS = max(
    0.0, float(os.environ.get("RECSYS_RATE_LIMIT_RPS", "0")))
_RATE_LIMIT_BURST = max(
    1, int(os.environ.get("RECSYS_RATE_LIMIT_BURST", "16")))
_METRICS_INTERVAL_S = max(
    0.1, float(os.environ.get("RECSYS_METRICS_INTERVAL_S", "1")))
_METRICS_STALE_S = max(
    1.0, float(os.environ.get("RECSYS_METRICS_STALE_S", "5")))
_METRICS_DIR = Path(os.environ.get(
    "RECSYS_METRICS_DIR", ROOT / "runtime" / "metrics" / "default"))
_FEEDBACK_DB = Path(os.environ.get(
    "RECSYS_FEEDBACK_DB", ROOT / "runtime" / "feedback" / "events.sqlite3"))
_MODEL_VERSION = os.environ.get(
    "RECSYS_MODEL_VERSION", "two-tower-retrieval-ml1m-v2")
_RANKING_POLICY = os.environ.get("RECSYS_RANKING_POLICY", "retrieval")
_RANKER_CHECKPOINT = os.environ.get("RECSYS_RANKER_CHECKPOINT") or None
_VERIFY_CHECKPOINT_HASHES = os.environ.get("RECSYS_VERIFY_CHECKPOINTS") == "1"
_EXPLORATION_RATE = float(os.environ.get(
    "RECSYS_EXPLORATION_RATE", "0"))
_EXPLORATION_POOL = max(
    1, int(os.environ.get("RECSYS_EXPLORATION_POOL", "20")))
_API_KEY = os.environ.get("RECSYS_API_KEY", "")
if not 0.0 <= _EXPLORATION_RATE <= 1.0:
    raise ValueError("RECSYS_EXPLORATION_RATE must be between 0 and 1")

# 进程级单例：懒加载，第一次请求（或启动钩子）时构建，之后常驻。
_MODEL = {"recommender": None, "error": None}
_MODEL_LOCK = threading.Lock()
_METRICS = ServiceMetrics()
_RUNTIME = RuntimeMetrics()
_REGISTRY = FileMetricsRegistry(_METRICS_DIR, _METRICS_STALE_S)
_FEEDBACK = FeedbackStore(_FEEDBACK_DB)


def get_recommender():
    if _MODEL["recommender"] is None and _MODEL["error"] is None:
        # 首批并发请求只能由一个线程加载模型，避免重复占用 CPU 和内存。
        with _MODEL_LOCK:
            if _MODEL["recommender"] is None and _MODEL["error"] is None:
                _METRICS.model_load_started()
                started = time.perf_counter()
                try:
                    _MODEL["recommender"] = Recommender.load(
                        ROOT, deepfm_ckpt=_RANKER_CHECKPOINT,
                        ranking_policy=_RANKING_POLICY,
                        verify_hashes=_VERIFY_CHECKPOINT_HASHES)
                except Exception as exc:  # 产物缺失/权重损坏：交给各接口降级
                    _MODEL["error"] = f"{type(exc).__name__}: {exc}"
                    # 保留完整堆栈：只存 type+message 会让冷启动失败几乎无法排查。
                    logging.getLogger("recsys.serving").exception(
                        "recommender model load failed")
                    _METRICS.model_load_finished(
                        False, (time.perf_counter() - started) * 1000,
                        _MODEL["error"])
                else:
                    _METRICS.model_load_finished(
                        True, (time.perf_counter() - started) * 1000)
    return _MODEL["recommender"]


def _json(payload, status=200):
    return JSONResponse(payload, status_code=status,
                        media_type="application/json; charset=utf-8")


def _metrics_snapshot(include_samples=False):
    rec = _MODEL["recommender"]
    return _METRICS.snapshot(
        model_loaded=rec is not None,
        load_error=_MODEL["error"],
        n_users=len(rec.list_users()) if rec else 0,
        n_items=rec.n_items if rec else 0,
        admission=_ADMISSION.stats(),
        rate_limit=_RATE_LIMIT.stats(),
        runtime=_RUNTIME.snapshot(include_samples=include_samples),
        include_samples=include_samples,
    )


async def _recommender_or_503():
    """在线程池里取（必要时加载）模型；未就绪时返回 503 响应。

    模型加载是同步阻塞的（冷启动 ~8s）；放到线程池里避免堵住事件循环，
    否则过载时连 /health、/metrics 都会被拖慢。返回 (rec, error_response)，
    两者恰有一个为 None。
    """
    rec = await run_in_threadpool(get_recommender)
    if rec is None:
        return None, _json({"error": _MODEL["error"] or "model unavailable"}, 503)
    return rec, None


async def health(request: Request):
    missing = check_artifacts(
        ROOT, ranking_policy=_RANKING_POLICY,
        deepfm_ckpt=_RANKER_CHECKPOINT)
    # 热路径只读取已经加载的单例，不竞争推理线程池令牌；因此即使准入
    # 槽位全满，健康探针仍能立即返回。仅冷启动时才在线程池执行重加载。
    rec = _MODEL["recommender"]
    if rec is None and _MODEL["error"] is None and not missing:
        rec = await run_in_threadpool(get_recommender)
    metrics = _metrics_snapshot()
    return _json({
        "status": "ok" if rec is not None else "degraded",
        "model_loaded": rec is not None,
        "ranking_policy": _RANKING_POLICY,
        "n_users": len(rec.list_users()) if rec else 0,
        "n_items": rec.n_items if rec else 0,
        "missing_artifacts": [rel for rel, _, _ in missing],
        "load_error": _MODEL["error"],
        "model_load_ms": metrics["model"]["last_load_ms"],
        "uptime_s": metrics["uptime_s"],
        "admission": metrics["admission"],
    }, status=200 if rec is not None else 503)


async def live(request: Request):
    return _json({
        "status": "alive",
        "worker_pid": os.getpid(),
        "uptime_s": _metrics_snapshot()["uptime_s"],
    })


async def ready(request: Request):
    rec = _MODEL["recommender"]
    missing = check_artifacts(
        ROOT, ranking_policy=_RANKING_POLICY,
        deepfm_ckpt=_RANKER_CHECKPOINT)
    is_ready = rec is not None and not missing and _MODEL["error"] is None
    return _json({
        "status": "ready" if is_ready else "not_ready",
        "worker_pid": os.getpid(),
        "model_loaded": rec is not None,
        "ranking_policy": _RANKING_POLICY,
        "missing_artifacts": [rel for rel, _, _ in missing],
        "load_error": _MODEL["error"],
    }, status=200 if is_ready else 503)


async def metrics(request: Request):
    # 该接口只读取内存状态，不触发模型加载，也不计入自身请求指标。
    return _json(_metrics_snapshot())


async def aggregate_metrics(request: Request):
    snapshot = await asyncio.to_thread(_REGISTRY.aggregate)
    return _json(snapshot, status=200 if snapshot["worker_count"] else 503)


async def prometheus_metrics(request: Request):
    snapshot = await asyncio.to_thread(_REGISTRY.aggregate)
    return Response(
        render_prometheus(snapshot),
        status_code=200 if snapshot["worker_count"] else 503,
        media_type="text/plain; version=0.0.4",
    )


async def _json_body(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return None, _json({"error": "request body must be valid JSON"}, 400)
    if not isinstance(payload, dict):
        return None, _json({"error": "request body must be a JSON object"}, 400)
    return payload, None


async def list_users(request: Request):
    rec, unavailable = await _recommender_or_503()
    if unavailable is not None:
        return unavailable
    try:
        limit = int(request.query_params.get("limit", 50))
    except ValueError:
        return _json({"error": "limit must be an integer"}, 400)
    users = await run_in_threadpool(rec.list_users)
    return _json({"n_users": len(users), "users": users[:max(0, limit)]})


async def get_user(request: Request):
    rec, unavailable = await _recommender_or_503()
    if unavailable is not None:
        return unavailable
    try:
        uid = int(request.path_params["uid"])
    except ValueError:
        return _json({"error": "user_id must be an integer"}, 400)
    try:
        profile, history = await run_in_threadpool(
            lambda: (rec.user_profile(uid), rec.user_history(uid)))
    except UnknownUserError:
        return _json({"error": f"unknown user_id {uid}"}, 404)
    return _json({"profile": profile, "history": history})


async def recommend(request: Request):
    params = request.query_params
    if "user_id" not in params:
        return _json({"error": "query param 'user_id' is required"}, 400)
    try:
        uid = int(params["user_id"])
        k = int(params.get("k", 10))
    except ValueError:
        return _json({"error": "user_id and k must be integers"}, 400)
    if k <= 0:
        return _json({"error": "k must be a positive integer"}, 400)
    if k > 100:
        return _json({"error": "k cannot exceed 100"}, 400)
    rec, unavailable = await _recommender_or_503()
    if unavailable is not None:
        return unavailable
    try:
        # 双塔召回（以及显式启用时的实验精排）是同步 CPU 计算，放线程池
        # 执行，让事件循环可以同时接纳其他请求 —— 这也让准入闸门看得到
        # 真实并发。
        pool_size = max(k, _EXPLORATION_POOL) \
            if _EXPLORATION_RATE > 0.0 else k
        result = await run_in_threadpool(
            rec.recommend, uid, pool_size, max(50, pool_size))
    except UnknownUserError:
        return _json({"error": f"unknown user_id {uid}"}, 404)
    candidates = result["recommendations"]
    if not candidates:
        return _json(result)

    served, logged = select_slate(candidates, k, _EXPLORATION_RATE)
    recommendation_id = uuid.uuid4().hex
    request_id = request.scope.get("recsys.request_id")
    policy_name = (
        f"{_RANKING_POLICY}_epsilon_slate_v2"
        if _EXPLORATION_RATE > 0.0
        else f"{_RANKING_POLICY}_deterministic_top_k_v2"
    )
    try:
        await run_in_threadpool(
            lambda: _FEEDBACK.record_recommendation(
                recommendation_id=recommendation_id,
                request_id=request_id,
                user_id=uid,
                model_version=_MODEL_VERSION,
                policy_name=policy_name,
                exploration_rate=_EXPLORATION_RATE,
                requested_k=k,
                candidates=logged,
                actor_id=request.headers.get("x-actor-id"),
            )
        )
    except Exception:
        logging.getLogger("recsys.feedback").exception(
            "failed to persist recommendation")
        return _json({"error": "feedback store unavailable"}, 503)
    return _json({
        "recommendation_id": recommendation_id,
        "request_id": request_id,
        "user_id": result["user_id"],
        "model_version": _MODEL_VERSION,
        "ranking_policy": result["ranking_policy"],
        "score_type": result["score_type"],
        "policy_name": policy_name,
        "exploration_rate": _EXPLORATION_RATE,
        "n_candidates": result["n_candidates"],
        "recommendations": served,
    })


async def record_impression(request: Request):
    payload, invalid = await _json_body(request)
    if invalid is not None:
        return invalid
    recommendation_id = payload.get("recommendation_id")
    if not isinstance(recommendation_id, str) or not recommendation_id:
        return _json({"error": "recommendation_id is required"}, 400)
    movie_ids = payload.get("movie_ids")
    if movie_ids is not None and not isinstance(movie_ids, list):
        return _json({"error": "movie_ids must be an array"}, 400)
    try:
        result = await run_in_threadpool(
            _FEEDBACK.record_impressions,
            recommendation_id,
            movie_ids,
            payload.get("impressed_at"),
        )
    except (FeedbackValidationError, TypeError, ValueError) as exc:
        return _json({"error": str(exc)}, 400)
    except Exception:
        logging.getLogger("recsys.feedback").exception(
            "failed to persist impressions")
        return _json({"error": "feedback store unavailable"}, 503)
    return _json(result, 201 if result["inserted"] else 200)


async def record_feedback(request: Request):
    payload, invalid = await _json_body(request)
    if invalid is not None:
        return invalid
    missing = [
        key for key in ("recommendation_id", "movie_id", "event_type")
        if key not in payload
    ]
    if missing:
        return _json({"error": f"missing fields: {', '.join(missing)}"}, 400)
    try:
        result = await run_in_threadpool(
            lambda: _FEEDBACK.record_feedback(
                recommendation_id=str(payload["recommendation_id"]),
                movie_id=int(payload["movie_id"]),
                event_type=str(payload["event_type"]),
                value=payload.get("value"),
                event_id=payload.get("event_id"),
                occurred_at=payload.get("occurred_at"),
            )
        )
    except (FeedbackValidationError, TypeError, ValueError) as exc:
        return _json({"error": str(exc)}, 400)
    except Exception:
        logging.getLogger("recsys.feedback").exception(
            "failed to persist feedback")
        return _json({"error": "feedback store unavailable"}, 503)
    return _json(result, 201 if result["inserted"] else 200)


async def feedback_stats(request: Request):
    try:
        snapshot = await run_in_threadpool(_FEEDBACK.stats)
    except Exception:
        logging.getLogger("recsys.feedback").exception(
            "failed to read feedback stats")
        return _json({"error": "feedback store unavailable"}, 503)
    return _json({
        "model_version": _MODEL_VERSION,
        **snapshot,
    })


@asynccontextmanager
async def _lifespan(app):
    """每个 worker 启动时执行：可选预热模型，并对齐线程池并发上限。"""
    logging.getLogger("recsys.access").setLevel(logging.INFO)
    # 线程池并发对齐准入上限：放行的请求都能立刻拿到线程，不再二次排队。
    try:
        import anyio
        anyio.to_thread.current_default_thread_limiter().total_tokens = \
            _MAX_IN_FLIGHT
    except Exception:  # pragma: no cover - anyio 总是随 starlette 提供
        pass
    await asyncio.to_thread(_FEEDBACK.initialize)
    if os.environ.get("RECSYS_PRELOAD"):
        await run_in_threadpool(get_recommender)

    _RUNTIME.observe(0.0, 0.0, process_rss_bytes())
    await asyncio.to_thread(
        _REGISTRY.publish, _metrics_snapshot(include_samples=True))
    stop_sampler = asyncio.Event()
    sampler = asyncio.create_task(
        _runtime_metrics_loop(stop_sampler),
        name=f"recsys-metrics-{os.getpid()}")
    try:
        yield
    finally:
        # Event-driven shutdown waits for any in-progress to_thread publish.
        # Cancelling asyncio.to_thread does not stop its OS thread and could
        # otherwise leave a late .tmp file after worker unregister.
        stop_sampler.set()
        await sampler
        await asyncio.to_thread(_REGISTRY.remove, os.getpid())


async def _runtime_metrics_loop(stop_event: asyncio.Event):
    """Sample loop lag/process usage and publish one atomic worker snapshot."""
    loop = asyncio.get_running_loop()
    previous_cpu = time.process_time()
    previous_wall = time.perf_counter()
    while not stop_event.is_set():
        expected = loop.time() + _METRICS_INTERVAL_S
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=_METRICS_INTERVAL_S)
            break
        except TimeoutError:
            pass
        now_loop = loop.time()
        now_wall = time.perf_counter()
        now_cpu = time.process_time()
        elapsed = max(now_wall - previous_wall, 1e-9)
        cpu_percent = (now_cpu - previous_cpu) / elapsed * 100.0
        _RUNTIME.observe(
            max(0.0, now_loop - expected) * 1000.0,
            cpu_percent,
            process_rss_bytes(),
        )
        previous_cpu, previous_wall = now_cpu, now_wall
        await asyncio.to_thread(
            _REGISTRY.publish, _metrics_snapshot(include_samples=True))


async def index(request: Request):
    """返回自包含的单文件前端（薄前端 demo，同源调用 REST API）。"""
    return FileResponse(str(ROOT / "web" / "index.html"))


_API_KEY_EXEMPT = frozenset({
    "/", "/health", "/live", "/ready",
    "/metrics", "/metrics/aggregate", "/metrics/prometheus",
})


class ApiKeyMiddleware:
    """可选 X-API-Key 鉴权（demo 级，非生产安全边界）。

    设置 RECSYS_API_KEY 后，除前端页与健康/指标探针外的接口都要求
    `X-API-Key` 头匹配，否则返回 401。未设置时完全透传，向后兼容。
    生产环境请替换为 JWT / API Gateway 的签名鉴权。
    """

    def __init__(self, app, api_key: str, *, exempt_paths=_API_KEY_EXEMPT):
        self.app = app
        self.api_key = api_key
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not self.api_key:
            await self.app(scope, receive, send)
            return
        if scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        supplied = headers.get(b"x-api-key", b"").decode("latin-1")
        # 常数时间比较：`==` 会在第一个不同字节处提前返回，理论上可被时序
        # 侧信道逐字节爆破。compare_digest 是标准库自带，零成本。
        if hmac.compare_digest(supplied, self.api_key):
            await self.app(scope, receive, send)
            return
        body = b'{"error":"missing or invalid X-API-Key","code":"unauthorized"}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"www-authenticate", b"ApiKey"),
            ],
        })
        await send({"type": "http.response.body", "body": body})


_starlette_app = Starlette(routes=[
    Route("/", index),
    Route("/health", health),
    Route("/live", live),
    Route("/ready", ready),
    Route("/metrics", metrics),
    Route("/metrics/aggregate", aggregate_metrics),
    Route("/metrics/prometheus", prometheus_metrics),
    Route("/users", list_users),
    Route("/users/{uid}", get_user),
    Route("/recommend", recommend),
    Route("/events/impression", record_impression, methods=["POST"]),
    Route("/events/feedback", record_feedback, methods=["POST"]),
    Route("/events/stats", feedback_stats),
], lifespan=_lifespan)

# 中间件顺序：可观测在最外层，速率限制在中间，准入闸门在里层。
# 两类 429 都会被计入各自错误分类并带上关联/计时响应头。
_ADMISSION = AdmissionControlMiddleware(_starlette_app, _MAX_IN_FLIGHT)
_RATE_LIMIT = RateLimitMiddleware(
    _ADMISSION, _RATE_LIMIT_RPS, _RATE_LIMIT_BURST)
_observable = ObservabilityMiddleware(_RATE_LIMIT, _METRICS)
# 鉴权放最外层：坏 key 立即 401，不消耗限流/准入资源。未设 key 时透传。
app = ApiKeyMiddleware(_observable, _API_KEY)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", action="store_true",
                        help="启动时即加载模型（默认懒加载到首个请求）")
    parser.add_argument("--max-in-flight", type=int, default=_MAX_IN_FLIGHT,
                        help="每个 worker 同时处理的在途请求上限，超出即 429")
    parser.add_argument("--workers", type=int, default=1,
                        help="uvicorn worker 进程数（横向扩展）")
    parser.add_argument(
        "--rate-limit", type=float, default=_RATE_LIMIT_RPS,
        help="每 worker 每秒令牌数；0 表示关闭速率限制")
    parser.add_argument(
        "--rate-burst", type=int, default=_RATE_LIMIT_BURST,
        help="每 worker 令牌桶突发容量")
    parser.add_argument(
        "--metrics-dir", default=None,
        help="跨 worker 指标目录；默认按 host-port 放到 runtime/metrics/")
    parser.add_argument(
        "--metrics-interval", type=float, default=_METRICS_INTERVAL_S,
        help="运行时采样与跨 worker 发布间隔（秒）")
    parser.add_argument(
        "--graceful-timeout", type=int, default=15,
        help="优雅停机等待在途请求的秒数")
    parser.add_argument(
        "--keep-alive", type=int, default=5,
        help="HTTP keep-alive 空闲超时秒数")
    parser.add_argument(
        "--feedback-db", default=None,
        help="SQLite feedback database; defaults to runtime/feedback/events.sqlite3")
    parser.add_argument(
        "--model-version", default=_MODEL_VERSION,
        help="version written with every recommendation event")
    parser.add_argument(
        "--ranking-policy", choices=("retrieval", "deepfm"),
        default=_RANKING_POLICY,
        help="retrieval=默认双塔 Top-K；deepfm=显式实验精排")
    parser.add_argument(
        "--ranker-checkpoint", default=_RANKER_CHECKPOINT,
        help="--ranking-policy deepfm 时使用的可选 ranker 权重")
    parser.add_argument(
        "--verify-checkpoint-hashes", action="store_true",
        default=_VERIFY_CHECKPOINT_HASHES,
        help="启动前按 release_manifest.json 校验已登记权重的 size+SHA-256；"
             "从零训练复现时不要开启（会因自训权重未登记而启动失败）")
    parser.add_argument(
        "--exploration-rate", type=float, default=_EXPLORATION_RATE,
        help="epsilon mixture for uniform slate exploration (default: 0)")
    parser.add_argument(
        "--exploration-pool", type=int, default=_EXPLORATION_POOL,
        help="candidate pool used when exploration is enabled")
    parser.add_argument(
        "--api-key", default=_API_KEY,
        help="optional X-API-Key required on API routes (empty = disabled)")
    args = parser.parse_args()
    if args.max_in_flight <= 0:
        parser.error("--max-in-flight must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.rate_limit < 0:
        parser.error("--rate-limit cannot be negative")
    if args.rate_burst <= 0:
        parser.error("--rate-burst must be positive")
    if args.metrics_interval <= 0:
        parser.error("--metrics-interval must be positive")
    if args.graceful_timeout < 0 or args.keep_alive < 0:
        parser.error("timeouts cannot be negative")
    if not 0.0 <= args.exploration_rate <= 1.0:
        parser.error("--exploration-rate must be between 0 and 1")
    if args.exploration_pool <= 0:
        parser.error("--exploration-pool must be positive")
    if args.ranker_checkpoint and args.ranking_policy != "deepfm":
        parser.error("--ranker-checkpoint requires --ranking-policy deepfm")

    # 通过环境变量把配置传给（可能是子进程重新 import 的）worker。
    os.environ["RECSYS_MAX_IN_FLIGHT"] = str(args.max_in_flight)
    os.environ["RECSYS_RATE_LIMIT_RPS"] = str(args.rate_limit)
    os.environ["RECSYS_RATE_LIMIT_BURST"] = str(args.rate_burst)
    os.environ["RECSYS_METRICS_INTERVAL_S"] = str(args.metrics_interval)
    os.environ["RECSYS_MODEL_VERSION"] = args.model_version
    os.environ["RECSYS_RANKING_POLICY"] = args.ranking_policy
    os.environ["RECSYS_RANKER_CHECKPOINT"] = args.ranker_checkpoint or ""
    os.environ["RECSYS_VERIFY_CHECKPOINTS"] = (
        "1" if args.verify_checkpoint_hashes else "")
    os.environ["RECSYS_EXPLORATION_RATE"] = str(args.exploration_rate)
    os.environ["RECSYS_EXPLORATION_POOL"] = str(args.exploration_pool)
    os.environ["RECSYS_API_KEY"] = args.api_key or ""
    feedback_db = (
        Path(args.feedback_db) if args.feedback_db
        else _FEEDBACK_DB
    )
    os.environ["RECSYS_FEEDBACK_DB"] = str(feedback_db.resolve())
    metrics_dir = (
        Path(args.metrics_dir) if args.metrics_dir
        else ROOT / "runtime" / "metrics" / f"{args.host}-{args.port}"
    )
    os.environ["RECSYS_METRICS_DIR"] = str(metrics_dir.resolve())
    if args.preload:
        os.environ["RECSYS_PRELOAD"] = "1"
    logging.getLogger("recsys.access").setLevel(logging.INFO)
    # 统一走 import 字符串：单/多 worker 一致，worker 从环境变量重建 app。
    uvicorn.run("serve:app", host=args.host, port=args.port,
                workers=args.workers, access_log=False,
                timeout_graceful_shutdown=args.graceful_timeout,
                timeout_keep_alive=args.keep_alive)
