# Production-oriented Recommender System

> 一个可运行、可评估、可观测的推荐系统项目：默认双塔 + Faiss Top-K、可选实验精排、
> 序列推荐与会话 RL、主动工具门控、REST 服务、容量治理，以及真实曝光/反馈的
> OPE 决策闭环。

## 面试入口

- **5–10 分钟项目答辩**：[面试手册](docs/INTERVIEW_PLAYBOOK.md)
- **可直接改写进简历**：[中英文简历描述](docs/RESUME_PROJECT.md)
- **指标与证据索引**：[项目证据清单](docs/PROJECT_EVIDENCE.md)
- **完整研究与工程结果**：[总报告](REPORT.md)
- **面试展示 PPT**：`docs/interview/Recsys_Interview_Deck_CN.pptx`

## 一句话业务结论

在 MovieLens-1M/25M 上完成“召回 → 排序 → 服务 → 反馈 → 决策”的端到端闭环。
双塔修正采样偏差后 Recall@10 达 0.098，较 MF 基线提升 58%；DeepFM 在
“已评分样本中预测评分≥4”的显式偏好任务上 AUC 0.754（user-level GAUC 0.735；
**MovieLens 无曝光日志，这不是实际 CTR**）。本地服务完成多 worker、限流、过载保护、Prometheus
与真实压测。当前默认双塔路径 `/recommend` HTTP 端到端 **5.4 ms / p95 9.7 ms**
（2026-08-16 WSL/Linux CPU、SQLite 在本地 /tmp 的 c1 复测）；推荐核心 **0.19 ms**。
旧的 Windows 精排路径 4.6 ms 仅作历史参考，不再代表默认服务。
个人反馈倾向低权重多样性重排，但 1,834 人外部问卷不支持全局替换；结合端到端
精排负增量，当前默认策略是**双塔 Top-K，多样性与 DeepFM 都只作为实验候选**。

**同时诚实记录一个负面结论及其修复过程**：补上端到端联合评估后发现，线上真实
链路（召回 50 → DeepFM 精排 → Top-10）的 Recall@10 只有 0.065，**比直接用双塔前
10 名（0.098）还差 33.6%**——精排把候选坍缩到了热门头部（去重物品 1853 → 773）。
随后按「难负样本 → listwise → 交叉特征 → 残差 → out-of-fold 交叉拟合」做了六版重训，
最好的 v6 达 0.094（**缺口补回 88.7%**），但配对 bootstrap 95% 区间
[−0.0132, +0.0053] **跨 0 = 与不精排持平，仍无证据说更好**。因此决策是：
**v0 下线，线上直接用双塔 Top-10**。

**另外补上了评估协议对照**：既有 Recall 0.098 来自 leave-one-out（存在时间穿越）；
换成全局时间切分后是 **0.069 [0.055, 0.085]，区间与 LOO 不重叠**。两个都报、
并标注差异，而不是挑好看的那个。全过程见 [REPORT.md](REPORT.md)。

## 招聘者 30 秒速览

```text
MovieLens 日志
  → 双塔向量召回（Faiss，3706 → 50）
  → 默认直接取双塔 Top-10
  → 可选 DeepFM/残差精排（仅离线晋升评估与失败复现）
  → Starlette API / Streamlit Demo
  → 曝光与反馈日志（SQLite/WAL + propensity）
  → IPS/SNIPS + ESS + bootstrap 策略门槛
  → 个人候选，而非未经证据的全量上线
```

| 能力面 | 可验证产出 |
|---|---|
| 推荐算法 | MF、DeepFM、双塔、SASRec、Bandit、会话 RL、离线 RL |
| 评估诊断 | 对齐 leave-one-out/负采样协议；定位采样偏差并做 logit 校正 |
| ML 工程 | 同源 Demo/API、模型常驻、产物自检、CPU-only 测试套件（数量以 `--test` 输出为准） |
| 生产治理 | 2/4 worker 扩展、429 准入保护、令牌桶、Prometheus、优雅停机 |
| 业务决策 | 真实反馈 OPE + 1,834 人外部问卷；明确“不全量上线”的边界 |
| Agentic RL | 主动工具门控 + 条件奖励消融；规则门控省 12.8% 调用并拒绝上线退化 RL |

---

下面保留完整学习与复现路径。

---

## 一、为什么选这个项目？

推荐/广告系统是互联网行业**商业价值最高的机器学习场景**，没有之一：

- 字节跳动的信息流推荐、淘宝的商品推荐、B站的视频推荐，背后都是同一套技术栈
- 岗位需求大：「推荐算法工程师」是算法岗中招聘量最大的方向之一
- 技术链条完整：数据处理 → 特征工程 → 深度模型 → 向量检索 → 在线策略（RL），
  几乎覆盖深度学习工程的所有核心环节
- **产出看得见**：AUC 指标、训练曲线、召回效果、可交互 Demo、实验报告

## 二、业务问题定义

> 给你 6000 个用户、4000 部电影、100 万条历史评分（MovieLens-1M 公开数据集），
> 你能不能预测"用户会不会喜欢一部没看过的电影"，并给出每个人最该看的 Top-10？

这正是所有推荐系统的最小原型。学完你会发现，把"电影"换成"短视频"、"商品"、
"广告"，问题的本质一模一样。

## 三、学习路线图（5 个阶段 + 1 个产出阶段）

| 阶段 | 主题 | 你会学到 | 产出 |
|------|------|----------|------|
| 0 | 环境搭建 & 数据探索 ✅ | pandas、数据直觉、推荐系统的数据长什么样 | 探索图表 + 笔记 |
| 1 | 传统召回基线 ✅ | 协同过滤、矩阵分解（手写 SGD，理解 Embedding 的本质） | Recall@10=0.062、过拟合实验 |
| 1.5 | RL 快速体验 ✅ | PPO 训练 LunarLander 登月（ Gymnasium + SB3 ） | 奖励曲线 + 着陆 GIF |
| 2 | 深度学习排序模型 ✅ | PyTorch 训练流程、LR/FM/DeepFM、显式偏好二分类、AUC/GAUC | AUC 0.754 / GAUC 0.735；非真实 CTR |
| 3 | 双塔召回 + 向量检索 ✅ | 工业界"召回→排序"两段式架构、Faiss、logit 校正 | Recall@10=0.098（+58%）|
| 4 | 强化学习扩展 ✅ | Contextual Bandit（ε-greedy / LinUCB）、探索-利用 | 在线 CTR 0.704 突破天花板 |
| 5 | 工程产出 ✅ | Streamlit 可交互 Demo、实验报告撰写 | `app.py` + `REPORT.md` |
| 6 | Agentic 推荐 ✅ | 多工具轨迹、预算/非法动作、调用成本、组内相对策略梯度 | 二元门控 + 多工具闭环 + 正式上线决策 |

**时间预估**：每天 1-2 小时，约 4-6 周完成。每个阶段结束后在 `notes/` 写学习日志。

## 四、目录结构

```
recsys-learning/
├── README.md            ← 你在这里（项目总览 + 路线图）
├── 项目导学.md          ← ★ 新手入口：从零看懂每一步（先读这个！）
├── REPORT.md            ← ★ 项目总报告（所有指标和结论）
├── app.py               ← ★ Streamlit 推荐 Demo（召回→排序完整流水线）
├── requirements.txt     ← 依赖清单
├── data/                ← 数据集（运行 src/download_data.py 自动下载）
├── checkpoints/         ← 训练好的模型（mf/deepfm/two_tower/ppo_lander）
├── experiments/         ← 实验日志 CSV + 推荐样例
├── figures/             ← 所有图表产出（12 张）
├── notes/               ← ★ 学习笔记（边学边记；最推荐 15-端到端评估与精排重建）
├── research/            ← ★ Phase 2 研究课题（课题设计书 + 研究报告）
└── src/                 ← 所有训练和绘图脚本（带逐行中文注释）
```

## 从零复现（新 clone 必读）

仓库**不含**数据与模型权重（在 `.gitignore` 中，避免提交大文件与数据集再分发）。
克隆后按下面顺序即可从零跑通：

```powershell
# 1) 按完整锁文件创建环境（严格 Python 3.12）
uv sync --locked

# 2) 下载 MovieLens-1M（约 6 MB，自动解压到 data/ml-1m/）
.venv/Scripts/python.exe src/download_data.py

# 3) 训练当前默认服务所需的双塔
.venv/Scripts/python.exe src/train_two_tower.py

# 可选研究产物：显式偏好 DeepFM（不是 CTR、不是默认服务）与 SASRec 仿真器
.venv/Scripts/python.exe src/train_deepfm.py --epochs 15
.venv/Scripts/python.exe src/train_sasrec.py

# 4) 起 Demo / REST API
.venv/Scripts/python.exe -m streamlit run app.py
.venv/Scripts/python.exe serve.py --port 8000 --preload

# 5) 自检
.venv/Scripts/python.exe scripts/ai_startup_harness.py --check
.venv/Scripts/python.exe scripts/ai_startup_harness.py --test
.venv/Scripts/python.exe scripts/ai_startup_harness.py --release
```

CI（`.github/workflows/ci.yml`）会在**无数据 / 无 GPU** 的干净环境跑语法检查 + 全部
单元测试，验证代码可被任何人复现。测试数量随开发变化，**不在文档里硬编码**——
以 `scripts/ai_startup_harness.py --test` 的实际输出为准（曾出现六处文档各写一个
数字、无一正确的情况）。

`uv.lock` 固定完整依赖图；`artifacts/release_manifest.json` 以 SHA-256 固定参考
数据与 checkpoint。默认 `--release` 检查真正在线的 `serve` profile（MovieLens +
双塔），`--release --release-profile research` 再检查 DeepFM/SASRec/v6 reranker。
`scripts/verify_release.py` / `--release` 会硬校验参考权重；部署服务时可再加
`serve.py --verify-checkpoint-hashes`（或 `RECSYS_VERIFY_CHECKPOINTS=1`）在启动前
对 manifest 中登记的权重做 size+SHA-256 校验。所有 checkpoint 加载统一走
`weights_only=True` 白名单，未登记的实验权重也不能用 pickle 执行任意代码。
从零训练复现时不要开启运行时哈希校验，因为自训权重尚未登记进 manifest。
详见 `docs/REPRODUCIBILITY.md`。

## 运行 Demo

```bash
.venv/Scripts/python.exe -m streamlit run app.py
# 浏览器打开 http://localhost:8501；默认展示双塔 Top-K，可显式切换失败精排对照
```

## 运行 REST API（服务化）

Demo 与 API 共用 `src/serving.py` 的推荐核心（单一事实来源）：

```bash
.venv/Scripts/python.exe serve.py --port 8000 --preload --workers 2 `
  --max-in-flight 8 --rate-limit 80 --rate-burst 8
# GET /health                     模型/产物状态
# GET /live 和 /ready             存活/就绪探针
# GET /metrics                    当前 worker JSON
# GET /metrics/aggregate          跨 worker 聚合 JSON
# GET /metrics/prometheus         Prometheus 文本
# GET /recommend?user_id=1&k=10   默认双塔 Top-k

# 仅用于失败复现/候选 ranker 验证；默认服务不要这样启动
.venv/Scripts/python.exe serve.py --port 8001 --preload `
  --ranking-policy deepfm --ranker-checkpoint checkpoints/deepfm_rerank_v6.pt

# 本机可重复压测；结果会保存到 experiments/serve_load_<tag>.json
.venv/Scripts/python.exe scripts/load_test_api.py --requests 500 --concurrency 16 --tag local-c16

# 1/2/4 worker 扩展与过载保护实验（会多次加载模型，使用唯一 tag）
.venv/Scripts/python.exe scripts/scaling_experiment.py --tag local-scaling

# 真实 2-worker 生产面完整验收
.venv/Scripts/python.exe scripts/production_smoke.py --tag local-production
```

每个业务响应都带 `X-Request-ID` 与 `Server-Timing`。

**当前默认双塔 Top-K 的延迟（2026-08-16 复测，`--concurrency 1`、200 请求）**：

| 环境 | HTTP mean / p95 | 推荐核心 | 说明 |
|---|---:|---:|---|
| WSL/Linux CPU + SQLite 位于 `/tmp` | **5.42 / 9.67 ms** | **0.19 ms** | 当前默认路径，200 请求实测 |
| WSL/Linux CPU + SQLite 位于仓库目录 | 30.04 / 32.80 ms | 0.19 ms | `/mnt/d` 跨文件系统写放大，不代表服务热路径 |
| 历史 Windows localhost + DeepFM | 4.6 / 5.2 ms | 0.97 ms | 旧路径，已不作为默认，待本机 Windows 复测 |

当前默认路径里曝光落盘仍是主要开销；核心推荐已无 pandas 热路径。SQLite 放在本地
高速盘可回到个位数毫秒。要把延迟继续往下压，下一步是批量/异步写曝光，但会改变
“曝光一定被记录”的持久化语义，属于产品决策。产物：
`experiments/serve_load_default-retrieval-linux-c1-tmpdb-20260815.json`。

早期版本里 `movies.loc[...]` 这类 pandas 标量索引占了推荐核心 **39.8%** 的时间，
而模型前向只占 7.9%——**推理从来不是瓶颈，数据访问层才是**。把物品标题/类型、
用户特征、已看集合在加载时预展开成 dict 之后，核心从 4.69 ms 降到 1.66 ms
（**2.8×**，且输出逐字节不变）。**现在的瓶颈是曝光落盘**：要继续压延迟，下一步是
把 SQLite 写入改成批量/异步——但那会改变"曝光一定被记录"的持久化语义，属于产品
决策而非纯性能优化。

`/metrics` 是单进程内存指标，
不会触发模型加载，也不会统计自身请求；多 worker 部署时需逐 worker 采集或接入共享指标后端。
业务请求超过每 worker 的 `max-in-flight` 会立即返回 `429 + Retry-After`，`/health`
和指标/探针始终豁免。每 worker 还可配置令牌桶速率与突发容量。当前 12 逻辑核本机
推荐从 2 workers × 8、80 RPS/worker 开始，约占 1.8 GiB RSS。完整运行手册见
[`docs/PRODUCTION_SERVING.md`](docs/PRODUCTION_SERVING.md)。

## 自动化测试

```bash
.venv/Scripts/python.exe scripts/ai_startup_harness.py --test   # CPU-only 冒烟套件
```

## 评估脚本（分段指标之外，必须看的两个）

```bash
# 1) 召回→精排端到端联合评估：线上真正跑的那条路径的 Recall/NDCG
#    同时输出召回天花板、"不精排"对照组、以及曝光集中度诊断
.venv/Scripts/python.exe scripts/eval_end_to_end.py --tag <new-tag>

# 2) Agentic 工具门控的成本相变扫描：默认 3 seed 只做探索（约 33 分钟）
#    正式胜负推断至少 10 seed，例如 --seeds 42,43,44,45,46,47,48,49,50,51
.venv/Scripts/python.exe scripts/agentic_cost_sweep.py --tag <new-tag>

# 2b) 预算约束的多工具轨迹：正式默认 10 个训练 seed
.venv/Scripts/python.exe scripts/agentic_multitool_experiment.py --tag <new-tag>

# 3) 重训精排层（修复上面那个负增量）；--deepfm-ckpt 让新旧版本走同一条评估路径
.venv/Scripts/python.exe src/train_reranker.py --tag v5 --loss listwise --residual --epochs 20
.venv/Scripts/python.exe scripts/eval_end_to_end.py --tag <new-tag> \
  --deepfm-ckpt checkpoints/deepfm_rerank_v5.pt
```

两个脚本的产物都按 `--tag` 隔离且拒绝覆盖。**结论请读 `REPORT.md`——
其中包括一个负面结论：当前的 DeepFM 精排层是端到端净损失。**

滚动时间、全物品库、强基线与多 seed 的召回基准：

```powershell
.venv/Scripts/python.exe scripts/benchmark_retrieval.py `
  --tag <new-tag> --models popular,item_knn,two_tower `
  --train-fracs 0.70,0.80,0.90 --horizon-frac 0.05 `
  --seeds 42,43,44 --epochs 10
```

当前 v2 结果：MostPopular / Item-kNN / TwoTower 的三窗 Recall@10 宏平均为
0.0718 / 0.0691 / 0.0689，相关性近似打平；双塔换来 17%–30% 的 catalog coverage
（Popular 仅 1.5%–2.6%）和更稳定的最差窗口。详见
`notes/16-滚动时间全库召回基准.md`。

其上的多路版本把 TwoTower / Item-kNN / Popular Top-200 用 RRF 融合，三窗宏平均
Recall@10 提到 0.0787，但 coverage@10 降到 3.5%–5.0%，仍未晋升。序列感知
残差 listwise ranker 严格只用更早窗口训练，在 W2/W3 的 6 个配对上仅 +0.00047，
没有稳定增益；Candidate Recall@200 仅 44%–51% 才是主要瓶颈。见 `notes/17`。

同协议还实现了一个明确标注差异的 TIGER-lite 生成式召回基线：内容元数据经
RQ-KMeans 形成 `64×64×64 + collision token` Semantic ID，Transformer 自回归
生成，并对全库 ID 精确计分。3 seed × 3 窗 Recall@10 宏平均为 **0.0557**，低于
Popular / Item-kNN / TwoTower 的 0.0718 / 0.0691 / 0.0689；coverage 仅 2.75%，
4 个真正冷物品目标全部未命中，因此不晋升。详见 `notes/19-TIGER语义ID生成式召回.md`。

## 五、环境说明

- **硬件**：RTX 5060 Ti + 32GB 内存 —— 对本项目绰绰有余
- **虚拟环境**：用 `uv sync --locked` 创建 `.venv`（严格 Python 3.12），运行脚本用：
  ```bash
  .venv/Scripts/python.exe src/<脚本名>.py
  ```
- **本机参考 PyTorch**：2.9.1+cu128，GPU 训练已验证 ✅。CPU/默认环境直接
  `uv sync --locked`；RTX 50 系先运行
  `uv sync --locked --no-install-package torch`，再安装 CUDA 版：
  （注意：RTX 50 系是 Blackwell 架构，必须 cu128+；且 Windows 下
  PyPI 默认源给的是 CPU 版。如需重装，用交大镜像：
  `.venv/Scripts/python.exe -m pip install --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128 torch==2.9.1`）
- 绘图样式已内置在 `src/plotting.py`，不依赖特定桌面运行时；系统没有中文字体时
  会回退到 DejaVu Sans（图可生成，但中文 glyph 可能缺失）。

## 六、如何使用这个项目

1. 按阶段顺序推进，不要跳步
2. 每跑一个脚本，先读懂代码里的注释，再运行
3. 遇到不懂的概念，随手问，并把解释整理进 `notes/`
4. 每个阶段结束，用 `notes/00-学习日志模板.md` 写一篇日志

## 推荐反馈闭环

`GET /recommend` 现在会生成可追踪的 `recommendation_id`，并把候选池、
模型版本、实际展示集合和行为策略概率写入 SQLite。客户端通过
`POST /events/impression` 确认曝光，通过 `POST /events/feedback` 提交行为；
`scripts/feedback_replay.py` 按 policy/model 队列导出 schema-v2 不可变 JSONL，并生成
IPS/SNIPS、交叉拟合 DR、支持度与数据质量报告；存在多队列时拒绝静默混算。

默认探索率为 0，不会静默改变推荐结果；只有显式设置
`--exploration-rate` 后，才可为新 Top-K 策略建立非零支持。完整协议与示例见
[`docs/FEEDBACK_LOOP.md`](docs/FEEDBACK_LOOP.md)。

真人反馈采集：

```powershell
.venv/Scripts/python.exe -m streamlit run feedback_app.py --server.port 8502
```

实验台默认采用“快速个人试验”：每批展示 5 部，点选 1 部即自动进入下一批，完成 20 批后
自动停止并给出个人试验信号。侧边栏仍可切换到多人正式评估；当前 v2 正式评估比较
双塔 Top-K 与类型多样性重排，至少要求 200 批推荐、5 位匿名体验者、95%
曝光覆盖率和 ESS 100，并按体验者做 cluster bootstrap。个人信号不会冒充正式
上线结论，未过正式门槛时也不会生成“新策略胜出”的伪结论。

OPE 本身也经过已知反事实真值的 5,000 次合成曝光校准：策略增量的绝对误差为
IPS 0.00028、SNIPS 0.00859、DR 0.00812。真实历史探索队列仅 21 次推荐/
1 位体验者，IPS 与 DR 方向冲突并被质量门判为 `collect_more_data`；它只验证闭环，
不构成当前默认双塔策略的收益证据。详见 `notes/18-曝光数据与OPE校准.md`。

## 七、Phase 2：研究生式研究课题（已完成 ✅）

在 Phase 1 的工程流水线之上，按"复现 → 消融 → 提出新问题 → 实验验证"的
研究范式完成了一个完整课题，详见 `research/课题设计书.md` 与
`research/研究报告.md`：

| 模块 | 内容 | 关键结果 |
|------|------|----------|
| W2-a | SASRec（自注意力序列推荐）论文复现 | test HR@10 0.791 / NDCG 0.543（论文 0.82/0.59）|
| W2-b | 消融实验矩阵（维度/层数/序列长度/dropout） | blocks=3 最优；序列截断 200→50 掉 2 个点 |
| W3-a | 推荐会话 MDP 仿真环境（SASRec 当用户模型，含疲劳与厌倦退出） | 贪心策略多样性 1.00、奖励仅 2.81 —— 短视被量化 |
| W3-b | 手写 Double DQN（v1 发散 → 定位致命三要素 → v2 修复收敛） | 7.63 > 随机 6.69 >> 贪心 2.81 |
| W3-c | REINFORCE + Actor-Critic（PPO 家族的祖先） | **采样部署 15.86，朴素贪心 5.6 倍**，会话长度 19.5/20 |
| W3-d | 贪心去重基线（公平对手，后补） | **17.93**，超过 DQN 7.63 与 PG 15.86 —— 疲劳奖励让「交错类型」近乎免费，RL 增量有限 |
| W4 | ML-25M（2500 万评分）可扩展性验证 | 59k 物品 / 310 万交互子样本 3 epoch 收敛，全量约 50 分钟 |
| W5 | **后训练**：离线 RL（CQL/naive DQN）+ SFT（行为克隆）| 离线 RL 越过 SFT 天花板（2.87→8~13.6）；CQL 保守性权衡见 `notes/12` |

核心研究结论：在含疲劳/厌倦动力学的推荐会话中，**最优策略本质上是随机的**
（必须交错类型），这解释了为什么"采样部署 15.86 vs argmax 部署 5.07"——
部署方式是 RL 推荐系统上线时被忽视的关键决策。
