# Production-oriented Recommender System

> 一个可运行、可评估、可观测的推荐系统项目：双塔 + Faiss 召回、DeepFM 精排、
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
双塔修正采样偏差后 Recall@10 达 0.097，较 MF 基线提升 56%；DeepFM AUC 0.754。
本地服务完成多 worker、限流、过载保护、Prometheus 与真实压测。个人反馈倾向
低权重多样性重排，但 1,834 人外部问卷不支持全局替换，因此最终选择
**保留 DeepFM 基线，把多样性作为个性化候选**。

## 招聘者 30 秒速览

```text
MovieLens 日志
  → 双塔向量召回（Faiss，3706 → 50）
  → DeepFM 精排（50 → Top-10）
  → Starlette API / Streamlit Demo
  → 曝光与反馈日志（SQLite/WAL + propensity）
  → IPS/SNIPS + ESS + bootstrap 策略门槛
  → 个人候选，而非未经证据的全量上线
```

| 能力面 | 可验证产出 |
|---|---|
| 推荐算法 | MF、DeepFM、双塔、SASRec、Bandit、会话 RL、离线 RL |
| 评估诊断 | 对齐 leave-one-out/负采样协议；定位采样偏差并做 logit 校正 |
| ML 工程 | 同源 Demo/API、模型常驻、产物自检、73 项测试（2 项集成测试按配置跳过） |
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
| 2 | 深度学习排序模型 ✅ | PyTorch 训练流程、LR/FM/DeepFM、CTR 预估、AUC | AUC 0.754 + 三模型对比曲线 |
| 3 | 双塔召回 + 向量检索 ✅ | 工业界"召回→排序"两段式架构、Faiss、logit 校正 | Recall@10=0.097（+56%）|
| 4 | 强化学习扩展 ✅ | Contextual Bandit（ε-greedy / LinUCB）、探索-利用 | 在线 CTR 0.704 突破天花板 |
| 5 | 工程产出 ✅ | Streamlit 可交互 Demo、实验报告撰写 | `app.py` + `REPORT.md` |
| 6 | Agentic 推荐 ✅ | 主动工具选择、调用成本、条件奖励、组内相对策略梯度 | 最小闭环 + 消融 + 上线决策 |

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
├── notes/               ← ★ 学习笔记（9 篇，边学边记）
├── research/            ← ★ Phase 2 研究课题（课题设计书 + 研究报告）
└── src/                 ← 所有训练和绘图脚本（带逐行中文注释）
```

## 运行 Demo

```bash
.venv/Scripts/python.exe -m streamlit run app.py
# 浏览器打开 http://localhost:8501，任选用户看"双塔召回→DeepFM精排"实时推荐
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
# GET /recommend?user_id=1&k=10   Top-k 推荐（召回 50 → 精排，单次约 5 ms）

# 本机可重复压测；结果会保存到 experiments/serve_load_<tag>.json
.venv/Scripts/python.exe scripts/load_test_api.py --requests 500 --concurrency 16 --tag local-c16

# 1/2/4 worker 扩展与过载保护实验（会多次加载模型，使用唯一 tag）
.venv/Scripts/python.exe scripts/scaling_experiment.py --tag local-scaling

# 真实 2-worker 生产面完整验收
.venv/Scripts/python.exe scripts/production_smoke.py --tag local-production
```

每个业务响应都带 `X-Request-ID` 与 `Server-Timing`。`/metrics` 是单进程内存指标，
不会触发模型加载，也不会统计自身请求；多 worker 部署时需逐 worker 采集或接入共享指标后端。
业务请求超过每 worker 的 `max-in-flight` 会立即返回 `429 + Retry-After`，`/health`
和指标/探针始终豁免。每 worker 还可配置令牌桶速率与突发容量。当前 12 逻辑核本机
推荐从 2 workers × 8、80 RPS/worker 开始，约占 1.8 GiB RSS。完整运行手册见
[`docs/PRODUCTION_SERVING.md`](docs/PRODUCTION_SERVING.md)。

## 自动化测试

```bash
.venv/Scripts/python.exe scripts/ai_startup_harness.py --test   # CPU-only 冒烟套件
```

## 五、环境说明

- **硬件**：RTX 5060 Ti + 32GB 内存 —— 对本项目绰绰有余
- **虚拟环境**：项目自带 `.venv`（Python 3.12），运行脚本用：
  ```bash
  .venv/Scripts/python.exe src/<脚本名>.py
  ```
- **PyTorch 已装好**：2.9.1+cu128，GPU 训练已验证 ✅
  （注意：RTX 50 系是 Blackwell 架构，必须 cu128+；且 Windows 下
  PyPI 默认源给的是 CPU 版。如需重装，用交大镜像：
  `pip install --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128 torch`）
- 画图统一用 Kimi Work 托管 Python（`python src/plot_*.py`），自带中文字体

## 六、如何使用这个项目

1. 按阶段顺序推进，不要跳步
2. 每跑一个脚本，先读懂代码里的注释，再运行
3. 遇到不懂的概念，随手问，并把解释整理进 `notes/`
4. 每个阶段结束，用 `notes/00-学习日志模板.md` 写一篇日志

## 推荐反馈闭环

`GET /recommend` 现在会生成可追踪的 `recommendation_id`，并把候选池、
模型版本、实际展示集合和行为策略概率写入 SQLite。客户端通过
`POST /events/impression` 确认曝光，通过 `POST /events/feedback` 提交行为；
`scripts/feedback_replay.py` 可导出不可变 JSONL 并生成 IPS/SNIPS 与支持度报告。

默认探索率为 0，不会静默改变推荐结果；只有显式设置
`--exploration-rate` 后，才可为新 Top-K 策略建立非零支持。完整协议与示例见
[`docs/FEEDBACK_LOOP.md`](docs/FEEDBACK_LOOP.md)。

真人反馈采集：

```powershell
.venv/Scripts/python.exe -m streamlit run feedback_app.py --server.port 8502
```

实验台默认采用“快速个人试验”：每批展示 5 部，点选 1 部即自动进入下一批，完成 20 批后
自动停止并给出个人试验信号。侧边栏仍可切换到多人正式评估；正式评估比较
DeepFM Top-K 与类型多样性重排，至少要求 200 批推荐、5 位匿名体验者、95%
曝光覆盖率和 ESS 100，并按体验者做 cluster bootstrap。个人信号不会冒充正式
上线结论，未过正式门槛时也不会生成“新策略胜出”的伪结论。

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
