# 简历项目描述

## 推荐标题

**生产导向推荐系统：双塔检索、端到端策略决策与真实反馈闭环**

技术栈：Python、PyTorch、Faiss、Starlette/Uvicorn、Streamlit、SQLite/WAL、
Prometheus、NumPy/pandas、unittest

## 中文精简版

- 独立构建“**双塔 + Faiss 召回 → 候选精排**”两阶段研究系统；统一
  leave-one-out/负采样评估并修复采样先验偏差，Recall@10 达 **0.098**，
  较 MF 基线 **提升 58%**；DeepFM 在已评分正向偏好任务上测试 AUC **0.754**
  （user-level GAUC 0.735，非真实 CTR）；
  另建全局时间切分协议对照，量化留一法的乐观偏差（**0.069 vs 0.098，区间不重叠**）。
- **建立端到端联合评估并据此推翻了一个既有架构决策**：分段指标体面，但线上
  真实链路（召回 50 → 精排 10）Recall@10 仅 **0.065**，**比不精排低 33.6%**；
  用曝光集中度（Top-10 去重物品 1853→773）定位为 sample selection bias 导致的
  热门坍缩，经难负样本 / listwise / 交叉特征 / 残差门控 / **out-of-fold 交叉拟合**
  六轮重训将缺口收窄 **88.7%**；最终依据残差门自学出的权重（仅占召回 6% 且为负）
  判定为**信息问题而非实现问题**，并因配对 bootstrap 95% 区间跨 0 而**拒绝上线**、
  改为直接使用召回结果。
- 将推理核心封装为 Streamlit/Starlette 同源服务，完成多 worker、准入保护、
  令牌桶限流、Prometheus、请求追踪与优雅停机；本机 2 worker、并发 16 下
  成功吞吐约 **270 RPS**，HTTP 200 p99 约 **72 ms**；用 profiler 定位到推荐
  核心 **39.8%** 耗时在 pandas 标量索引（模型前向仅 7.9%），改为加载期预展开
  后核心延迟 **4.69 → 1.66 ms（2.8×）** 且输出逐字节不变；默认双塔路径再降至
  **0.19 ms**，HTTP 端到端 **5.42 ms / p95 9.67 ms**（WSL/Linux c1 复测）。
- 建立候选池、曝光、propensity 和真人反馈日志，使用 IPS/SNIPS、ESS 与
  cluster bootstrap 比较策略；结合个人试验和 **1,834 人**外部问卷，作出
  “保留个人多样性候选、不做全局上线”的可解释决策。
- 参考 DeepEyes 主动感知范式实现带调用成本的 Agentic 推荐工具门控及
  group-relative REINFORCE / GRPO；沿工具成本轴做 **6 成本 × 2 算法 × 3 seed**
  探索性扫描，发现相变候选（均值在 cost≈0.5 附近穿越零点、cost=0.8 时
  RL 高 **+1.92**）；明确 3 seed 不足以宣称显著，并把 ≥10 seed 设为推断门槛。

## 中文一行版

独立实现可运行的两阶段推荐系统与反馈 OPE 闭环，覆盖双塔/Faiss 召回、
DeepFM 精排、生产服务治理和真实策略决策；Recall@10 较 MF 提升 58%，
并通过端到端漏斗评估发现并量化了精排层的负增量，按置信区间门槛作出不上线决策。

## English version

**Production-oriented Recommender System — Retrieval, Ranking, and Feedback OPE**

- Built an end-to-end two-stage recommender with two-tower/Faiss retrieval and
  DeepFM ranking; aligned leave-one-out evaluation and corrected sampling-prior
  bias, reaching **0.098 Recall@10 (+58% vs. MF)** and **0.754 test AUC**
  (user-level GAUC 0.735) on positive-rating classification among observed
  ratings; MovieLens has no impression labels, so this is not CTR.
- **Built funnel-level evaluation that overturned an existing design decision:**
  per-stage metrics looked healthy, yet the served path (retrieve 50 → rank 10)
  reached only **0.065 Recall@10, 33.6% below not reranking at all**. Diagnosed
  it as sample-selection-bias-driven popularity collapse via exposure
  concentration (distinct Top-10 items 1853 → 773), closed **88.7%** of the gap
  across six retraining iterations (retrieval-aligned hard negatives, listwise
  loss, cross features, residual-from-retrieval gating, **out-of-fold
  cross-fitted features**), then used the gate's own learned weight (6% of the
  retrieval weight, and negative) to conclude this is an **information** rather
  than an implementation problem, and **declined to ship** because the paired
  bootstrap 95% CI still straddled zero. Also added a global time-split
  protocol quantifying leave-one-out's optimism (**0.069 vs 0.098, disjoint
  intervals**).
- Productionized a shared inference core behind Streamlit and Starlette,
  including multi-worker serving, admission control, token-bucket rate limiting,
  Prometheus metrics, request tracing, and graceful shutdown; measured
  **~270 successful RPS and ~72 ms HTTP-200 p99** at two workers/concurrency 16
  on a local 12-logical-CPU machine. Profiling showed **39.8%** of recommender
  latency in pandas scalar indexing versus **7.9%** in model forward; hoisting
  lookups to load time cut the core from **4.69 ms to 1.66 ms (2.8×)** with
  byte-identical output; the promoted retrieval-only default reaches a
  **0.19 ms** core and **5.42 ms / 9.67 ms p95** HTTP end-to-end (WSL/Linux,
  c1, SQLite on a local fast disk).
- Implemented candidate, impression, propensity, and real-feedback logging with
  IPS/SNIPS, ESS, and cluster-bootstrap policy gates; combined a personal pilot
  with a **1,834-user** external survey to reject an unsupported global rollout
  and retain diversity reranking as a personalized candidate.
- Built a DeepEyes-inspired, cost-aware agentic tool gate with group-relative
  REINFORCE and GRPO; ran a **6-cost × 2-algorithm × 3-seed** phase-transition
  exploratory sweep that identified a candidate crossover near cost 0.5
  (**+1.92 mean net reward** at cost 0.8); withheld significance claims and set
  a minimum 10-seed threshold for formal inference.

## 关键词

推荐系统、召回、排序、双塔、Embedding、Faiss、DeepFM、负采样、logit correction、
端到端漏斗评估、sample selection bias、hard negative mining、listwise ranking、GAUC、
SASRec、Bandit、offline RL、Agentic RL、Tool Use、OPE、IPS、SNIPS、ESS、A/B Testing、Model Serving、
Observability、Prometheus、Rate Limiting、Load Testing

## 使用说明

- 校招/实习简历优先放前三条完整 bullet（第 2 条最有区分度：它证明会做
  架构级诊断，而不只是训模型）。
- 社招或项目较多时保留第 2 条与第 4 条。
- **第 2 条不要改写成"我把精排优化了 45%"**。真实结论是"缺口收窄 88.7%、
  最终仍与不精排持平、因此没有上线"——**"收窄"是过程，不是效果**，
  改写成正向收益会在追问中崩掉。
- 引用召回指标务必带协议名（leave-one-out 0.098 / 全局时间切分 0.069）。
- 投纯 LLM/RAG 岗时把本项目放在“机器学习系统”位置，不要改写成大模型项目。
- 所有数字都应能指向 `docs/PROJECT_EVIDENCE.md` 中的文件证据。
