# 简历项目描述

## 推荐标题

**生产导向推荐系统：双塔召回、DeepFM 精排与真实反馈决策闭环**

技术栈：Python、PyTorch、Faiss、Starlette/Uvicorn、Streamlit、SQLite/WAL、
Prometheus、NumPy/pandas、unittest

## 中文精简版

- 独立构建“**双塔 + Faiss 召回 → DeepFM 精排**”两阶段推荐系统；统一
  leave-one-out/负采样评估并修复采样先验偏差，Recall@10 达 **0.097**，
  较 MF 基线 **提升 56%**，DeepFM 测试 AUC **0.754**。
- 将推理核心封装为 Streamlit/Starlette 同源服务，完成多 worker、准入保护、
  令牌桶限流、Prometheus、请求追踪与优雅停机；本机 2 worker、并发 16 下
  成功吞吐约 **270 RPS**，HTTP 200 p99 约 **72 ms**。
- 建立候选池、曝光、propensity 和真人反馈日志，使用 IPS/SNIPS、ESS 与
  cluster bootstrap 比较策略；结合个人试验和 **1,834 人**外部问卷，作出
  “保留个人多样性候选、不做全局上线”的可解释决策。
- 参考 DeepEyes 主动感知范式实现带调用成本的 Agentic 推荐工具门控及
  group-relative REINFORCE；300 个仿真会话中，规则门控在保持奖励 **17.947**
  的同时减少 **12.8%** 调用，并根据消融拒绝上线退化的学习策略。

## 中文一行版

独立实现可运行的两阶段推荐系统与反馈 OPE 闭环，覆盖双塔/Faiss 召回、
DeepFM 精排、生产服务治理和真实策略决策；Recall@10 较 MF 提升 56%。

## English version

**Production-oriented Recommender System — Retrieval, Ranking, and Feedback OPE**

- Built an end-to-end two-stage recommender with two-tower/Faiss retrieval and
  DeepFM ranking; aligned leave-one-out evaluation and corrected sampling-prior
  bias, reaching **0.097 Recall@10 (+56% vs. MF)** and **0.754 test AUC**.
- Productionized a shared inference core behind Streamlit and Starlette,
  including multi-worker serving, admission control, token-bucket rate limiting,
  Prometheus metrics, request tracing, and graceful shutdown; measured
  **~270 successful RPS and ~72 ms HTTP-200 p99** at two workers/concurrency 16
  on a local 12-logical-CPU machine.
- Implemented candidate, impression, propensity, and real-feedback logging with
  IPS/SNIPS, ESS, and cluster-bootstrap policy gates; combined a personal pilot
  with a **1,834-user** external survey to reject an unsupported global rollout
  and retain diversity reranking as a personalized candidate.
- Built a DeepEyes-inspired, cost-aware agentic tool gate with group-relative
  REINFORCE; an interpretable gain gate preserved **17.947 simulator reward**
  while saving **12.8% of tool calls**, and the learned gate was rejected after
  reward ablation exposed deterministic always-tool collapse.

## 关键词

推荐系统、召回、排序、双塔、Embedding、Faiss、DeepFM、负采样、logit correction、
SASRec、Bandit、offline RL、Agentic RL、Tool Use、OPE、IPS、SNIPS、ESS、A/B Testing、Model Serving、
Observability、Prometheus、Rate Limiting、Load Testing

## 使用说明

- 校招/实习简历优先放三条完整 bullet。
- 社招或项目较多时保留第一条与第三条。
- 投纯 LLM/RAG 岗时把本项目放在“机器学习系统”位置，不要改写成大模型项目。
- 所有数字都应能指向 `docs/PROJECT_EVIDENCE.md` 中的文件证据。
