# TIGER 语义 ID 生成式召回

## 1. 实现边界

[TIGER（Rajput et al., NeurIPS 2023）](https://arxiv.org/abs/2305.05065)把推荐
召回改写为生成任务：先把物品内容压缩成多级 Semantic ID，再让 Transformer 根据
用户历史自回归生成下一物品 ID。本项目实现的是 **TIGER-lite 受控基线**，不是逐配置
复现：MovieLens-1M 只有短标题和类型，因而用 deterministic hashed
title/genre/decade 向量与 residual K-Means，替代论文的 SentenceT5 与 learned
RQ-VAE。保留下来的核心机制是：

- 分层残差量化，每一级只编码上一级未解释的 residual；
- collision ordinal 作为最后 token，保证一个 ID 只映射一个物品；
- Transformer encoder-decoder 按 token 自回归分解物品概率；
- 受 catalog Semantic-ID 前缀约束，对全部未看物品做精确概率排序。

## 2. Semantic ID 诊断

低容量 `16×16×16` 冒烟只形成 631 个基础元组，最大碰撞 243，最后 token 几乎
退化成原子物品 ID，因此被否决。正式 `64×64×64` 索引得到：

- 3,706 个物品、2,754 个基础元组；
- 433 个元组发生碰撞，最大碰撞 15；
- 追加 collision token 后 3,706 个最终 ID 全部唯一；
- 三级 residual MSE：0.003161 → 0.002196 → 0.001432。

这说明量化器结构有效，但“ID 唯一”只解决可寻址性，不保证内容 token 与用户交互
目标一致。

## 3. 防止测试集调参

早期 1-seed pilot 显示更长训练会降低外层 Recall、提高 coverage。不能据此直接挑
epoch，因为那等于看测试答案。正式协议在每个 P1 外层训练窗内部再切最后 10% 时间片：

1. 只在更早的 development 交互训练 1–5 epoch；
2. 用内层未来的 generative loss 选 epoch；
3. 从相同 seed 在完整外层训练窗重训选定轮数；
4. 只在外层后续 5% 时间窗评估一次。

所有 9 个 run 的内层验证均选择 epoch 5。候选集仍是全部 3,706 个物品减完整已看
历史；计算每个唯一 Semantic ID 的精确自回归 log-probability，不用 sampled negatives
或 beam pruning。

## 4. 同协议结果

| 模型 | W1 | W2 | W3 | Recall@10 宏平均 | 平均 coverage@10 |
|---|---:|---:|---:|---:|---:|
| TIGER-lite（3 seed） | 0.0749 | 0.0463 | 0.0460 | **0.0557** | 2.75% |
| Popular | 0.0963 | 0.0675 | 0.0516 | 0.0718 | 2.13% |
| Item-kNN | 0.0917 | 0.0516 | 0.0640 | 0.0691 | 5.68% |
| TwoTower（3 seed） | 0.0734 | 0.0675 | 0.0658 | 0.0689 | 21.79% |

TIGER-lite 的宏平均 NDCG@10 / MRR@10 为 0.0289 / 0.0208，新颖度 8.42 bits，
long-tail share 近零。它只比 Popular 多一点覆盖，远低于 Item-kNN 和 TwoTower，也没有
相关性收益。W3 有 4 个训练期未出现的目标物品，三个 seed 都没有命中；样本太小，
既不能证明也不能系统否定 TIGER 的冷启动主张。

## 5. 结论与下一步

本阶段证明了生成式召回链路可以在统一协议下被真实比较，也得到一个有约束的负结果：
在 ML-1M 的稀疏内容上，Semantic ID 的内容邻近性不足以替代协同信号，生成概率还会
沿热门 token 路径集中。当前模型不晋升。

若继续研究，优先级应是含丰富文本且有足够冷物品的 Amazon/Steam 数据，复现
SentenceT5 + RQ-VAE，并分别报告 warm/cold；也可做 dense + generative hybrid。
不建议继续在同一 ML-1M 外层测试上调 codebook、epoch 或模型宽度。

正式证据：`experiments/generative_retrieval_p2-tiger-lite-rolling-3seed-v1.json`。
所有 smoke/pilot 只保留作失败与协议演进审计，不得替代正式结果。
