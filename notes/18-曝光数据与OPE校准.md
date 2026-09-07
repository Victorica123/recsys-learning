# 曝光数据与 OPE 校准

## 1. 这一步解决什么问题

MovieLens 评分不是曝光日志，不能把 DeepFM 的 `rating>=4` 分类 AUC 当 CTR，也不能
从确定性 Top-K 日志中随意评估没有被展示过的策略。本阶段把“有个反馈表”升级为可审计
的候选级曝光数据集，并让离线策略评估先在已知真值上接受校准。

## 2. 数据协议

一次推荐保存完整候选池、实际展示集合、rank、score、`policy_name`、
`model_version` 和物品被纳入 slate 的真实边际 propensity。客户端随后确认 impression，
只有已曝光物品才能写 reward。schema v2 导出会检查：

- 同一推荐内 movie/rank 唯一；
- propensity 位于 `(0, 1]`；
- 未曝光候选没有 reward，已有 reward 的候选必须曝光；
- 不同 policy/model 队列不能静默混合。

当前 epsilon-slate 行为策略以 `1-epsilon` 选择 scorer Top-K，以 `epsilon` 从 pool
均匀无放回选 K 个。因此 Top-K 物品边际概率为
`1-epsilon + epsilon*K/pool_size`，其余物品为 `epsilon*K/pool_size`。这里记录的是
item inclusion propensity，不是整个有序 slate 的联合概率。

## 3. 三个估计器与决策角色

- IPS：用 `observed_reward / propensity` 还原目标策略的 item-level 期望奖励。
- SNIPS：再按权重和归一化，通常降方差，但会引入有限样本偏差。
- DR：结果模型的直接预测，加上已曝光行的 propensity 加权残差修正。

DR 的 nuisance model 按 recommendation 做交叉拟合，只用曝光行学习 reward，给所有
候选产生 out-of-fold 预测，避免同一条结果既训练又评估。但“双重稳健”不等于“任何
时候方差和误差都更小”。项目保留 IPS 作为预注册正式决策门，DR 只作交叉检查；两者
方向冲突时必须停止解释。

## 4. 已知真值校准

`scripts/validate_ope_estimators.py` 为每个 actor/movie 构造固定的反事实点击概率，
独立计算目标策略 oracle 值，再根据 epsilon-slate 只采样实际曝光和 Bernoulli reward。
固定协议为 5,000 次推荐、10 候选、展示 3 个、epsilon=0.30、20 actors、seed=42。

| 指标 | Oracle / 估计值 | 策略增量绝对误差 |
|---|---:|---:|
| Oracle delta | 0.026839 | — |
| IPS delta | 0.026564 | **0.000275** |
| SNIPS delta | 0.018249 | 0.008589 |
| DR delta | 0.018721 | 0.008117 |

本次精确 propensity 和较大样本让 IPS 很准；302 维线性 ridge 结果模型对合成响应面
仍有失配，因此 DR 没有胜出。这是有意义的负面校准：不能因为估计器名字更先进就
默认相信它。

证据：`experiments/ope_synthetic_p2-ope-calibration-5000-v2.jsonl` 与相邻 report。

## 5. 真实历史队列审计

数据库中存在多个旧队列，未指定过滤时回放脚本已验证会拒绝运行。可支持 OPE 的个人
探索队列只有 21 次推荐、1 位 actor、105 个曝光 item，候选策略 ESS 17.6：

- IPS delta：+0.0212；
- DR delta：-0.0489；
- 正式门：`collect_more_data`。

单 actor 会让 cluster bootstrap 退化成常数区间；估计器又方向冲突。这份历史
DeepFM-v1 数据只能作为数据与评估链路审计，不能代表当前 retrieval-v2，也不能作为
上线证据。

## 6. 下一步采集准入

要形成可解释的当前策略证据，必须重新用 `two-tower-retrieval-ml1m-v2` 和唯一的
retrieval-v2 policy cohort 显式探索，至少满足 200 次完整推荐、5 位匿名体验者、
95% impression coverage、两个目标策略 ESS 均不低于 100，且无零支持物品。即使过门，
仍需 IPS 配对区间、DR 方向和业务护栏共同检查，再进入 shadow/A-B，而非直接全量。
