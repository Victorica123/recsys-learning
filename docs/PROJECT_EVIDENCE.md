# 项目指标与证据索引

面试中只使用下面可复查的数字。不同数据集和协议的结果不得直接横向比较。

| 结论 | 数值 | 协议/边界 | 证据 |
|---|---:|---|---|
| MF 基线 | Recall@10 0.062 | ML-1M leave-one-out | `REPORT.md` |
| DeepFM | AUC 0.754 | ML-1M 已评分样本中 `rating>=4` 的时间切分分类；**非 CTR** | `experiments/deepfm_log.csv` |
| DeepFM | **user-level GAUC 0.735** | 同上；1108 用户参与、91.6% 覆盖 | `src/train_deepfm.py --model deepfm` |
| 双塔召回 | Recall@10 0.098 | ML-1M 校正评估，**leave-one-out（有时间穿越）** | `figures/08-双塔召回对比.png` |
| **端到端 召回50→精排10** | **Recall@10 0.065** | 线上真实链路，3552 用户 leave-one-out | `experiments/eval_end_to_end_v0-ci.json` |
| **精排增量（原版 v0）** | **−0.0329，CI [−0.0445, −0.0220]** | 配对 bootstrap，**显著劣于不精排** | 同上 |
| **精排增量（重训 v6，最优）** | **−0.0037，CI [−0.0132, +0.0053]** | 区间跨 0 = **与不精排持平，未晋升**；缺口补回 88.7% | `experiments/eval_end_to_end_rerank-v6.json` |
| **当前默认服务策略** | **双塔 Top-K，不加载精排器** | `ranking_policy=retrieval`；DeepFM/v6 只可显式实验启用 | `src/serving.py`, `tests/test_serving.py` |
| **残差门学到了什么** | `w_retrieval` 19.65 / `gate` **−1.22**（6.2%） | 模型自学结论="基本照抄召回"，且把 DeepFM 分量当**惩罚项** | `checkpoints/deepfm_rerank_v6.pt` |
| **双塔（全局时间切分）** | **Recall@10 0.0691 [0.0553, 0.0847]** | 无时间穿越；1157 测试用户、31 冷启动剔除 | `experiments/eval_split_protocols_protocol-v1.json` |
| **协议乐观偏差** | 时间切分 = LOO 的 **70.6%**，区间**不重叠** | 差值混合了"消除穿越"与"数据更少/任务更难"，不可拆 | 同上 |
| **滚动全库召回** | Popular / Item-kNN / TwoTower 三窗 Recall 宏平均 **0.0718 / 0.0691 / 0.0689** | 全局时间 3 窗、未来 5%、无采样负例；双塔 3 seed × 10 epoch | `experiments/retrieval_benchmark_p1-rolling-fullcatalog-3seed-v2.json` |
| **双塔分布收益** | coverage@10 **17%–30%** vs Popular **1.5%–2.6%** | 相关性未胜，但覆盖/新颖度/最差窗口更稳 | 同上；`notes/16-滚动时间全库召回基准.md` |
| **三路 RRF** | 三窗 Recall 宏平均 **0.0787**；coverage **3.5%–5.0%** | TwoTower + Item-kNN + Popular，3 seed；相关性方向提升但覆盖下降，未晋升 | `experiments/multiroute_ranker_p1-multiroute-seqrank-3seed-v1.json` |
| **序列残差 listwise** | W2/W3 配对均值 **+0.00047**，范围 −0.00337~+0.00397 | 只用历史窗口训练；无稳定增益；Candidate Recall@200 仅 44%–51% | 同上；`notes/17-多路召回与序列残差精排.md` |
| **TIGER-lite 生成召回** | 三窗 Recall@10 **0.0557**；coverage **2.75%** | 3 seed、全库精确 Semantic-ID 生成、嵌套时间选 epoch；低于三类强基线，不晋升 | `experiments/generative_retrieval_p2-tiger-lite-rolling-3seed-v1.json`；`notes/19-TIGER语义ID生成式召回.md` |
| 曝光集中度 | 去重物品 773 → 2308 → 2011 → 1437 | v0 / v1 / v5 / v6，Top-10；双塔本身 1853 | `experiments/eval_end_to_end_*.json` |
| SASRec | HR@10 0.7909 / NDCG@10 0.5428 | ML-1M 序列留一 | `research/研究报告.md` |
| 策略梯度 | 会话奖励 15.86 | 冻结 SASRec 仿真器 | `experiments/pg_eval.csv` |
| 2 worker 服务 | 270.1 RPS / p99 72.4 ms | 本机并发 16、600 请求 | `experiments/serve_scaling_verified-20260726.json` |
| 4 worker 服务 | 385.5 RPS / p99 51.9 ms | 本机并发 16、600 请求 | 同上 |
| **推荐核心延迟** | **4.69 → 1.66 ms（2.8×）→ 当前默认 0.19 ms** | 去 pandas 热路径 + 默认不再精排 | `experiments/serve_load_verify-c1-postopt.json`；2026-08-16 Linux 复测 |
| **HTTP 端到端延迟** | **5.42 ms / p95 9.67 ms**（默认双塔 Top-K；SQLite@/tmp、c1×200） | 旧 Windows DeepFM 路径 4.6 ms 仅历史参考 | `experiments/serve_load_default-retrieval-linux-c1-tmpdb-20260815.json` |
| 生产冒烟 | 13/13 通过 | 两 worker 真实 checkpoint（历史 DeepFM 路径；当前默认双塔路径待 Windows 复跑） | `experiments/serve_production_smoke_final-v2-20260726.json` |
| 个人反馈 | 多样性 IPS 方向 +11.1% | 单人 20 批，只是个人信号 | `runtime/feedback/events.sqlite3` |
| **OPE 合成校准** | oracle 增量 0.02684；IPS / SNIPS / DR 增量误差 **0.00028 / 0.00859 / 0.00812** | 5,000 推荐、精确 epsilon-slate propensity；结果模型失配时 DR 不保证更优 | `experiments/ope_synthetic_p2-ope-calibration-5000-v2_report.json` |
| **历史探索队列审计** | IPS +0.0212，DR -0.0489；`collect_more_data` | 仅 21 推荐/1 actor/ESS 17.6；历史 DeepFM v1，不能代表当前双塔 v2 | `experiments/feedback_p2-personal-pilot-v1-dr-audit_ope.json` |
| 外部问卷 | 高多样性个性化感 -0.332/5 | 176 vs 208，Holm p=0.0144 | `experiments/external_survey_personality-2018-diversity-v1-20260727.json` |
| Agentic 规则门控 | 用户奖励 17.947 / 净奖励 16.203 | ML-1M 冻结 SASRec 仿真，300 会话；工具成本 0.10 | `experiments/agentic_rec_deepeyes-minloop-v2-20260727_report.json` |
| **Agentic 成本相变候选** | 3-seed 均值：cost≤0.3 规则更高；cost=0.8 时 RL 高 +1.92 | 探索性 6 成本 × 2 算法 × 3 seed；不足以宣称显著，正式复验需 ≥10 seed | `experiments/agentic_cost_sweep_cost-phase-v1.json`（旧产物中的区间仅作描述） |
| **Agentic 多工具正式对照** | 规则 15.371 vs learned 10.725；配对差 **-4.646 [-4.974, -4.322]** | 10 训练 seed × 120 同 seed 会话；3 工具、预算/非法动作；仿真证据，规则正式胜出 | `experiments/agentic_multitool_p2-multitool-10seed-v1.json`；`notes/20-Agentic多工具预算轨迹.md` |
| 自动化测试 | 以 `--test` 实际输出为准 | CPU-only；集成测试需显式开启 | `scripts/ai_startup_harness.py --test` |
| **权重供应链** | `weights_only=True` 强制执行；开启 `--verify-checkpoint-hashes` 后 manifest SHA-256 不匹配启动失败 | 未登记实验权重仍禁 pickle 白名单外对象；`--release` 为离线硬校验 | `src/checkpoint_io.py`、`tests/test_checkpoint_io.py` |

## 可展示图表

- `figures/08-双塔召回对比.png`：评估修正前后。
- `figures/12-DQN稳定性对比.png`：失败诊断与修复。
- `figures/13-离线RL后训练对比.png`：SFT 天花板与离线 RL。
- `figures/serve_worker_scaling_verified-20260726.png`：吞吐、p99 与内存权衡。
- `notes/13-Agentic推荐-主动工具调用.md`：工具门控、奖励消融与不上线决策。
- `notes/15-端到端评估与精排重建.md`：**端到端负增量的发现、诊断与六版修复**
  （最推荐讲的一条线，含一个被自己实验否定的假设、以及"门控自己学出照抄召回"
  这个决定性诊断）。

## 不能合并的指标

- AUC 与 Recall@K 不同。
- **分段指标与端到端指标不同**：双塔 Recall@10 0.098 与 DeepFM AUC 0.754
  都不代表链路质量；线上真实链路是 0.065（原版）/ 0.094（v6，最优）。
- **leave-one-out 与全局时间切分不同**：0.098 vs 0.069，区间不重叠。引用召回
  指标时必须带协议名，否则等于挑了好看的那个。
- ML-1M 与 ML-25M 不同。
- 真实点击、问卷 Likert 分和仿真器奖励不同。
- 本机客户端 p99 与服务中间件耗时不同；SLO 应优先引用客户端数据。
- **推荐核心延迟与 HTTP 端到端延迟不同**：当前默认路径核心 0.19 ms；WSL/Linux
  HTTP 5.42 ms（SQLite@/tmp），差额主要是同步写曝光日志与 HTTP/JSON 开销。
  SQLite 放在仓库所在的 `/mnt/d` 时 WSL 跨文件系统写放大到约 30 ms，不能当作服务退化。

## 均值向好但不能宣称收益的两处

本项目统一门槛：**配对 bootstrap 95% 区间不含 0 才允许宣称收益 / 推全量。**

- 精排 v6 相对不精排：均值 −0.0037，区间 [−0.0132, +0.0053] **跨 0**
  → 表述为"与不精排持平"，不得表述为"接近打平"、"基本追上"或"补回 88.7%
  所以快成了"。缺口收窄是过程描述，不是效果结论。
- Agentic 成本扫描只有 3 个独立 seed：可报告均值方向和候选相变，**不得把旧
  bootstrap 区间当正式 CI，也不得使用“显著胜/负”**。正式推断需先扩到 ≥10 seed。
- 多工具实验已达到 10-seed 门槛，可报告“当前 masked trajectory-level GRPO 低于
  预算规则门”；但它使用新的隐藏偏好、三种工具成本和会话预算，不能反推旧二元成本
  相变，也不能外推线上用户。
- 滚动召回 v2：三种模型 Recall 宏平均接近，**不得说双塔击败强基线**；可以说
  双塔大幅扩大 catalog coverage，并提高跨窗口最差表现。v1 Item-kNN 数字无效。
- 三路 RRF 的 0.0787 是探索性均值且 coverage 明显下降；序列 ranker 只有
  +0.00047 的混合方向。两者都不得表述为已晋升或稳定收益。
- 历史真人 OPE 只有 1 位 actor，cluster bootstrap 区间退化，且 IPS/DR 方向冲突；
  不得引用 +0.0212 作为策略增益，也不得把该 DeepFM v1 队列归因于当前双塔 v2。
- TIGER-lite 是保留核心机制的受控基线，不是 SentenceT5 + RQ-VAE 的论文复现；
  0.0557 低于强基线，且 4 个冷目标均未命中，不得宣称生成式召回或冷启动收益。
