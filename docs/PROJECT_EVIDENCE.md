# 项目指标与证据索引

面试中只使用下面可复查的数字。不同数据集和协议的结果不得直接横向比较。

| 结论 | 数值 | 协议/边界 | 证据 |
|---|---:|---|---|
| MF 基线 | Recall@10 0.062 | ML-1M leave-one-out | `REPORT.md` |
| DeepFM | AUC 0.754 | ML-1M CTR 时间切分分类 | `experiments/deepfm_log.csv` |
| 双塔召回 | Recall@10 0.097 | ML-1M 校正评估 | `figures/08-双塔召回对比.png` |
| SASRec | HR@10 0.7909 / NDCG@10 0.5428 | ML-1M 序列留一 | `research/研究报告.md` |
| 策略梯度 | 会话奖励 15.86 | 冻结 SASRec 仿真器 | `experiments/pg_eval.csv` |
| 2 worker 服务 | 270.1 RPS / p99 72.4 ms | 本机并发 16、600 请求 | `experiments/serve_scaling_verified-20260726.json` |
| 4 worker 服务 | 385.5 RPS / p99 51.9 ms | 本机并发 16、600 请求 | 同上 |
| 生产冒烟 | 13/13 通过 | 两 worker 真实 checkpoint | `experiments/serve_production_smoke_final-v2-20260726.json` |
| 个人反馈 | 多样性 IPS 方向 +11.1% | 单人 20 批，只是个人信号 | `runtime/feedback/events.sqlite3` |
| 外部问卷 | 高多样性个性化感 -0.332/5 | 176 vs 208，Holm p=0.0144 | `experiments/external_survey_personality-2018-diversity-v1-20260727.json` |
| Agentic 规则门控 | 用户奖励 17.947 / 净奖励 16.203 | ML-1M 冻结 SASRec 仿真，300 会话；工具成本 0.10 | `experiments/agentic_rec_deepeyes-minloop-v2-20260727_report.json` |
| 自动化测试 | 73 项发现，71 通过、2 跳过 | CPU-only；集成测试需显式开启 | `scripts/ai_startup_harness.py --test` |

## 可展示图表

- `figures/08-双塔召回对比.png`：评估修正前后。
- `figures/12-DQN稳定性对比.png`：失败诊断与修复。
- `figures/13-离线RL后训练对比.png`：SFT 天花板与离线 RL。
- `figures/serve_worker_scaling_verified-20260726.png`：吞吐、p99 与内存权衡。
- `notes/13-Agentic推荐-主动工具调用.md`：工具门控、奖励消融与不上线决策。

## 不能合并的指标

- AUC 与 Recall@K 不同。
- ML-1M 与 ML-25M 不同。
- 真实点击、问卷 Likert 分和仿真器奖励不同。
- 本机客户端 p99 与服务中间件耗时不同；SLO 应优先引用客户端数据。
