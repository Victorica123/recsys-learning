# -*- coding: utf-8 -*-
"""阶段 2：推荐系统"后训练"——离线 RL + SFT 微调。

对齐大厂 LLM 后训练范式（预训练 → SFT → RLHF），落在推荐的真实业务约束上：
线上系统**不能自由探索**，只能用固定的历史曝光/反馈**日志**离线训练。

- SFT ≈ 行为克隆（监督式模仿日志中的"好"示范）
- RLHF/RL ≈ 离线 RL（在固定数据上做策略提升，且必须对未见动作保持保守）

模块：
- dataset.py           固化"生产日志"为离线数据集（含 return-to-go）
- behavior_cloning.py  SFT：行为克隆 + 奖励加权行为克隆
- cql.py               离线 RL：保守 Q 学习（CQL）+ naive 离线 DQN 对照
- train.py             编排：采日志 → SFT → 离线 RL → 仿真器评估
"""
