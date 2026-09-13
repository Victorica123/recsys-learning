# 视频 / 内容推荐系统

面向内容分发的推荐系统工程实践，当前使用 **MovieLens 电影评分数据**实现和验证。
用户选择一个身份后，系统根据画像与历史偏好返回相关内容，并提供网页、API 和反馈实验台。

**主线：用户偏好 → 向量推荐 → 页面/API → 曝光与反馈 → 评估改进。**

在“搜索、广告、推荐”中，本仓库已实现的是推荐部分。视频关键词搜索和广告投放属于后续方向。

## 先抓住三个重点

- **推荐内容**：双塔模型与 Faiss 检索，过滤已评分项后返回 Top-K。
- **交付应用**：页面与 API 复用推荐核心，配有反馈记录、限流和监控。
- **验证改进**：比较最终推荐效果，保留有效基线和可复现的实验。

当前默认采用双塔直接推荐；DeepFM 精排、序列模型和强化学习保留为研究与对照材料。

## 先跑 Demo

进入 `recsys-learning` 仓库根目录，在已有项目环境中执行：

```powershell
.venv/Scripts/python.exe -m streamlit run app.py
```

打开 [本地 Demo](http://localhost:8501)，选择用户，看历史偏好与 Top-10 推荐。
新克隆需要准备 Python 3.12、MovieLens 数据和模型权重，步骤见 [完整运行指南](docs/MOVIELENS_GUIDE.md)。

## 从这两页开始

1. [15 分钟项目导学](项目导学.md)：看懂页面、模型、API 和反馈的关系。
2. [简版面试手册](docs/INTERVIEW_PLAYBOOK.md)：按 AI 全栈应用岗准备项目介绍和技术追问。

<details>
<summary>按需要查阅</summary>

- [八股复习地图](docs/INTERVIEW_STUDY_MAP.md) · [指标证据](docs/PROJECT_EVIDENCE.md)。
- [反馈流程](docs/FEEDBACK_LOOP.md) · [服务运行说明](docs/PRODUCTION_SERVING.md)。
- [完整学习路线](docs/MOVIELENS_LEARNING_PATH.md) · [进阶面试资料](docs/INTERVIEW_DEEP_DIVE.md)。
- [企业知识种子扩展](docs/ENTERPRISE_PILOT_GUIDE.md)：保留另一种业务场景的迁移实验。

</details>
