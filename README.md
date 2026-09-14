# 视频 / 内容推荐系统

面向内容分发的推荐系统工程实践，当前使用 **MovieLens 电影评分数据**实现和验证。
用户选择一个身份后，系统根据画像与历史偏好返回相关内容，在同一网页完成推荐、展示与反馈。

**主线：用户偏好 → 向量推荐 → 页面/API → 曝光与反馈 → 评估改进。**

在“搜索、广告、推荐”中，本仓库已实现的是推荐部分。视频关键词搜索和广告投放属于后续方向。

## 先抓住三个重点

- **推荐内容**：双塔模型与 Faiss 检索，过滤已评分项后返回 Top-K。
- **交付应用**：网页通过 HTTP API 调用推荐核心，配有反馈记录、限流和监控。
- **验证改进**：比较最终推荐效果，保留有效基线和可复现的实验。

当前默认采用双塔直接推荐；DeepFM 精排、序列模型和强化学习保留为研究与对照材料。

## 先跑 Demo

进入 `recsys-learning` 仓库根目录，在已有项目环境中执行：

```powershell
.venv/Scripts/python.exe serve.py --preload
```

打开 [本地 Demo](http://127.0.0.1:8000)，选择用户 1，点击“生成推荐”，再对一部电影点“感兴趣”。
页面默认展示 5 条；只有进入视口的内容才记录曝光，反馈失败后可以安全重试。
新克隆需要准备 Python 3.12、MovieLens 数据和模型权重，步骤见 [完整运行指南](docs/MOVIELENS_GUIDE.md)。

## 从这两页开始

1. [15 分钟项目导学](项目导学.md)：看懂页面、模型、API 和反馈的关系。
2. [简版面试手册](docs/INTERVIEW_PLAYBOOK.md)：按 AI 全栈应用岗准备项目介绍和技术追问。

<details>
<summary>按需要查阅</summary>

- 容器启动：准备数据和权重后运行 `docker compose up --build -d`，访问同一网页。
  [生产运行手册](docs/PRODUCTION_SERVING.md) · [本轮交付验证](docs/APPLICATION_DELIVERY.md)。
- 原 Streamlit 展示与反馈实验台继续保留，入口见 [项目导学](项目导学.md)。
- [八股复习地图](docs/INTERVIEW_STUDY_MAP.md) · [指标证据](docs/PROJECT_EVIDENCE.md)。
- [反馈流程](docs/FEEDBACK_LOOP.md) · [服务运行说明](docs/PRODUCTION_SERVING.md)。
- [完整学习路线](docs/MOVIELENS_LEARNING_PATH.md) · [进阶面试资料](docs/INTERVIEW_DEEP_DIVE.md)。
- [企业知识种子扩展](docs/ENTERPRISE_PILOT_GUIDE.md)：保留另一种业务场景的迁移实验。

</details>
