# 企业知识种子扩展：15 分钟演练

本页保留企业知识场景的扩展练习。项目主线是视频/内容推荐，首次上手见 [项目导学](../项目导学.md)。

读完这页，目标是能跑出一次结果，并讲清楚“输入是什么、怎样挑资料、输出是什么”。

## 1. 先看一个已经跑出的例子

种子数据里，一位安全审核用户想找“统一身份与 OIDC”的可信知识和历史方案。
OIDC 可以先理解成企业登录相关的一类技术，了解这个业务背景就够了。

系统检查这位用户所属的公司、工作区和角色，以及文档是否仍有效，得到 31 份可用资料。
接着按相关程度等因素排序，展示前 5 份。这个样例的前三项是：

| 顺序 | 推荐资料 | 能帮用户做什么 |
|---|---|---|
| 1 | 知识库：统一身份与 OIDC实施手册 v2 | 查具体做法 |
| 2 | 事故复盘：统一身份与 OIDC链路异常 | 了解过去出过什么问题 |
| 3 | PRD：统一身份与 OIDC企业版改造 | 找历史需求和方案 |

这些标题来自固定种子的实际输出。按下一步运行后，可在
`experiments/enterprise_pilot_my-first-run/agent_service_example.json` 查看请求与结果。
该文件是本地运行产物，新设备按同一命令即可重新生成。

整个过程可以记成：

```mermaid
flowchart LR
    A[当前想找的资料] --> B[筛选有权限且有效的文档]
    B --> C[把相关资料排在前面]
    C --> D[记录展示与后续反馈]
```

随机种子像保存了一套练习题：每次都能生成同一批公司、用户、文档和行为，方便重复运行、比较方案和排查问题。
本地代码会实际执行校验、排序和反馈入库；这批数据用于验证企业场景的处理流程。

## 2. 自己跑一遍

在仓库根目录打开 PowerShell，使用已有的项目虚拟环境：

```powershell
.venv/Scripts/python.exe scripts/enterprise_pilot.py --tag my-first-run
```

如果已经运行过这条命令，可以直接看输出。再次运行时换成 `my-first-run-2`，原有结果会保留。
新机器需要 Python 3.12；尚无虚拟环境时先执行 `py -3.12 -m venv .venv`。
这条企业演练命令使用 Python 标准库，无需训练或下载模型。

看到 `PASS: enterprise synthetic pilot` 表示运行成功。
后面的 `not_promoted_synthetic_only` 是试点状态，表示场景演练完成，暂不发布到生产环境。

结果在 `experiments/enterprise_pilot_my-first-run/`，先打开两个文件：

| 文件 | 先看哪里 |
|---|---|
| `agent_service_example.json` | `request.context.intent` 是当前需求；`response.candidates` 里看标题和 `served_rank`，有展示序号的就是展示结果 |
| `report.md` | “同请求、同候选池的测试结果”：三种方案在同一套题上的表现 |

样例文件演示从该用户全部可用资料中排序；报告比较的是种子中已经标注的候选资料。
两者用途不同，报告里的分数用于描述这套模拟数据上的排序表现。

## 3. 先理解三个实现点

**先看权限。** 文档再相关，也要先检查用户能不能看。已经失效的旧版本也要排除。

**再看相关性。** 系统先比较当前需求与文档内容，再参考新鲜度、历史热度等信息。
BM25 是这里使用的文本匹配方法，可以先理解为“给词语匹配程度打分”。

**最后留记录。** 一次请求推荐了什么、展示了什么、后来有没有打开或引用，都能对应起来。
同一批反馈重复导入，不会重复计数；记录有冲突时会报错。

读代码也按这个顺序找，第一遍只看对应入口：

| 想理解的事情 | 文件与入口 |
|---|---|
| 一次演练怎样串起来 | [enterprise_pilot.py](../scripts/enterprise_pilot.py) 的 `run()` |
| 怎样筛选和排序 | [enterprise_recommendation.py](../src/enterprise_recommendation.py) 的 `KnowledgeRanker.recommend()` |
| 怎样保存和重放反馈 | [enterprise_feedback.py](../src/enterprise_feedback.py) 的 `import_dataset()` |

## 4. 用一分钟检查是否看懂

试着用自己的话说明三件事：当前用户想找什么；为什么有些资料不能进入推荐；重复导入反馈为什么不能重复计数。

本扩展用种子数据完成企业场景开发和演示。真实企业平台接入、真实用户效果验证是后续阶段。
这条扩展的面试参考见 [企业知识扩展问答](ENTERPRISE_INTERVIEW_NOTES.md)。

需要学习电影推荐模型时，进入 [MovieLens 完整学习路线](MOVIELENS_LEARNING_PATH.md)；
需要接入业务代码时，再查 [企业接入说明](ENTERPRISE_RECOMMENDATION.md)。
