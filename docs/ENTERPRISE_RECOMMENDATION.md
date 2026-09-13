# 企业知识推荐：离线闭环与接入契约

当前交付面是 `recsys-learning` 的企业数据处理、离线评估、通用反馈回放和可复用排序函数。
目标场景是用户阅读分析结果后，找到当前工作区内有权限、仍有效的知识、历史 PRD 和会议资料。
在线入口使用 Enterprise Agent Service；本次未修改企业平台仓库，也未增加在线后端。

## 一键复现

从仓库根目录、Python 3.12 环境运行：

```powershell
# 自动复现原种子，再校验、导入、评估与导出
.venv/Scripts/python.exe scripts/enterprise_pilot.py --tag enterprise-demo-v1

# 已有网页端原包时，只读取其 data/，不执行包内脚本
.venv/Scripts/python.exe scripts/enterprise_pilot.py --dataset data/enterprise_seed_original/data --tag imported-demo-v1

# 只生成数据：必须指定不存在的输出目录
.venv/Scripts/python.exe src/enterprise_seed.py --output data/enterprise_seed_v1 --seed 20260913 --requests 640
```

所有新目录和实验 tag 拒绝覆盖。网页端原生成器会递归删除输出目录；本仓库已移除该行为。
种子主题、文案、采样与标签规则保留，默认种子生成的七个 JSONL 文件逐个通过原包 SHA-256 对照。
来源、原始 manifest 和公开规范链接保存在
[enterprise_seed_provenance.json](../artifacts/enterprise_seed_provenance.json)。

`--requests 40` 可运行最小冒烟；正式参考使用原始 640 次请求。企业模块只依赖 Python 标准库。
历史 MovieLens 启动检查仍检查电影数据与权重；企业复现命令无需这些产物。

## 模块与职责

| 文件 | 职责 |
|---|---|
| [enterprise_seed.py](../src/enterprise_seed.py) | 可复现合成数据生成；oracle 仅在此生成标签和模拟行为 |
| [enterprise_dataset.py](../src/enterprise_dataset.py) | 源文件哈希、引用完整性、授权、版本、概率、事件时间与标签覆盖验证 |
| [enterprise_recommendation.py](../src/enterprise_recommendation.py) | 授权候选、文本排序、版本去重、多样性、降级和稳定实验分桶 |
| [enterprise_feedback.py](../src/enterprise_feedback.py) | 本地 SQLite 通用事件表、事务导入、幂等、作用域导出与留存原语 |
| [enterprise_evaluation.py](../src/enterprise_evaluation.py) | 同请求同候选池对照、历史特征截止、业务指标分组与晋升边界 |
| [enterprise_pilot.py](../scripts/enterprise_pilot.py) | 串联上述步骤并交付版本化策略和回放产物 |

## 数据契约

企业记录使用独立的 `schema_version=1`，不能与 MovieLens feedback schema-v2 混算。
标识统一使用字符串；物品标识由 `item_type + item_id + item_version` 组成，全部位于租户和工作区内。

| 范围 | 字段与约束 |
|---|---|
| 请求身份 | `recommendation_id`, `tenant_id`, `workspace_id`, `workspace_type`, `actor_id` |
| 当前上下文 | `context_type`, `context_id`, `context.intent`, `requested_at` |
| 可追溯版本 | `policy_version`, `model_version`, `feature_snapshot_version`, `synthetic` |
| 候选 | `item_type`, `item_id`, `item_version`, `candidate_rank`, `score`, `recommendation_reason` |
| 实际展示 | `served_rank`；未展示为 null；展示集合不得重复，排名必须连续 |
| 探索 | `epsilon`, `explored`, `behavior_propensity`；校验全部候选，包含未展示项 |
| 事件 | `event_id`, `event_type`, `occurred_at`, `served_rank`, `value`, `metadata`，以及请求的身份与版本字段 |
| 离线窗口 | `split`, `split_start`, `split_end`；请求时间须落在实际边界内 |

事件支持 `impression/open/dwell/cite/save/dismiss/create_action/complete_action`。
每个反馈必须指向同一主体、同一推荐批次中已经曝光的物品版本。完成行动项须带
`metadata.action_id`，并关联同一物品的先前创建事件。时间判断使用带时区的时间戳，不依赖文件行顺序。
相同 ID 与相同内容重放不新增记录；相同 ID 携带不同内容时拒绝。整批导入失败会回滚。

原始种子有两个需要明确的适配约定：

- 每个租户只有一个 team 工作区，成员与角色不随时间变化；缺省工作区由 `tenants.jsonl` 映射。
  显式提供但不匹配的工作区、类型和版本不会被静默改写。
- 源数据没有实际 PRD/context ID，因此导入生成 `synthetic-context:<recommendation_id>`。
  源 `open.value` 是合成停留秒数，并在 `metadata.unit` 标明；打开率按事件计数，不对该数值求和。

真实平台导出应直接使用真实版本和上下文 ID，并保留当时的授权、特征和事件快照，
不能从最终数据库状态倒推历史权限。

## 授权与排序

调用方必须从既有 JWT/用户/工作区成员关系解析可信 `Principal`。
`actor_id`、角色或租户不能直接采用前端传入值。角色匹配使用服务端映射的业务角色，
不把“匹配程度”当作授权权限。

顺序固定为：

1. 检查请求与可信主体的 tenant/workspace/actor/workspace_type 一致。
2. 检查对象角色权限；personal 工作区和 owner 可见对象同时检查 owner。
3. 以请求时间检查创建、撤销和替代边界；未知状态拒绝。
4. 去重并选择有效物品版本，然后才读取文本、计算 BM25 词频和归一化热度。
5. 生成 Top-K、推荐原因、实际策略版本和特征快照哈希。

历史回放中，今天已替代的旧版本在替代发生前仍可有效；边界时刻开始只允许新版本。
实际在线调用须传入服务端时间与最新授权快照；打开、引用和创建行动项时还需重新校验当前权限。

三种固定策略：

- `baseline_v0`：新鲜度 0.55、训练期热度 0.45。
- `semantic_v1`：BM25 文本相关性。中文用二元词组、英文用词项，没有神经向量模型。
- `hybrid_v1`：文本 0.70、新鲜度 0.15、热度 0.10、角色类型匹配 0.05，
  叠加 0.06 的文本重叠惩罚，以减少列表冗余。

参数在评估前固定。排序不读取合成 `topic`、用户隐藏偏好、`popularity_prior`、
原始候选分数/特征或真值标签。内容打分器返回不可用或非有限值时，降级到同一授权池的
`baseline_v0`，记录请求策略、实际策略与 `fallback_reason`；鉴权失败不会触发放宽权限的降级。

`experiment_bucket()` 按 tenant/workspace/actor/experiment_id 稳定分桶；现阶段只提供组件，
未在真实用户上启用实验。策略文件校验版本与权重；回滚时选择已验证的旧策略文件与相应代码版本。

## Agent Service 嵌入接口

产物目录中的 `enterprise_recommendation.py` 与 `policy.json` 可交付给 Agent Service。
以下是嵌入接口示例，`auth`、`workspace`、请求及文档快照由既有服务提供：

```python
from pathlib import Path
from enterprise_recommendation import KnowledgeRanker, Principal

ranker = KnowledgeRanker.from_artifact(Path(__file__).with_name("policy.json"))
principal = Principal(
    auth.tenant_id, workspace.id, auth.subject, auth.business_role, workspace.type
)
result = ranker.recommend(
    principal, request_snapshot, authorized_documents,
    policy="hybrid_v1", k=5, popularity=historical_popularity_snapshot,
)
```

`agent_service_example.json` 提供实际跑出的合成请求与响应。该示例对完整授权目录排序；
离线指标仅评估原包有标签的候选池，两者的范围在产物中明确区分。
真实集成需要复用企业平台现有数据库、授权和前端事件流程，不应公开一个依赖客户端主体信息的接口。

## 回放、指标与数据保留

`KnowledgeFeedbackStore` 是本地实验存储。在线业务事件应由企业平台写入 MySQL，
导出后再进入本项目。`replay(tenant_id, workspace_id)` 是显式工作区范围的离线导出操作，
调用者承担导出授权。

`delete_actor()` 和 `purge_before()` 只对指定租户/工作区执行本地删除，候选与反馈级联清理。
它们不是完整的企业数据删除服务；备份、SQLite WAL、法定留存和审计仍需平台处理。

业务指标按原始 `split/policy_version/model_version` 分组，打开、引用、收藏、驳回、
创建行动项按唯一“请求＋物品版本”计数。完成率按唯一 action ID 计数。
观测截止后事件保留在回放中，但不计入截止前的指标；延迟完成存在右删失。
新策略没有真实行为结果，不会凭离线排序结果生成“新策略打开率”。

## 验收与下一步

```powershell
.venv/Scripts/python.exe -m unittest tests.test_enterprise_recommendation tests.test_enterprise_dataset tests.test_enterprise_evaluation -v
.venv/Scripts/python.exe scripts/ai_startup_harness.py --check
.venv/Scripts/python.exe scripts/ai_startup_harness.py --test
```

企业侧验收覆盖源数据复现、越权与版本边界、未来事件隔离、概率一致性、事务回滚、
幂等冲突、主体删除、降级与同请求对照。详细实验结论见
[种子审计笔记](../notes/21-企业知识推荐-种子审计与业务迁移.md)。

后续企业平台工作是挂接现有分析/PRD 页面、将曝光与引用等事件写入业务库，并做少量真实使用者试点。
在真实样本与支持度不足时继续保留内容基线；MovieLens 权重和合成 oracle 均不进入企业生产模型。
