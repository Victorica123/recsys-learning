# 推荐反馈最小闭环

## 闭环范围

当前闭环覆盖：

```text
GET /recommend
  -> SQLite 记录候选池、实际展示集合、模型版本和真实行为策略概率
  -> POST /events/impression 确认客户端实际曝光
  -> POST /events/feedback 写入点击/喜欢/不喜欢/跳过/停留
  -> scripts/feedback_replay.py 按 policy/model 队列导出不可覆盖的 JSONL
  -> item-level IPS / SNIPS + 交叉拟合 DR + 支持度/数据质量门
```

SQLite 使用 WAL 模式和短连接，同一台主机上的多个 Uvicorn worker 可以共享。
默认数据库是 `runtime/feedback/events.sqlite3`，可通过 `--feedback-db`
指定。事件接口只接受已经被该次推荐实际返回并确认曝光的电影，避免反馈与推荐
错配。

## 启动

### 真人采集实验台

最直接的采集入口是独立 Streamlit 页面：

```powershell
.venv/Scripts/python.exe -m streamlit run feedback_app.py --server.port 8502
```

打开 `http://localhost:8502` 后，体验者先填写不含个人信息的匿名代号。页面默认
进入“快速个人试验”：每批从 10 个候选中展示 5 个，点选最想看的一部后立即进入
下一批。系统把选择记录为一次 `click`，其余四部记录为显式 `skip`；完成 20 批后自动停止，
并用 90% 置信区间和 ESS 5 门槛给出个人试验信号。该信号只用于决定下一步，不能
替代多人正式上线结论。

侧边栏可切换到“多人正式评估”，恢复逐物品“想看、喜欢、不喜欢、跳过”反馈和
正式门槛。两种模式使用不同的 `policy_name`；个人模式还按当前匿名体验者过滤，
只评估每个展示物品均已收到反馈的完整批次，不会与旧数据或未提交批次混合。

第一版建议先使用确定性策略收集数据：

```powershell
.venv/Scripts/python.exe serve.py `
  --port 8000 --preload --workers 2 `
  --feedback-db runtime/feedback/events.sqlite3 `
  --model-version two-tower-retrieval-ml1m-v2
```

需要评估不同 Top-K 策略时，必须显式开启有限探索，例如：

```powershell
.venv/Scripts/python.exe serve.py `
  --port 8000 --preload `
  --exploration-rate 0.05 `
  --exploration-pool 20 `
  --model-version two-tower-retrieval-ml1m-v2
```

行为策略以 `1-epsilon` 的概率展示排序 Top-K，以 `epsilon` 的概率从候选池
均匀采样 K 个物品。日志里的 `behavior_propensity` 是每个物品被纳入展示集合的
真实边际概率，适用于当前 item-level IPS/SNIPS，不是联合 slate probability。

## 客户端事件协议

推荐响应新增：

- `recommendation_id`：连接推荐、曝光和反馈的主键；
- `request_id`：HTTP 关联 ID；
- `model_version`、`policy_name`、`exploration_rate`；
- 每个物品的 `candidate_rank` 和 `behavior_propensity`。

确认全部返回物品都已实际曝光：

```http
POST /events/impression
Content-Type: application/json

{"recommendation_id":"<id>"}
```

只确认部分物品时增加 `movie_ids` 数组。重复确认是幂等的。

提交反馈：

```http
POST /events/feedback
Content-Type: application/json

{
  "event_id": "client-generated-idempotency-key",
  "recommendation_id": "<id>",
  "movie_id": 318,
  "event_type": "click"
}
```

支持 `click`、`like`、`dislike`、`skip`、`dwell_seconds`。`event_id`
重复且内容一致时返回幂等成功；同一 ID 对应不同内容会被拒绝。反馈必须对应已
确认的曝光。`GET /events/stats` 返回推荐数、曝光覆盖率、事件数和观测 CTR。

## 导出与离线回放

每次导出必须使用新 tag，脚本拒绝覆盖已有数据集或报告：

```powershell
.venv/Scripts/python.exe scripts/feedback_replay.py `
  --db runtime/feedback/events.sqlite3 `
  --tag 20260727-v1 `
  --policy-name retrieval_epsilon_slate_v2 `
  --model-version two-tower-retrieval-ml1m-v2 `
  --target-k 10 `
  --diversity-weight 0.15 `
  --min-recommendations 200 `
  --min-actors 5 `
  --min-ess 100
```

输出：

- `experiments/feedback_<tag>.jsonl`：候选级不可变快照，包含未曝光候选；
- `experiments/feedback_<tag>_ope.json`：协议、SHA-256、数据画像、
  IPS/SNIPS、交叉拟合 DR、有效样本量、bootstrap 置信区间、支持度和上线门槛。

回放 schema 当前为 v2。每行都带 `schema_version`，导出器会检查同一推荐内的
`movie_id`/rank 唯一性、propensity 范围、曝光与奖励的一致性。数据库存在多个
`policy_name/model_version` 队列时，脚本会拒绝无过滤导出；两项过滤器都必须显式
指定，避免把历史 DeepFM、当前双塔或不同探索率产生的日志静默混算。

当前 v2 候选策略为 `双塔相似度 + 类型多样性重排`：贪心选择时，在双塔分数上加入
尚未覆盖电影类型的奖励。报告按推荐批次做配对 item-inclusion IPS，并以匿名
体验者为 cluster 对“候选策略 - 双塔 Top-K”的差值做 percentile
bootstrap，避免把同一人的大量重复评价误当作独立用户。默认门槛：

历史 `*_v1` policy 使用 DeepFM 基线，仍原样保留在 SQLite 中。当前采集器使用
`retrieval_*_v2` policy_name，回放时不得跨版本混算。

- 推荐批次不少于 200；
- 匿名体验者不少于 5；
- 曝光覆盖率不低于 95%；
- 基线和候选策略的最小有效样本量（ESS）不低于 100；
- 两个目标策略都没有零 propensity 物品。

通过门槛后，差值置信区间全大于 0 才输出 `promote_candidate`；全小于 0 输出
`keep_baseline`；跨过 0 输出 `continue_experiment`。未过数据门槛一律输出
`collect_more_data`。

正式决策门保持预注册的 IPS；DR 是稳健性复核，不替换决策门。结果模型使用
recommendation-level 交叉拟合 ridge，只在已曝光行上学习奖励，却为全部候选生成
out-of-fold 预测。它只使用 score、rank、genre hash、匿名 actor bucket 及其交叉等
不含结果的特征。若 IPS 与 DR 方向冲突，应停止解释并诊断覆盖率、propensity 和
结果模型，而不是挑选更好看的估计器。

奖励协议固定为：click +1、like +2、dislike -1、skip 0、
`dwell_seconds` 最多 +1。修改奖励定义时应升级协议版本，不应静默重算旧报告。

## 解释边界

- 默认探索率为 0，避免未经决策直接改变线上体验。
- 确定性日志只能评估其已有支持范围；目标策略包含零概率物品时，报告会将
  `support_ok` 置为 `false`，并停止给出 IPS/SNIPS 数值。
- 一次或少量冒烟事件只能证明链路正确，不能证明策略更好。正式决策需要足够
  样本、置信区间、时间窗口和业务护栏。
- “真实反馈”指真人在 `feedback_app.py` 中产生的事件；隔离 QA 数据库与
  压测数据库不得并入默认反馈库。
- 当前 SQLite 方案面向单机最小闭环。跨主机写入、事件流、隐私删除和数据保留
  策略仍需要外部基础设施。

## OPE 校准与当前证据

`scripts/validate_ope_estimators.py` 生成带已知反事实奖励均值的候选级合成曝光数据，
用独立的 oracle 直接计算两个目标策略真值，再对 IPS、SNIPS 和交叉拟合 DR 做误差
校准：

```powershell
.venv/Scripts/python.exe scripts/validate_ope_estimators.py `
  --tag <新的唯一标签> --recommendations 5000 `
  --pool-size 10 --slate-k 3 --exploration-rate 0.30 `
  --actors 20 --bootstrap-samples 500 --seed 42
```

固定种子正式产物 `p2-ope-calibration-5000-v2` 中，oracle 策略增量为
0.02684；IPS / SNIPS / DR 的增量绝对误差分别为 0.00028 / 0.00859 /
0.00812。这里精确 propensity 与大样本使 IPS 最准，而线性结果模型存在失配，
所以 **DR 并不天然优于 IPS**；校准的目的正是暴露这种边界。

现有真实库中唯一可回放的历史探索队列审计为
`p2-personal-pilot-v1-dr-audit`：21 次推荐、1 位体验者、候选策略 ESS 17.6。
IPS 增量 +0.0212，而 DR 增量 -0.0489，且 actor cluster 只有一个，正式质量门
正确输出 `collect_more_data`。这批数据只证明链路可运行，不能作为当前 v2 双塔策略
或上线收益的证据；当前 v2 队列尚需重新采集。

## 外部问卷可迁移性检查

找不到足够本地体验者时，可使用许可明确的公开真人研究数据做“外部一致性检查”，
但不能把它冒充当前产品的线上 A/B。项目已接入 GroupLens Personality 2018：

```powershell
.venv/Scripts/python.exe scripts/validate_external_survey.py `
  --tag <新的唯一标签>
```

1,834 人全量数据中，默认/多样性低中高四组共 728 人进入预设对照。中等多样性
相对默认的“愿意观看”均值高 0.113/5 分，但 95% 区间跨 0；高多样性并未提高
观看意愿，且“感到个性化”低 0.332 分（Holm 校正 `p=0.0144`）。因此外部数据
不支持全局上线，但与“低权重策略可作为个人候选”一致。详见
`notes/13-外部问卷验证多样性策略.md`。
