# 阶段 0：GridWorld 与 Q-learning

## 本阶段目标

先不碰神经网络。完成后，你应该能用自己的话解释：状态、动作、奖励、回报、
策略、Q 值、探索，以及一次 Bellman 更新为什么有效。

## 环境

```text
S . . .
. X . .
. . X .
. . . G
```

- `S`：起点；`G`：目标，奖励 `+1`；`X`：陷阱，奖励 `-1`。
- 其他移动每步奖励 `-0.02`，所以智能体还需要学习“尽快到达”。
- 状态是 16 个格子，动作是上、右、下、左。

一次交互就是：

```text
当前状态 s -> 选择动作 a -> 环境返回奖励 r、下一状态 s'、是否结束 done
```

## 唯一需要先记住的公式

```text
target = r                              episode 已结束
target = r + gamma * max Q(s', a')      episode 未结束

Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
```

- `alpha`：这次新经验要相信多少。
- `gamma`：未来奖励有多重要。
- `epsilon`：有多大概率随机探索，而不是选当前最优动作。

## 运行顺序

```powershell
cd "D:\machine  learning\recsys-learning"
.venv/Scripts/python.exe -m unittest research_v2.phase0_q_learning.test_q_learning -v
.venv/Scripts/python.exe -m research_v2.phase0_q_learning.train
```

## 练习 1：观察默认实验

记录下面四项，不急着看全部代码：

1. 随机策略成功率是多少？
2. 学习后成功率是多少？
3. 最终策略从 `S` 到 `G` 需要几步？
4. 为什么每步 `-0.02` 能让路径更短？

## 练习 2：手算一次更新

假设：`Q(s,a)=0`、`r=-0.02`、下一状态最大 Q 值为 `0.4`、
`alpha=0.1`、`gamma=0.95`。

```text
target = -0.02 + 0.95 * 0.4 = 0.36
new Q  = 0 + 0.1 * (0.36 - 0) = 0.036
```

对应测试是 `test_one_bellman_update`。尝试把 `gamma` 改成 `0`，重新手算。

## 练习 3：改变一个变量

每次只改一个参数：

```powershell
.venv/Scripts/python.exe -m research_v2.phase0_q_learning.train --gamma 0
.venv/Scripts/python.exe -m research_v2.phase0_q_learning.train --episodes 20
.venv/Scripts/python.exe -m research_v2.phase0_q_learning.train --alpha 0
```

| 实验 | 成功率 | 平均回报 | 你的解释 |
|---|---:|---:|---|
| 默认 |  |  |  |
| gamma=0 |  |  |  |
| episodes=20 |  |  |  |
| alpha=0 |  |  |  |

## 完成标准

- 能独立写出一次 Q-learning 更新。
- 能说明 `alpha/gamma/epsilon` 分别控制什么。
- 能解释随机策略和学习策略的差别。
- 自动测试通过，默认学习策略成功率不低于 95%。

推荐阅读：Sutton & Barto 第二版第 3、6 章。PPO 和 GRPO 都建立在这里的
状态、动作、奖励、回报和优势概念之上。
