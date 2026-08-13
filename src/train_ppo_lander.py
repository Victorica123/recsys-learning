# -*- coding: utf-8 -*-
"""
RL 快速体验：PPO 训练月球着陆器（LunarLander）
=============================================

【任务】控制着陆器（4 个离散动作：熄火/左推/主推/右推），
平稳降落在两面旗帜之间。奖励：降落成功 +100，坠毁 -100，
每帧点火有微小油耗惩罚 —— 智能体要自己学会"省着烧、轻轻落"。

【PPO 直觉】
- 策略网络（Policy）：输入 8 维观测（位置/速度/角度/腿触地），
  输出 4 个动作的概率分布 —— 它就是"大脑"
- 每个 rollout 收集一批 (观测, 动作, 奖励)，估计每个动作的"优势"
  （比平均好多少），然后把好动作的概率往上推、差动作往下压
- PPO 的精髓在"裁剪"（clip）：每次只许策略小步更新，
  防止一步迈太大把已经学会的东西毁掉（和推荐里 lr 太大过拟合一个道理）

【运行方式】（在项目根目录；分段跑，每段自动保存，可无限续训）
    .venv/Scripts/python.exe src/train_ppo_lander.py --timesteps 150000
    续训：加 --resume
"""
import argparse
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import set_seed, ExperimentConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
EXP_DIR = ROOT / "experiments"
CKPT_DIR.mkdir(exist_ok=True)
EXP_DIR.mkdir(exist_ok=True)
MODEL_PATH = CKPT_DIR / "ppo_lander"
LOG_PATH = EXP_DIR / "ppo_log.csv"   # 列：总步数,回合奖励,回合长度

N_ENVS = 16  # 16 个环境并行采样（PPO 是 on-policy 算法，并行能大幅提速）


class RewardLogCallback(BaseCallback):
    """每结束一个回合，往 CSV 追加一行（总步数, 回合奖励, 回合长度）。"""

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:  # Monitor 包装后，回合结束时 info 里会有统计
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(f"{self.num_timesteps},{info['episode']['r']:.2f},"
                            f"{info['episode']['l']}\n")
        return True


def make_venv():
    # Monitor 不加文件名：只负责统计回合奖励，不写文件
    return DummyVecEnv(
        [lambda: Monitor(gym.make("LunarLander-v3")) for _ in range(N_ENVS)])


def report_recent(n=20):
    """读 CSV，汇报最近 n 个回合的平均奖励。"""
    if not LOG_PATH.exists():
        return
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    recent = [float(l.split(",")[1]) for l in lines[-n:]]
    total_steps = int(lines[-1].split(",")[0])
    print(f"总步数 {total_steps:>8,} | 最近 {len(recent)} 回合平均奖励: "
          f"{sum(recent) / len(recent):.1f}  (满分参考: 成功着陆约 +200~300)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="????")
    ap.add_argument("--timesteps", type=int, default=150_000,
                    help="本段训练的步数")
    ap.add_argument("--resume", action="store_true", help="加载上次模型继续训练")
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(EXP_DIR)
    print(f"实验配置已保存: {cfg.run_name}")

    venv = make_venv()
    # 这组超参数来自 SB3 官方调优库（rl-zoo），针对 LunarLander 调过
    kwargs = dict(n_steps=1024, batch_size=64, gae_lambda=0.98,
                  gamma=0.999, n_epochs=4, ent_coef=0.01,
                  verbose=0, seed=42,
                  device="cpu")  # 策略网络很小，CPU 反而比 GPU 快
                   # （小网络每次更新数据量小，GPU 的数据搬运开销大于计算收益）

    if args.resume and MODEL_PATH.with_suffix(".zip").exists():
        model = PPO.load(MODEL_PATH, env=venv, device="cpu")
        print(f"已加载 checkpoint，继续训练 {args.timesteps:,} 步 ...")
    else:
        model = PPO("MlpPolicy", venv, **kwargs)
        print(f"从头开始训练 {args.timesteps:,} 步 ...")

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps,
                callback=RewardLogCallback(),
                reset_num_timesteps=not args.resume)
    model.save(MODEL_PATH)
    print(f"本段耗时 {time.time() - t0:.0f}s，模型已保存到 {MODEL_PATH}.zip")
    report_recent()


if __name__ == "__main__":
    main()
