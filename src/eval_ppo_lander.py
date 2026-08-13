# -*- coding: utf-8 -*-
"""
RL 快速体验：看训练好的 PPO 智能体降落（生成 GIF 动画）

运行方式：.venv/Scripts/python.exe src/eval_ppo_lander.py
产出：figures/05-PPO着陆动画.gif
"""
from pathlib import Path

import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "checkpoints" / "ppo_lander"
GIF_PATH = ROOT / "figures" / "05-PPO着陆动画.gif"


def run_episode(model, seed):
    """跑一个回合，返回 (总奖励, 帧列表)。"""
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, total_r = [], 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)  # 评估时不探索
        obs, r, terminated, truncated, _ = env.step(int(action))
        frames.append(Image.fromarray(env.render()))
        total_r += r
        if terminated or truncated:
            break
    env.close()
    return total_r, frames


def main():
    model = PPO.load(MODEL_PATH)
    # 跑 3 个不同种子，挑奖励最高的一局做成 GIF
    best_r, best_frames = -1e9, None
    for seed in (7, 11, 2024):
        r, frames = run_episode(model, seed)
        print(f"seed {seed}: 回合奖励 {r:.1f}，共 {len(frames)} 帧")
        if r > best_r:
            best_r, best_frames = r, frames

    GIF_PATH.parent.mkdir(exist_ok=True)
    best_frames[0].save(GIF_PATH, save_all=True, append_images=best_frames[1:],
                        duration=40, loop=0, optimize=True)
    print(f"\n最佳一局奖励 {best_r:.1f}，动画已保存到 {GIF_PATH}")


if __name__ == "__main__":
    main()
