#!/usr/bin/env python3
"""
FetchPickAndPlaceDense-v4 DDPG训练脚本 - 基于本地 Stable-Baselines3
参考 [Fetch Pick & Place] Deep Deterministic Policy Gradient (DDPG).ipynb 实现

注意：默认启用 HER（Hindsight Experience Replay），这对 Pick&Place 任务至关重要！

使用方法:
    # 默认配置（启用 HER）
    python local_sb3_pick_and_place.py
    
    # 快速测试（10万步，启用 HER）
    python local_sb3_pick_and_place.py --episodes 100000
    
    # 完整训练 + 渲染测试
    python local_sb3_pick_and_place.py --episodes 1000000 --render
    
    # 不使用 HER（不推荐，仅用于对比实验）
    python local_sb3_pick_and_place.py --no-her
"""

import sys
import os
import argparse

# 添加本地 stable-baselines3 源码路径（优先于系统安装的版本）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_local_sb3_path = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", "..", "stable-baselines3"))
if _local_sb3_path not in sys.path:
    sys.path.insert(0, _local_sb3_path)

import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
import time
from datetime import datetime

# 从本地 stable_baselines3 导入
from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

# 注册环境
gym.register_envs(gymnasium_robotics)

# ─────────────────────────────────────────────
# 环境与训练配置（参考 Notebook）
# ─────────────────────────────────────────────
ENV_NAME = "FetchPickAndPlaceDense-v4"

# Notebook 推荐的超参数
DEFAULT_GAMMA = 0.95      # 折扣因子（Notebook 使用 0.95）
DEFAULT_TAU = 0.05        # 目标网络软更新率（Notebook 使用 0.05）
DEFAULT_LR = 1e-3         # 学习率
DEFAULT_BUFFER_SIZE = int(1e6)
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_STARTS = 1000
DEFAULT_N_ENVS = 4        # 并行环境数量
DEFAULT_EVAL_FREQ = 25000
DEFAULT_TOTAL_TIMESTEPS = 1000000


def train_ddpg_pick_and_place(
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    eval_freq: int = DEFAULT_EVAL_FREQ,
    use_her: bool = True,  # 默认启用 HER，Pick&Place 任务强烈推荐
    render: bool = False,
    n_envs: int = DEFAULT_N_ENVS,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_TAU,
):
    """
    使用 Stable-Baselines3 训练 FetchPickAndPlaceDense-v4 的 DDPG

    Args:
        total_timesteps: 总训练步数
        eval_freq: 评估频率
        use_her: 是否使用 HER（Hindsight Experience Replay）
        render: 是否在训练后进行渲染测试
        n_envs: 并行环境数量
        gamma: 折扣因子
        tau: 目标网络软更新率
    """
    print("🤖 开始 FetchPickAndPlaceDense-v4 DDPG 训练 (基于本地 Stable-Baselines3)...")
    print("=" * 70)

    # 打印 SB3 版本和路径
    import stable_baselines3
    print(f"📦 使用本地 stable-baselines3: {os.path.dirname(stable_baselines3.__file__)}")

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"pick_and_place_ddpg_models/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    monitor_dir = os.path.join(log_dir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    print(f"\n📊 训练配置:")
    print(f"   环境:              {ENV_NAME}")
    print(f"   算法:              DDPG {'+ HER' if use_her else ''}")
    print(f"   日志目录:          {log_dir}")
    print(f"   总步数:            {total_timesteps:,}")
    print(f"   评估频率:          {eval_freq:,}")
    print(f"   并行环境数:        {n_envs}")
    print(f"   Gamma (折扣因子):  {gamma}")
    print(f"   Tau (软更新率):    {tau}")
    print(f"   使用 HER:          {'是' if use_her else '否'}")
    print(f"   渲染测试:          {'是' if render else '否'}")

    # 创建训练环境 (向量化环境)
    print("\n🔄 创建训练环境...")
    env = make_vec_env(ENV_NAME, n_envs=n_envs, monitor_dir=monitor_dir)

    # 创建评估环境
    print("🔄 创建评估环境...")
    env_val = make_vec_env(ENV_NAME, n_envs=1)

    # 打印环境信息
    single_env = gym.make(ENV_NAME)
    print(f"\n📋 环境信息:")
    print(f"   观测空间: {single_env.observation_space}")
    print(f"   动作空间: {single_env.action_space}")
    single_env.close()

    # 创建动作噪声（参考 Notebook：sigma=0.1）
    n_actions = env.action_space.shape[-1]
    mean = np.zeros(n_actions)
    sigma = 0.1 * np.ones(n_actions)
    action_noise = NormalActionNoise(mean=mean, sigma=sigma)

    # 创建评估回调
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        render=False,
        deterministic=True,
        n_eval_episodes=20
    )

    # 构建 DDPG 模型参数
    model_kwargs = dict(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_starts=DEFAULT_LEARNING_STARTS,
        learning_rate=DEFAULT_LR,
        buffer_size=DEFAULT_BUFFER_SIZE,
        action_noise=action_noise,
        gamma=gamma,
        tau=tau,
        batch_size=DEFAULT_BATCH_SIZE,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
    )

    # 如果使用 HER，添加 HER 配置
    if use_her:
        model_kwargs["replay_buffer_class"] = HerReplayBuffer
        model_kwargs["replay_buffer_kwargs"] = dict(
            n_sampled_goal=8,  # 增加到 8（参考 FetchReach 成功经验）
            goal_selection_strategy=GoalSelectionStrategy.FUTURE
        )

    # 实例化 DDPG 代理
    print("\n🧠 初始化 DDPG 代理...")
    model = DDPG(**model_kwargs)

    print("\n🚀 开始训练...")
    start_time = time.time()

    try:
        # 训练模型
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )

        # 保存最终模型
        final_model_path = os.path.join(log_dir, "final_model")
        model.save(final_model_path)
        print(f"\n💾 最终模型已保存到: {final_model_path}.zip")

        # 训练完成统计
        training_time = time.time() - start_time
        print(f"\n🎉 训练完成!")
        print(f"   总训练时间: {training_time:.1f}秒 ({training_time/3600:.2f}小时)")

        # 评估最终模型
        print("\n🎯 评估最终模型...")
        env_eval = make_vec_env(ENV_NAME, n_envs=1)
        mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=20)
        print(f"   最终模型 - 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
        env_eval.close()

        # 加载并评估最佳模型
        print("\n🌟 评估最佳模型...")
        best_model_path = os.path.join(log_dir, "best_model")
        if os.path.exists(best_model_path + ".zip"):
            try:
                # 等待文件写入完成
                time.sleep(0.5)
                best_model = DDPG.load(best_model_path, env=env_val)
                mean_reward_best, std_reward_best = evaluate_policy(best_model, env_val, n_eval_episodes=20)
                print(f"   最佳模型 - 平均奖励: {mean_reward_best:.2f} +/- {std_reward_best:.2f}")
            except Exception as e:
                print(f"   加载最佳模型时出错: {e}")
                print("   使用最终模型进行后续操作")
                mean_reward_best, std_reward_best = mean_reward, std_reward
        else:
            print("   未找到最佳模型文件")
            mean_reward_best, std_reward_best = mean_reward, std_reward

        # 清理环境
        env.close()
        env_val.close()

        # 渲染测试
        if render:
            print("\n🎬 开始渲染测试...")
            test_trained_model(best_model_path if os.path.exists(best_model_path + ".zip") else final_model_path)

        return log_dir, mean_reward, std_reward

    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断")
        # 保存中断时的模型
        interrupt_model_path = os.path.join(log_dir, "interrupted_model")
        try:
            model.save(interrupt_model_path)
            print(f"💾 中断模型已保存到: {interrupt_model_path}.zip")
        except Exception as save_error:
            print(f"❌ 保存中断模型时出错: {save_error}")

        env.close()
        env_val.close()
        return log_dir, None, None

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

        # 尝试保存当前模型状态
        try:
            error_model_path = os.path.join(log_dir, "error_model")
            model.save(error_model_path)
            print(f"💾 错误模型已保存到: {error_model_path}.zip")
        except:
            print("❌ 无法保存错误模型")

        env.close()
        env_val.close()
        return log_dir, None, None


def test_trained_model(model_path: str, num_episodes: int = 5):
    """测试训练好的模型（带渲染）"""
    print("\n🧪 开始测试训练好的模型...")
    print("=" * 50)

    if not os.path.exists(model_path + ".zip"):
        print(f"⚠️  模型文件不存在: {model_path}")
        return

    # 创建带渲染的环境
    env = gym.make(ENV_NAME, render_mode='human')

    # 加载模型
    model = DDPG.load(model_path, env=env)
    print(f"✅ 成功加载模型: {model_path}")

    # 测试结果统计
    total_rewards = []
    successes = []

    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            success = False

            print(f"\n🚀 测试 Episode {episode + 1}/{num_episodes}")

            for step in range(50):  # FetchPickAndPlace 最大 50 步
                # 获取动作
                action, _ = model.predict(obs, deterministic=True)

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # 累积奖励
                total_reward += reward

                # 检查成功
                if info.get('is_success', False):
                    success = True
                    print(f"   ✅ Step {step}: 成功完成任务!")

                # 渲染
                try:
                    env.render()
                except:
                    pass

                if done:
                    break

            # 记录结果
            total_rewards.append(total_reward)
            successes.append(success)

            print(f"   📊 Episode {episode + 1} 结果:")
            print(f"      总奖励: {total_reward:.2f}")
            print(f"      成功: {'是' if success else '否'}")

    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    finally:
        # 计算总体统计
        if total_rewards:
            avg_reward = np.mean(total_rewards)
            success_rate = np.mean(successes)

            print(f"\n🎯 测试完成! 性能汇总:")
            print(f"   平均奖励: {avg_reward:.2f}")
            print(f"   成功率: {success_rate:.2%}")
            print(f"   最佳单次奖励: {np.max(total_rewards):.2f}")

        env.close()
        print("✅ 测试结束，环境已关闭")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FetchPickAndPlaceDense-v4 DDPG 训练脚本（使用本地 SB3）"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help=f"总训练步数 (默认: {DEFAULT_TOTAL_TIMESTEPS:,})")
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ,
                        help=f"评估频率 (默认: {DEFAULT_EVAL_FREQ:,})")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS,
                        help=f"并行环境数量 (默认: {DEFAULT_N_ENVS})")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help=f"折扣因子 (默认: {DEFAULT_GAMMA})")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help=f"目标网络软更新率 (默认: {DEFAULT_TAU})")
    parser.add_argument("--no-her", action="store_true",
                        help="禁用 HER (默认启用 HER，Pick&Place 强烈推荐)")
    parser.add_argument("--render", action="store_true",
                        help="训练完成后进行渲染测试")
    parser.add_argument("--test-only", type=str, default=None,
                        help="仅测试指定的模型路径（跳过训练）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("🚀 启动 FetchPickAndPlaceDense-v4 DDPG 训练...")
    print("=" * 70)

    if args.test_only:
        # 仅测试模式
        test_trained_model(args.test_only, num_episodes=5)
    else:
        # 训练模式
        log_dir, mean_reward, std_reward = train_ddpg_pick_and_place(
            total_timesteps=args.episodes,
            eval_freq=args.eval_freq,
            use_her=not args.no_her,  # 默认启用，--no-her 才禁用
            render=args.render,
            n_envs=args.n_envs,
            gamma=args.gamma,
            tau=args.tau,
        )

        if mean_reward is not None:
            print(f"\n🏆 训练总结:")
            print(f"   日志目录: {log_dir}")
            print(f"   最终性能: {mean_reward:.2f} +/- {std_reward:.2f}")
        else:
            print("训练未完成或出现错误")
