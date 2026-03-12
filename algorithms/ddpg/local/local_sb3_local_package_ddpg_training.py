#!/usr/bin/env python3
"""
改进的DDPG训练脚本 - 基于Stable-Baselines3和HER
参考Fetch Reach DDPG notebook实现

注意：使用本地 stable-baselines3 源码包
"""

import sys
import os

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
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

# 注册环境
gym.register_envs(gymnasium_robotics)

def train_improved_ddpg(episodes=1000000, eval_freq=25000):
    """使用Stable-Baselines3训练改进的DDPG"""
    print("🤖 开始改进的DDPG训练 (基于Stable-Baselines3 + HER)...")
    print("=" * 60)
    
    # 环境配置
    env_str = "FetchReachDense-v4"  # 使用dense奖励版本
    rl_type = "DDPG"
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"improved_ddpg_models/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    monitor_dir = os.path.join(log_dir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    
    print(f"📊 训练配置:")
    print(f"   环境: {env_str}")
    print(f"   算法: {rl_type}")
    print(f"   日志目录: {log_dir}")
    print(f"   总步数: {episodes}")
    print(f"   评估频率: {eval_freq}")
    
    # 创建训练环境 (向量化环境)
    print("🔄 创建训练环境...")
    env = make_vec_env(env_str,
                       n_envs=4,  # 并行环境数量
                       monitor_dir=monitor_dir)
    
    # 创建评估环境
    print("🔄 创建评估环境...")
    env_val = make_vec_env(env_str, n_envs=1)
    
    # 创建评估回调
    eval_callback = EvalCallback(env_val,
                                best_model_save_path=log_dir,
                                log_path=log_dir,
                                eval_freq=eval_freq,
                                render=False,
                                deterministic=True,
                                n_eval_episodes=20)
    
    # 打印环境信息
    single_env = gym.make(env_str)
    print(f"   观测空间: {single_env.observation_space}")
    print(f"   动作空间: {single_env.action_space}")
    single_env.close()
    
    # 实例化改进的DDPG代理
    print("🧠 初始化DDPG代理...")
    model = DDPG("MultiInputPolicy",  # 处理goal-based observation
                 env,
                 verbose=1,
                 learning_starts=1000,  # 学习开始步数
                 learning_rate=1e-3,    # 学习率
                 buffer_size=int(1e6),  # 经验回放缓冲区大小
                 replay_buffer_class=HerReplayBuffer,  # 使用HER
                 replay_buffer_kwargs=dict(
                     n_sampled_goal=4,  # HER采样目标数量
                     goal_selection_strategy=GoalSelectionStrategy.FUTURE
                 ),
                 gamma=0.95,    # 折扣因子
                 tau=0.05,      # 目标网络更新速率
                 batch_size=256, # 批次大小
                 tensorboard_log=os.path.join(log_dir, "tensorboard"))
    
    print("🚀 开始训练...")
    start_time = time.time()
    
    try:
        # 训练模型
        model.learn(total_timesteps=episodes,
                   callback=eval_callback,
                   progress_bar=True)
        
        # 保存最终模型
        final_model_path = os.path.join(log_dir, "final_model")
        model.save(final_model_path)
        print(f"💾 最终模型已保存到: {final_model_path}.zip")
        
        # 训练完成统计
        training_time = time.time() - start_time
        print(f"\n🎉 训练完成!")
        print(f"总训练时间: {training_time:.1f}秒 ({training_time/3600:.2f}小时)")
        
        # 评估最终模型
        print("🎯 评估最终模型...")
        env_eval = make_vec_env(env_str, n_envs=1)
        mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=50)
        print(f"最终模型 - 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
        env_eval.close()
        
        # 加载并评估最佳模型
        print("🌟 评估最佳模型...")
        best_model_path = os.path.join(log_dir, "best_model")
        if os.path.exists(best_model_path + ".zip"):  # stable-baselines3保存为.zip文件
            best_model = DDPG.load(best_model_path, env=env_val)
            mean_reward, std_reward = evaluate_policy(best_model, env_val, n_eval_episodes=50)
            print(f"最佳模型 - 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
        else:
            print("未找到最佳模型文件")
        
        # 清理环境
        env.close()
        env_val.close()
        
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
            # 尝试保存检查点
            try:
                model.save_replay_buffer(os.path.join(log_dir, "replay_buffer"))
                model.save(os.path.join(log_dir, "checkpoint_model"))
                print("💾 已保存检查点模型和回放缓冲区")
            except:
                print("❌ 无法保存任何模型文件")
        
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


def test_trained_model(model_path, num_episodes=10):
    """测试训练好的模型"""
    print("🧪 开始测试训练好的模型...")
    print("=" * 50)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"⚠️  模型文件不存在: {model_path}")
        return
    
    # 创建环境
    env_str = "FetchReachDense-v4"
    env = gym.make(env_str, render_mode='human')
    
    # 加载模型
    model = DDPG.load(model_path, env=env)
    print(f"✅ 成功加载模型: {model_path}")
    
    # 测试结果统计
    total_rewards = []
    success_counts = []
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            success_count = 0
            
            print(f"\n🚀 测试 Episode {episode + 1}/{num_episodes}")
            
            for step in range(100):  # 最大100步
                # 获取动作
                action, _ = model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 累积奖励
                total_reward += reward
                
                # 检查成功
                if 'is_success' in info and info['is_success']:
                    success_count += 1
                    print(f"   ✅ Step {step}: 成功到达目标!")
                
                # 渲染环境
                try:
                    env.render()
                except:
                    pass
                
                # 显示进展
                if step % 20 == 0:
                    print(f"   Step {step}: 奖励={reward:.2f}")
                
                if done:
                    break
            
            # 记录结果
            total_rewards.append(total_reward)
            success_counts.append(success_count)
            
            print(f"   📊 Episode {episode + 1} 结果:")
            print(f"      总奖励: {total_reward:.2f}")
            print(f"      成功次数: {success_count}")
    
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    finally:
        # 计算总体统计
        if total_rewards:
            avg_total_reward = np.mean(total_rewards)
            avg_success_rate = np.mean(success_counts)
            overall_success_rate = np.mean([1 if sc > 0 else 0 for sc in success_counts])
            
            print(f"\n🎯 测试完成! 性能汇总:")
            print(f"   平均总奖励: {avg_total_reward:.2f}")
            print(f"   平均成功次数: {avg_success_rate:.2f}")
            print(f"   整体成功率: {overall_success_rate:.2%}")
            print(f"   最佳单次奖励: {np.max(total_rewards):.2f}")
        
        env.close()
        print("✅ 测试结束，环境已关闭")


if __name__ == "__main__":
    print("🚀 启动改进的DDPG机械臂训练...")
    print("基于Stable-Baselines3 + HER (Hindsight Experience Replay)")
    print("=" * 60)
    
    # 训练模型
    log_dir, mean_reward, std_reward = train_improved_ddpg(episodes=1000000, eval_freq=25000)
    
    if mean_reward is not None:
        print(f"\n🏆 训练总结:")
        print(f"   日志目录: {log_dir}")
        print(f"   最终性能: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # 询问是否测试模型
        test_choice = input("\n是否要测试训练好的模型? (y/n): ")
        if test_choice.lower() == 'y':
            model_path = os.path.join(log_dir, "best_model")
            test_trained_model(model_path, num_episodes=5)
    else:
        print("训练未完成或出现错误")