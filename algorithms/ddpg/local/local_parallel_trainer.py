"""本地并行DDPG训练器 - 使用多个本地Gymnasium环境并行收集数据（含HER）

直接运行：
    python local_parallel_trainer.py
    python local_parallel_trainer.py --workers 4 --episodes 2000 --env FetchReachDense-v4
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from datetime import datetime
import threading

# 直接加载 custom_trainer.py 文件（跳过 trainers/__init__.py 的副作用）
import importlib.util as _ilu

_ct_path = os.path.join(os.path.dirname(__file__), "..", "trainers", "custom_trainer.py")
_ct_spec = _ilu.spec_from_file_location("custom_trainer", os.path.abspath(_ct_path))
_ct_mod  = _ilu.module_from_spec(_ct_spec)
_ct_spec.loader.exec_module(_ct_mod)

DDPGAgent     = _ct_mod.DDPGAgent
TrainingStats = _ct_mod.TrainingStats


# ─────────────────────────────────────────────
# 环境配置常量（FetchReachDense-v4）
# ─────────────────────────────────────────────
ENV_NAME          = "FetchReachDense-v4"

OBS_DIM           = 10   # observation 维度
GOAL_DIM          = 3    # achieved_goal / desired_goal 维度
STATE_DIM         = OBS_DIM + GOAL_DIM   # 输入网络的状态维度 = 13
ACTION_DIM        = 4    # 机械臂动作维度
MAX_ACTION        = 1.0
MAX_STEPS         = 50   # FetchReach 每个 episode 最多 50 步


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def make_env(env_name: str, render_mode: str = None):
    """创建并注册 Gymnasium 环境
    
    Args:
        env_name: 环境名称
        render_mode: 渲染模式 (None, "human", "rgb_array" 等)
    """
    import gymnasium
    import gymnasium_robotics
    gymnasium.register_envs(gymnasium_robotics)
    return gymnasium.make(env_name, render_mode=render_mode)


def obs_to_state(obs: Dict) -> np.ndarray:
    """将 dict 型观测拼接为网络输入状态：[observation, desired_goal]"""
    return np.concatenate([
        np.array(obs["observation"], dtype=np.float32),
        np.array(obs["desired_goal"],  dtype=np.float32)
    ])


# ─────────────────────────────────────────────
# LocalWorker：单个本地环境工作器
# ─────────────────────────────────────────────

class LocalWorker:
    """单个本地 Gymnasium 环境工作器
    
    每个 worker 持有一个独立的环境实例，运行完整的 episode 并收集 transitions。
    Agent（网络权重）在所有 worker 之间共享，但读操作加锁以保证推理安全。
    """

    def __init__(self, worker_id: int, env_name: str, agent: DDPGAgent,
                 read_lock: threading.Lock, max_steps: int = MAX_STEPS):
        self.worker_id  = worker_id
        self.env_name   = env_name
        self.agent      = agent
        self.read_lock  = read_lock   # 推理时保护 agent.select_action
        self.max_steps  = max_steps
        self.env        = None

    def init_env(self):
        """初始化环境（线程内调用）"""
        self.env = make_env(self.env_name)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def run_episode(self, noise_scale: float = 0.1) -> Dict[str, Any]:
        """运行一个完整 episode，返回 transitions 和统计信息
        
        Returns:
            {
                'worker_id': int,
                'episode_buffer': List[Dict],   # 用于 HER 处理
                'episode_reward': float,
                'steps': int,
                'success': bool
            }
        """
        if self.env is None:
            self.init_env()

        obs, _ = self.env.reset()
        state         = obs_to_state(obs)
        achieved_goal = np.array(obs["achieved_goal"], dtype=np.float32)
        desired_goal  = np.array(obs["desired_goal"],  dtype=np.float32)

        episode_buffer = []
        episode_reward = 0.0
        success        = False

        for _ in range(self.max_steps):
            # 推理加锁（PyTorch 跨线程推理需共享锁）
            with self.read_lock:
                action = self.agent.select_action(state, noise_scale)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            next_state         = obs_to_state(next_obs)
            next_achieved_goal = np.array(next_obs["achieved_goal"], dtype=np.float32)

            transition = {
                "state":              state.copy(),
                "action":             action.copy(),
                "reward":             float(reward),
                "next_state":         next_state.copy(),
                "done":               bool(done),
                "terminated":         bool(terminated),   # 仅真实终止（FetchReach 为 False）
                "achieved_goal":      achieved_goal.copy(),
                "desired_goal":       desired_goal.copy(),
                "next_achieved_goal": next_achieved_goal.copy(),
            }
            episode_buffer.append(transition)

            episode_reward += float(reward)
            if info.get("is_success", False):
                success = True

            state         = next_state
            achieved_goal = next_achieved_goal

            if done:
                break

        return {
            "worker_id":      self.worker_id,
            "episode_buffer": episode_buffer,
            "episode_reward": episode_reward,
            "steps":          len(episode_buffer),
            "success":        success,
        }


# ─────────────────────────────────────────────
# LocalParallelTrainer：并行训练主类
# ─────────────────────────────────────────────

class LocalParallelTrainer:
    """多环境并行 DDPG 训练器（本地版）
    
    并行策略：
    - 每个 episode round 中，所有 workers 同时在各自的环境中跑完整 episode
    - 使用 ThreadPoolExecutor 并行化（IO密集部分并行，PyTorch 推理通过锁顺序化）
    - 所有 episodes 结束后统一做 HER 处理和网络更新
    - 网络更新在主线程进行，无需额外锁

    推荐配置（FetchReachDense-v4 能看到明显收敛）：
        workers=4, episodes=10000, updates=100, n_goals=8, warmup=2000
    """

    def __init__(
        self,
        num_workers:          int   = 4,
        episodes:             int   = 10000,
        eval_freq:            int   = 200,
        learning_rate:        float = 1e-3,
        use_her:              bool  = True,
        env_name:             str   = ENV_NAME,
        updates_per_episode:  int   = 100,
        n_sampled_goal:       int   = 8,
        warmup_rounds:        int   = 50,      # 增加预热让 Buffer 积累更多数据
        noise_start:          float = 0.3,     # 降低初始噪声（SB3 推荐值）
        noise_end:            float = 0.02,    # 降低最终噪声以便收敛
        buffer_size:          int   = 1_000_000,  # SB3 默认 Buffer 容量
        log_interval:         int   = 20,      # 日志输出间隔（轮数，默认 20，方便调试）
        demo_episodes:        int   = 3,       # 每次演示执行的 episode 数（默认 3）
        save_dir:             str   = None,    # 模型保存目录（默认当前目录下的 models/）
    ):
        """
        参数说明:
            num_workers:          并行环境数量（默认 4）
            episodes:             总训练轮数（每轮同时跑 num_workers 个 episode，默认 10000）
            eval_freq:            详细评估打印间隔（轮数，默认 200）
            learning_rate:        学习率（默认 1e-3，对齐 SB3）
            use_her:              是否使用 Hindsight Experience Replay（默认开启）
            env_name:             Gymnasium 环境名称
            updates_per_episode:  每轮结束后的梯度更新次数（默认 100）
            n_sampled_goal:       HER 每条 transition 重标记的目标数量（默认 8）
            warmup_rounds:        预热轮数（默认 50，让 Buffer 先积累足够数据）
            noise_start:          探索噪声初始值（默认 0.3，SB3 推荐）
            noise_end:            探索噪声最小值（默认 0.02）
            buffer_size:          Replay Buffer 容量（默认 1,000,000，对齐 SB3）
            log_interval:         日志输出间隔（轮数，默认 20，方便调试）
            demo_episodes:        每次演示执行的 episode 数（默认 3）
            save_dir:             模型保存目录（默认当前目录下的 models/）
        """
        self.num_workers          = num_workers
        self.episodes             = episodes
        self.eval_freq            = eval_freq
        self.learning_rate        = learning_rate
        self.use_her              = use_her
        self.env_name             = env_name
        self.updates_per_episode  = updates_per_episode
        self.n_sampled_goal       = n_sampled_goal
        self.warmup_rounds        = warmup_rounds
        self.noise_start          = noise_start
        self.noise_end            = noise_end
        self.buffer_size          = buffer_size
        self.log_interval         = log_interval
        self.demo_episodes        = demo_episodes
        
        # 模型保存目录（带时间戳，避免覆盖）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), "models")
            self.save_dir = os.path.join(base_dir, f"run_{timestamp}")
        else:
            self.save_dir = os.path.join(save_dir, f"run_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📁 模型保存目录: {self.save_dir}")

        self.agent:   Optional[DDPGAgent]     = None
        self.workers: List[LocalWorker]       = []
        self.stats                            = TrainingStats(window_size=100)
        self._read_lock                       = threading.Lock()

    # ──────────────────── 初始化 ────────────────────

    def _init_agent(self):
        self.agent = DDPGAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            max_action=MAX_ACTION,
            learning_rate=self.learning_rate,
            use_her=self.use_her,
            goal_dim=GOAL_DIM,
            n_sampled_goal=self.n_sampled_goal,
            buffer_size=self.buffer_size,
        )
        print(f"✅ DDPG Agent 初始化完成 {'(HER 启用)' if self.use_her else ''}")
        print(f"   状态维度: {STATE_DIM}  动作维度: {ACTION_DIM}")
        print(f"   HER 重标记目标数: {self.n_sampled_goal}")
        print(f"   Replay Buffer 容量: {self.buffer_size:,}")
        print(f"   Batch Size: {self.agent.batch_size}, Gamma: {self.agent.gamma}, Tau: {self.agent.tau}")

    def _create_workers(self):
        self.workers = [
            LocalWorker(i, self.env_name, self.agent, self._read_lock)
            for i in range(self.num_workers)
        ]
        # 提前初始化所有环境（顺序，避免 mujoco 的并发初始化问题）
        print(f"⚙️  初始化 {self.num_workers} 个本地环境...")
        for w in self.workers:
            w.init_env()
        print(f"✅ 全部环境就绪")

    # ──────────────────── 一轮并行 episode ────────────────────

    def _run_parallel_round(self, noise_scale: float) -> List[Dict]:
        """并行跑所有 workers 的 episode，返回结果列表"""
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(w.run_episode, noise_scale): w.worker_id
                for w in self.workers
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    wid = futures[future]
                    print(f"   ⚠️ Worker {wid} episode 执行失败: {e}")
        # 按 worker_id 排序，保持确定性
        results.sort(key=lambda r: r["worker_id"])
        return results
    
    # ──────────────────── 模型演示 ────────────────────
    
    def _run_demo(self, round_idx: int, model_path: str):
        """运行模型演示（可视化当前训练效果）
        
        Args:
            round_idx: 当前训练轮次
            model_path: 模型文件路径
        """
        print(f"\n{'🎬 '*20}")
        print(f"🎬 开始演示：轮次 {round_idx}，模型: {os.path.basename(model_path)}")
        print(f"{'🎬 '*20}\n")
        
        # 创建可渲染环境（指定 render_mode="human"）
        demo_env = make_env(self.env_name, render_mode="human")
        
        demo_results = []
        total_success = 0
        
        try:
            for ep_idx in range(self.demo_episodes):
                obs, _ = demo_env.reset()
                state = obs_to_state(obs)
                episode_reward = 0.0
                success = False
                
                print(f"   Episode {ep_idx + 1}/{self.demo_episodes} - ", end="", flush=True)
                
                for step in range(MAX_STEPS):
                    # 渲染当前帧
                    demo_env.render()
                    
                    # 使用训练好的策略选择动作（无噪声）
                    with torch.no_grad():
                        with self._read_lock:
                            action = self.agent.select_action(state, noise_scale=0.0)
                    
                    next_obs, reward, terminated, truncated, info = demo_env.step(action)
                    done = terminated or truncated
                    
                    next_state = obs_to_state(next_obs)
                    episode_reward += float(reward)
                    
                    if info.get("is_success", False):
                        success = True
                    
                    state = next_state
                    
                    if done:
                        break
                
                total_success += int(success)
                demo_results.append({
                    "episode": ep_idx + 1,
                    "reward": episode_reward,
                    "success": success,
                    "steps": step + 1
                })
                
                # 打印结果
                status = "✅ 成功" if success else "❌ 失败"
                print(f"{status}  奖励: {episode_reward:.3f}  步数: {step + 1}")
            
            # 汇总统计
            avg_reward = np.mean([r["reward"] for r in demo_results])
            success_rate = total_success / self.demo_episodes
            
            print(f"\n{'─'*50}")
            print(f"📊 演示汇总：")
            print(f"   平均奖励:    {avg_reward:.3f}")
            print(f"   成功率:      {success_rate:.1%} ({total_success}/{self.demo_episodes})")
            print(f"{'─'*50}\n")
            
        finally:
            demo_env.close()
        
        return demo_results

    # ──────────────────── 主训练循环 ────────────────────

    def train(self) -> Dict[str, Any]:
        """执行完整训练流程"""
        self._init_agent()
        self._create_workers()

        print(f"\n{'='*60}")
        print(f"🚀 开始本地并行 DDPG 训练")
        print(f"   环境:            {self.env_name}")
        print(f"   并行环境数:       {self.num_workers}")
        print(f"   总训练轮数:       {self.episodes}")
        print(f"   预计总环境步数:   {self.episodes * self.num_workers * MAX_STEPS:,}")
        print(f"   每轮梯度更新:     {self.updates_per_episode} 次")
        print(f"   预热轮数:         {self.warmup_rounds}")
        print(f"   探索噪声范围:     {self.noise_start} → {self.noise_end}")
        print(f"   日志输出间隔:     每 {self.log_interval} 轮")
        print(f"   HER:             {'开启 (重标记目标数=' + str(self.n_sampled_goal) + ')' if self.use_her else '关闭'}")
        print(f"{'='*60}\n")

        start_time = time.time()
        all_results: List[Dict] = []

        # 滑动窗口统计（用于阶段性输出）
        recent_rewards   = deque(maxlen=100)
        recent_successes = deque(maxlen=100)
        recent_steps     = deque(maxlen=100)

        # 累积统计
        total_timesteps = 0
        total_episodes  = 0
        n_updates       = 0
        last_actor_loss  = 0.0
        last_critic_loss = 0.0

        try:
            for round_idx in range(self.episodes):
                # 噪声衰减策略：线性衰减
                warmup_fraction = 0.3
                if round_idx < self.episodes * warmup_fraction:
                    noise_scale = self.noise_start
                else:
                    decay_progress = (round_idx - self.episodes * warmup_fraction) / (self.episodes * (1 - warmup_fraction))
                    noise_scale = self.noise_start - (self.noise_start - self.noise_end) * decay_progress
                    noise_scale = max(self.noise_end, noise_scale)

                # ① 并行收集所有 workers 的 episode 数据
                round_results = self._run_parallel_round(noise_scale)

                # ② HER 处理
                for res in round_results:
                    if res["episode_buffer"]:
                        self.agent.process_worker_episode(
                            res["episode_buffer"], res["worker_id"]
                        )

                # ③ 梯度更新
                update_losses = []
                in_warmup = round_idx < self.warmup_rounds
                if not in_warmup and len(self.agent.replay_buffer) >= self.agent.batch_size:
                    for _ in range(self.updates_per_episode):
                        loss_info = self.agent.update()
                        if loss_info:
                            update_losses.append(loss_info)
                            n_updates += 1

                # ④ 汇总统计
                avg_reward   = np.mean([r["episode_reward"] for r in round_results])
                avg_steps    = np.mean([r["steps"]          for r in round_results])
                any_success  = any(r["success"]             for r in round_results)
                success_cnt  = sum(r["success"]             for r in round_results)
                round_steps  = sum(r["steps"]               for r in round_results)

                total_timesteps += round_steps
                total_episodes  += self.num_workers

                self.stats.add_episode(avg_reward, any_success, int(avg_steps))

                # 更新滑动窗口
                recent_rewards.append(avg_reward)
                recent_successes.append(success_cnt)
                recent_steps.append(avg_steps)

                # 记录最新 loss
                if update_losses:
                    last_actor_loss  = np.mean([l.get("actor_loss", 0) for l in update_losses])
                    last_critic_loss = np.mean([l["critic_loss"] for l in update_losses])

                round_info = {
                    "round":         round_idx,
                    "avg_reward":    round(avg_reward, 3),
                    "avg_steps":     round(avg_steps, 1),
                    "success_count": success_cnt,
                    "total_steps":   round_steps,
                    "buffer_size":   len(self.agent.replay_buffer),
                    "noise_scale":   round(noise_scale, 3),
                }
                all_results.append(round_info)

                # ⑤ 阶段性输出（类似 SB3 格式，每 log_interval 轮输出一次）
                if (round_idx + 1) % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = int(total_timesteps / elapsed) if elapsed > 0 else 0
                    
                    # 计算近期统计
                    ep_rew_mean  = np.mean(recent_rewards) if recent_rewards else 0
                    ep_len_mean  = np.mean(recent_steps) if recent_steps else 0
                    success_rate = sum(recent_successes) / (self.num_workers * len(recent_successes)) if recent_successes else 0

                    # 打印 SB3 风格的表格
                    print("-" * 40)
                    print(f"| {'rollout/':<20} | {'':<13} |")
                    print(f"|    {'ep_len_mean':<17} | {ep_len_mean:<13.0f} |")
                    print(f"|    {'ep_rew_mean':<17} | {ep_rew_mean:<13.3f} |")
                    print(f"|    {'success_rate':<17} | {success_rate:<13.3f} |")
                    print(f"| {'time/':<20} | {'':<13} |")
                    print(f"|    {'episodes':<17} | {total_episodes:<13} |")
                    print(f"|    {'fps':<17} | {fps:<13} |")
                    print(f"|    {'time_elapsed':<17} | {int(elapsed):<13} |")
                    print(f"|    {'total_timesteps':<17} | {total_timesteps:<13} |")
                    print(f"| {'train/':<20} | {'':<13} |")
                    print(f"|    {'actor_loss':<17} | {last_actor_loss:<13.5f} |")
                    print(f"|    {'critic_loss':<17} | {last_critic_loss:<13.5f} |")
                    print(f"|    {'learning_rate':<17} | {self.learning_rate:<13} |")
                    print(f"|    {'n_updates':<17} | {n_updates:<13} |")
                    print(f"|    {'noise_scale':<17} | {noise_scale:<13.3f} |")
                    print(f"|    {'buffer_size':<17} | {len(self.agent.replay_buffer):<13} |")
                    print("-" * 40)
                    
                    # 💾 保存模型
                    model_filename = f"ddpg_model_round_{round_idx+1}.pth"
                    model_path = os.path.join(self.save_dir, model_filename)
                    self.agent.save_model(model_path)
                    print(f"💾 模型已保存: {model_path}")
                    
                    # 🎬 运行演示（跳过预热阶段）
                    if round_idx >= self.warmup_rounds:
                        self._run_demo(round_idx + 1, model_path)
                    else:
                        print(f"⏭️  跳过演示（预热阶段：{round_idx+1}/{self.warmup_rounds}）\n")

        except KeyboardInterrupt:
            print(f"\n⚠️  训练被用户中断 (Ctrl-C)")

        finally:
            for w in self.workers:
                w.close()

        # ─── 最终统计 ───
        final_summary = self.stats.get_summary()
        elapsed_total = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"🎉 训练完成！")
        print(f"   总轮数:        {len(all_results)}")
        print(f"   总耗时:        {elapsed_total/60:.1f} 分钟")
        print(f"   全局平均奖励:  {final_summary.get('global_avg_reward', 0):.3f}")
        print(f"   最终成功率:    {final_summary.get('success_rate', 0):.1%}")
        print(f"   训练效果:      {final_summary.get('effectiveness_cn', '未知')}")
        print(f"{'='*60}")

        return {
            "status":         "completed",
            "total_rounds":   len(all_results),
            "elapsed_seconds": elapsed_total,
            "final_summary":  final_summary,
            "results":        all_results[-200:],
        }

    # ──────────────────── 辅助 ────────────────────

    def _print_eval_summary(self, round_idx: int, summary: Dict, elapsed: float):
        """打印详细评估信息"""
        trend_map = {"improving": "📈 上升", "declining": "📉 下降", "stable": "➡️ 平稳"}
        trend_str = trend_map.get(summary.get("trend", ""), "➡️ 平稳")
        print(f"\n{'─'*55}")
        print(f"📊 评估 @ 轮次 {round_idx}  （已训练 {elapsed/60:.1f} 分钟）")
        print(f"   近{self.stats.window_size}轮平均奖励:  {summary.get('avg_reward', 0):.3f}")
        print(f"   近{self.stats.window_size}轮成功率:    {summary.get('success_rate', 0):.1%}")
        print(f"   全局平均奖励:         {summary.get('global_avg_reward', 0):.3f}")
        print(f"   奖励趋势:             {trend_str}")
        print(f"   Buffer 大小:          {len(self.agent.replay_buffer)}")
        print(f"{'─'*55}\n")


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="本地并行 DDPG 训练器 (FetchReach 系列环境，对齐 SB3 参数)"
    )
    parser.add_argument("--workers",   type=int,   default=4,
                        help="并行环境数量 (默认: 4)")
    parser.add_argument("--episodes",  type=int,   default=10000,
                        help="训练轮数，每轮并行跑 --workers 个 episode (默认: 10000)")
    parser.add_argument("--eval-freq", type=int,   default=200,
                        help="详细评估打印间隔（轮数）(默认: 200)")
    parser.add_argument("--lr",        type=float, default=1e-3,
                        help="学习率 (默认: 1e-3，对齐 SB3)")
    parser.add_argument("--no-her",    action="store_true",
                        help="禁用 Hindsight Experience Replay")
    parser.add_argument("--env",       type=str,   default=ENV_NAME,
                        help=f"Gymnasium 环境名称 (默认: {ENV_NAME})")
    parser.add_argument("--updates",   type=int,   default=100,
                        help="每轮 episode 结束后的梯度更新次数 (默认: 100)")
    parser.add_argument("--n-goals",   type=int,   default=8,
                        help="HER 每条 transition 重标记的目标数量 (默认: 8)")
    parser.add_argument("--warmup",    type=int,   default=50,
                        help="预热轮数，Buffer 积累足够后再开始梯度更新 (默认: 50)")
    parser.add_argument("--noise-start", type=float, default=0.3,
                        help="探索噪声初始值 (默认: 0.3)")
    parser.add_argument("--noise-end",   type=float, default=0.02,
                        help="探索噪声最小值 (默认: 0.02)")
    parser.add_argument("--buffer-size", type=int,   default=1_000_000,
                        help="Replay Buffer 容量 (默认: 1,000,000)")
    parser.add_argument("--log-interval", type=int, default=20,
                        help="日志输出间隔（轮数）(默认: 20，方便调试)")
    parser.add_argument("--demo-episodes", type=int, default=3,
                        help="每次演示执行的 episode 数 (默认: 3)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="模型保存目录 (默认: ./models/)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainer = LocalParallelTrainer(
        num_workers          = args.workers,
        episodes             = args.episodes,
        eval_freq            = args.eval_freq,
        learning_rate        = args.lr,
        use_her              = not args.no_her,
        env_name             = args.env,
        updates_per_episode  = args.updates,
        n_sampled_goal       = args.n_goals,
        warmup_rounds        = args.warmup,
        noise_start          = args.noise_start,
        noise_end            = args.noise_end,
        buffer_size          = args.buffer_size,
        log_interval         = args.log_interval,
        demo_episodes        = args.demo_episodes,
        save_dir             = args.save_dir,
    )
    trainer.train()
