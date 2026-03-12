"""并行DDPG训练器 - 支持多沙箱并行数据收集（含HER）

关键修复（对齐 local_parallel_trainer 的有效实现）：
1. 状态构建：state = concat([obs, desired_goal])，确保网络输入包含目标信息
2. terminated 字段：FetchReach 总是 truncated 终止，terminated=False，避免截断 Q 值
3. 超参数对齐 SB3：buffer_size=1M, n_sampled_goal=8, batch_size=256, tau=0.005
4. 训练循环：每轮 episode 结束后做 updates_per_episode 次梯度更新
5. 分段线性噪声衰减（前 30% 保持高探索，之后线性降到 noise_end）
6. 阶段性模型保存和测试沙箱演示
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import asyncio
from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime

from .custom_trainer import Actor, Critic, DDPGAgent, TrainingStats


# ─────────────────────────────────────────────
# 环境配置常量（FetchReachDense-v4）
# ─────────────────────────────────────────────
OBS_DIM    = 10   # observation 维度
GOAL_DIM   = 3    # achieved_goal / desired_goal 维度
STATE_DIM  = OBS_DIM + GOAL_DIM  # 网络输入维度 = 13
ACTION_DIM = 4
MAX_ACTION = 1.0
MAX_STEPS  = 50   # FetchReach 每个 episode 最多 50 步


def _build_state(obs, desired_goal) -> np.ndarray:
    """将环境观测和目标拼接为网络输入状态 [obs(10d), desired_goal(3d)] = 13d"""
    obs_arr  = np.array(obs,          dtype=np.float32).flatten()
    goal_arr = np.array(desired_goal, dtype=np.float32).flatten()
    return np.concatenate([obs_arr, goal_arr])


class ParallelWorker:
    """并行工作器 - 管理单个沙箱的环境交互（支持HER episode隔离）"""

    def __init__(self, worker_id: int, env_proxy, agent: DDPGAgent):
        self.worker_id  = worker_id
        self.env_proxy  = env_proxy
        self.agent      = agent          # 共享 agent 引用
        self.current_state = None
        self.episode_reward = 0.0
        self.episode_steps  = 0
        self.success        = False
        self.done           = False
        # HER 相关
        self.current_achieved_goal = None
        self.desired_goal          = None
        # 独立 episode buffer（HER 隔离：每个 worker 的 episode 独立处理）
        self.episode_buffer: List[Dict] = []

    async def reset(self) -> bool:
        """重置环境（获取 goal 信息，清空 episode buffer）"""
        try:
            obs, reset_info = await self.env_proxy.reset()
            # ── 关键修复：state = concat([obs, desired_goal]) ──
            desired_goal = reset_info.get('desired_goal')
            if desired_goal is None:
                # fallback：从 obs dict 中取
                if isinstance(obs, dict):
                    desired_goal = obs.get('desired_goal', np.zeros(GOAL_DIM, dtype=np.float32))
                else:
                    desired_goal = np.zeros(GOAL_DIM, dtype=np.float32)

            self.desired_goal = np.array(desired_goal, dtype=np.float32)

            # obs 可能是 dict 或 flat array
            if isinstance(obs, dict):
                raw_obs = np.array(obs.get('observation', obs), dtype=np.float32).flatten()
                self.current_achieved_goal = np.array(obs.get('achieved_goal', np.zeros(GOAL_DIM)), dtype=np.float32)
            else:
                raw_obs = np.array(obs, dtype=np.float32).flatten()
                self.current_achieved_goal = reset_info.get('achieved_goal')
                if self.current_achieved_goal is not None:
                    self.current_achieved_goal = np.array(self.current_achieved_goal, dtype=np.float32)

            # 构建包含 goal 的完整状态
            self.current_state = _build_state(raw_obs, self.desired_goal)

            self.episode_reward = 0.0
            self.episode_steps  = 0
            self.success        = False
            self.done           = False
            self.episode_buffer = []
            return self.current_state.size > 0
        except Exception as e:
            print(f"   ⚠️ Worker {self.worker_id} reset 失败: {e}")
            return False

    async def collect_batch(self, batch_size: int, noise_scale: float) -> List[Dict]:
        """收集一批经验数据（包含 goal 信息用于 HER）

        返回的 experiences 仅用于统计；实际的训练数据通过 episode_buffer 在
        episode 结束后统一调用 process_worker_episode 写入 replay buffer。
        """
        if self.done or self.current_state is None or self.current_state.size == 0:
            return []

        # 基于当前真实状态生成一批动作
        actions = [self.agent.select_action(self.current_state, noise_scale)
                   for _ in range(batch_size)]

        try:
            batch_results = await self.env_proxy.batch_step(actions)
        except Exception as e:
            print(f"   ⚠️ Worker {self.worker_id} batch_step 失败: {e}")
            return []

        if not batch_results:
            return []

        experiences = []
        batch_state         = self.current_state.copy()
        batch_achieved_goal = self.current_achieved_goal

        for i, result in enumerate(batch_results):
            if isinstance(result, tuple) and len(result) >= 4:
                next_obs, reward, step_done, info = result[:4]
            else:
                continue

            # ── 关键修复：next_state 同样需要拼接 desired_goal ──
            if next_obs is not None:
                if isinstance(next_obs, dict):
                    raw_next = np.array(next_obs.get('observation', next_obs), dtype=np.float32).flatten()
                    next_achieved_goal = np.array(
                        next_obs.get('achieved_goal', np.zeros(GOAL_DIM)), dtype=np.float32)
                else:
                    raw_next = np.array(next_obs, dtype=np.float32).flatten()
                    next_achieved_goal = info.get('achieved_goal') if isinstance(info, dict) else None
                    if next_achieved_goal is not None:
                        next_achieved_goal = np.array(next_achieved_goal, dtype=np.float32)

                # desired_goal 在整个 episode 中不变
                next_state = _build_state(raw_next, self.desired_goal)
            else:
                next_state         = np.zeros_like(self.current_state)
                next_achieved_goal = batch_achieved_goal

            step_desired_goal = self.desired_goal

            # ── 关键修复：FetchReach 总是 truncated，terminated=False ──
            # 不截断 Q 值估计，对齐 SB3 handle_timeout_termination=True 的行为
            terminated = False  # FetchReach 从不真实终止

            exp = {
                'state':              batch_state.copy(),
                'action':             actions[i],
                'reward':             float(reward),
                'next_state':         next_state.copy(),
                'done':               bool(step_done),
                'terminated':         terminated,
                'achieved_goal':      batch_achieved_goal.copy() if batch_achieved_goal is not None else None,
                'desired_goal':       step_desired_goal.copy() if step_desired_goal is not None else None,
                'next_achieved_goal': next_achieved_goal.copy() if next_achieved_goal is not None else None,
            }

            self.episode_buffer.append(exp)
            experiences.append(exp)

            self.episode_reward += float(reward)
            self.episode_steps  += 1

            if isinstance(info, dict) and info.get('is_success'):
                self.success = True

            batch_state         = next_state
            batch_achieved_goal = next_achieved_goal

            if step_done:
                self.done = True
                break

        # 更新当前状态和 goal
        self.current_state         = batch_state
        self.current_achieved_goal = batch_achieved_goal
        return experiences

    def finalize_episode(self) -> List[Dict]:
        """完成 episode，返回 episode buffer 并清空"""
        episode_data        = self.episode_buffer.copy()
        self.episode_buffer = []
        return episode_data

    def get_episode_result(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "reward":    round(self.episode_reward, 3),
            "steps":     self.episode_steps,
            "success":   self.success,
            "done":      self.done,
        }


class ParallelTrainer:
    """并行DDPG训练器 - 多沙箱并行数据收集（支持HER，对齐 SB3 训练效果）

    对齐 local_parallel_trainer 的有效配置：
    - buffer_size=1_000_000, n_sampled_goal=8, batch_size=256
    - updates_per_episode=100（每轮 episode 结束后更新 100 次）
    - warmup_rounds=50（预热让 buffer 积累足够经验再更新）
    - 分段线性噪声衰减：前 30% 保持 noise_start，之后线性衰减到 noise_end
    """

    # 每个 worker 每次调用 batch_step 的步数
    MINI_BATCH_SIZE = 10
    # episode 最大步数
    MAX_STEPS_PER_EPISODE = MAX_STEPS

    def __init__(
        self,
        num_workers:         int   = 5,
        episodes:            int   = 2000,
        eval_freq:           int   = 100,
        learning_rate:       float = 1e-3,
        use_her:             bool  = True,
        # ─── 新增：对齐 local_parallel_trainer 的关键超参数 ───
        updates_per_episode: int   = 100,      # 每轮 episode 后的梯度更新次数
        n_sampled_goal:      int   = 8,        # HER 重标记目标数（SB3推荐8）
        warmup_rounds:       int   = 10,       # 预热轮数（降低以便快速验证演示）
        noise_start:         float = 0.3,      # 初始探索噪声
        noise_end:           float = 0.02,     # 最终探索噪声
        buffer_size:         int   = 1_000_000,# SB3 默认 buffer 容量
        log_interval:        int   = 10,       # 日志输出间隔（轮数，便于调试）
        demo_episodes:       int   = 3,        # 每次演示执行的 episode 数
        save_dir:            str   = None,     # 模型保存目录
    ):
        self.num_workers         = num_workers
        self.episodes            = episodes
        self.eval_freq           = eval_freq
        self.learning_rate       = learning_rate
        self.use_her             = use_her
        self.updates_per_episode = updates_per_episode
        self.n_sampled_goal      = n_sampled_goal
        self.warmup_rounds       = warmup_rounds
        self.noise_start         = noise_start
        self.noise_end           = noise_end
        self.buffer_size         = buffer_size
        self.log_interval        = log_interval
        self.demo_episodes       = demo_episodes
        
        # 模型保存目录（带时间戳，避免覆盖）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            self.save_dir = os.path.join(base_dir, f"run_{timestamp}")
        else:
            self.save_dir = os.path.join(save_dir, f"run_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📁 模型保存目录: {self.save_dir}")

        self.agent:   Optional[DDPGAgent]   = None
        self.workers: List[ParallelWorker]  = []
        # 统计窗口大小与日志输出频率一致
        self.stats                          = TrainingStats(window_size=self.log_interval)
        
        # 测试沙箱相关
        self.test_sandbox_proxy = None
        self.on_demo_complete: Optional[Callable] = None

    # ──────────────────── 初始化 ────────────────────

    def initialize_agent(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        """初始化共享的 DDPG 代理（支持 HER）"""
        self.agent = DDPGAgent(
            state_dim,
            action_dim,
            max_action,
            learning_rate=self.learning_rate,
            use_her=self.use_her,
            goal_dim=GOAL_DIM,
            n_sampled_goal=self.n_sampled_goal,
            buffer_size=self.buffer_size,
        )
        self.state_dim  = state_dim
        self.action_dim = action_dim

        print(f"✅ 并行DDPG训练器初始化完成 {'(HER 启用)' if self.use_her else ''}")
        print(f"   并行沙箱数量:     {self.num_workers}")
        print(f"   状态维度:         {state_dim}  动作维度: {action_dim}")
        print(f"   HER 重标记目标数: {self.n_sampled_goal}")
        print(f"   Replay Buffer 容量: {self.buffer_size:,}")
        print(f"   Batch Size: {self.agent.batch_size}, Gamma: {self.agent.gamma}, Tau: {self.agent.tau}")
        print(f"   每轮梯度更新:     {self.updates_per_episode} 次")

    def create_workers(self, env_proxies: List) -> bool:
        """创建并行工作器"""
        if not self.agent:
            raise RuntimeError("请先调用 initialize_agent() 初始化代理")
        self.workers = [ParallelWorker(i, ep, self.agent) for i, ep in enumerate(env_proxies)]
        print(f"✅ 创建了 {len(self.workers)} 个并行工作器")
        return len(self.workers) > 0

    # ──────────────────── 噪声计算 ────────────────────

    def _get_noise_scale(self, episode_idx: int) -> float:
        """分段线性衰减：前 30% 保持 noise_start，之后线性降到 noise_end"""
        warmup_frac = 0.3
        if episode_idx < self.episodes * warmup_frac:
            return self.noise_start
        decay = (episode_idx - self.episodes * warmup_frac) / (self.episodes * (1 - warmup_frac))
        return max(self.noise_end, self.noise_start - (self.noise_start - self.noise_end) * decay)
    
    # ──────────────────── 测试沙箱与演示 ────────────────────
    
    def set_test_sandbox(self, env_proxy, on_demo_complete: Callable = None):
        """设置测试沙箱代理，用于阶段性演示
        
        Args:
            env_proxy: 测试沙箱的环境代理
            on_demo_complete: 演示完成后的回调函数，接收演示结果 dict
        """
        self.test_sandbox_proxy = env_proxy
        self.on_demo_complete = on_demo_complete
        print(f"✅ 已设置测试沙箱，将在每次日志输出时进行演示")
    
    async def _run_demo(self, episode_idx: int, model_path: str) -> Dict[str, Any]:
        """在测试沙箱中运行模型演示
        
        Args:
            episode_idx: 当前训练轮次
            model_path: 模型文件路径
            
        Returns:
            演示结果字典
        """
        if not self.test_sandbox_proxy:
            return {"status": "skipped", "reason": "no_test_sandbox"}
        
        print(f"\n🎬 开始演示（轮次 {episode_idx}）...")
        
        demo_results = []
        total_success = 0
        
        try:
            for ep_idx in range(self.demo_episodes):
                # 重置测试沙箱环境
                try:
                    obs, reset_info = await self.test_sandbox_proxy.reset()
                except Exception as e:
                    print(f"   Episode {ep_idx + 1} reset 失败: {e}")
                    continue
                
                # 构建状态
                desired_goal = reset_info.get('desired_goal')
                if desired_goal is None and isinstance(obs, dict):
                    desired_goal = obs.get('desired_goal', np.zeros(GOAL_DIM, dtype=np.float32))
                desired_goal = np.array(desired_goal, dtype=np.float32) if desired_goal is not None else np.zeros(GOAL_DIM, dtype=np.float32)
                
                if isinstance(obs, dict):
                    raw_obs = np.array(obs.get('observation', obs), dtype=np.float32).flatten()
                else:
                    raw_obs = np.array(obs, dtype=np.float32).flatten()
                
                state = _build_state(raw_obs, desired_goal)
                episode_reward = 0.0
                success = False
                
                for step in range(MAX_STEPS):
                    # 使用训练好的策略选择动作（无噪声）
                    with torch.no_grad():
                        action = self.agent.select_action(state, noise_scale=0.0)
                    
                    try:
                        # env_proxy.step 返回 4 个值：(obs, reward, done, info)
                        next_obs, reward, done, info = await self.test_sandbox_proxy.step(action)
                    except Exception as e:
                        print(f"step 失败: {e}")
                        break
                    
                    # 处理下一状态
                    if isinstance(next_obs, dict):
                        raw_next = np.array(next_obs.get('observation', next_obs), dtype=np.float32).flatten()
                    else:
                        raw_next = np.array(next_obs, dtype=np.float32).flatten()
                    next_state = _build_state(raw_next, desired_goal)
                    
                    episode_reward += float(reward)
                    
                    # 检查成功条件：通过 info 或基于距离判断
                    if isinstance(info, dict):
                        if info.get('is_success'):
                            success = True
                        # 也检查 achieved_goal 和 desired_goal 的距离
                        achieved = info.get('achieved_goal')
                        if achieved is not None:
                            distance = np.linalg.norm(np.array(achieved) - desired_goal)
                            if distance < 0.05:  # FetchReach 成功阈值
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
            
            # 汇总统计
            if demo_results:
                avg_reward = np.mean([r["reward"] for r in demo_results])
                success_rate = total_success / self.demo_episodes
            else:
                avg_reward = 0
                success_rate = 0
            
            print(f"   演示结果: 平均奖励={avg_reward:.3f}, 成功率={total_success}(成功数)/{self.demo_episodes}(演示数)={success_rate:.1%}")
            
            result = {
                "status": "completed",
                "episode_idx": episode_idx,
                "model_path": model_path,
                "avg_reward": avg_reward,
                "success_rate": success_rate,
                "success_count": total_success,
                "demo_count": self.demo_episodes,
                "demo_results": demo_results,
            }
            
            # 调用回调
            if self.on_demo_complete and callable(self.on_demo_complete):
                try:
                    cb = self.on_demo_complete(result)
                    if asyncio.iscoroutine(cb):
                        await cb
                except Exception as e:
                    print(f"   ⚠️ 演示回调执行失败: {e}")
            
            return result
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ──────────────────── 单轮并行 episode ────────────────────

    async def train_episode_parallel(self, episode: int) -> Dict[str, Any]:
        """并行训练单个 episode（支持 HER episode 隔离）

        流程：
        1. 并行 reset 所有 worker
        2. 循环 mini-batch 收集，直到所有 worker done 或达到最大步数
        3. 每个 worker done 时立即触发 HER 处理（隔离，不跨 worker）
        4. 所有 worker 完成后，做 updates_per_episode 次梯度更新
        """
        if not self.workers:
            raise RuntimeError("没有可用的工作器，请先调用 create_workers()")

        # 1. 并行重置
        reset_results = await asyncio.gather(
            *[w.reset() for w in self.workers], return_exceptions=True
        )
        active_workers = [
            w for w, ok in zip(self.workers, reset_results) if ok is True
        ]
        if not active_workers:
            return {"episode": episode, "reward": 0, "steps": 0,
                    "success": False, "error": "所有 worker reset 失败"}

        noise_scale       = self._get_noise_scale(episode)
        total_steps       = 0
        processed_workers = set()

        # 2. 循环收集 mini-batch
        while total_steps < self.MAX_STEPS_PER_EPISODE * len(active_workers):
            running = [w for w in active_workers if not w.done]
            if not running:
                break

            batch_results = await asyncio.gather(
                *[w.collect_batch(self.MINI_BATCH_SIZE, noise_scale) for w in running],
                return_exceptions=True,
            )
            for w, exps in zip(running, batch_results):
                if isinstance(exps, list):
                    total_steps += len(exps)

            # 3. 每个 worker done 后独立触发 HER（不在此处做梯度更新）
            for w in active_workers:
                if w.done and w.worker_id not in processed_workers:
                    ep_data = w.finalize_episode()
                    if ep_data:
                        self.agent.process_worker_episode(ep_data, w.worker_id)
                    processed_workers.add(w.worker_id)

        # 处理尚未 done 但已达最大步数的 worker
        for w in active_workers:
            if w.worker_id not in processed_workers:
                ep_data = w.finalize_episode()
                if ep_data:
                    self.agent.process_worker_episode(ep_data, w.worker_id)
                processed_workers.add(w.worker_id)

        # 4. 梯度更新（episode 结束后集中更新，对齐 local 版本）
        in_warmup = episode < self.warmup_rounds
        update_losses = []
        if not in_warmup and len(self.agent.replay_buffer) >= self.agent.batch_size:
            for _ in range(self.updates_per_episode):
                loss_info = self.agent.update()
                if loss_info:
                    update_losses.append(loss_info)

        # 5. 汇总
        total_reward = sum(w.episode_reward for w in active_workers)
        avg_reward   = total_reward / len(active_workers)
        avg_steps    = sum(w.episode_steps for w in active_workers) / len(active_workers)
        any_success  = any(w.success for w in active_workers)
        success_cnt  = sum(w.success for w in active_workers)

        result = {
            "episode":        episode,
            "reward":         round(avg_reward, 3),
            "total_reward":   round(total_reward, 3),
            "steps":          int(avg_steps),
            "success":        any_success,
            "success_cnt":    success_cnt,
            "active_workers": len(active_workers),
            "buffer_size":    len(self.agent.replay_buffer),
            "noise_scale":    round(noise_scale, 3),
            "in_warmup":      in_warmup,
            "update_losses":  update_losses,
        }
        return result

    # ──────────────────── 主训练循环 ────────────────────

    async def train(
        self,
        env_proxies:         List,
        state_dim:           int,
        action_dim:          int,
        on_episode_complete: Callable = None,
    ) -> Dict[str, Any]:
        """执行并行训练

        Args:
            env_proxies:         环境代理列表（每个沙箱一个）
            state_dim:           状态维度
            action_dim:          动作维度
            on_episode_complete: episode 完成后的回调函数（支持异步）
        """
        self.initialize_agent(state_dim, action_dim)
        if not self.create_workers(env_proxies):
            return {"status": "error", "error": "创建工作器失败"}

        print(f"\n{'='*60}")
        print(f"🚀 开始并行 DDPG 训练（沙箱模式）")
        print(f"   并行沙箱数:       {len(self.workers)}")
        print(f"   总训练轮数:       {self.episodes}")
        print(f"   每轮梯度更新:     {self.updates_per_episode} 次")
        print(f"   预热轮数:         {self.warmup_rounds}")
        print(f"   探索噪声范围:     {self.noise_start} → {self.noise_end}")
        print(f"   日志输出间隔:     每 {self.log_interval} 轮")
        print(f"   HER:             {'开启 (重标记目标数=' + str(self.n_sampled_goal) + ')' if self.use_her else '关闭'}")
        print(f"{'='*60}")
        print(f"📊 指标口径说明:")
        print(f"   成功率 = 成功的Worker数 / 活跃Worker数（最近 {self.log_interval} 轮平均）")
        print(f"   平均奖励 = 最近 {self.log_interval} 轮的平均奖励")
        print(f"{'='*60}\n")

        start_time      = time.time()
        all_results     = []
        total_timesteps = 0
        total_episodes  = 0
        n_updates       = 0
        last_actor_loss  = 0.0
        last_critic_loss = 0.0

        try:
            for episode in range(self.episodes):
                result = await self.train_episode_parallel(episode)
                all_results.append(result)

                # 更新统计（使用 worker 成功率，而非 episode 成功率）
                # success_cnt / active_workers 表示这一轮中成功的 worker 比例
                worker_success_rate = result.get('success_cnt', 0) / max(result.get('active_workers', 1), 1)
                self.stats.add_episode(result['reward'], worker_success_rate, result['steps'])
                summary = self.stats.get_summary()

                # 累积计数
                round_steps      = result['steps'] * result.get('active_workers', self.num_workers)
                total_timesteps += round_steps
                total_episodes  += result.get('active_workers', self.num_workers)
                n_updates       += len(result.get('update_losses', []))

                losses = result.get('update_losses', [])
                if losses:
                    last_actor_loss  = np.mean([l.get('actor_loss', 0) for l in losses])
                    last_critic_loss = np.mean([l['critic_loss'] for l in losses])

                # 构建进度信息（用于前端回调）
                progress_info = {
                    "episode":          episode,
                    "total_episodes":   self.episodes,
                    "progress_percent": round((episode + 1) / self.episodes * 100, 1),
                    "current_result":   result,
                    "summary":          summary,
                    "parallel_info": {
                        "num_workers":    len(self.workers),
                        "active_workers": result.get('active_workers', 0),
                    },
                }

                if on_episode_complete and callable(on_episode_complete):
                    try:
                        cb = on_episode_complete(progress_info)
                        if asyncio.iscoroutine(cb):
                            await cb
                    except Exception as e:
                        pass  # 回调失败不中断训练

                # ⑤ 阶段性 SB3 风格输出（使用统一数据源 self.stats）
                if (episode + 1) % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = int(total_timesteps / elapsed) if elapsed > 0 else 0

                    # 从统一的 TrainingStats 获取指标
                    ep_rew_mean  = summary.get('avg_reward', 0)
                    ep_len_mean  = summary.get('avg_steps', 0)
                    success_rate = summary.get('success_rate', 0)
                    window_size  = summary.get('window_size', self.log_interval)
                    warmup_tag = " 【预热中】" if result.get('in_warmup') else ""

                    print("-" * 42)
                    print(f"| {'rollout/':<22} | {'':<13} |")
                    print(f"|    {'ep_len_mean':<19} | {ep_len_mean:<13.0f} |")
                    print(f"|    {'ep_rew_mean':<19} | {ep_rew_mean:<13.3f} |")
                    print(f"|    {'success_rate':<19} | {success_rate:<13.3f} |")
                    print(f"| {'time/':<22} | {'':<13} |")
                    print(f"|    {'episodes':<19} | {episode + 1:<13} |")
                    print(f"|    {'window_size':<19} | {window_size:<13} |")
                    print(f"|    {'fps':<19} | {fps:<13} |")
                    print(f"|    {'time_elapsed':<19} | {int(elapsed):<13} |")
                    print(f"|    {'total_timesteps':<19} | {total_timesteps:<13} |")
                    print(f"| {'train/':<22} | {'':<13} |")
                    print(f"|    {'actor_loss':<19} | {last_actor_loss:<13.5f} |")
                    print(f"|    {'critic_loss':<19} | {last_critic_loss:<13.5f} |")
                    print(f"|    {'learning_rate':<19} | {self.learning_rate:<13} |")
                    print(f"|    {'n_updates':<19} | {n_updates:<13} |")
                    print(f"|    {'noise_scale':<19} | {result['noise_scale']:<13.3f} |")
                    print(f"|    {'buffer_size':<19} | {result['buffer_size']:<13} |")
                    print(f"-" * 42 + warmup_tag)
                    
                    # 💾 保存模型
                    model_filename = f"ddpg_model_episode_{episode+1}.pth"
                    model_path = os.path.join(self.save_dir, model_filename)
                    self.agent.save_model(model_path)
                    
                    # 🎬 运行演示（跳过预热阶段）
                    if self.test_sandbox_proxy and episode >= self.warmup_rounds:
                        await self._run_demo(episode + 1, model_path)

        except Exception as e:
            print(f"❌ 训练出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status":        "error",
                "error":         str(e),
                "results":       all_results,
                "final_summary": self.stats.get_summary(),
            }

        final_summary = self.stats.get_summary()
        elapsed_total = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"🎉 并行训练完成！")
        print(f"   总轮数:        {len(all_results)}")
        print(f"   并行沙箱数:    {len(self.workers)}")
        print(f"   总耗时:        {elapsed_total/60:.1f} 分钟")
        print(f"   全局平均奖励:  {final_summary.get('global_avg_reward', 0):.3f}")
        print(f"   最终成功率:    {final_summary.get('success_rate', 0):.1%}")
        print(f"   训练效果:      {final_summary.get('effectiveness_cn', '未知')}")
        print(f"{'='*60}")

        return {
            "status":          "completed",
            "total_episodes":  len(all_results),
            "average_reward":  final_summary.get('global_avg_reward', 0),
            "success_rate":    final_summary.get('success_rate', 0),
            "final_summary":   final_summary,
            "parallel_info": {
                "num_workers":      len(self.workers),
                "total_experiences": len(self.agent.replay_buffer) if self.agent else 0,
            },
            "results": all_results[-100:],
        }
