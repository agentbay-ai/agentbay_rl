"""自定义PyTorch DDPG训练器（支持HER）"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import random
from typing import Optional, Dict, Any, Callable, List, Tuple
from collections import deque


class Actor(nn.Module):
    """Actor网络"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    """Critic网络"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.l1(sa))
        q = torch.relu(self.l2(q))
        return self.l3(q)


class HERReplayBuffer:
    """
    Hindsight Experience Replay Buffer
    
    核心思想：对于失败的episode，用实际达成的位置作为"假装的目标"，
    这样即使没达到真正目标也能学到有用的东西。
    
    目标重标记策略：FUTURE - 用episode后续步骤中达成的位置作为新目标
    """
    
    def __init__(self, capacity: int = 1_000_000, goal_dim: int = 3, 
                 n_sampled_goal: int = 4, distance_threshold: float = 0.05):
        """
        Args:
            capacity: 缓冲区容量
            goal_dim: 目标维度（FetchReach为3，即xyz坐标）
            n_sampled_goal: 每个transition重标记的目标数量
            distance_threshold: 判断成功的距离阈值
        """
        self.capacity = capacity
        self.goal_dim = goal_dim
        self.n_sampled_goal = n_sampled_goal
        self.distance_threshold = distance_threshold
        
        # 主缓冲区：存储原始经验
        self.buffer = deque(maxlen=capacity)
        
        # episode缓冲区：临时存储当前episode的transition用于HER处理
        self.episode_buffer: List[Dict] = []
        
    def add_transition(self, state: np.ndarray, action: np.ndarray, 
                      reward: float, next_state: np.ndarray, done: bool,
                      achieved_goal: np.ndarray, desired_goal: np.ndarray,
                      next_achieved_goal: np.ndarray):
        """添加单个transition到episode缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 原始奖励
            next_state: 下一状态
            done: 是否结束
            achieved_goal: 当前达成的目标位置
            desired_goal: 期望的目标位置
            next_achieved_goal: 下一步达成的目标位置
        """
        transition = {
            'state': state.copy() if hasattr(state, 'copy') else np.array(state),
            'action': action.copy() if hasattr(action, 'copy') else np.array(action),
            'reward': reward,
            'next_state': next_state.copy() if hasattr(next_state, 'copy') else np.array(next_state),
            'done': done,
            'achieved_goal': achieved_goal.copy() if hasattr(achieved_goal, 'copy') else np.array(achieved_goal) if achieved_goal is not None else None,
            'desired_goal': desired_goal.copy() if hasattr(desired_goal, 'copy') else np.array(desired_goal) if desired_goal is not None else None,
            'next_achieved_goal': next_achieved_goal.copy() if hasattr(next_achieved_goal, 'copy') else np.array(next_achieved_goal) if next_achieved_goal is not None else None,
        }
        self.episode_buffer.append(transition)
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """计算reward（基于距离）
        
        FetchReach环境使用的dense reward计算方式
        """
        if achieved_goal is None or desired_goal is None:
            return -1.0
        distance = np.linalg.norm(np.array(achieved_goal) - np.array(desired_goal))
        # Dense reward：负距离
        return -distance
    
    def end_episode(self):
        """episode结束时，执行HER目标重标记（处理内部episode_buffer）
        
        使用FUTURE策略：对于每个transition，从其后续步骤中随机选择
        n_sampled_goal个achieved_goal作为新的desired_goal
        """
        if len(self.episode_buffer) == 0:
            return
        self.process_episode(self.episode_buffer)
        self.episode_buffer = []
    
    def process_episode(self, episode_transitions: List[Dict], worker_id: int = -1):
        """处理一个episode的transitions，执行HER目标重标记
        
        此方法可处理外部传入的episode buffer（用于并行训练场景）
        
        关键修复（对齐 SB3 HER 行为）：
        - 原始经验使用 terminated（而非 done=terminated|truncated）作为 TD target 的 done 标志
          FetchReach 总是 truncated 终止，不应截断 Q 值估计
        - HER 重标记时必须将 state/next_state 末尾的 desired_goal 部分替换为 new_goal
          否则网络看到 state="目标X" 但 reward 按 "目标Y" 计算，造成严重矛盾
        
        Args:
            episode_transitions: episode的transition列表
            worker_id: worker标识（用于日志，-1表示非并行模式）
        """
        episode_length = len(episode_transitions)
        if episode_length == 0:
            return
        
        # 1. 先添加原始经验到主缓冲区
        # 用 terminated（不含 timeout）作为 done，避免截断 FetchReach 的 Q 值估计
        for t in episode_transitions:
            done_for_td = t.get('terminated', t['done'])  # 优先使用 terminated
            self.buffer.append((
                t['state'],
                t['action'],
                t['reward'],
                t['next_state'],
                done_for_td
            ))
        
        # 2. HER：为每个transition生成额外的重标记经验
        her_count = 0
        for idx, transition in enumerate(episode_transitions):
            # 跳过没有goal信息的transition
            if transition['achieved_goal'] is None or transition['next_achieved_goal'] is None:
                continue
            
            # FUTURE策略：从后续步骤中选择目标（含当前步，对齐 SB3 inclusive 策略）
            future_indices = list(range(idx, episode_length))
            if not future_indices:
                continue
            
            # 随机选择 n_sampled_goal 个future目标
            n_goals = min(self.n_sampled_goal, len(future_indices))
            selected_indices = random.sample(future_indices, n_goals)
            
            for future_idx in selected_indices:
                # 用future step的achieved_goal作为新的desired_goal
                new_goal = episode_transitions[future_idx]['achieved_goal']
                if new_goal is None:
                    continue
                
                # 重新计算reward（用 next_achieved_goal 对齐 SB3 语义）
                new_reward = self.compute_reward(
                    transition['next_achieved_goal'], 
                    new_goal
                )
                
                # ── 关键修复：重建 state/next_state，替换末尾 goal 部分 ──
                # state = concat([obs (10d), desired_goal (3d)])
                # HER 后 state 应为 concat([obs (10d), new_goal (3d)])
                obs_dim = len(transition['state']) - self.goal_dim
                her_state = np.concatenate([
                    transition['state'][:obs_dim], new_goal
                ]).astype(np.float32)
                her_next_state = np.concatenate([
                    transition['next_state'][:obs_dim], new_goal
                ]).astype(np.float32)
                
                done_for_td = transition.get('terminated', transition['done'])
                self.buffer.append((
                    her_state,
                    transition['action'],
                    new_reward,
                    her_next_state,
                    done_for_td
                ))
                her_count += 1
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """DDPG代理（支持HER）
    
    对齐 SB3 DDPG 默认参数：
    - learning_rate: 1e-3 (actor 和 critic 使用相同学习率)
    - buffer_size: 1_000_000
    - batch_size: 256
    - tau: 0.005
    - gamma: 0.99
    """
    def __init__(self, state_dim, action_dim, max_action, learning_rate=1e-3,
                 use_her: bool = True, goal_dim: int = 3, n_sampled_goal: int = 4,
                 buffer_size: int = 1_000_000, gamma: float = 0.99, tau: float = 0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.use_her = use_her
        
        # 网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # 复制权重到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器（SB3 默认：actor 和 critic 使用相同学习率 1e-3）
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Replay Buffer（SB3 默认容量 1_000_000）
        if use_her:
            self.replay_buffer = HERReplayBuffer(
                capacity=buffer_size,
                goal_dim=goal_dim,
                n_sampled_goal=n_sampled_goal
            )
        else:
            self.replay_buffer = deque(maxlen=buffer_size)
        
        # 超参数（支持自定义，默认对齐 SB3 DDPG）
        self.batch_size = 256   # SB3 默认 256
        self.gamma = gamma      # 折扣因子（默认 0.99，FetchPickAndPlace 推荐 0.95）
        self.tau = tau          # 软更新率（默认 0.005，FetchPickAndPlace 推荐 0.05）
        self.grad_clip = 1.0    # 梯度裁剪阈值
        
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """选择动作（带探索噪声）"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_tensor).cpu().numpy()[0]
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        return action
    
    def add_experience(self, state, action, reward, next_state, done,
                      achieved_goal=None, desired_goal=None, next_achieved_goal=None):
        """添加经验到 Replay Buffer"""
        if self.use_her and isinstance(self.replay_buffer, HERReplayBuffer):
            self.replay_buffer.add_transition(
                state, action, reward, next_state, done,
                achieved_goal, desired_goal, next_achieved_goal
            )
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))
    
    def end_episode(self):
        """episode结束时调用（触发HER处理）"""
        if self.use_her and isinstance(self.replay_buffer, HERReplayBuffer):
            self.replay_buffer.end_episode()
    
    def process_worker_episode(self, episode_transitions: List[Dict], worker_id: int):
        """处理某个worker的独立episode（用于并行训练的HER隔离）
        
        Args:
            episode_transitions: 该worker的episode transition列表
            worker_id: worker ID
        """
        if self.use_her and isinstance(self.replay_buffer, HERReplayBuffer):
            self.replay_buffer.process_episode(episode_transitions, worker_id)
    
    def update(self) -> Dict[str, float]:
        """从 Replay Buffer 采样并更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 采样
        if self.use_her and isinstance(self.replay_buffer, HERReplayBuffer):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        else:
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 更新 Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # 更新 Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }
    
    def _soft_update(self, target_net, source_net):
        """软更新目标网络"""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """保存模型权重和训练参数
        
        Args:
            filepath: 保存路径（.pth 文件）
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'max_action': self.max_action,
                'gamma': self.gamma,
                'tau': self.tau,
                'batch_size': self.batch_size,
                'use_her': self.use_her,
            }
        }
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        """加载模型权重
        
        Args:
            filepath: 模型文件路径（.pth 文件）
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class TrainingStats:
    """训练统计管理器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards_history = deque(maxlen=window_size)
        self.success_history = deque(maxlen=window_size)
        self.steps_history = deque(maxlen=window_size)
        self.all_rewards = []  # 保存所有奖励用于趋势分析
        
    def add_episode(self, reward: float, success: float, steps: int):
        """添加 episode 数据
        
        Args:
            reward: episode 奖励
            success: 成功率（0-1 之间的浮点数，兼容布尔值）
            steps: episode 步数
        """
        self.rewards_history.append(reward)
        # 支持布尔值和浮点数两种输入
        if isinstance(success, bool):
            self.success_history.append(1.0 if success else 0.0)
        else:
            self.success_history.append(float(success))
        self.steps_history.append(steps)
        self.all_rewards.append(reward)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练进度总结"""
        if not self.rewards_history:
            return {}
        
        # 最近窗口统计
        recent_rewards = list(self.rewards_history)
        recent_success = list(self.success_history)
        recent_steps = list(self.steps_history)
        
        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 0
        max_reward = np.max(recent_rewards)
        min_reward = np.min(recent_rewards)
        success_rate = np.mean(recent_success)
        avg_steps = np.mean(recent_steps)
        
        # 趋势分析（比较前半窗口和后半窗口）
        trend = "stable"
        trend_value = 0
        if len(recent_rewards) >= 20:
            mid = len(recent_rewards) // 2
            first_half_avg = np.mean(recent_rewards[:mid])
            second_half_avg = np.mean(recent_rewards[mid:])
            trend_value = second_half_avg - first_half_avg
            
            if trend_value > 0.5:
                trend = "improving"
            elif trend_value < -0.5:
                trend = "declining"
            else:
                trend = "stable"
        
        # 全局统计
        total_episodes = len(self.all_rewards)
        global_avg_reward = np.mean(self.all_rewards) if self.all_rewards else 0
        
        # 训练效果评估
        if avg_reward > -5:
            effectiveness = "excellent"
            effectiveness_cn = "优秀"
        elif avg_reward > -10:
            effectiveness = "good"
            effectiveness_cn = "良好"
        elif avg_reward > -20:
            effectiveness = "moderate"
            effectiveness_cn = "中等"
        else:
            effectiveness = "needs_improvement"
            effectiveness_cn = "需改进"
        
        return {
            "window_size": len(recent_rewards),
            "avg_reward": round(avg_reward, 3),
            "std_reward": round(std_reward, 3),
            "max_reward": round(max_reward, 3),
            "min_reward": round(min_reward, 3),
            "success_rate": round(success_rate, 3),
            "avg_steps": round(avg_steps, 1),
            "trend": trend,
            "trend_value": round(trend_value, 3),
            "total_episodes": total_episodes,
            "global_avg_reward": round(global_avg_reward, 3),
            "effectiveness": effectiveness,
            "effectiveness_cn": effectiveness_cn
        }


class CustomTrainer:
    """自定义PyTorch DDPG训练器（支持HER）"""
    
    # Episode最大步数（也用作batch大小，减少沙箱通信次数）
    MAX_STEPS_PER_EPISODE = 100
    
    def __init__(self, episodes: int = 1000000, eval_freq: int = 25000, 
                 learning_rate: float = 1e-3, use_her: bool = True):
        self.episodes = episodes
        self.eval_freq = eval_freq
        # batch大小设为episode最大步数，每个episode只需一次批量通信
        self.batch_size = self.MAX_STEPS_PER_EPISODE
        self.learning_rate = learning_rate
        self.use_her = use_her  # 是否使用HER
        self.agent = None
        self.stats = TrainingStats(window_size=100)
        
    def initialize_agent(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        """初始化DDPG代理
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 最大动作值
        """
        self.agent = DDPGAgent(
            state_dim, action_dim, max_action, 
            learning_rate=self.learning_rate,
            use_her=self.use_her,
            goal_dim=3,  # FetchReach目标维度为3（xyz坐标）
            n_sampled_goal=4  # 每个transition重标记4次
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        print(f"✅ DDPG代理初始化完成 {'(HER启用)' if self.use_her else ''}")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
        print(f"   Episode最大步数: {self.MAX_STEPS_PER_EPISODE}")
        print(f"   小批量大小: {self.MINI_BATCH_SIZE} (每episode约{self.MAX_STEPS_PER_EPISODE // self.MINI_BATCH_SIZE}次沙箱通信)")
        if self.use_her:
            print(f"   HER配置: goal_dim=3, n_sampled_goal=4")
    
    # 小批量大小：每次基于当前真实状态生成这么多动作
    # 较小的值 → 动作质量更高，但沙箱通信更多
    # 较大的值 → 沙箱通信更少，但动作质量下降
    MINI_BATCH_SIZE = 10  # 每10步重新基于真实状态生成动作
    
    async def train_episode_batch(self, env_proxy, episode: int) -> Dict[str, Any]:
        """使用小批量交互训练单个episode（支持HER）
        
        改进的批量训练流程（平衡效率与质量）：
        - 每 MINI_BATCH_SIZE 步基于真实状态重新生成动作
        - 相比单步执行：通信次数减少 MINI_BATCH_SIZE 倍
        - 相比全episode批量：动作质量大幅提升
        - HER：episode结束后进行目标重标记，大幅提升样本效率
        
        Args:
            env_proxy: 环境代理
            episode: episode编号
            
        Returns:
            Dict[str, Any]: episode结果
        """
        if not self.agent:
            raise RuntimeError("请先调用initialize_agent()初始化代理")
        
        # 1. 重置环境获取初始状态和goal信息
        obs, reset_info = await env_proxy.reset()
        current_state = np.array(obs)
        
        # 获取初始goal信息（HER需要）
        current_achieved_goal = reset_info.get('achieved_goal')
        desired_goal = reset_info.get('desired_goal')
        
        if current_state.size == 0:
            print(f"   ⚠️ Episode {episode}: reset返回空状态")
            return {
                "episode": episode,
                "reward": 0,
                "steps": 0,
                "success": False,
                "error": "reset返回空状态"
            }
        
        # 探索噪声：随训练进行逐渐减小
        noise_scale = max(0.05, 0.3 - episode / (self.episodes * 2))
        
        episode_reward = 0
        episode_steps = 0
        success = False
        done = False
        
        # 2. 小批量循环：每次基于当前真实状态生成一小批动作
        while episode_steps < self.MAX_STEPS_PER_EPISODE and not done:
            # 计算这一批要执行多少步
            remaining_steps = self.MAX_STEPS_PER_EPISODE - episode_steps
            batch_size = min(self.MINI_BATCH_SIZE, remaining_steps)
            
            # 基于当前真实状态生成一批动作
            actions = []
            for _ in range(batch_size):
                action = self.agent.select_action(current_state, noise_scale)
                actions.append(action)
            
            # 批量执行这一小批动作
            batch_results = await env_proxy.batch_step(actions)
            
            if not batch_results:
                print(f"   ⚠️ Episode {episode}: 小批量执行返回空结果")
                break
            
            # 处理批量结果
            batch_state = current_state.copy()
            batch_achieved_goal = current_achieved_goal
            
            for i, result in enumerate(batch_results):
                if isinstance(result, tuple) and len(result) >= 4:
                    next_obs, reward, step_done, info = result[:4]
                else:
                    continue
                
                next_state = np.array(next_obs) if next_obs else np.zeros_like(current_state)
                
                # 获取goal信息（HER需要）
                next_achieved_goal = info.get('achieved_goal') if isinstance(info, dict) else None
                step_desired_goal = info.get('desired_goal', desired_goal) if isinstance(info, dict) else desired_goal
                
                # 累计统计
                episode_reward += reward
                episode_steps += 1
                
                # 检查成功
                if isinstance(info, dict) and info.get('is_success'):
                    success = True
                
                # 添加经验到Replay Buffer（包含goal信息用于HER）
                self.agent.add_experience(
                    batch_state,
                    actions[i],
                    reward,
                    next_state,
                    step_done,
                    achieved_goal=batch_achieved_goal,
                    desired_goal=step_desired_goal,
                    next_achieved_goal=next_achieved_goal
                )
                
                batch_state = next_state
                batch_achieved_goal = next_achieved_goal
                
                if step_done:
                    done = True
                    break
            
            # 更新当前状态和goal
            current_state = batch_state
            current_achieved_goal = batch_achieved_goal
            
            # 每个小批量后进行模型更新
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                self.agent.update()
        
        # 3. episode结束：触发HER处理（目标重标记）
        self.agent.end_episode()
        
        # episode结束后额外更新几次
        extra_updates = min(episode_steps // 10, 5)
        for _ in range(extra_updates):
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                self.agent.update()
        
        return {
            "episode": episode,
            "reward": round(episode_reward, 3),
            "steps": episode_steps,
            "success": success,
            "buffer_size": len(self.agent.replay_buffer)
        }
    
    async def train(self, env_proxy, state_dim: int, action_dim: int,
                   on_episode_complete: Callable = None) -> Dict[str, Any]:
        """执行训练
        
        Args:
            env_proxy: 环境代理
            state_dim: 状态维度
            action_dim: 动作维度
            on_episode_complete: episode完成后的回调函数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        # 初始化代理
        self.initialize_agent(state_dim, action_dim)
        
        print(f"🚀 开始自定义DDPG训练（批量交互优化）")
        print(f"   总episodes: {self.episodes}")
        print(f"   评估频率: 每{self.eval_freq}个episode")
        print(f"   每episode最大步数: {self.MAX_STEPS_PER_EPISODE}")
        
        all_results = []
        
        try:
            for episode in range(self.episodes):
                # 使用批量交互训练episode
                result = await self.train_episode_batch(env_proxy, episode)
                all_results.append(result)
                
                # 更新统计
                self.stats.add_episode(
                    result['reward'], 
                    result['success'], 
                    result['steps']
                )
                
                # 获取训练进度总结
                summary = self.stats.get_summary()
                
                # 构建进度信息
                progress_info = {
                    "episode": episode,
                    "total_episodes": self.episodes,
                    "progress_percent": round((episode + 1) / self.episodes * 100, 1),
                    "current_result": result,
                    "summary": summary
                }
                
                # 回调通知
                if on_episode_complete and callable(on_episode_complete):
                    try:
                        callback_result = on_episode_complete(progress_info)
                        if asyncio.iscoroutine(callback_result):
                            await callback_result
                    except Exception as e:
                        print(f"⚠️ Episode回调执行失败: {e}")
                
                # 每个episode都打印基本进度
                buffer_size = result.get('buffer_size', 0)
                print(f"🏃 Episode {episode}/{self.episodes} | "
                      f"奖励: {result['reward']:.2f} | "
                      f"步数: {result['steps']} | "
                      f"成功: {'✅' if result['success'] else '❌'} | "
                      f"Buffer: {buffer_size}")
                
                # 每10个episode打印详细统计
                if episode > 0 and episode % 10 == 0:
                    trend_emoji = "📈" if summary.get('trend') == 'improving' else \
                                 "📉" if summary.get('trend') == 'declining' else "➡️"
                    print(f"   📊 统计: 平均奖励={summary.get('avg_reward', 0):.2f}, "
                          f"成功率={summary.get('success_rate', 0):.1%}, "
                          f"趋势: {trend_emoji}")
                
                # 定期评估
                if episode > 0 and episode % self.eval_freq == 0:
                    print(f"\n🔄 评估模型性能 @ Episode {episode}")
                    print(f"   效果评估: {summary.get('effectiveness_cn', '未知')}")
                    print(f"   趋势: {summary.get('trend', 'unknown')} ({summary.get('trend_value', 0):+.2f})")
                    print()
                    
        except Exception as e:
            print(f"❌ 训练出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "results": all_results,
                "final_summary": self.stats.get_summary()
            }
        
        # 最终统计
        final_summary = self.stats.get_summary()
        
        result = {
            "status": "completed",
            "total_episodes": len(all_results),
            "average_reward": final_summary.get('global_avg_reward', 0),
            "success_rate": final_summary.get('success_rate', 0),
            "final_summary": final_summary,
            "results": all_results[-100:]  # 只保留最后100个结果
        }
        
        print(f"\n🎉 训练完成!")
        print(f"   总episodes: {len(all_results)}")
        print(f"   全局平均奖励: {final_summary.get('global_avg_reward', 0):.2f}")
        print(f"   最终成功率: {final_summary.get('success_rate', 0):.1%}")
        print(f"   训练效果: {final_summary.get('effectiveness_cn', '未知')}")
        print(f"   训练趋势: {final_summary.get('trend', 'unknown')}")
        
        return result