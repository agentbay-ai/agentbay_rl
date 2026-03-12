"""Stable-Baselines3 DDPG训练器 - 支持多沙箱并行执行"""

import os
import json
import time
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor

# 启用嵌套事件循环支持（允许在已运行的事件循环中再次调用 run_until_complete）
import nest_asyncio
nest_asyncio.apply()

from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import gymnasium as gym
from gymnasium import spaces

from ..communication.env_proxy import EnvProxy
from ..communication.sandbox_gym_env import SandboxGymEnv
from ..sandbox_components.sandbox_setup import SandboxSetup


class AsyncSandboxVecEnv(VecEnv):
    """异步并行沙箱向量化环境
    
    真正并行执行多个沙箱的 step 操作，而不是像 DummyVecEnv 那样顺序执行。
    """
    
    def __init__(self, env_proxies: List[EnvProxy], sandbox_manager, main_loop=None,
                 obs_dim=10, goal_dim=3, action_dim=4, max_action=1.0,
                 env_name: str = "FetchReachDense-v4"):
        self.env_proxies = env_proxies
        self.sandbox_manager = sandbox_manager
        self.num_envs = len(env_proxies)
        
        # 保存主事件循环的引用（MCP 客户端绑定到这个循环）
        self._main_loop = main_loop
        
        # 环境名称（用于 compute_reward 判断密集/稀疏奖励类型）
        self._env_name = env_name
        
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 定义 observation 和 action space
        observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32),
        })
        action_space = spaces.Box(low=-max_action, high=max_action, shape=(action_dim,), dtype=np.float32)
        
        super().__init__(self.num_envs, observation_space, action_space)
        
        # 存储待发送的 actions
        self._actions = None
        
        # 当前状态
        self._obs = None
        self._infos = [{} for _ in range(self.num_envs)]
        
        # 主事件循环引用（必须在创建 VecEnv 时设置）
        self._main_loop = main_loop
    
    def _run_async(self, coro):
        """将协程提交到主事件循环执行（从工作线程安全调用）
        
        工作流：model.learn() 在独立线程运行 → VecEnv 同步方法调用此函数 
        → run_coroutine_threadsafe 提交协程到主 uvloop → MCP 正常工作
        """
        if self._main_loop is None:
            raise RuntimeError("主事件循环未设置，请在 create_vec_env 时传入 main_loop")
        
        # 从工作线程提交协程到主事件循环（不会阻塞主循环）
        future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        return future.result(timeout=120)  # 等待结果（最多 120 秒）
    
    def close(self):
        """关闭环境"""
        pass
    
    async def _parallel_step(self, actions: np.ndarray):
        """并行执行所有沙箱的 step"""
        async def single_step(idx, action):
            try:
                result = await self.env_proxies[idx].step(action.tolist())
                return idx, result
            except Exception as e:
                print(f"⚠️ 沙箱 {idx} step 失败: {e}")
                # 返回默认值
                return idx, (self._make_default_obs(), -1.0, True, {"error": str(e)})
        
        # 并行发送所有 step 请求
        tasks = [single_step(i, actions[i]) for i in range(self.num_envs)]
        results = await asyncio.gather(*tasks)
        
        # 按索引排序结果
        results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in results]
    
    async def _parallel_auto_reset(self, done_indices: list):
        """对 done=True 的环境进行并行 reset（遵循 DummyVecEnv 规范）"""
        async def single_reset(idx):
            try:
                result = await self.env_proxies[idx].reset()
                return idx, result
            except Exception as e:
                print(f"⚠️ 沙箱 {idx} auto-reset 失败: {e}")
                return idx, (self._make_default_obs(), {})
        
        tasks = [single_reset(i) for i in done_indices]
        results = await asyncio.gather(*tasks)
        # 返回 {idx: (obs, info)} 字典
        return {idx: res for idx, res in results}
    
    async def _parallel_reset(self):
        """并行重置所有沙箱"""
        print(f"   📡 _parallel_reset: 开始并行发送 reset 请求...")
        
        async def single_reset(idx):
            try:
                print(f"      沙箱 {idx}: 发送 reset 请求...")
                result = await self.env_proxies[idx].reset()
                print(f"      沙箱 {idx}: reset 完成")
                return idx, result
            except Exception as e:
                print(f"⚠️ 沙箱 {idx} reset 失败: {e}")
                return idx, (self._make_default_obs(), {"error": str(e)})
        
        tasks = [single_reset(i) for i in range(self.num_envs)]
        print(f"   📡 _parallel_reset: 等待 {len(tasks)} 个任务完成...")
        results = await asyncio.gather(*tasks)
        print(f"   📡 _parallel_reset: 所有任务完成")
        results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in results]
    
    def _make_default_obs(self):
        """创建默认 observation"""
        return {
            "observation": np.zeros(self.obs_dim, dtype=np.float32),
            "achieved_goal": np.zeros(self.goal_dim, dtype=np.float32),
            "desired_goal": np.zeros(self.goal_dim, dtype=np.float32),
        }
    
    def _format_obs(self, raw_obs, info):
        """格式化 observation 为标准字典格式"""
        if isinstance(raw_obs, dict) and "observation" in raw_obs:
            return {
                "observation": np.array(raw_obs["observation"], dtype=np.float32),
                "achieved_goal": np.array(raw_obs.get("achieved_goal", np.zeros(self.goal_dim)), dtype=np.float32),
                "desired_goal": np.array(raw_obs.get("desired_goal", np.zeros(self.goal_dim)), dtype=np.float32),
            }
        elif isinstance(raw_obs, (list, np.ndarray)):
            obs_array = np.array(raw_obs, dtype=np.float32)
            return {
                "observation": obs_array[:self.obs_dim] if len(obs_array) >= self.obs_dim else np.zeros(self.obs_dim, dtype=np.float32),
                "achieved_goal": info.get("achieved_goal", np.zeros(self.goal_dim, dtype=np.float32)),
                "desired_goal": info.get("desired_goal", np.zeros(self.goal_dim, dtype=np.float32)),
            }
        return self._make_default_obs()
    
    def step_async(self, actions: np.ndarray):
        """异步发送 step 请求（SB3 VecEnv 接口）"""
        self._actions = actions
    
    def step_wait(self):
        """等待 step 结果（SB3 VecEnv 接口）
        
        遵循 DummyVecEnv 规范：
        - 当 done=True 时，把终止观测存入 infos["terminal_observation"]
        - 自动调用 reset() 获取下一 episode 的初始观测并返回
        这对于 HER 缓冲区正确存储 next_obs 至关重要。
        """
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        
        # 并行执行所有 step
        results = self._run_async(self._parallel_step(self._actions))
        self._actions = None
        
        # 解析结果
        obs_list = []
        rewards = []
        dones = []
        infos = []
        
        for i, result in enumerate(results):
            raw_obs, reward, done, info = result
            obs_list.append(self._format_obs(raw_obs, info))
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info if isinstance(info, dict) else {})
        
        # ===== 关键：模拟 DummyVecEnv 的自动 reset 行为 =====
        # 当 done=True 时：
        # 1. 保存终止观测到 infos["terminal_observation"]（SB3 _store_transition 会用它）
        # 2. 并行 reset 这些环境，返回新 episode 的初始观测
        done_indices = [i for i, d in enumerate(dones) if d]
        if done_indices:
            # 保存终止观测（格式与观测空间一致的字典）
            for idx in done_indices:
                infos[idx]["terminal_observation"] = {
                    "observation": obs_list[idx]["observation"].copy(),
                    "achieved_goal": obs_list[idx]["achieved_goal"].copy(),
                    "desired_goal": obs_list[idx]["desired_goal"].copy(),
                }
            
            # 并行 reset done 的环境，获取下一 episode 的初始观测
            reset_results = self._run_async(self._parallel_auto_reset(done_indices))
            
            # 用初始观测替换 done 环境的观测
            for idx in done_indices:
                raw_reset_obs, reset_info = reset_results[idx]
                obs_list[idx] = self._format_obs(raw_reset_obs, reset_info)
        
        # 转换为 VecEnv 格式
        obs = {
            "observation": np.array([o["observation"] for o in obs_list], dtype=np.float32),
            "achieved_goal": np.array([o["achieved_goal"] for o in obs_list], dtype=np.float32),
            "desired_goal": np.array([o["desired_goal"] for o in obs_list], dtype=np.float32),
        }
        
        self._obs = obs
        self._infos = infos
        
        return obs, np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos
    
    def reset(self):
        """重置所有环境"""
        print(f"🔄 AsyncSandboxVecEnv.reset() 开始并行重置 {self.num_envs} 个环境...")
        results = self._run_async(self._parallel_reset())
        print(f"🔄 AsyncSandboxVecEnv.reset() 收到 {len(results)} 个结果")
        
        obs_list = []
        infos = []
        
        for i, result in enumerate(results):
            raw_obs, info = result
            obs_list.append(self._format_obs(raw_obs, info))
            infos.append(info if isinstance(info, dict) else {})
        
        obs = {
            "observation": np.array([o["observation"] for o in obs_list], dtype=np.float32),
            "achieved_goal": np.array([o["achieved_goal"] for o in obs_list], dtype=np.float32),
            "desired_goal": np.array([o["desired_goal"] for o in obs_list], dtype=np.float32),
        }
        
        self._obs = obs
        self._infos = infos
        
        print(f"✅ AsyncSandboxVecEnv.reset() 完成")
        return obs
    
    def close(self):
        """关闭环境"""
        pass
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """调用环境方法
        
        HER 会调用 compute_reward 重新计算虚拟目标的奖励，必须正确实现。
        FetchReachDense-v4 的奖励 = -||achieved_goal - desired_goal||（欧式距离的负值）
        FetchReach-v4 的奖励 = -(||achieved_goal - desired_goal|| > 阈值)（稀疏，-1或0）
        """
        if method_name == "compute_reward":
            # method_args: (achieved_goal, desired_goal, info)
            # 形状: achieved_goal (N, 3), desired_goal (N, 3)
            achieved_goal = method_args[0]
            desired_goal = method_args[1]
            info = method_args[2] if len(method_args) > 2 else None
            
            # 计算欧式距离
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            
            # 判断是密集奖励还是稀疏奖励
            # 根据沙箱环境配置，默认使用密集奖励（FetchReachDense-v4）
            env_name = getattr(self, '_env_name', 'FetchReachDense-v4')
            if 'Dense' in env_name:
                rewards = -d  # 密集奖励：负距离
            else:
                rewards = -(d > 0.05).astype(np.float32)  # 稀疏奖励：阈值0.05米
            
            # HER 期望返回 [rewards_array] 格式（长度等于 num_envs 的列表，每个元素是一个数组）
            # 但实际上 HER 调用时 indices 参数指定了哪个 env，通常是全部
            n_indices = self.num_envs if indices is None else len(indices)
            # HER 实际上期望 env_method 返回 list，每个元素对应一个 env 的结果
            # 但由于是批量计算，直接返回包含整个 rewards 数组的列表
            return [rewards]
        
        return [None] * self.num_envs
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """检查环境是否被包装"""
        return [False] * self.num_envs
    
    def get_attr(self, attr_name, indices=None):
        """获取环境属性"""
        return [None] * self.num_envs
    
    def set_attr(self, attr_name, value, indices=None):
        """设置环境属性"""
        pass
    
    def seed(self, seed=None):
        """设置随机种子"""
        return [seed] * self.num_envs


class ProgressCallback(BaseCallback):
    """训练进度回调 - 包含成功率、奖励等指标"""
    
    def __init__(self, total_timesteps: int, on_progress: Callable = None, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.on_progress = on_progress
        self.last_log_step = 0
        self.log_interval = 5  # 每5步打印一次（实时反馈）
        
        # 统计指标
        self.episode_rewards = []
        self.episode_successes = []
        self.current_episode_reward = 0
    
    def _on_step(self) -> bool:
        # 累计当前 episode 的奖励
        rewards = self.locals.get('rewards', [])
        if len(rewards) > 0:
            self.current_episode_reward += float(rewards[0])
        
        # 检查 episode 是否结束
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        if len(dones) > 0 and dones[0]:
            # Episode 结束，记录奖励
            self.episode_rewards.append(self.current_episode_reward)
            
            # 检查是否成功（FetchReach 环境通过 info['is_success'] 判断）
            if len(infos) > 0 and 'is_success' in infos[0]:
                self.episode_successes.append(float(infos[0]['is_success']))
            
            self.current_episode_reward = 0
        
        # 定期发送进度更新
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            progress = self.num_timesteps / self.total_timesteps * 100
            
            # 计算统计指标（最近2个episode）
            recent_rewards = self.episode_rewards[-2:] if self.episode_rewards else []
            recent_successes = self.episode_successes[-2:] if self.episode_successes else []
            
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            success_rate = np.mean(recent_successes) if recent_successes else 0
            num_episodes = len(self.episode_rewards)
            
            print(f"🏃 SB3 训练: {self.num_timesteps}/{self.total_timesteps} ({progress:.1f}%) "
                  f"| Episodes: {num_episodes} | 成功率: {success_rate:.1%} | 平均奖励: {avg_reward:.2f}")
            
            if self.on_progress and callable(self.on_progress):
                self.on_progress({
                    "timesteps": self.num_timesteps,
                    "total_timesteps": self.total_timesteps,
                    "progress_percent": progress,
                    "num_episodes": num_episodes,
                    "success_rate": success_rate,
                    "avg_reward": avg_reward,
                    "recent_rewards": recent_rewards[-10:] if recent_rewards else []
                })
            
            self.last_log_step = self.num_timesteps
        return True


class SB3SandboxTrainer:
    """基于 Stable-Baselines3 的多沙箱并行 DDPG 训练器
    
    复用现有的沙箱基础设施:
    - SimpleSandboxManager: 创建和管理沙箱
    - EnvProxy: 与沙箱环境通信
    - SandboxSetup: 安装依赖和启动环境执行器
    - SandboxGymEnv: 将异步 EnvProxy 包装为同步 Gym 接口
    
    使用 SB3 的 DDPG + HER 进行训练，每个沙箱运行一个独立的 Gym 环境。
    """
    
    # FetchReach 环境维度
    OBS_DIM = 10
    GOAL_DIM = 3
    ACTION_DIM = 4
    MAX_ACTION = 1.0
    
    def __init__(
        self,
        sandbox_manager,
        num_sandboxes: int = 4,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        learning_rate: float = 1e-3,
        buffer_size: int = 100000,
        on_sandbox_created: Callable = None  # 沙箱创建回调
    ):
        """
        Args:
            sandbox_manager: SimpleSandboxManager 实例
            num_sandboxes: 并行沙箱数量（默认 4 个）
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            learning_rate: 学习率
            buffer_size: 经验回放缓冲区大小
            on_sandbox_created: 沙箱创建成功后的回调函数
        """
        self.sandbox_manager = sandbox_manager
        self.num_sandboxes = num_sandboxes
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.on_sandbox_created = on_sandbox_created
        
        self.sandbox_setup = SandboxSetup(sandbox_manager)
        
        # 沙箱会话和环境代理
        self.sandbox_sessions: List = []
        self.env_proxies: List[EnvProxy] = []
        self.sandbox_envs: List[SandboxGymEnv] = []
        
        # 事件循环（用于异步操作）
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # SB3 模型
        self.model: Optional[DDPG] = None
        self.vec_env: Optional[VecEnv] = None
        self.eval_env: Optional[VecEnv] = None
    
    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """获取或创建事件循环"""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def _run_async(self, coro):
        """运行异步协程（使用 nest_asyncio 支持嵌套事件循环）"""
        print(f"   🔄 _run_async: 开始执行协程...", flush=True)
        loop = self._get_or_create_loop()
        result = loop.run_until_complete(coro)
        print(f"   🔄 _run_async: 协程执行完成，返回: {type(result)}", flush=True)
        return result
    
    async def _setup_single_sandbox(self, sandbox_idx: int) -> Optional[Dict]:
        """设置单个沙箱
        
        Returns:
            Dict: {'session': session, 'env_proxy': env_proxy, 'sandbox_env': SandboxGymEnv}
            或 None（如果设置失败）
        """
        try:
            print(f"   🔧 设置沙箱 {sandbox_idx + 1}/{self.num_sandboxes}...")
            
            # 1. 创建沙箱
            session = await self.sandbox_manager.create_sandbox()
            if not session:
                print(f"   ❌ 沙箱 {sandbox_idx + 1} 创建失败")
                return None
            
            sandbox_id = session.sandbox_id
            session_id = session.session_id  # 内部会话 ID，用于 execute_command
            print(f"   ✅ 沙箱 {sandbox_idx + 1} 创建成功: {sandbox_id[:8]}...")
            
            # 2. 获取流化 URL
            stream_url = await self.sandbox_manager.get_sandbox_url(sandbox_id)
            if stream_url:
                print(f"   🌐 沙箱 {sandbox_idx + 1} 流化URL: {stream_url}")
            
            # 3. 安装依赖（使用 session_id）
            if not await self.sandbox_setup.install_dependencies(session_id):
                print(f"   ❌ 沙箱 {sandbox_idx + 1} 依赖安装失败")
                return None
            
            # 4. 设置环境执行器（使用 session_id）
            if not await self.sandbox_setup.setup_env_executor(session_id):
                print(f"   ❌ 沙箱 {sandbox_idx + 1} 环境执行器设置失败")
                return None
            
            # 5. 创建 EnvProxy（使用 session_id）
            env_proxy = EnvProxy(self.sandbox_manager, session_id)
            
            # 6. 创建 SandboxGymEnv（使用当前线程的事件循环）
            loop = self._get_or_create_loop()
            sandbox_env = SandboxGymEnv(
                env_proxy=env_proxy,
                obs_dim=self.OBS_DIM,
                goal_dim=self.GOAL_DIM,
                action_dim=self.ACTION_DIM,
                max_action=self.MAX_ACTION,
                loop=loop
            )
            
            print(f"   ✅ 沙箱 {sandbox_idx + 1} 设置完成")
            
            return {
                'session': session,
                'env_proxy': env_proxy,
                'sandbox_env': sandbox_env
            }
            
        except Exception as e:
            print(f"   ❌ 沙箱 {sandbox_idx + 1} 设置异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def setup_sandboxes(self) -> bool:
        """并行设置所有沙箱
        
        Returns:
            bool: 是否成功设置至少一个沙箱
        """
        print(f"\n🚀 开始并行设置 {self.num_sandboxes} 个沙箱...")
        start_time = time.time()
        
        # 并行创建所有沙箱
        tasks = [self._setup_single_sandbox(i) for i in range(self.num_sandboxes)]
        results = await asyncio.gather(*tasks)
        
        # 处理结果
        for i, result in enumerate(results):
            if result:
                self.sandbox_sessions.append(result['session'])
                self.env_proxies.append(result['env_proxy'])
                self.sandbox_envs.append(result['sandbox_env'])
                
                # 通知前端沙箱已创建
                if self.on_sandbox_created and callable(self.on_sandbox_created):
                    try:
                        session = result['session']
                        stream_url = await self.sandbox_manager.get_sandbox_url(session.sandbox_id)
                        sandbox_info = {
                            "session_id": session.session_id,
                            "sandbox_id": session.sandbox_id,
                            "resource_url": stream_url or "",
                            "index": i + 1,
                            "total": self.num_sandboxes,
                            "sandbox_type": "sb3_training"
                        }
                        callback_result = self.on_sandbox_created(sandbox_info)
                        if asyncio.iscoroutine(callback_result):
                            await callback_result
                    except Exception as e:
                        print(f"   ⚠️ 沙箱创建回调执行失败: {e}")
        
        if not self.sandbox_envs:
            print("❌ 没有沙箱设置成功", flush=True)
            return False
        
        elapsed = time.time() - start_time
        print(f"\n✅ 并行创建完成: {len(self.sandbox_envs)}/{self.num_sandboxes} 个沙箱 (耗时 {elapsed:.1f}s)", flush=True)
        print("   📤 setup_sandboxes 准备返回 True...", flush=True)
        return True
    
    def create_vec_env(self, main_loop=None) -> Optional[VecEnv]:
        """创建异步并行向量化环境
        
        Args:
            main_loop: 主事件循环（MCP 客户端绑定到这个循环）
        
        Returns:
            VecEnv: SB3 向量化环境（支持真正的并行 step）
        """
        if not self.env_proxies:
            print("❌ 没有可用的沙箱环境")
            return None
        
        print(f"   正在创建 AsyncSandboxVecEnv...")
        
        # 保存主事件循环引用
        if main_loop:
            self._main_loop = main_loop
        print(f"   主事件循环: {main_loop}")
        
        self.vec_env = AsyncSandboxVecEnv(
            env_proxies=self.env_proxies,
            sandbox_manager=self.sandbox_manager,
            main_loop=main_loop,
            obs_dim=self.OBS_DIM,
            goal_dim=self.GOAL_DIM,
            action_dim=self.ACTION_DIM,
            max_action=self.MAX_ACTION,
            env_name="FetchReachDense-v4"  # 传入环境名称，用于 compute_reward 判断奖励类型
        )
        
        print(f"✅ 创建异步并行向量化环境，包含 {len(self.env_proxies)} 个并行沙箱")
        print(f"   🚀 step 操作将并行执行，充分利用多沙箱优势")
        return self.vec_env
    
    def initialize_model(self, log_dir: str) -> bool:
        """初始化 SB3 DDPG 模型
        
        Args:
            log_dir: 日志目录
            
        Returns:
            bool: 是否成功初始化
        """
        if not self.vec_env:
            print("❌ 请先创建向量化环境")
            return False
        
        print("\n🧠 初始化 SB3 DDPG 模型...")
        print("   ⚠️ DDPG 初始化时会调用 env.reset()，这需要与沙箱通信...")
        import sys
        sys.stdout.flush()
        
        try:
            tensorboard_log = os.path.join(log_dir, "tensorboard")
            os.makedirs(tensorboard_log, exist_ok=True)
            
            print("   创建 DDPG 模型对象...")
            sys.stdout.flush()
            
            self.model = DDPG(
                "MultiInputPolicy",  # 处理 goal-based observation
                self.vec_env,
                verbose=1,
                learning_starts=1000,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                replay_buffer_class=HerReplayBuffer,  # 使用 HER
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy=GoalSelectionStrategy.FUTURE
                ),
                gamma=0.95,
                tau=0.05,
                batch_size=256,
                tensorboard_log=tensorboard_log
            )
            
            print(f"✅ SB3 DDPG 模型初始化完成")
            print(f"   并行沙箱数: {len(self.sandbox_envs)}")
            print(f"   学习率: {self.learning_rate}")
            print(f"   缓冲区大小: {self.buffer_size}")
            print(f"   HER: 启用 (FUTURE 策略, n_sampled_goal=4)")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def train(
        self,
        log_dir: Optional[str] = None,
        on_progress: Callable = None
    ) -> Dict[str, Any]:
        """执行训练（异步方法，直接在调用者的事件循环中运行）
        
        Args:
            log_dir: 日志目录
            on_progress: 进度回调函数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        # 创建日志目录
        if not log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"data/sb3_sandbox_ddpg/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🚀 SB3 多沙箱并行 DDPG 训练")
        print(f"   并行沙箱数: {self.num_sandboxes}")
        print(f"   总步数: {self.total_timesteps}")
        print(f"   评估频率: {self.eval_freq}")
        print(f"   日志目录: {log_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            import sys
            
            # 获取当前运行的事件循环（MCP 客户端绑定到这个循环）
            main_loop = asyncio.get_running_loop()
            print(f"   获取主事件循环: {main_loop}", flush=True)
            
            # 1. 设置沙箱（直接 await，在调用者的事件循环中运行）
            print("\n📦 步骤 1: 设置沙箱...", flush=True)
            setup_result = await self.setup_sandboxes()
            print(f"   setup_sandboxes 返回: {setup_result}", flush=True)
            if not setup_result:
                return {"status": "error", "error": "沙箱设置失败"}
            
            print("   ✅ 沙箱设置完成，准备创建向量化环境...", flush=True)
            
            # 2. 创建向量化环境（传递主事件循环）
            print("\n📦 步骤 2: 创建向量化环境...", flush=True)
            vec_env_result = self.create_vec_env(main_loop=main_loop)
            print(f"   create_vec_env 返回: {vec_env_result}", flush=True)
            if not vec_env_result:
                return {"status": "error", "error": "创建向量化环境失败"}
            
            print("   ✅ 向量化环境创建完成，准备初始化模型...", flush=True)
            
            # 3. 初始化模型（这里会调用 env.reset()）
            # 注意：必须先在主协程中初始化，之后再把 learn() 放到线程中
            print("\n📦 步骤 3: 初始化模型...", flush=True)
            print("   ⚠️ 初始化时会调用 env.reset()，可能需要等待沙箱响应...", flush=True)
            if not self.initialize_model(log_dir):
                return {"status": "error", "error": "模型初始化失败"}
            
            # 4. 创建回调
            callbacks = [
                ProgressCallback(self.total_timesteps, on_progress)
            ]
            
            # 5. 在独立线程中运行 model.learn()
            # 关键：主事件循环必须保持空闲，才能处理 VecEnv._run_async 提交的协程
            # 使用 run_in_executor 将同步的 learn() 放到工作线程，主循环继续调度
            print("\n🏃 开始训练（在独立线程中运行 model.learn()）...")
            loop = asyncio.get_running_loop()
            
            def run_learn():
                return self.model.learn(
                    total_timesteps=self.total_timesteps,
                    callback=callbacks,
                    progress_bar=True
                )
            
            await loop.run_in_executor(None, run_learn)
            
            # 6. 保存模型
            model_path = os.path.join(log_dir, "final_model")
            self.model.save(model_path)
            print(f"\n💾 模型已保存到: {model_path}.zip")
            
            training_time = time.time() - start_time
            
            result = {
                "status": "completed",
                "training_time": training_time,
                "model_path": model_path,
                "log_dir": log_dir,
                "total_timesteps": self.total_timesteps,
                "num_sandboxes": len(self.sandbox_envs)
            }
            
            print(f"\n🎉 训练完成!")
            print(f"   总时间: {training_time:.1f} 秒 ({training_time/60:.1f} 分钟)")
            print(f"   使用沙箱数: {len(self.sandbox_envs)}")
            
            return result
            
        except KeyboardInterrupt:
            print("\n🛑 训练被用户中断")
            return {
                "status": "interrupted",
                "training_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"\n❌ 训练出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "training_time": time.time() - start_time
            }
    
    async def cleanup(self):
        """清理资源"""
        print("\n🧹 清理沙箱资源...")
        
        # 关闭所有环境
        for env in self.sandbox_envs:
            try:
                env.close()
            except:
                pass
        
        # 释放所有沙箱
        for session in self.sandbox_sessions:
            try:
                await self.sandbox_manager.release_sandbox(session.sandbox_id)
                print(f"   ✅ 释放沙箱: {session.sandbox_id[:8]}...")
            except Exception as e:
                print(f"   ⚠️ 释放沙箱失败: {e}")
        
        self.sandbox_sessions = []
        self.env_proxies = []
        self.sandbox_envs = []
        
        print("✅ 资源清理完成")
    
    def cleanup_sync(self):
        """同步清理资源"""
        self._run_async(self.cleanup())


# ============== 保留原有的本地训练器（不依赖沙箱）==============

class SB3Trainer:
    """基于Stable-Baselines3的DDPG训练器（纯本地，不使用沙箱）
    
    保留此类用于本地快速验证和对比测试。
    """
    
    def __init__(self, episodes: int = 1000000, eval_freq: int = 25000, 
                 learning_rate: float = 1e-3, buffer_size: int = 1000000):
        self.episodes = episodes
        self.eval_freq = eval_freq
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.model = None
        self.env = None
        self.env_val = None
        
        # 延迟导入本地 gymnasium
        import gymnasium as gym
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
        self._gym = gym
    
    def setup_environments(self, env_name: str = "FetchReachDense-v4"):
        """设置训练和评估环境"""
        from stable_baselines3.common.env_util import make_vec_env
        
        print("🔄 创建本地训练环境...")
        self.env = make_vec_env(env_name, n_envs=4)
        
        print("🔄 创建本地评估环境...")
        self.env_val = make_vec_env(env_name, n_envs=1)
        
        single_env = self._gym.make(env_name)
        print(f"   观测空间: {single_env.observation_space}")
        print(f"   动作空间: {single_env.action_space}")
        single_env.close()
    
    def initialize_model(self, env_name: str = "FetchReachDense-v4"):
        """初始化DDPG模型"""
        print("🧠 初始化本地DDPG代理...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"/tmp/sb3_ddpg_local/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_log = os.path.join(log_dir, "tensorboard")
        
        self.model = DDPG(
            "MultiInputPolicy",
            self.env,
            verbose=1,
            learning_starts=1000,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=GoalSelectionStrategy.FUTURE
            ),
            gamma=0.95,
            tau=0.05,
            batch_size=256,
            tensorboard_log=tensorboard_log
        )
        
        print(f"✅ 本地DDPG模型初始化完成")
        print(f"   日志目录: {log_dir}")
    
    def create_callbacks(self, log_dir: str):
        """创建训练回调"""
        eval_callback = EvalCallback(
            self.env_val,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=self.eval_freq,
            render=False,
            deterministic=True,
            n_eval_episodes=20
        )
        return eval_callback
    
    def train(self, log_dir: Optional[str] = None) -> Dict[str, Any]:
        """执行本地训练"""
        if not self.env or not self.env_val:
            raise RuntimeError("请先调用setup_environments()设置环境")
        
        if not self.model:
            raise RuntimeError("请先调用initialize_model()初始化模型")
        
        if not log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"/tmp/sb3_ddpg_local/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        print("🚀 开始本地训练...")
        start_time = time.time()
        
        try:
            eval_callback = self.create_callbacks(log_dir)
            
            self.model.learn(
                total_timesteps=self.episodes,
                callback=eval_callback,
                progress_bar=True
            )
            
            final_model_path = os.path.join(log_dir, "final_model")
            self.model.save(final_model_path)
            print(f"💾 模型已保存到: {final_model_path}.zip")
            
            training_time = time.time() - start_time
            
            return {
                "status": "completed",
                "training_time": training_time,
                "model_path": final_model_path,
                "log_dir": log_dir,
                "episodes": self.episodes
            }
            
        except Exception as e:
            print(f"❌ 训练出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "training_time": time.time() - start_time
            }
    
    def cleanup(self):
        """清理资源"""
        if self.env:
            self.env.close()
        if self.env_val:
            self.env_val.close()
        print("✅ 本地SB3训练器资源已清理")
