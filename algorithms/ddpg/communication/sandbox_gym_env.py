"""沙箱 Gym 环境包装器 - 将异步 EnvProxy 包装为同步 Gym 接口供 SB3 使用"""

import asyncio
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple


class SandboxGymEnv(gym.Env):
    """
    将沙箱 EnvProxy 包装为标准 Gym 环境接口。
    
    支持 goal-based 环境（Dict observation space），可与 SB3 HER 配合使用。
    
    关键特性：
    - 将异步 EnvProxy 的 reset/step 转换为同步调用
    - 支持 FetchReach 的 Dict observation space (observation, achieved_goal, desired_goal)
    - 每个实例维护独立的事件循环
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        env_proxy,
        obs_dim: int = 10,
        goal_dim: int = 3,
        action_dim: int = 4,
        max_action: float = 1.0,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        """
        Args:
            env_proxy: EnvProxy 实例（与沙箱通信）
            obs_dim: 观测维度（FetchReach 为 10）
            goal_dim: 目标维度（FetchReach 为 3，xyz 坐标）
            action_dim: 动作维度（FetchReach 为 4）
            max_action: 动作最大值
            loop: 事件循环（如果为 None 则创建新的）
        """
        super().__init__()
        
        self.env_proxy = env_proxy
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 创建或使用传入的事件循环
        if loop is not None:
            self._loop = loop
            self._owns_loop = False
        else:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True
        
        # 定义 action space
        self.action_space = spaces.Box(
            low=-max_action,
            high=max_action,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # 定义 observation space（Goal-based，用于 HER）
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            "achieved_goal": spaces.Box(
                low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
            ),
        })
        
        # 缓存当前状态
        self._current_obs = None
        self._achieved_goal = None
        self._desired_goal = None
    
    def _run_async(self, coro):
        """在事件循环中运行异步协程"""
        return self._loop.run_until_complete(coro)
    
    def _format_observation(self, obs, info: Dict) -> Dict[str, np.ndarray]:
        """格式化观测为 Dict 格式（goal-based）"""
        # 处理观测值
        if isinstance(obs, (list, tuple)):
            obs_array = np.array(obs, dtype=np.float32)
        elif isinstance(obs, np.ndarray):
            obs_array = obs.astype(np.float32)
        else:
            obs_array = np.zeros(self.obs_dim, dtype=np.float32)
        
        # 确保维度正确
        if obs_array.shape[0] != self.obs_dim:
            if obs_array.shape[0] > self.obs_dim:
                obs_array = obs_array[:self.obs_dim]
            else:
                obs_array = np.pad(obs_array, (0, self.obs_dim - obs_array.shape[0]))
        
        # 提取 goal 信息
        achieved_goal = info.get('achieved_goal')
        desired_goal = info.get('desired_goal')
        
        if achieved_goal is not None:
            achieved_goal = np.array(achieved_goal, dtype=np.float32)
        else:
            # 默认使用观测的前 goal_dim 维作为 achieved_goal
            achieved_goal = obs_array[:self.goal_dim].copy()
        
        if desired_goal is not None:
            desired_goal = np.array(desired_goal, dtype=np.float32)
        else:
            desired_goal = np.zeros(self.goal_dim, dtype=np.float32)
        
        # 缓存
        self._current_obs = obs_array
        self._achieved_goal = achieved_goal
        self._desired_goal = desired_goal
        
        return {
            "observation": obs_array,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """重置环境
        
        Returns:
            Tuple[Dict, Dict]: (observation dict, info dict)
        """
        super().reset(seed=seed)
        
        try:
            obs, info = self._run_async(self.env_proxy.reset())
            formatted_obs = self._format_observation(obs, info)
            return formatted_obs, info
        except Exception as e:
            print(f"[SandboxGymEnv] reset 失败: {e}")
            # 返回零观测
            return self._format_observation([], {}), {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """执行一步动作
        
        Args:
            action: 动作数组
            
        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        try:
            # 确保动作格式正确
            if isinstance(action, np.ndarray):
                action_list = action.tolist()
            else:
                action_list = list(action)
            
            # 执行动作
            obs, reward, done, info = self._run_async(self.env_proxy.step(action_list))
            
            # 格式化观测
            formatted_obs = self._format_observation(obs, info)
            
            # SB3 需要区分 terminated 和 truncated
            terminated = done and info.get('is_success', False)
            truncated = done and not terminated
            
            return formatted_obs, float(reward), terminated, truncated, info
            
        except Exception as e:
            print(f"[SandboxGymEnv] step 失败: {e}")
            # 返回错误状态
            return self._format_observation([], {}), 0.0, True, False, {}
    
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict
    ) -> float:
        """计算奖励（用于 HER）
        
        基于距离的 dense reward
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -distance
    
    def render(self):
        """渲染（沙箱环境通过流化页面展示）"""
        pass
    
    def close(self):
        """关闭环境"""
        try:
            self._run_async(self.env_proxy.close())
        except:
            pass
        
        # 关闭自己创建的事件循环
        if self._owns_loop and self._loop and not self._loop.is_closed():
            self._loop.close()


class SandboxVecEnvFactory:
    """沙箱向量化环境工厂
    
    创建多个 SandboxGymEnv 并包装为 SB3 可用的 VecEnv
    """
    
    @staticmethod
    def create_env_fn(env_proxy, obs_dim: int, goal_dim: int, action_dim: int, loop):
        """创建环境工厂函数"""
        def _init():
            return SandboxGymEnv(
                env_proxy=env_proxy,
                obs_dim=obs_dim,
                goal_dim=goal_dim,
                action_dim=action_dim,
                loop=loop
            )
        return _init
