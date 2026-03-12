"""SB3环境适配器 - 将沙箱环境包装为SB3兼容格式"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any
from gymnasium import spaces


class SB3EnvAdapter(gym.Env):
    """SB3环境适配器"""
    
    def __init__(self, env_proxy):
        super(SB3EnvAdapter, self).__init__()
        self.env_proxy = env_proxy
        
        # 定义观察空间和动作空间
        # 这些需要根据实际环境进行调整
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        
        self.current_obs = None
        self.current_info = {}
    
    async def async_reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """异步重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            Tuple[np.ndarray, Dict]: (观测值, info字典)
        """
        obs, info = await self.env_proxy.reset()
        
        # 转换观测值格式
        if isinstance(obs, dict) and 'observation' in obs:
            # 处理复合观测结构
            self.current_obs = np.array(obs['observation'], dtype=np.float32)
        elif isinstance(obs, (list, tuple)):
            self.current_obs = np.array(obs, dtype=np.float32)
        else:
            self.current_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        self.current_info = info or {}
        
        return self.current_obs, self.current_info
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境（同步版本，用于SB3兼容性）
        
        注意：这个方法在SB3中会被调用，但我们实际使用异步版本
        """
        # 这里返回默认值，实际训练时会使用异步版本
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    
    async def async_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """异步执行动作
        
        Args:
            action: 动作数组
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: (观测值, 奖励, 终止, 截断, info)
        """
        obs, reward, done, info = await self.env_proxy.step(action.tolist())
        
        # 转换观测值格式
        if isinstance(obs, dict) and 'observation' in obs:
            next_obs = np.array(obs['observation'], dtype=np.float32)
        elif isinstance(obs, (list, tuple)):
            next_obs = np.array(obs, dtype=np.float32)
        else:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 处理终止和截断
        terminated = done
        truncated = False  # 根据需要调整
        
        self.current_obs = next_obs
        self.current_info = info or {}
        
        return next_obs, float(reward), terminated, truncated, self.current_info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作（同步版本，用于SB3兼容性）
        
        注意：这个方法在SB3中会被调用，但我们实际使用异步版本
        """
        # 这里返回默认值，实际训练时会使用异步版本
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            True,
            False,
            {}
        )
    
    def render(self):
        """渲染环境（不实现，由沙箱处理）"""
        pass
    
    def close(self):
        """关闭环境"""
        pass
    
    @property
    def unwrapped(self):
        """返回底层环境"""
        return self