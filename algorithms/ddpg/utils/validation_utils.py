"""验证工具 - 数据验证和格式检查工具"""

import numpy as np
from typing import Any, Union, List, Dict


class ValidationUtils:
    """验证工具类"""
    
    @staticmethod
    def validate_observation(obs: Any, expected_shape: tuple = None) -> bool:
        """验证观测值格式
        
        Args:
            obs: 观测值
            expected_shape: 期望的形状
            
        Returns:
            bool: 是否有效
        """
        # 检查基本类型
        if obs is None:
            return False
        
        # 处理字典格式的观测值
        if isinstance(obs, dict):
            if 'observation' in obs:
                obs = obs['observation']
            else:
                return False
        
        # 转换为numpy数组
        try:
            obs_array = np.array(obs)
        except:
            return False
        
        # 检查是否为空
        if obs_array.size == 0:
            return False
        
        # 检查形状
        if expected_shape and obs_array.shape != expected_shape:
            return False
        
        # 检查数值范围（基本检查）
        if not np.isfinite(obs_array).all():
            return False
            
        return True
    
    @staticmethod
    def validate_action(action: Any, action_space_low: float = -1.0, 
                       action_space_high: float = 1.0) -> bool:
        """验证动作格式
        
        Args:
            action: 动作值
            action_space_low: 动作空间下界
            action_space_high: 动作空间上界
            
        Returns:
            bool: 是否有效
        """
        if action is None:
            return False
        
        try:
            action_array = np.array(action)
        except:
            return False
        
        # 检查是否为空
        if action_array.size == 0:
            return False
        
        # 检查数值范围
        if np.any(action_array < action_space_low) or np.any(action_array > action_space_high):
            return False
            
        # 检查是否为有限数值
        if not np.isfinite(action_array).all():
            return False
            
        return True
    
    @staticmethod
    def validate_reward(reward: Any) -> bool:
        """验证奖励格式
        
        Args:
            reward: 奖励值
            
        Returns:
            bool: 是否有效
        """
        if reward is None:
            return False
        
        try:
            reward_float = float(reward)
        except:
            return False
        
        # 检查是否为有限数值
        if not np.isfinite(reward_float):
            return False
            
        return True
    
    @staticmethod
    def validate_info(info: Any) -> bool:
        """验证info字典格式
        
        Args:
            info: info字典
            
        Returns:
            bool: 是否有效
        """
        if info is None:
            return True  # info可以为空
        
        if not isinstance(info, dict):
            return False
            
        return True
    
    @staticmethod
    def sanitize_observation(obs: Any) -> np.ndarray:
        """清理和标准化观测值
        
        Args:
            obs: 原始观测值
            
        Returns:
            np.ndarray: 标准化后的观测值
        """
        # 处理字典格式
        if isinstance(obs, dict):
            if 'observation' in obs:
                obs = obs['observation']
            else:
                # 如果没有observation键，尝试使用第一个键
                if obs:
                    first_key = next(iter(obs.keys()))
                    obs = obs[first_key]
                else:
                    return np.array([])
        
        # 转换为numpy数组
        try:
            obs_array = np.array(obs, dtype=np.float32)
        except:
            return np.array([])
        
        # 处理NaN和无穷大值
        if not np.isfinite(obs_array).all():
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return obs_array
    
    @staticmethod
    def format_training_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化训练结果
        
        Args:
            result: 原始训练结果
            
        Returns:
            Dict[str, Any]: 格式化后的结果
        """
        formatted = result.copy()
        
        # 确保数值字段为正确类型
        numeric_fields = ['average_reward', 'success_rate', 'training_time']
        for field in numeric_fields:
            if field in formatted and formatted[field] is not None:
                try:
                    formatted[field] = float(formatted[field])
                except:
                    formatted[field] = 0.0
        
        # 确保状态字段存在
        if 'status' not in formatted:
            formatted['status'] = 'unknown'
            
        return formatted