"""配置管理器 - 训练配置和模式管理"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class TrainingMode(Enum):
    """训练模式枚举"""
    CUSTOM_BATCH = "custom_batch"            # 自定义网络 + 批量交互模式
    STABLE_BASELINES3 = "stable_baselines3"  # Stable-Baselines3实现


@dataclass
class TrainingConfig:
    """训练配置数据类"""
    mode: TrainingMode
    episodes: int = 1000000
    eval_freq: int = 25000
    batch_size: int = 10
    learning_rate: float = 1e-3
    buffer_size: int = 1000000
    gamma: float = 0.95
    tau: float = 0.05
    force_cleanup: bool = True
    render_mode: str = "human"


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.default_configs = {
            TrainingMode.CUSTOM_BATCH: TrainingConfig(
                mode=TrainingMode.CUSTOM_BATCH,
                episodes=1000000,
                eval_freq=25000,
                batch_size=10  # 批量交互大小
            ),
            TrainingMode.STABLE_BASELINES3: TrainingConfig(
                mode=TrainingMode.STABLE_BASELINES3,
                episodes=1000000,
                eval_freq=25000,
                learning_rate=1e-3,
                buffer_size=1000000
            )
        }
    
    def get_config(self, mode: TrainingMode, custom_config: Optional[Dict[str, Any]] = None) -> TrainingConfig:
        """获取训练配置
        
        Args:
            mode: 训练模式
            custom_config: 自定义配置参数
            
        Returns:
            TrainingConfig: 配置对象
        """
        # 获取默认配置
        config = self.default_configs.get(mode, self.default_configs[TrainingMode.CUSTOM_BATCH])
        
        # 应用自定义配置
        if custom_config:
            config_dict = config.__dict__.copy()
            config_dict.update(custom_config)
            # 重新创建配置对象
            config = TrainingConfig(**config_dict)
        
        return config
    
    def validate_config(self, config: TrainingConfig) -> bool:
        """验证配置有效性
        
        Args:
            config: 训练配置
            
        Returns:
            bool: 配置是否有效
        """
        # 基本验证
        if config.episodes <= 0:
            return False
        if config.eval_freq <= 0:
            return False
        if config.learning_rate <= 0:
            return False
        if config.buffer_size <= 0:
            return False
            
        # 模式特定验证
        if config.mode == TrainingMode.CUSTOM_BATCH:
            if config.batch_size <= 0:
                return False
                
        return True