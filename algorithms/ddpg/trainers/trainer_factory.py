"""训练器工厂 - 根据配置创建不同的训练器"""

from typing import Any
from .custom_trainer import CustomTrainer
from ..core.config_manager import TrainingMode, TrainingConfig


class TrainerFactory:
    """训练器工厂类"""

    @staticmethod
    def create_trainer(config: TrainingConfig, **kwargs) -> Any:
        """根据配置创建训练器

        Args:
            config: 训练配置
            **kwargs: 其他参数

        Returns:
            训练器实例
        """
        if config.mode == TrainingMode.STABLE_BASELINES3:
            # 延迟导入，避免启动时强依赖 stable_baselines3
            from .sb3_trainer import SB3Trainer
            return SB3Trainer(
                episodes=config.episodes,
                eval_freq=config.eval_freq,
                learning_rate=config.learning_rate,
                buffer_size=config.buffer_size,
                **kwargs
            )
        elif config.mode == TrainingMode.CUSTOM_BATCH:
            return CustomTrainer(
                episodes=config.episodes,
                eval_freq=config.eval_freq,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的训练模式: {config.mode}")

    @staticmethod
    def get_available_modes() -> list:
        """获取可用的训练模式列表"""
        return [mode.value for mode in TrainingMode]

    @staticmethod
    def get_mode_description(mode: TrainingMode) -> str:
        """获取训练模式描述"""
        descriptions = {
            TrainingMode.CUSTOM_BATCH: "自定义网络 + 批量交互模式，减少通信开销",
            TrainingMode.STABLE_BASELINES3: "Stable-Baselines3实现，使用成熟的RL框架"
        }
        return descriptions.get(mode, "未知模式")
