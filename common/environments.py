"""强化学习环境定义和管理"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod


class RLEnvironment(ABC):
    """强化学习环境抽象基类"""
    
    def __init__(self):
        self.state = None
        self.done = False
        self.step_count = 0
        
    @abstractmethod
    def reset(self) -> Any:
        """重置环境到初始状态"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """执行动作并返回 (next_state, reward, done, info)"""
        pass
    
    @abstractmethod
    def get_action_space(self) -> int:
        """返回动作空间大小"""
        pass
    
    @abstractmethod
    def get_state_space(self) -> Any:
        """返回状态空间信息"""
        pass
    
    @abstractmethod
    def render(self) -> str:
        """返回环境的文本表示"""
        pass


class BanditEnvironment(RLEnvironment):
    """多臂老虎机环境"""
    
    def __init__(self, n_arms: int = 10, seed: Optional[int] = None):
        super().__init__()
        self.n_arms = n_arms
        self.rng = np.random.RandomState(seed)
        
        # 每个臂的真实奖励分布均值
        self.arm_means = self.rng.normal(0, 1, n_arms)
        
    def reset(self) -> int:
        """重置环境（老虎机环境状态始终为0）"""
        self.state = 0
        self.done = False
        self.step_count = 0
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """拉动手臂获得奖励"""
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"动作 {action} 超出范围 [0, {self.n_arms})")
        
        # 从对应臂的分布中采样奖励
        reward = self.rng.normal(self.arm_means[action], 1.0)
        
        self.step_count += 1
        self.done = False  # 老虎机环境通常不会结束
        
        info = {
            "arm_mean": self.arm_means[action],
            "optimal_arm": np.argmax(self.arm_means),
            "is_optimal": action == np.argmax(self.arm_means)
        }
        
        return self.state, reward, self.done, info
    
    def get_action_space(self) -> int:
        return self.n_arms
    
    def get_state_space(self) -> int:
        return 1  # 单一状态
    
    def render(self) -> str:
        """显示各臂的统计信息"""
        optimal_arm = np.argmax(self.arm_means)
        arm_info = []
        for i in range(self.n_arms):
            status = "⭐" if i == optimal_arm else "  "
            arm_info.append(f"{status}臂{i}: 均值={self.arm_means[i]:.2f}")
        return "\n".join(arm_info)


class GridWorldEnvironment(RLEnvironment):
    """网格世界环境（用于DQN、PPO等算法）"""
    
    def __init__(self, width: int = 5, height: int = 5, seed: Optional[int] = None):
        super().__init__()
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)
        
        # 动作定义: 0=上, 1=右, 2=下, 3=左
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        self.reset()
    
    def reset(self) -> Tuple[int, int]:
        """重置到随机起始位置"""
        self.player_pos = (self.rng.randint(0, self.width), 
                          self.rng.randint(0, self.height))
        self.goal_pos = (self.width - 1, self.height - 1)
        self.done = False
        self.step_count = 0
        return self.player_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """执行移动动作"""
        if action < 0 or action >= len(self.actions):
            raise ValueError(f"动作 {action} 超出范围")
        
        dx, dy = self.actions[action]
        new_x = max(0, min(self.width - 1, self.player_pos[0] + dx))
        new_y = max(0, min(self.height - 1, self.player_pos[1] + dy))
        
        self.player_pos = (new_x, new_y)
        self.step_count += 1
        
        # 计算奖励
        if self.player_pos == self.goal_pos:
            reward = 10.0
            self.done = True
        else:
            # 负距离奖励鼓励向目标移动
            distance = abs(new_x - self.goal_pos[0]) + abs(new_y - self.goal_pos[1])
            reward = -0.1 - distance * 0.01
        
        # 最大步数限制
        if self.step_count >= 100:
            self.done = True
            
        info = {
            "distance_to_goal": abs(self.player_pos[0] - self.goal_pos[0]) + 
                              abs(self.player_pos[1] - self.goal_pos[1]),
            "at_goal": self.player_pos == self.goal_pos
        }
        
        return self.player_pos, reward, self.done, info
    
    def get_action_space(self) -> int:
        return len(self.actions)
    
    def get_state_space(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def render(self) -> str:
        """文本渲染网格世界"""
        grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) == self.player_pos:
                    row.append('P')  # 玩家
                elif (x, y) == self.goal_pos:
                    row.append('G')  # 目标
                else:
                    row.append('.')
            grid.append(' '.join(row))
        return '\n'.join(grid)


def create_environment(env_type: str, **kwargs) -> RLEnvironment:
    """工厂函数创建环境"""
    env_map = {
        "bandit": BanditEnvironment,
        "gridworld": GridWorldEnvironment,
    }
    
    if env_type not in env_map:
        raise ValueError(f"未知环境类型: {env_type}")
    
    return env_map[env_type](**kwargs)