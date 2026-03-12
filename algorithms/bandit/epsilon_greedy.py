"""ε-贪婪多臂老虎机算法实现"""

import numpy as np
from typing import List, Tuple, Optional
from common.environments import BanditEnvironment


class EpsilonGreedyBandit:
    """ε-贪婪策略的多臂老虎机算法"""
    
    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: Optional[int] = None):
        """
        初始化ε-贪婪老虎机算法
        
        Args:
            n_arms: 老虎机臂数量
            epsilon: 探索概率 (0-1)
            seed: 随机种子
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        
        # 统计信息
        self.counts = np.zeros(n_arms, dtype=int)  # 每个臂被拉动的次数
        self.values = np.zeros(n_arms, dtype=float)  # 每个臂的平均奖励估计
        
        # 历史记录
        self.history: List[Tuple[int, float, bool]] = []  # (action, reward, is_optimal)
        
    def select_action(self) -> int:
        """选择要拉动的臂"""
        if self.rng.random() < self.epsilon:
            # 探索：随机选择一个臂
            return self.rng.randint(0, self.n_arms)
        else:
            # 利用：选择当前估计最好的臂
            return np.argmax(self.values)
    
    def update(self, action: int, reward: float):
        """更新对选定臂的估计"""
        self.counts[action] += 1
        
        # 增量更新平均值
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = ((n - 1) / n) * value + (1 / n) * reward
    
    def train(self, env: BanditEnvironment, n_episodes: int) -> List[float]:
        """
        训练算法
        
        Args:
            env: 老虎机环境
            n_episodes: 训练回合数
            
        Returns:
            每回合的奖励列表
        """
        rewards = []
        
        for episode in range(n_episodes):
            # 重置环境
            env.reset()
            
            # 选择动作
            action = self.select_action()
            
            # 执行动作
            _, reward, _, info = env.step(action)
            
            # 更新估计
            self.update(action, reward)
            
            # 记录历史
            self.history.append((action, reward, info["is_optimal"]))
            rewards.append(reward)
            
            # 每100回合打印一次统计
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                optimal_rate = np.mean([1 if is_opt else 0 
                                      for _, _, is_opt in self.history[-100:]])
                print(f"回合 {episode + 1:4d}: "
                      f"平均奖励={avg_reward:.3f}, "
                      f"最优臂选择率={optimal_rate:.2%}")
        
        return rewards
    
    def get_statistics(self) -> dict:
        """获取算法统计信息"""
        if not self.history:
            return {}
            
        total_rewards = [r for _, r, _ in self.history]
        optimal_choices = [1 if is_opt else 0 for _, _, is_opt in self.history]
        
        return {
            "total_episodes": len(self.history),
            "total_reward": sum(total_rewards),
            "average_reward": np.mean(total_rewards),
            "optimal_selection_rate": np.mean(optimal_choices),
            "arm_pull_counts": self.counts.tolist(),
            "arm_value_estimates": self.values.tolist()
        }


class UCB1Bandit:
    """UCB1（置信上限）算法"""
    
    def __init__(self, n_arms: int, c: float = 2.0, seed: Optional[int] = None):
        """
        初始化UCB1算法
        
        Args:
            n_arms: 老虎机臂数量
            c: 置信参数，控制探索程度
            seed: 随机种子
        """
        self.n_arms = n_arms
        self.c = c
        self.rng = np.random.RandomState(seed)
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.total_count = 0
        self.history: List[Tuple[int, float, bool]] = []
    
    def select_action(self) -> int:
        """使用UCB1准则选择动作"""
        # 如果有任何臂还没被拉动过，优先拉动它
        if self.total_count < self.n_arms:
            return self.total_count
        
        # 计算UCB值
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.counts[i] > 0:
                # 置信上限 = 平均奖励 + 置信区间
                confidence_bound = self.c * np.sqrt(np.log(self.total_count) / self.counts[i])
                ucb_values[i] = self.values[i] + confidence_bound
            else:
                # 给未探索的臂很高的优先级
                ucb_values[i] = float('inf')
        
        return np.argmax(ucb_values)
    
    def update(self, action: int, reward: float):
        """更新统计信息"""
        self.counts[action] += 1
        self.total_count += 1
        
        # 增量更新平均值
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = ((n - 1) / n) * value + (1 / n) * reward
    
    def train(self, env: BanditEnvironment, n_episodes: int) -> List[float]:
        """训练UCB1算法"""
        rewards = []
        
        for episode in range(n_episodes):
            env.reset()
            action = self.select_action()
            _, reward, _, info = env.step(action)
            
            self.update(action, reward)
            self.history.append((action, reward, info["is_optimal"]))
            rewards.append(reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                optimal_rate = np.mean([1 if is_opt else 0 
                                      for _, _, is_opt in self.history[-100:]])
                print(f"UCB1 回合 {episode + 1:4d}: "
                      f"平均奖励={avg_reward:.3f}, "
                      f"最优选择率={optimal_rate:.2%}")
        
        return rewards
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        if not self.history:
            return {}
            
        total_rewards = [r for _, r, _ in self.history]
        optimal_choices = [1 if is_opt else 0 for _, _, is_opt in self.history]
        
        return {
            "total_episodes": len(self.history),
            "total_reward": sum(total_rewards),
            "average_reward": np.mean(total_rewards),
            "optimal_selection_rate": np.mean(optimal_choices),
            "arm_pull_counts": self.counts.tolist(),
            "arm_value_estimates": self.values.tolist()
        }


def compare_algorithms(env: BanditEnvironment, n_episodes: int = 1000):
    """比较不同老虎机算法的性能"""
    print("=" * 60)
    print("多臂老虎机算法比较")
    print("=" * 60)
    
    # ε-贪婪算法
    print("\n1. ε-贪婪算法 (ε=0.1)")
    print("-" * 30)
    eps_greedy = EpsilonGreedyBandit(env.n_arms, epsilon=0.1)
    eps_rewards = eps_greedy.train(env, n_episodes)
    eps_stats = eps_greedy.get_statistics()
    
    # UCB1算法
    print("\n2. UCB1算法 (c=2.0)")
    print("-" * 30)
    ucb1 = UCB1Bandit(env.n_arms, c=2.0)
    ucb_rewards = ucb1.train(env, n_episodes)
    ucb_stats = ucb1.get_statistics()
    
    # 结果比较
    print("\n" + "=" * 60)
    print("性能比较总结")
    print("=" * 60)
    print(f"{'算法':<15} {'平均奖励':<12} {'最优选择率':<15} {'总奖励':<12}")
    print("-" * 60)
    print(f"{'ε-贪婪':<15} {eps_stats['average_reward']:<12.3f} "
          f"{eps_stats['optimal_selection_rate']:<15.2%} "
          f"{eps_stats['total_reward']:<12.2f}")
    print(f"{'UCB1':<15} {ucb_stats['average_reward']:<12.3f} "
          f"{ucb_stats['optimal_selection_rate']:<15.2%} "
          f"{ucb_stats['total_reward']:<12.2f}")


if __name__ == "__main__":
    # 示例使用
    env = BanditEnvironment(n_arms=10, seed=42)
    print("老虎机环境初始化完成")
    print(f"最优臂: {np.argmax(env.arm_means)}")
    print(f"各臂真实均值: {env.arm_means}")
    
    # 比较算法
    compare_algorithms(env, n_episodes=1000)