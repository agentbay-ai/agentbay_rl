"""强化学习教学材料管理模块
包含算法理论讲解、实践指导和交互式教程
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class DifficultyLevel(Enum):
    """难度等级"""
    BEGINNER = "入门"
    INTERMEDIATE = "初级" 
    ADVANCED = "中级"
    EXPERT = "高级"

@dataclass
class TeachingConcept:
    """教学概念"""
    title: str
    description: str
    explanation: str
    examples: List[str]
    key_points: List[str]

@dataclass
class InteractiveDemo:
    """交互式演示"""
    name: str
    description: str
    parameters: Dict[str, Any]
    visualization_type: str
    explanation: str

@dataclass
class AlgorithmCourse:
    """算法课程"""
    algorithm_name: str
    algorithm_key: str
    difficulty: DifficultyLevel
    estimated_time: str
    icon: str
    color: str
    
    # 理论部分
    theory_concepts: List[TeachingConcept]
    
    # 实践部分
    practical_examples: List[InteractiveDemo]
    
    # 学习目标
    learning_objectives: List[str]
    
    # 先修知识
    prerequisites: List[str]
    
    # 评估标准
    assessment_criteria: List[str]

class TeachingMaterialsManager:
    """教学材料管理器"""
    
    def __init__(self):
        self.courses: Dict[str, AlgorithmCourse] = {}
        self._initialize_courses()
    
    def _initialize_courses(self):
        """初始化所有课程"""
        # 多臂老虎机课程
        self.courses['bandit'] = self._create_bandit_course()
        
        # DQN课程
        self.courses['dqn'] = self._create_dqn_course()
        
        # PPO课程
        self.courses['ppo'] = self._create_ppo_course()
        
        # SAC课程
        self.courses['sac'] = self._create_sac_course()
    
    def _create_bandit_course(self) -> AlgorithmCourse:
        """创建多臂老虎机课程"""
        return AlgorithmCourse(
            algorithm_name="多臂老虎机",
            algorithm_key="bandit",
            difficulty=DifficultyLevel.BEGINNER,
            estimated_time="5分钟",
            icon="🎰",
            color="#6366f1",
            learning_objectives=[
                "理解探索与利用的基本概念",
                "掌握ε-贪婪策略的工作原理",
                "学会评估不同策略的效果",
                "理解累积奖励最大化的目标"
            ],
            prerequisites=[],
            assessment_criteria=[
                "能够解释探索与利用的权衡",
                "能够实现基本的ε-贪婪算法",
                "能够分析不同ε值对性能的影响"
            ],
            theory_concepts=[
                TeachingConcept(
                    title="什么是多臂老虎机问题",
                    description="经典的强化学习入门问题，模拟在多个选择中寻找最优选项的过程",
                    explanation="""多臂老虎机(Multi-Armed Bandit)是强化学习中最基础的问题之一。
                    想象你在赌场面对多个老虎机，每台机器有不同的获奖概率，你的目标是在有限的尝试次数内
                    获得最大的总奖励。这个问题抽象了现实世界中的许多决策场景：广告投放、医疗治疗选择、
                    投资组合优化等。""",
                    examples=[
                        "在线广告：选择哪个广告位展示以获得最高点击率",
                        "推荐系统：向用户推荐哪个商品以获得最高购买转化",
                        "临床试验：选择哪种药物治疗方案以获得最佳疗效"
                    ],
                    key_points=[
                        "这是一个序列决策问题",
                        "需要在探索(尝试新选项)和利用(选择已知好选项)之间平衡",
                        "目标是最大化长期累积奖励",
                        "没有状态转移，每次选择都是独立的"
                    ]
                ),
                TeachingConcept(
                    title="探索与利用的权衡",
                    description="强化学习的核心挑战：如何平衡尝试新事物和利用已知信息",
                    explanation="""这是强化学习最基本也是最重要的概念之一。
                    - 探索(Exploration)：尝试从未选择过的选项，可能发现更好的选择
                    - 利用(Exploitation)：选择当前认为最好的选项，获得稳定收益
                    
                    如果只探索不利用，永远无法获得好的回报；
                    如果只利用不探索，可能会错过真正最优的选择。""",
                    examples=[
                        "餐厅选择：是去熟悉的餐厅还是尝试新餐厅？",
                        "投资决策：是投资熟悉的股票还是探索新兴行业？",
                        "学习路径：是深化现有技能还是学习新技能？"
                    ],
                    key_points=[
                        "这是所有强化学习算法都需要解决的核心问题",
                        "不同的算法采用不同的探索策略",
                        "探索率通常随时间递减",
                        "最优策略需要在两者间找到平衡"
                    ]
                ),
                TeachingConcept(
                    title="ε-贪婪策略",
                    description="最简单的探索策略：以ε概率随机探索，以(1-ε)概率利用",
                    explanation=f"""ε-贪婪(Epsilon-Greedy)策略是最直观的探索方法：
                    - 以概率ε随机选择任意一个臂（探索）
                    - 以概率(1-ε)选择当前估计价值最高的臂（利用）
                    
                    例如：ε=0.1表示10%的时间随机探索，90%的时间选择当前最佳选择。
                    随着学习的进行，通常会让ε逐渐减小，减少探索比例。""",
                    examples=[
                        "ε=0.1：保守策略，主要用于利用",
                        "ε=0.5：平衡策略，探索和利用各占一半", 
                        "ε=0.9：激进策略，主要进行探索"
                    ],
                    key_points=[
                        "实现简单，易于理解和调试",
                        "ε值的选择对性能影响很大",
                        "可以通过退火(annealing)让ε随时间递减",
                        "是许多复杂探索策略的基础"
                    ]
                )
            ],
            practical_examples=[
                InteractiveDemo(
                    name="ε值对比实验",
                    description="观察不同ε值对算法性能的影响",
                    parameters={
                        "epsilon_values": [0.01, 0.1, 0.3, 0.7, 0.9],
                        "episodes": 1000,
                        "arms": 10
                    },
                    visualization_type="line_chart_comparison",
                    explanation="""通过这个实验，你可以直观地看到：
                    - ε过小(如0.01)：收敛快但可能陷入局部最优
                    - ε适中(如0.1)：平衡的探索利用效果
                    - ε过大(如0.7)：过度探索，收敛慢且不稳定
                    
                    观察不同ε值下的累积奖励曲线和最优臂选择率。"""
                ),
                InteractiveDemo(
                    name="臂价值估计可视化",
                    description="实时观察算法如何学习各臂的真实价值",
                    parameters={
                        "epsilon": 0.1,
                        "episodes": 500,
                        "arms": 5
                    },
                    visualization_type="bar_chart_animation",
                    explanation="""这个可视化展示了算法的学习过程：
                    - 每个柱状图代表一个臂的价值估计
                    - 随着试验进行，估计值逐渐接近真实值
                    - 高亮显示当前选择的臂和获得的奖励
                    - 可以看到探索如何帮助发现真正最优的臂"""
                )
            ]
        )
    
    def _create_dqn_course(self) -> AlgorithmCourse:
        """创建DQN课程"""
        return AlgorithmCourse(
            algorithm_name="深度Q网络 (DQN)",
            algorithm_key="dqn",
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_time="15分钟",
            icon="🧠",
            color="#8b5cf6",
            learning_objectives=[
                "理解值函数方法的基本思想",
                "掌握Q-learning算法原理",
                "学会使用神经网络近似Q函数",
                "理解经验回放和目标网络的作用"
            ],
            prerequisites=["多臂老虎机基础概念"],
            assessment_criteria=[
                "能够解释Bellman方程和Q函数",
                "能够描述DQN相比传统Q-learning的优势",
                "能够实现基本的DQN算法框架"
            ],
            theory_concepts=[
                TeachingConcept(
                    title="值函数方法概述",
                    description="通过学习状态(或状态-动作)价值来指导决策的方法",
                    explanation="""值函数方法是强化学习的重要分支，核心思想是：
                    - 学习一个价值函数V(s)或Q(s,a)，评估状态或状态-动作的好坏
                    - 根据价值函数选择最优动作
                    - 通过贝尔曼方程进行迭代更新
                    
                    相比于策略搜索方法，值函数方法通常更容易收敛和分析。""",
                    examples=[
                        "Q-learning：学习动作价值函数Q(s,a)",
                        "Value Iteration：学习状态价值函数V(s)",
                        "Policy Evaluation：评估给定策略的价值"
                    ],
                    key_points=[
                        "价值函数提供了对未来奖励的预测",
                        "基于贪心策略选择当前最优动作",
                        "通过贝尔曼方程建立递推关系",
                        "适合离散动作空间的问题"
                    ]
                )
            ],
            practical_examples=[]
        )
    
    def _create_ppo_course(self) -> AlgorithmCourse:
        """创建PPO课程"""
        return AlgorithmCourse(
            algorithm_name="近端策略优化 (PPO)",
            algorithm_key="ppo",
            difficulty=DifficultyLevel.ADVANCED,
            estimated_time="20分钟",
            icon="⚡",
            color="#10b981",
            learning_objectives=[],
            prerequisites=["策略梯度基础"],
            assessment_criteria=[],
            theory_concepts=[],
            practical_examples=[]
        )
    
    def _create_sac_course(self) -> AlgorithmCourse:
        """创建SAC课程"""
        return AlgorithmCourse(
            algorithm_name="软Actor-Critic (SAC)",
            algorithm_key="sac",
            difficulty=DifficultyLevel.EXPERT,
            estimated_time="25分钟",
            icon="🔥",
            color="#f59e0b",
            learning_objectives=[],
            prerequisites=["最大熵RL理论"],
            assessment_criteria=[],
            theory_concepts=[],
            practical_examples=[]
        )
    
    def get_course(self, algorithm_key: str) -> Optional[AlgorithmCourse]:
        """获取指定算法的课程"""
        return self.courses.get(algorithm_key)
    
    def list_courses(self) -> List[AlgorithmCourse]:
        """获取所有课程列表"""
        return list(self.courses.values())
    
    def get_learning_path(self) -> List[AlgorithmCourse]:
        """获取推荐学习路径（按难度排序）"""
        difficulty_order = {
            DifficultyLevel.BEGINNER: 1,
            DifficultyLevel.INTERMEDIATE: 2, 
            DifficultyLevel.ADVANCED: 3,
            DifficultyLevel.EXPERT: 4
        }
        return sorted(self.list_courses(), key=lambda c: difficulty_order[c.difficulty])

# 全局实例
teaching_manager = TeachingMaterialsManager()