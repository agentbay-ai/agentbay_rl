/**
 * RL Teaching Platform - 配置和常量
 */

// 算法配置
// 注：DQN、PPO、SAC 暂时移除，待后端示例流程实现后再添加
// 注：bandit 暂时隐藏，但后台功能保留
export const algorithmConfigs = {
    'guider': {
        name: '强化学习入门指南',
        icon: '📚',
        color: '#8b5cf6',
        description: '系统性强化学习入门：从原理到实践',
        difficulty: '全阶段',
        estimated_time: '20 分钟',
        tutorial_file: 'guider_tutorial',
        experiment_template: 'none'  // 特殊标记：不需要实验界面
    },
    // 'bandit': {
    //     name: '多臂老虎机',
    //     icon: '🎰',
    //     color: '#6366f1',
    //     description: '强化学习入门：探索与利用的平衡',
    //     difficulty: '入门',
    //     estimated_time: '5 分钟',
    //     tutorial_file: 'bandit_tutorial',
    //     experiment_template: 'bandit_experiment'
    // },
    'ddpg': {
        name: '深度确定性策略梯度 (DDPG)',
        icon: '🦾',
        color: '#ec4899',
        description: '连续控制方法：确定性策略与价值函数结合',
        difficulty: '中级',
        estimated_time: '30分钟',
        tutorial_file: 'ddpg_tutorial',
        experiment_template: 'ddpg_experiment'
    }
};

// 训练模式描述
// 注：custom_batch 和 stable_baselines3 模式暂时移除，待后续完善后再添加
export const trainingModeDescriptions = {
    'parallel': '<strong style="color: var(--text);">并行训练(推荐)：</strong>多沙箱并行收集数据，自定义DDPG+HER实现，大幅提升训练效率。'
};

// 状态映射
export const statusMap = {
    'idle': { text: '空闲', class: 'status-idle' },
    'initializing': { text: '初始化中...', class: 'status-initializing' },
    'training': { text: '训练中...', class: 'status-training' },
    'completed': { text: '训练完成', class: 'status-completed' },
    'error': { text: '错误', class: 'status-error' },
    'stopped': { text: '已停止', class: 'status-idle' }
};
