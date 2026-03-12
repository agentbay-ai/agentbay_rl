/**
 * RL Teaching Platform - 训练控制模块
 */

import { updateStatus, addLog, toggleElement, setButtonState, updateProgress, updateStats } from './ui.js';
import { algorithmConfigs } from './config.js';

// 训练数据
let trainingData = {
    rewards: [],
    episode_lengths: [],
    policy_losses: [],
    value_losses: []
};

// 训练历史记录（每10个episode的记录）
let trainingHistory = [];

let selectedAlgorithm = '';

/**
 * 设置选中的算法
 */
export function setSelectedAlgorithm(algorithm) {
    selectedAlgorithm = algorithm;
}

/**
 * 获取选中的算法
 */
export function getSelectedAlgorithm() {
    return selectedAlgorithm;
}

/**
 * 开始训练
 */
export async function startTraining() {
    if (!selectedAlgorithm) {
        addLog('请先选择一个算法', 'warning');
        return;
    }
    
    // 获取动态生成的表单元素
    const armsInput = document.getElementById('armsInput');
    const epsilonInput = document.getElementById('epsilonInput');
    const trainingModeSelect = document.getElementById('trainingModeSelect');
    const parallelWorkersInput = document.getElementById('parallelWorkersInput');
    
    // DDPG 不再从前端获取 episodes 和 eval_freq，使用后台默认配置
    const arms = armsInput ? parseInt(armsInput.value) : 10;
    const epsilon = epsilonInput ? parseFloat(epsilonInput.value) : 0.1;
    const trainingMode = trainingModeSelect ? trainingModeSelect.value : 'parallel';
    const parallelWorkers = parallelWorkersInput ? parseInt(parallelWorkersInput.value) : 5;
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                algorithm: selectedAlgorithm,
                episodes: 100,  // bandit 使用的默认值，DDPG 会在后台覆盖
                parallel_sandboxes: parallelWorkers,  // 并行沙箱数量
                config: {
                    n_arms: arms,
                    epsilon: epsilon,
                    training_mode: trainingMode,
                    parallel_workers: parallelWorkers
                    // 移除 eval_freq，使用后台默认配置
                }
            })
        });
        
        if (response.ok) {
            setButtonState('startBtn', false);
            setButtonState('stopBtn', true);
            toggleElement('progressCard', true);
            
            updateStatus('initializing');
            addLog(`开始训练 ${algorithmConfigs[selectedAlgorithm].name}`, 'info');
        } else {
            const error = await response.json();
            addLog(`训练启动失败: ${error.detail}`, 'error');
        }
    } catch (error) {
        addLog(`网络错误: ${error.message}`, 'error');
    }
}

/**
 * 停止训练
 */
export async function stopTraining() {
    try {
        const response = await fetch('/api/stop', {
            method: 'POST'
        });
        
        if (response.ok) {
            setButtonState('startBtn', true);
            setButtonState('stopBtn', false);
            
            addLog('正在停止训练...', 'warning');
        }
    } catch (error) {
        addLog(`停止训练失败: ${error.message}`, 'error');
    }
}

/**
 * 处理训练开始
 */
export function handleTrainingStarted(data) {
    setButtonState('startBtn', false);
    setButtonState('stopBtn', true);
    toggleElement('progressCard', true);
    toggleElement('trainingHistoryCard', true);
    
    const totalEpisodes = document.getElementById('totalEpisodes');
    if (totalEpisodes) totalEpisodes.textContent = data.total_episodes;
    
    trainingData = {
        rewards: [],
        episode_lengths: [],
        policy_losses: [],
        value_losses: []
    };
    
    // 清空训练历史
    trainingHistory = [];
    updateHistoryDisplay();
    
    addLog(`开始训练 ${data.algorithm}，共 ${data.total_episodes} 回合`, 'success');
}

/**
 * 处理每个episode的简单进度更新
 */
export function handleEpisodeProgress(data) {
    try {
        const episode = data.episode || 0;
        const totalEpisodes = data.total_episodes || 0;
        const progressPercent = data.progress_percent || 0;
        const currentReward = data.current_reward || 0;
        
        // 更新进度条
        updateProgress(progressPercent);
        
        // 更新当前episode显示
        const currentEpisodeEl = document.getElementById('currentEpisode');
        const totalEpisodesEl = document.getElementById('totalEpisodes');
        if (currentEpisodeEl) currentEpisodeEl.textContent = episode;
        if (totalEpisodesEl) totalEpisodesEl.textContent = totalEpisodes;
        
        // 更新当前奖励
        const currentRewardEl = document.getElementById('currentReward');
        if (currentRewardEl) currentRewardEl.textContent = currentReward.toFixed(2);
        
        console.log(`🏃 Episode ${episode}/${totalEpisodes} (${progressPercent.toFixed(1)}%)`);
    } catch (error) {
        console.error('❌ Episode进度更新出错:', error);
    }
}

/**
 * 处理进度更新（每10个episode的完整历史记录）
 */
export function handleProgressUpdate(data) {
    console.log('📊 处理进度更新数据:', data);
    
    try {
        // 更新进度条
        const progressPercent = data.progress_percent || 0;
        updateProgress(progressPercent);
        
        // 更新当前episode
        const currentEpisode = document.getElementById('currentEpisode');
        if (currentEpisode) currentEpisode.textContent = data.episode || 0;
        
        // 更新当前统计
        const summary = data.summary || {};
        const currentReward = data.current_reward || 0;
        updateStats({
            currentReward: currentReward.toFixed(2),
            avgReward: summary.avg_reward ? summary.avg_reward.toFixed(2) : '0.00',
            successRate: summary.success_rate ? (summary.success_rate * 100).toFixed(1) + '%' : '0.0%'
        });
        
        // 添加到历史记录
        const historyEntry = {
            episode: data.episode || 0,
            reward: currentReward,
            avgReward: summary.avg_reward || 0,
            successRate: summary.success_rate || 0,
            trend: summary.trend || 'stable',
            trendValue: summary.trend_value || 0,
            effectiveness: summary.effectiveness_cn || '未知',
            timestamp: new Date().toLocaleTimeString()
        };
        
        trainingHistory.push(historyEntry);
        updateHistoryDisplay();
        
        // 存储数据用于图表
        trainingData.rewards.push(currentReward);
        
        console.log('✅ 进度更新处理完成, 历史记录数:', trainingHistory.length);
    } catch (error) {
        console.error('❌ 进度更新处理出错:', error);
    }
}

/**
 * 更新历史记录显示
 */
function updateHistoryDisplay() {
    const historyContainer = document.getElementById('trainingHistoryContainer');
    if (!historyContainer) return;
    
    if (trainingHistory.length === 0) {
        historyContainer.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 20px;">暂无训练历史</div>';
        return;
    }
    
    // 倒序显示（最新的在上面）
    const html = trainingHistory.slice().reverse().map(entry => {
        const trendIcon = entry.trend === 'improving' ? '📈' : 
                         entry.trend === 'declining' ? '📉' : '➡️';
        const trendColor = entry.trend === 'improving' ? '#10b981' : 
                          entry.trend === 'declining' ? '#ef4444' : '#6b7280';
        
        return `
            <div class="history-entry">
                <div class="history-header">
                    <span class="history-episode">Episode ${entry.episode}</span>
                    <span class="history-time">${entry.timestamp}</span>
                </div>
                <div class="history-stats">
                    <div class="history-stat">
                        <span class="history-label">当前奖励:</span>
                        <span class="history-value">${entry.reward.toFixed(2)}</span>
                    </div>
                    <div class="history-stat">
                        <span class="history-label">平均奖励:</span>
                        <span class="history-value">${entry.avgReward.toFixed(2)}</span>
                    </div>
                    <div class="history-stat">
                        <span class="history-label">成功率:</span>
                        <span class="history-value">${(entry.successRate * 100).toFixed(1)}%</span>
                    </div>
                    <div class="history-stat">
                        <span class="history-label">趋势:</span>
                        <span class="history-value" style="color: ${trendColor}">
                            ${trendIcon} ${entry.effectiveness}
                        </span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    historyContainer.innerHTML = html;
}

/**
 * 处理训练完成
 */
export function handleTrainingCompleted(data) {
    setButtonState('startBtn', true);
    setButtonState('stopBtn', false);
    
    updateStatus('completed');
    addLog(`训练完成！最终平均奖励: ${data.final_stats.final_avg_reward.toFixed(2)}`, 'success');
}

/**
 * 处理训练停止
 */
export function handleTrainingStopped() {
    setButtonState('startBtn', true);
    setButtonState('stopBtn', false);
    
    updateStatus('idle');
    addLog('训练已停止', 'warning');
}

/**
 * 处理训练错误
 */
export function handleTrainingError(data) {
    setButtonState('startBtn', true);
    setButtonState('stopBtn', false);
    
    updateStatus('error');
    addLog(`训练出错: ${data.error}`, 'error');
}

/**
 * 获取训练数据
 */
export function getTrainingData() {
    return trainingData;
}
