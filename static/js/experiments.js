/**
 * RL Teaching Platform - 实验界面渲染模块
 */

import { algorithmConfigs, trainingModeDescriptions } from './config.js';
import { startTraining, stopTraining } from './training.js';
import { createSandbox, clearSandboxes } from './sandbox.js';
import { addLog } from './ui.js';

/**
 * 渲染实验界面
 */
export function renderExperimentInterface(algorithm) {
    const experimentContent = document.getElementById('experimentDynamicContent');
    const config = algorithmConfigs[algorithm];
    
    let html = '';
    
    switch(algorithm) {
        case 'bandit':
            html = renderBanditExperiment(config);
            break;
        case 'dqn':
            html = renderDQNExperiment(config);
            break;
        case 'ppo':
            html = renderPPOExperiment(config);
            break;
        case 'sac':
            html = renderSACExperiment(config);
            break;
        case 'ddpg':
            html = renderDDPGExperiment(config);
            break;
        default:
            html = `<div style="text-align: center; padding: 40px; color: var(--text-muted);">
                <i class="fas fa-cog" style="font-size: 32px; margin-bottom: 16px;"></i>
                <p>${config.name} 实验功能正在开发中...</p>
            </div>`;
    }
    
    experimentContent.innerHTML = html;
    
    // 使用setTimeout确保DOM更新完成后再绑定事件
    setTimeout(() => {
        bindExperimentEvents();
    }, 100);
}

/**
 * 绑定实验事件
 */
function bindExperimentEvents() {
    // 重新获取 DOM 元素
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const trainingModeSelect = document.getElementById('trainingModeSelect');
    
    // 移除可能存在的旧事件监听器
    if (startBtn) {
        const newStartBtn = startBtn.cloneNode(true);
        startBtn.parentNode.replaceChild(newStartBtn, startBtn);
        newStartBtn.addEventListener('click', startTraining);
        console.log('✅ 开始训练按钮事件绑定成功');
    }
    
    if (stopBtn) {
        const newStopBtn = stopBtn.cloneNode(true);
        stopBtn.parentNode.replaceChild(newStopBtn, stopBtn);
        newStopBtn.addEventListener('click', stopTraining);
        console.log('✅ 停止训练按钮事件绑定成功');
    }
    
    // 训练模式选择器事件绑定
    if (trainingModeSelect) {
        const newModeSelect = trainingModeSelect.cloneNode(true);
        trainingModeSelect.parentNode.replaceChild(newModeSelect, trainingModeSelect);
        newModeSelect.addEventListener('change', updateModeDescription);
        console.log('✅ 训练模式选择器事件绑定成功');
    }
}

/**
 * 更新训练模式说明
 */
function updateModeDescription() {
    const modeSelect = document.getElementById('trainingModeSelect');
    const descDiv = document.getElementById('modeDescription');
    if (!modeSelect || !descDiv) return;
    
    const mode = modeSelect.value;
    const description = trainingModeDescriptions[mode] || trainingModeDescriptions['custom_batch'];
    
    descDiv.innerHTML = `<div style="font-size: 13px; color: var(--text-muted);">${description}</div>`;
    addLog(`已切换训练模式: ${mode}`, 'info');
}

// ============ 实验模板函数 ============
// 注意：这些模板将从原 index.html.backup 中提取
// 为了保持代码简洁，完整模板请参考 templates/experiment-templates/ 目录

function renderBanditExperiment(config) {
    // Bandit专用模板 - 不包含训练进度和历史模块（沙箱内GUI已包含这些信息）
    return `
        <!-- 控制面板 -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #10b981, #34d399);">
                    <i class="fas fa-sliders-h"></i>
                </div>
                <h2 class="card-title">老虎机训练控制</h2>
            </div>
            
            <div class="control-panel">
                <button class="btn btn-primary" id="startBtn" disabled>
                    <i class="fas fa-play"></i>
                    开始训练
                </button>
                <button class="btn btn-error" id="stopBtn" disabled>
                    <i class="fas fa-stop"></i>
                    停止训练
                </button>
                
                <div style="flex: 1;"></div>
                
                <div class="input-group">
                    <label for="episodesInput">训练回合数:</label>
                    <input type="number" id="episodesInput" value="100" min="10" max="1000">
                </div>
                <div class="input-group">
                    <label for="armsInput">老虎机臂数:</label>
                    <input type="number" id="armsInput" value="10" min="2" max="50">
                </div>
                <div class="input-group">
                    <label for="epsilonInput">探索率 (ε):</label>
                    <input type="number" id="epsilonInput" value="0.1" min="0" max="1" step="0.05">
                </div>
            </div>
        </div>

        <!-- 实时日志 -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #3b82f6, #60a5fa);">
                    <i class="fas fa-terminal"></i>
                </div>
                <h2 class="card-title">训练日志</h2>
            </div>
            
            <div class="logs-container" id="logsContainer">
                <div class="log-entry log-info">
                    <span class="log-timestamp">[系统]</span>
                    <span class="log-message">准备开始 ${config.name} 训练，请点击开始训练按钮</span>
                </div>
            </div>
        </div>

        <!-- 沙箱流化界面 -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #8b5cf6, #a855f7);">
                    <i class="fas fa-desktop"></i>
                </div>
                <h2 class="card-title">沙箱环境 - 实时训练监控</h2>
                <div class="status-indicator status-training" style="margin-left: auto;">
                    <i class="fas fa-eye"></i>
                    <span>实时预览</span>
                </div>
                <div style="margin-left: auto; display: flex; gap: 10px;">
                    <button class="btn btn-secondary" id="createSandboxBtn">
                        <i class="fas fa-plus"></i>
                        创建沙箱
                    </button>
                    <button class="btn btn-warning" id="clearSandboxesBtn">
                        <i class="fas fa-trash"></i>
                        清理沙箱
                    </button>
                </div>
            </div>
            
            <div class="sandbox-grid" id="sandboxGrid">
                <div class="sandbox-placeholder">
                    <i class="fas fa-cloud sandbox-placeholder-icon"></i>
                    <p>暂无活动沙箱</p>
                    <p style="color: var(--text-muted); font-size: 14px;">点击"开始训练"按钮开始创建</p>
                    <p style="color: var(--text-muted); font-size: 12px; max-width: 300px; text-align: center;">💡 提示：训练开始后，沙箱中将显示实时训练GUI界面，包含进度、统计等信息</p>
                </div>
            </div>
        </div>
    `;
}

function renderDQNExperiment(config) {
    return `
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #8b5cf6, #a78bfa);">
                    <i class="fas fa-brain"></i>
                </div>
                <h2 class="card-title">${config.name} 实验</h2>
            </div>
            <div style="padding: 24px; text-align: center; color: var(--text-muted);">
                <i class="fas fa-cogs" style="font-size: 48px; margin-bottom: 20px; opacity: 0.5;"></i>
                <h3>${config.name} 实验功能正在开发中</h3>
                <p>敬请期待...</p>
            </div>
        </div>
    `;
}

function renderPPOExperiment(config) {
    return `
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #10b981, #34d399);">
                    <i class="fas fa-bolt"></i>
                </div>
                <h2 class="card-title">${config.name} 实验</h2>
            </div>
            <div style="padding: 24px; text-align: center; color: var(--text-muted);">
                <i class="fas fa-cogs" style="font-size: 48px; margin-bottom: 20px; opacity: 0.5;"></i>
                <h3>${config.name} 实验功能正在开发中</h3>
                <p>敬请期待...</p>
            </div>
        </div>
    `;
}

function renderSACExperiment(config) {
    return `
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #f59e0b, #fbbf24);">
                    <i class="fas fa-fire"></i>
                </div>
                <h2 class="card-title">${config.name} 实验</h2>
            </div>
            <div style="padding: 24px; text-align: center; color: var(--text-muted);">
                <i class="fas fa-cogs" style="font-size: 48px; margin-bottom: 20px; opacity: 0.5;"></i>
                <h3>${config.name} 实验功能正在开发中</h3>
                <p>敬请期待...</p>
            </div>
        </div>
    `;
}

function renderDDPGExperiment(config) {
    return `
        <!-- 控制面板 -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #ec4899, #f0abfc);">
                    <i class="fas fa-robot"></i>
                </div>
                <h2 class="card-title">${config.name} 实验</h2>
            </div>
            
            <div class="control-panel">
                <button class="btn btn-primary" id="startBtn" disabled>
                    <i class="fas fa-play"></i>
                    开始训练
                </button>
                <button class="btn btn-error" id="stopBtn" disabled>
                    <i class="fas fa-stop"></i>
                    停止训练
                </button>
                
                <div style="flex: 1;"></div>
                
                <!-- 训练模式：仅支持并行训练 -->
                <input type="hidden" id="trainingModeSelect" value="parallel">
                <div class="input-group" style="background: var(--bg-card-hover); padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border);">
                    <span style="color: var(--text); font-weight: 500;"><i class="fas fa-check-circle" style="color: #10b981; margin-right: 6px;"></i>并行训练模式</span>
                </div>
                <div class="input-group" id="parallelWorkersGroup">
                    <label for="parallelWorkersInput">训练沙箱数:</label>
                    <input type="number" id="parallelWorkersInput" value="5" min="1" max="10" 
                        style="padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-card-hover); color: var(--text); width: 80px;">
                    <span style="color: var(--text-muted); font-size: 12px; margin-left: 8px;">1-10</span>
                </div>
            </div>
            
            <div id="modeDescription" style="margin-top: 16px; padding: 12px 16px; background: var(--bg-card-hover); border-radius: 8px; border-left: 3px solid var(--primary);">
                <div style="font-size: 13px; color: var(--text-muted);">
                    使用多个训练沙箱并行收集数据，显著提升训练速度。测试沙箱独立创建，用于阶段性演示模型效果。集成自定义 DDPG+HER 实现。
                </div>
            </div>
        </div>

        <!-- 实时日志 -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #3b82f6, #60a5fa);">
                    <i class="fas fa-terminal"></i>
                </div>
                <h2 class="card-title">训练日志</h2>
            </div>
            
            <div class="logs-container" id="logsContainer">
                <div class="log-entry log-info">
                    <span class="log-timestamp">[系统]</span>
                    <span class="log-message">准备开始 ${config.name} 训练，请点击开始训练按钮</span>
                </div>
            </div>
        </div>

        <!-- ═══════════════════════════════════════════════════════════════ -->
        <!-- 🧪 验证沙箱区域（上方） - 模型效果演示 -->
        <!-- ═══════════════════════════════════════════════════════════════ -->
        <div style="margin: 24px 0 16px 0; padding: 12px 0; border-bottom: 2px solid var(--primary); display: flex; align-items: center; gap: 12px;">
            <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulse 2s infinite;"></div>
            <span style="font-size: 16px; font-weight: 600; color: var(--text);">
                <i class="fas fa-flask" style="color: #10b981; margin-right: 8px;"></i>
                验证沙箱区域
            </span>
            <span style="font-size: 13px; color: var(--text-muted);">（定期使用当前模型测试效果）</span>
        </div>
        
        <div class="card" style="border: 2px solid rgba(16, 185, 129, 0.3); background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), transparent);">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #10b981, #34d399);">
                    <i class="fas fa-vial"></i>
                </div>
                <h2 class="card-title">测试沙箱 - 模型效果演示</h2>
                <div class="status-indicator status-idle" id="testSandboxStatus" style="margin-left: auto;">
                    <i class="fas fa-hourglass-half"></i>
                    <span>等待训练</span>
                </div>
            </div>
            
            <div class="sandbox-grid" id="ddpgTestSandboxGrid">
                <div class="sandbox-placeholder">
                    <i class="fas fa-vial sandbox-placeholder-icon" style="color: #10b981;"></i>
                    <p>模型效果测试区域</p>
                    <p style="color: var(--text-muted); font-size: 14px;">训练过程中，会定期使用当前模型在此沙箱中执行任务</p>
                    <p style="color: var(--text-muted); font-size: 12px; max-width: 300px; text-align: center;">💡 提示：每次打印训练状态时，会在测试沙箱中演示 3 个 episode，展示当前模型的实际效果</p>
                </div>
            </div>
            
            <!-- 演示结果统计 -->
            <div id="demoStatsContainer" style="display: none; margin-top: 16px; padding: 16px; background: var(--bg-card-hover); border-radius: 8px;">
                <h4 style="margin: 0 0 12px 0; color: var(--text);">
                    <i class="fas fa-chart-bar" style="color: #10b981; margin-right: 8px;"></i>
                    最近演示结果
                </h4>
                <div class="stats-grid" style="grid-template-columns: repeat(3, 1fr);">
                    <div class="stat-card">
                        <div class="stat-value" id="demoAvgReward">-</div>
                        <div class="stat-label">平均奖励</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="demoSuccessRate">-</div>
                        <div class="stat-label" title="测试沙箱演示3个Episode的成功率">成功率 <i class="fas fa-info-circle" style="font-size: 10px; opacity: 0.6;"></i></div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="demoEpisode">-</div>
                        <div class="stat-label">训练轮次</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ═══════════════════════════════════════════════════════════════ -->
        <!-- 🤖 训练沙箱区域（下方） - 数据收集 -->
        <!-- ═══════════════════════════════════════════════════════════════ -->
        <div style="margin: 32px 0 16px 0; padding: 12px 0; border-bottom: 2px solid #8b5cf6; display: flex; align-items: center; gap: 12px;">
            <div style="width: 8px; height: 8px; background: #8b5cf6; border-radius: 50%; animation: pulse 2s infinite;"></div>
            <span style="font-size: 16px; font-weight: 600; color: var(--text);">
                <i class="fas fa-robot" style="color: #8b5cf6; margin-right: 8px;"></i>
                训练沙箱区域
            </span>
            <span style="font-size: 13px; color: var(--text-muted);">（并行收集训练数据）</span>
        </div>
        
        <div class="card" style="border: 2px solid rgba(139, 92, 246, 0.3); background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), transparent);">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #8b5cf6, #a855f7);">
                    <i class="fas fa-desktop"></i>
                </div>
                <h2 class="card-title">训练沙箱 - 机械臂训练可视化</h2>
                <div class="status-indicator status-training" style="margin-left: auto;">
                    <i class="fas fa-eye"></i>
                    <span>自动管理</span>
                </div>
            </div>
                    
            <div class="sandbox-grid" id="ddpgSandboxGrid">
                <div class="sandbox-placeholder">
                    <i class="fas fa-robot sandbox-placeholder-icon"></i>
                    <p>DDPG 机械臂训练沙箱</p>
                    <p style="color: var(--text-muted); font-size: 14px;">训练开始后，机械臂实时控制将在沙箱中显示</p>
                    <p style="color: var(--text-muted); font-size: 12px; max-width: 300px; text-align: center;">💡 提示：DDPG 算法将训练机械臂到达目标位置，可在沙箱中实时观察机械臂运动<br>🔄 沙箱将由系统自动创建和管理，无需手动操作</p>
                </div>
            </div>
        </div>
    `;
}

// 通用实验模板函数
function renderCommonExperimentTemplate(config, options) {
    const {
        title,
        icon,
        iconColor,
        sandboxGridId,
        extraInputs = '',
        modeDescription = '',
        sandboxTitle = '沙箱环境 - 实时训练监控',
        sandboxPlaceholder = `
            <i class="fas fa-cloud sandbox-placeholder-icon"></i>
            <p>暂无活动沙箱</p>
            <p style="color: var(--text-muted); font-size: 14px;">点击"创建沙箱"按钮开始创建</p>
            <p style="color: var(--text-muted); font-size: 12px; max-width: 300px; text-align: center;">💡 提示：训练开始后，实时训练过程将在此区域显示，无需额外打开页面</p>
        `
    } = options;
    
    return `
        <!-- 控制面板 -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, ${iconColor});">
                    <i class="fas fa-${icon}"></i>
                </div>
                <h2 class="card-title">${title}</h2>
            </div>
            
            <div class="control-panel">
                <button class="btn btn-primary" id="startBtn" disabled>
                    <i class="fas fa-play"></i>
                    开始训练
                </button>
                <button class="btn btn-error" id="stopBtn" disabled>
                    <i class="fas fa-stop"></i>
                    停止训练
                </button>
                
                <div style="flex: 1;"></div>
                
                ${extraInputs}
            </div>
            
            ${modeDescription}
        </div>

        <!-- 训练进度 -->
        <div class="card" id="progressCard" style="display: none;">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #f59e0b, #fbbf24);">
                    <i class="fas fa-chart-line"></i>
                </div>
                <h2 class="card-title">训练进度</h2>
                <div class="status-indicator status-training" style="margin-left: auto;">
                    <i class="fas fa-sync fa-spin"></i>
                    <span>训练中...</span>
                </div>
            </div>
            
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text">
                    <span>进度 <span id="currentEpisode">0</span>/<span id="totalEpisodes">100</span></span>
                    <span id="progressPercent">0%</span>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="currentReward">0.00</div>
                    <div class="stat-label">当前奖励</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avgReward">0.00</div>
                    <div class="stat-label">平均奖励</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="successRate">0.00%</div>
                    <div class="stat-label" title="计算口径：成功的Worker数 / 活跃Worker数，统计最近10个Episode">成功率 <i class="fas fa-info-circle" style="font-size: 10px; opacity: 0.6;"></i></div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="timeElapsed">0s</div>
                    <div class="stat-label">已用时间</div>
                </div>
            </div>
        </div>

        <!-- 训练历史 -->
        <div class="card" id="trainingHistoryCard" style="display: none;">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #06b6d4, #22d3ee);">
                    <i class="fas fa-history"></i>
                </div>
                <h2 class="card-title">训练历史（每10个Episode）</h2>
            </div>
            
            <div class="training-history-container" id="trainingHistoryContainer">
                <div style="text-align: center; color: var(--text-muted); padding: 20px;">
                    暂无训练历史
                </div>
            </div>
        </div>

        <!-- 实时日志 -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #3b82f6, #60a5fa);">
                    <i class="fas fa-terminal"></i>
                </div>
                <h2 class="card-title">训练日志</h2>
            </div>
            
            <div class="logs-container" id="logsContainer">
                <div class="log-entry log-info">
                    <span class="log-timestamp">[系统]</span>
                    <span class="log-message">准备开始 ${config.name} 训练，请点击开始训练按钮</span>
                </div>
            </div>
        </div>

        <!-- 沙箱流化界面 -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: linear-gradient(135deg, #8b5cf6, #a855f7);">
                    <i class="fas fa-desktop"></i>
                </div>
                <h2 class="card-title">${sandboxTitle}</h2>
                <div class="status-indicator status-training" style="margin-left: auto;">
                    <i class="fas fa-eye"></i>
                    <span>实时预览</span>
                </div>
                <div style="margin-left: auto; display: flex; gap: 10px;">
                    <button class="btn btn-secondary" id="createSandboxBtn">
                        <i class="fas fa-plus"></i>
                        创建沙箱
                    </button>
                    <button class="btn btn-warning" id="clearSandboxesBtn">
                        <i class="fas fa-trash"></i>
                        清理沙箱
                    </button>
                </div>
            </div>
            
            <div class="sandbox-grid" id="${sandboxGridId}">
                <div class="sandbox-placeholder">
                    ${sandboxPlaceholder}
                </div>
            </div>
        </div>
    `;
}
