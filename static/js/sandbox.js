/**
 * RL Teaching Platform - 沙箱管理模块
 */

import { addLog } from './ui.js';
import { getSelectedAlgorithm } from './training.js';

/**
 * 创建沙箱查看器
 */
export function addSandboxViewer(sandbox, algorithm = 'bandit') {
    // 根据算法类型和沙箱类型选择不同的沙箱容器
    let gridId = 'sandboxGrid';
    if (algorithm === 'ddpg') {
        // 区分测试沙箱和训练沙箱
        if (sandbox.sandbox_type === 'testing') {
            gridId = 'ddpgTestSandboxGrid';
        } else {
            gridId = 'ddpgSandboxGrid';
        }
    }
    
    const grid = document.getElementById(gridId);
    if (!grid) return;
    
    // 清除占位符
    const placeholder = grid.querySelector('.sandbox-placeholder');
    if (placeholder) {
        grid.innerHTML = '';
    }
    
    const viewer = document.createElement('div');
    viewer.className = 'sandbox-viewer';
    viewer.id = `sandbox-${sandbox.session_id}`;
    
    // 测试沙箱使用不同的样式
    const isTestSandbox = sandbox.sandbox_type === 'testing';
    const labelColor = isTestSandbox ? '#10b981' : '#8b5cf6';
    const labelText = isTestSandbox ? '测试沙箱' : '训练沙箱';
    
    viewer.innerHTML = `
        <div class="sandbox-header">
            <span class="sandbox-platform" style="background: ${labelColor}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">
                ${labelText} #${sandbox.session_id.substring(0, 8)}
            </span>
        </div>
        <iframe class="sandbox-frame" src="${sandbox.resource_url}" 
                sandbox="allow-scripts allow-same-origin" allowfullscreen></iframe>
    `;
    
    grid.appendChild(viewer);
    addLog(`${labelText}已创建: ${sandbox.session_id.substring(0, 8)}`, 'success');
}

/**
 * 处理沙箱创建
 */
export function handleSandboxCreated(sandbox) {
    // 根据沙箱类型决定使用哪个算法上下文
    let algorithmContext = getSelectedAlgorithm();
    if (sandbox.sandbox_type) {
        // 如果是DDPG的特定沙箱类型，确保使用ddpg算法上下文
        if ((sandbox.sandbox_type === 'training' || sandbox.sandbox_type === 'testing') && algorithmContext === 'ddpg') {
            algorithmContext = 'ddpg';
        }
    }
    addSandboxViewer(sandbox, algorithmContext);
    
    const activeSandboxes = document.getElementById('activeSandboxes');
    if (activeSandboxes) {
        activeSandboxes.textContent = document.querySelectorAll('.sandbox-viewer').length;
    }
}

/**
 * 处理沙箱清理
 */
export function handleSandboxesCleared() {
    const grid = document.getElementById('sandboxGrid');
    const activeSandboxes = document.getElementById('activeSandboxes');
    
    if (grid) {
        grid.innerHTML = `
            <div class="sandbox-placeholder">
                <i class="fas fa-cloud sandbox-placeholder-icon"></i>
                <p>暂无活动沙箱</p>
                <p style="color: var(--text-muted); font-size: 14px;">点击"创建沙箱"按钮开始创建</p>
            </div>
        `;
    }
    if (activeSandboxes) {
        activeSandboxes.textContent = '0';
    }
    addLog('所有沙箱已清理', 'info');
}

/**
 * 创建沙箱
 */
export async function createSandbox() {
    const createSandboxBtn = document.getElementById('createSandboxBtn');
    if (!createSandboxBtn) return;
    
    try {
        createSandboxBtn.disabled = true;
        createSandboxBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 创建中...';
        
        const response = await fetch('/api/create-sandbox', {
            method: 'POST'
        });
        
        if (response.ok) {
            const sandbox = await response.json();
            addLog(`沙箱创建成功: ${sandbox.session_id.substring(0, 8)}`, 'success');
        } else {
            const error = await response.json();
            addLog(`沙箱创建失败: ${error.detail}`, 'error');
        }
    } catch (error) {
        addLog(`网络错误: ${error.message}`, 'error');
    } finally {
        createSandboxBtn.disabled = false;
        createSandboxBtn.innerHTML = '<i class="fas fa-plus"></i> 创建沙箱';
    }
}

/**
 * 清理沙箱
 */
export async function clearSandboxes() {
    const clearSandboxesBtn = document.getElementById('clearSandboxesBtn');
    if (!clearSandboxesBtn) return;
    
    if (confirm('确定要清理所有沙箱会话吗？')) {
        try {
            clearSandboxesBtn.disabled = true;
            clearSandboxesBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 清理中...';
            
            const response = await fetch('/api/clear-sandboxes', {
                method: 'POST'
            });
            
            if (response.ok) {
                const result = await response.json();
                addLog(result.message, 'success');
            } else {
                const error = await response.json();
                addLog(`清理失败: ${error.detail}`, 'error');
            }
        } catch (error) {
            addLog(`网络错误: ${error.message}`, 'error');
        } finally {
            clearSandboxesBtn.disabled = false;
            clearSandboxesBtn.innerHTML = '<i class="fas fa-trash"></i> 清理沙箱';
        }
    }
}

/**
 * 处理演示完成事件
 */
export function handleDemoComplete(demoResult) {
    // 更新测试沙箱状态指示器
    const statusIndicator = document.getElementById('testSandboxStatus');
    if (statusIndicator) {
        statusIndicator.className = 'status-indicator status-success';
        statusIndicator.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>演示完成</span>
        `;
        
        // 3秒后恢复为等待状态
        setTimeout(() => {
            statusIndicator.className = 'status-indicator status-idle';
            statusIndicator.innerHTML = `
                <i class="fas fa-hourglass-half"></i>
                <span>等待下一轮</span>
            `;
        }, 3000);
    }
    
    // 更新演示结果统计
    const demoStatsContainer = document.getElementById('demoStatsContainer');
    if (demoStatsContainer && demoResult.status === 'completed') {
        demoStatsContainer.style.display = 'block';
        
        const avgRewardEl = document.getElementById('demoAvgReward');
        const successRateEl = document.getElementById('demoSuccessRate');
        const episodeEl = document.getElementById('demoEpisode');
        
        if (avgRewardEl) avgRewardEl.textContent = demoResult.avg_reward?.toFixed(3) || '-';
        if (successRateEl) successRateEl.textContent = ((demoResult.success_rate || 0) * 100).toFixed(1) + '%';
        if (episodeEl) episodeEl.textContent = demoResult.episode_idx || '-';
    }
    
    addLog(`演示完成: 平均奖励=${demoResult.avg_reward?.toFixed(3)}, 成功率=${((demoResult.success_rate || 0) * 100).toFixed(1)}%`, 'success');
}

/**
 * 更新测试沙箱状态
 */
export function updateTestSandboxStatus(status, message = '') {
    const statusIndicator = document.getElementById('testSandboxStatus');
    if (!statusIndicator) return;
    
    switch(status) {
        case 'running':
            statusIndicator.className = 'status-indicator status-training';
            statusIndicator.innerHTML = `
                <i class="fas fa-sync fa-spin"></i>
                <span>演示中...</span>
            `;
            break;
        case 'success':
            statusIndicator.className = 'status-indicator status-success';
            statusIndicator.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <span>${message || '演示完成'}</span>
            `;
            break;
        case 'error':
            statusIndicator.className = 'status-indicator status-error';
            statusIndicator.innerHTML = `
                <i class="fas fa-exclamation-circle"></i>
                <span>${message || '演示失败'}</span>
            `;
            break;
        default:
            statusIndicator.className = 'status-indicator status-idle';
            statusIndicator.innerHTML = `
                <i class="fas fa-hourglass-half"></i>
                <span>${message || '等待训练'}</span>
            `;
    }
}
