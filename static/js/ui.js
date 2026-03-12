/**
 * RL Teaching Platform - UI 工具模块
 */

import { statusMap } from './config.js';

// DOM 元素缓存
let elements = {
    statusIndicator: null,
    tutorialContent: null,
    algorithmsGrid: null,
    logsContainer: null
};

/**
 * 初始化 DOM 元素引用
 */
export function initElements() {
    elements.statusIndicator = document.getElementById('statusIndicator');
    elements.tutorialContent = document.getElementById('tutorialContent');
    elements.algorithmsGrid = document.getElementById('algorithmsGrid');
    elements.logsContainer = document.getElementById('logsContainer');
}

/**
 * 获取 DOM 元素
 */
export function getElements() {
    return elements;
}

/**
 * 更新状态指示器
 */
export function updateStatus(stage) {
    const status = statusMap[stage] || statusMap['idle'];
    elements.statusIndicator.className = `status-indicator ${status.class}`;
    elements.statusIndicator.innerHTML = stage === 'training' ? 
        '<i class="fas fa-sync fa-spin"></i><span>' + status.text + '</span>' : 
        '<span>' + status.text + '</span>';
}

/**
 * 添加日志
 */
export function addLog(message, level = 'info') {
    const logsContainer = document.getElementById('logsContainer');
    if (!logsContainer) return;
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${level}`;
    
    const timestamp = new Date().toLocaleTimeString();
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-message">${message}</span>
    `;
    
    logsContainer.appendChild(logEntry);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

/**
 * 清空日志
 */
export function clearLogs() {
    if (elements.logsContainer) {
        elements.logsContainer.innerHTML = '';
    }
}

/**
 * 显示/隐藏元素
 */
export function toggleElement(elementId, show) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = show ? 'block' : 'none';
    }
}

/**
 * 启用/禁用按钮
 */
export function setButtonState(buttonId, enabled) {
    const button = document.getElementById(buttonId);
    if (button) {
        button.disabled = !enabled;
    }
}

/**
 * 更新进度条
 */
export function updateProgress(progress) {
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (progressPercent) progressPercent.textContent = `${Math.round(progress)}%`;
}

/**
 * 更新统计数据
 */
export function updateStats(stats) {
    Object.entries(stats).forEach(([key, value]) => {
        const element = document.getElementById(key);
        if (element) {
            element.textContent = value;
        }
    });
}
