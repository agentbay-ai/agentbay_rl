/**
 * RL Teaching Platform - WebSocket 通信模块
 */

import { statusMap } from './config.js';
import { updateStatus, addLog } from './ui.js';
import { handleTrainingStarted, handleProgressUpdate, handleEpisodeProgress, handleTrainingCompleted, handleTrainingStopped, handleTrainingError } from './training.js';
import { handleSandboxCreated, handleSandboxesCleared, handleDemoComplete, updateTestSandboxStatus } from './sandbox.js';

let socket = null;

/**
 * 连接 WebSocket
 */
export function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = function(event) {
        console.log('WebSocket 连接已建立');
        addLog('WebSocket 连接成功', 'success');
    };
    
    socket.onmessage = function(event) {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };
    
    socket.onclose = function(event) {
        console.log('WebSocket 连接已关闭');
        addLog('WebSocket 连接断开，尝试重连...', 'warning');
        setTimeout(connectWebSocket, 3000);
    };
    
    socket.onerror = function(error) {
        console.error('WebSocket 错误:', error);
        addLog('WebSocket 连接错误', 'error');
    };
}

/**
 * 处理 WebSocket 消息
 */
function handleMessage(message) {
    console.log('📨 收到WebSocket消息:', message.type, message.data);
    
    switch(message.type) {
        case 'init':
            console.log('收到初始化消息:', message.data);
            break;
        case 'log':
            addLog(message.data.message, message.data.level);
            break;
        case 'training_started':
            handleTrainingStarted(message.data);
            break;
        case 'episode_progress':
            // 每个episode的简单进度更新
            handleEpisodeProgress(message.data);
            break;
        case 'progress_update':
            // 每10个episode的完整历史记录
            console.log('📊 收到进度更新:', message.data);
            handleProgressUpdate(message.data);
            break;
        case 'training_completed':
            handleTrainingCompleted(message.data);
            break;
        case 'training_stopped':
            handleTrainingStopped();
            break;
        case 'training_error':
            handleTrainingError(message.data);
            break;
        case 'stage_update':
            updateStatus(message.data.stage);
            break;
        case 'sandbox_created':
            handleSandboxCreated(message.data);
            break;
        case 'sandboxes_cleared':
            handleSandboxesCleared();
            break;
        case 'demo_started':
            // 演示开始
            updateTestSandboxStatus('running');
            addLog(`开始模型演示 (轮次 ${message.data.episode_idx || '-'})`, 'info');
            break;
        case 'demo_complete':
            // 演示完成
            handleDemoComplete(message.data);
            break;
        default:
            console.log('未知消息类型:', message.type);
    }
}

/**
 * 获取 WebSocket 实例
 */
export function getSocket() {
    return socket;
}
