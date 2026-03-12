/**
 * RL Teaching Platform - 主入口文件
 */

import { initElements } from './ui.js';
import { connectWebSocket } from './websocket.js';
import { renderAlgorithms } from './algorithm.js';

/**
 * 应用初始化
 */
async function init() {
    // 初始化 DOM 元素引用
    initElements();
    
    // 渲染算法选择卡片
    renderAlgorithms();
    
    // 连接 WebSocket
    connectWebSocket();
    
    console.log('✅ RL Teaching Platform 初始化完成');
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);
