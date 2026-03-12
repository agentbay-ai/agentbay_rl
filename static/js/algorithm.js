/**
 * RL Teaching Platform - 算法选择和教程模块
 */

import { algorithmConfigs } from './config.js';
import { getElements, addLog } from './ui.js';
import { setSelectedAlgorithm, getSelectedAlgorithm } from './training.js';
import { renderExperimentInterface } from './experiments.js';

/**
 * 加载 Markdown 教学内容
 */
export async function loadTutorialContent(tutorialName) {
    const elements = getElements();
    
    try {
        const response = await fetch(`/api/tutorial/${tutorialName}`);
        const data = await response.json();
        
        if (data.content) {
            // 使用marked解析Markdown
            const htmlContent = marked.parse(data.content);
            elements.tutorialContent.innerHTML = htmlContent;
        } else {
            elements.tutorialContent.innerHTML = `
                <div style="text-align: center; padding: 40px; color: var(--error);">
                    <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-bottom: 16px;"></i>
                    <p>加载教学内容失败: ${data.error || '未知错误'}</p>
                    <p style="font-size: 14px; margin-top: 12px;">请确保 data/${tutorialName}.md 文件存在</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('加载教学内容失败:', error);
        elements.tutorialContent.innerHTML = `
            <div style="text-align: center; padding: 40px; color: var(--error);">
                <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-bottom: 16px;"></i>
                <p>网络错误: ${error.message}</p>
            </div>
        `;
    }
}

/**
 * 渲染算法选择卡片
 */
export function renderAlgorithms() {
    const elements = getElements();
    elements.algorithmsGrid.innerHTML = '';
    
    Object.entries(algorithmConfigs).forEach(([key, config]) => {
        const card = document.createElement('div');
        card.className = 'algorithm-card';
        card.dataset.algorithm = key;
        card.style.cssText = `
            background: var(--bg-card);
            border: 2px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            height: 100px;
        `;
        
        card.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px; height: 100%;">
                <div style="width: 36px; height: 36px; border-radius: 10px; background: ${config.color}; 
                     display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;">
                    ${config.icon}
                </div>
                <div style="flex: 1; min-width: 0;">
                    <div style="font-size: 16px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${config.name}</div>
                    <div style="display: flex; gap: 12px; font-size: 12px; color: var(--text-muted); margin: 4px 0;">
                        <span style="display: flex; align-items: center; gap: 4px;">
                            <i class="fas fa-signal"></i>
                            ${config.difficulty}
                        </span>
                        <span style="display: flex; align-items: center; gap: 4px;">
                            <i class="fas fa-clock"></i>
                            ${config.estimated_time}
                        </span>
                    </div>
                    <div style="color: var(--text-muted); font-size: 13px; line-height: 1.3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${config.description}</div>
                </div>
            </div>
        `;
        
        card.addEventListener('click', () => selectAlgorithm(key, card));
        card.addEventListener('mouseenter', () => {
            card.style.borderColor = config.color;
            card.style.transform = 'translateY(-4px)';
            card.style.boxShadow = `0 12px 32px ${config.color}40`;
        });
        card.addEventListener('mouseleave', () => {
            if (!card.classList.contains('selected')) {
                card.style.borderColor = 'var(--border)';
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = 'none';
            }
        });
        
        elements.algorithmsGrid.appendChild(card);
    });
}

/**
 * 选择算法
 */
export async function selectAlgorithm(algorithm, cardElement) {
    // 清除之前的选择
    document.querySelectorAll('.algorithm-card').forEach(card => {
        card.classList.remove('selected');
        card.style.borderColor = 'var(--border)';
        card.style.background = 'var(--bg-card)';
        card.style.transform = 'translateY(0)';
        card.style.boxShadow = 'none';
    });
    
    // 选择新算法
    cardElement.classList.add('selected');
    cardElement.style.borderColor = algorithmConfigs[algorithm].color;
    cardElement.style.background = `linear-gradient(135deg, ${algorithmConfigs[algorithm].color}20, ${algorithmConfigs[algorithm].color}10)`;
    
    setSelectedAlgorithm(algorithm);
    
    // 更新标题
    document.getElementById('tutorialHeaderTitle').textContent = `${algorithmConfigs[algorithm].name} 教学`;
    document.getElementById('experimentHeaderTitle').textContent = `${algorithmConfigs[algorithm].name} 实验`;
    
    // 加载对应的教学内容
    await loadTutorialContent(algorithmConfigs[algorithm].tutorial_file);
    
    // 生成对应的实验界面
    renderExperimentInterface(algorithm);
    
    addLog(`已选择算法: ${algorithmConfigs[algorithm].name}`, 'info');
    
    // 延迟启用开始按钮（等待界面渲染完成）
    setTimeout(() => {
        const startBtn = document.getElementById('startBtn');
        if (startBtn) {
            startBtn.disabled = false;
            console.log('✅ 开始按钮已启用');
        }
    }, 200);
}
