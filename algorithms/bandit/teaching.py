"""
多臂老虎机教学演示模块

将教学逻辑与 Web 框架解耦，便于维护和测试。
"""

import logging
from typing import Dict, Any, Callable, Awaitable, Optional
from dataclasses import dataclass

from .sandbox_bandit import SandboxBanditRunner

logger = logging.getLogger(__name__)


@dataclass
class TeachingContext:
    """教学演示上下文，封装与 Web 框架的交互接口"""
    
    # 沙箱管理器
    sandbox_manager: Any
    
    # 日志发送函数: async def send_log(message: str, level: str)
    send_log: Callable[[str, str], Awaitable[None]]
    
    # 广播消息函数: async def broadcast_message(msg_type: str, data: dict)
    broadcast_message: Callable[[str, Dict], Awaitable[None]]
    
    # 状态更新回调
    on_stage_change: Optional[Callable[[str], Awaitable[None]]] = None
    
    # 配置
    config: Optional[Dict[str, Any]] = None


async def run_bandit_teaching(
    request_episodes: int,
    request_config: Optional[Dict[str, Any]],
    ctx: TeachingContext
) -> Dict[str, Any]:
    """
    运行多臂老虎机教学演示（沙箱版本）
    
    Args:
        request_episodes: 训练回合数
        request_config: 算法配置参数
        ctx: 教学上下文，包含沙箱管理器和回调函数
        
    Returns:
        训练结果字典，包含最终统计数据
    """
    result_data = {}
    sandbox_runner = None
    
    try:
        # 检查沙箱管理器
        if not ctx.sandbox_manager:
            raise RuntimeError("沙箱管理器未初始化")
        
        # 初始化教学演示
        await ctx.send_log("🚨 初始化沙箱版多臂老虎机教学演示（训练全过程在沙箱中执行）", "info")
        if ctx.on_stage_change:
            await ctx.on_stage_change("initializing")
        await ctx.broadcast_message("stage_update", {"stage": "initializing"})
        
        # 创建沙箱老虎机运行器
        sandbox_runner = SandboxBanditRunner(ctx.sandbox_manager)
        
        # 创建沙箱会话
        success = await sandbox_runner.create_sandbox_session()
        if not success:
            raise RuntimeError("创建老虎机沙箱会话失败")
        
        await ctx.send_log(f"老虎机沙箱创建成功，sandbox_id: {sandbox_runner.sandbox_id}", "success")
        
        # 获取完整的沙箱信息
        sandbox_info = {
            "sandbox_id": sandbox_runner.sandbox_id,
            "session_id": sandbox_runner.session.session_id if sandbox_runner.session else None,
            "resource_url": sandbox_runner.session.resource_url if sandbox_runner.session else None
        }
        
        await ctx.broadcast_message("sandbox_created", sandbox_info)
        
        # 获取算法参数
        algo_params = request_config or {}
        n_arms = algo_params.get("n_arms", 10)
        epsilon = algo_params.get("epsilon", 0.1)
        
        # 开始训练阶段
        if ctx.on_stage_change:
            await ctx.on_stage_change("training")
        await ctx.broadcast_message("stage_update", {"stage": "training"})
        await ctx.send_log(f"开始在沙箱中运行老虎机训练 ({request_episodes} 回合) - 完整训练过程将在沙箱内可视化展示", "info")
        
        # 在沙箱中运行老虎机
        result = await sandbox_runner.run_epsilon_greedy_demo(
            n_arms=n_arms,
            n_episodes=request_episodes,
            epsilon=epsilon
        )
        
        # 根据配置决定是否清理沙箱
        preserve_sandbox = ctx.config.get("preserve_sandbox_after_training", False) if ctx.config else False
        await sandbox_runner.cleanup(force_cleanup=not preserve_sandbox)
        
        # 处理训练结果
        if result:
            final_opt = result.get('final_opt', 0)
            final_avg = result.get('final_avg', 0)
            optimal_arm = result.get('optimal_arm', 0)
            counts = result.get('counts', [0] * n_arms)
            values = result.get('values', [0.0] * n_arms)
            means = result.get('means', [0.0] * n_arms)
            
            # 计算信息增益指标
            random_rate = 1.0 / n_arms
            improvement = final_opt / random_rate if random_rate > 0 else 0
            min_count = min(counts) if counts else 0
            optimal_count = counts[optimal_arm] if optimal_arm < len(counts) else 0
            
            # 发送直观总结到前端
            await ctx.send_log("📊 训练效果总结:", "info")
            await ctx.send_log(f"✅ 核心成果: 从'完全不知道' → '{final_opt:.0%}把握识别最优臂(臂{optimal_arm})'", "success")
            await ctx.send_log(f"✅ 效率提升: 比随机选择({random_rate:.0%})提升了{improvement:.1f}倍", "success")
            await ctx.send_log(f"✅ 最优臂选择: {optimal_count}次/总{request_episodes}次，占比{final_opt:.1%}", "success")
            if min_count >= 1:
                await ctx.send_log(f"✅ 探索充分性: 每个臂至少被探索{min_count}次，估计较可靠", "success")
            else:
                await ctx.send_log(f"⚠️ 探索充分性: 部分臂仅探索{min_count}次，估计可能不稳健", "warning")
            
            # 直白解释
            await ctx.send_log("💡 直白解释:", "info")
            if final_opt >= 0.6:
                await ctx.send_log("   训练效果很好！算法成功找到了最优臂，并大部分时候选择了它。", "success")
            elif final_opt >= 0.4:
                await ctx.send_log("   训练效果良好。算法识别出了较优的臂，但探索还不够充分。", "info")
            else:
                await ctx.send_log("   训练效果一般。算法还在探索阶段，没有稳定选择最优臂。", "warning")
            
            await ctx.send_log(f"   真实最优臂是臂{optimal_arm}(均值{means[optimal_arm]:.2f})，算法估计为{values[optimal_arm]:.2f}", "info")
            if abs(values[optimal_arm] - means[optimal_arm]) < 0.3:
                await ctx.send_log("   估计值与真实值接近，说明学习有效。", "success")
            else:
                await ctx.send_log("   估计值与真实值有偏差，可能需要更多训练回合。", "warning")
            
            # 构建结果数据
            result_data = {
                "sandbox_used": True,
                "sandbox_id": sandbox_runner.sandbox_id,
                "total_episodes": request_episodes,
                "algorithm": "epsilon_greedy",
                "parameters": {
                    "n_arms": n_arms,
                    "epsilon": epsilon
                },
                "summary": {
                    "optimal_rate": final_opt,
                    "improvement_factor": improvement,
                    "optimal_arm": optimal_arm,
                    "exploration_sufficient": min_count >= 1
                }
            }
        
        await ctx.send_log("🚨 沙箱版多臂老虎机教学演示完成！（训练全过程在沙箱中执行并可视化）", "success")
        
        return result_data
        
    except Exception as e:
        logger.error(f"老虎机教学演示错误: {e}")
        await ctx.send_log(f"教学演示出错: {str(e)}", "error")
        if ctx.on_stage_change:
            await ctx.on_stage_change("error")
        await ctx.broadcast_message("training_error", {"error": str(e)})
        raise
