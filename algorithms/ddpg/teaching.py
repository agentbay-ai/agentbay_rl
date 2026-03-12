"""
DDPG 教学演示模块

将教学逻辑与 Web 框架解耦，便于维护和测试。
"""

import logging
from typing import Dict, Any, Callable, Awaitable, Optional
from dataclasses import dataclass

from .sandbox_ddpg import SandboxDDPG

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


# DDPG 训练最优默认配置
DDPG_DEFAULT_EPISODES = 20000  # 需要足够步数才能让 DDPG+HER 收敛，FetchReach-v4 通常需要 10000-50000 步
DDPG_DEFAULT_EVAL_FREQ = 1000  # 每1000步评估一次


async def run_ddpg_teaching(
    request_episodes: int,
    request_parallel_sandboxes: int,
    request_config: Optional[Dict[str, Any]],
    ctx: TeachingContext
) -> Dict[str, Any]:
    """
    运行 DDPG 教学演示（使用新的模块化架构）
    
    Args:
        request_episodes: 训练回合数（会被后台最优默认值覆盖）
        request_parallel_sandboxes: 并行沙箱数量
        request_config: 算法配置参数
        ctx: 教学上下文，包含沙箱管理器和回调函数
        
    Returns:
        训练结果字典
    """
    result_data = {}
    
    try:
        # 检查沙箱管理器
        if not ctx.sandbox_manager:
            raise RuntimeError("沙箱管理器未初始化")
        
        # 初始化教学演示
        await ctx.send_log("🎯 初始化模块化DDPG机械臂控制教学演示", "info")
        if ctx.on_stage_change:
            await ctx.on_stage_change("initializing")
        await ctx.broadcast_message("stage_update", {"stage": "initializing"})
        
        # 创建沙箱DDPG运行器（使用新的模块化架构）
        sandbox_runner = SandboxDDPG(ctx.sandbox_manager)
        
        # 获取算法参数
        algo_params = request_config or {}
        # DDPG 使用后台配置的最优默认值（忽略前端传入的 episodes）
        episodes = DDPG_DEFAULT_EPISODES
        eval_freq = DDPG_DEFAULT_EVAL_FREQ
        # 支持两种参数名：training_mode（前端使用）和 mode（向后兼容）
        # 默认使用并行训练模式
        training_mode = algo_params.get("training_mode", algo_params.get("mode", "parallel"))
        # 并行沙箱数量（默认5个）
        parallel_workers = algo_params.get("parallel_workers", request_parallel_sandboxes) or 5
        
        if training_mode == "parallel":
            await ctx.send_log(f"📋 DDPG并行训练配置: {episodes} episodes, {parallel_workers} 个并行沙箱", "info")
        else:
            await ctx.send_log(f"📋 DDPG训练配置: {episodes} episodes, 每 {eval_freq} episodes 评估一次", "info")
        
        # 定义沙箱创建后的回调函数
        async def on_sandbox_created(sandbox_info):
            """沙箱创建后广播消息给前端"""
            await ctx.send_log(f"🖥️ DDPG训练沙箱已创建", "info")
            await ctx.broadcast_message("sandbox_created", sandbox_info)
            # 更新阶段状态
            if ctx.on_stage_change:
                await ctx.on_stage_change("training")
            await ctx.broadcast_message("stage_update", {"stage": "training"})
        
        # 定义episode完成后的回调函数
        async def on_episode_complete(progress_info):
            """每个episode完成后的回调"""
            # 处理 SB3 的 step-based 进度
            if progress_info.get('type') == 'sb3_progress':
                timesteps = progress_info.get('timesteps', 0)
                total_timesteps = progress_info.get('total_timesteps', 0)
                progress_percent = progress_info.get('progress_percent', 0)
                num_episodes = progress_info.get('num_episodes', 0)
                success_rate = progress_info.get('success_rate', 0)
                avg_reward = progress_info.get('avg_reward', 0)
                
                # 打印控制台日志
                print(f"🏃 SB3 训练: {timesteps}/{total_timesteps} ({progress_percent:.1f}%) "
                      f"| Episodes: {num_episodes} | 成功率: {success_rate:.1%} | 平均奖励: {avg_reward:.2f}")
                
                # 发送进度更新到前端（包含成功率和奖励）
                simple_progress = {
                    "episode": num_episodes,
                    "total_episodes": total_timesteps,
                    "progress_percent": progress_percent,
                    "current_reward": avg_reward,
                    "success_rate": success_rate,
                    "timesteps": timesteps,
                    "mode": "sb3"
                }
                await ctx.broadcast_message("episode_progress", simple_progress)
                
                # 每 50 步发送日志（包含成功率信息）
                if timesteps % 50 == 0 and timesteps > 0:
                    await ctx.send_log(
                        f"SB3 训练: {timesteps} 步 | {num_episodes} Episodes | "
                        f"成功率: {success_rate:.1%} | 平均奖励: {avg_reward:.2f}",
                        "info"
                    )
                return
            
            # 原有的 episode-based 进度处理
            episode = progress_info.get('episode', 0)
            total_episodes = progress_info.get('total_episodes', 0)
            progress_percent = progress_info.get('progress_percent', 0)
            summary = progress_info.get('summary', {})
            current_result = progress_info.get('current_result', {})
            
            # 每个episode都打印控制台日志
            print(f"🏃 Episode {episode}/{total_episodes} ({progress_percent:.1f}%) - "
                  f"奖励: {current_result.get('reward', 0):.2f}, "
                  f"成功: {current_result.get('success', False)}")
            
            # 每个episode都发送简单的进度更新（用于更新进度条和当前episode显示）
            simple_progress = {
                "episode": episode,
                "total_episodes": total_episodes,
                "progress_percent": progress_percent,
                "current_reward": current_result.get('reward', 0),
            }
            await ctx.broadcast_message("episode_progress", simple_progress)
            
            # 每10个episode发送完整的历史记录（与评估矩阵同步）
            log_interval = ctx.config.get("log_interval", 10) if ctx.config else 10
            if (episode + 1) % log_interval == 0:
                # 构建前端友好的进度消息（包含完整统计信息）
                progress_message = {
                    "episode": episode,
                    "total_episodes": total_episodes,
                    "progress_percent": progress_percent,
                    "current_reward": current_result.get('reward', 0),
                    "current_steps": current_result.get('steps', 0),
                    "current_success": current_result.get('success', False),
                    "summary": {
                        "avg_reward": summary.get('avg_reward', 0),
                        "success_rate": summary.get('success_rate', 0),
                        "trend": summary.get('trend', 'stable'),
                        "trend_value": summary.get('trend_value', 0),
                        "effectiveness": summary.get('effectiveness', 'unknown'),
                        "effectiveness_cn": summary.get('effectiveness_cn', '未知'),
                        "window_size": summary.get('window_size', 0)
                    }
                }
                
                # 广播训练进度更新（用于历史记录）
                print(f"📤 广播历史记录: Episode {episode}")
                await ctx.broadcast_message("progress_update", progress_message)
                
                # 发送日志
                trend_emoji = "📈" if summary.get('trend') == 'improving' else \
                             "📉" if summary.get('trend') == 'declining' else "➡️"
                await ctx.send_log(
                    f"Episode {episode}: 奖励={current_result.get('reward', 0):.2f}, "
                    f"平均={summary.get('avg_reward', 0):.2f}, "
                    f"成功率={summary.get('success_rate', 0):.1%} {trend_emoji}",
                    "info"
                )
        
        # 定义演示完成后的回调函数
        async def on_demo_complete(demo_result):
            """模型演示完成后的回调"""
            status = demo_result.get('status', 'unknown')
            if status == 'completed':
                episode_idx = demo_result.get('episode_idx', 0)
                avg_reward = demo_result.get('avg_reward', 0)
                success_rate = demo_result.get('success_rate', 0)
                success_count = demo_result.get('success_count', 0)
                demo_count = demo_result.get('demo_count', 0)
                
                # 广播演示结果到前端
                await ctx.broadcast_message("demo_complete", {
                    "status": status,
                    "episode_idx": episode_idx,
                    "avg_reward": avg_reward,
                    "success_rate": success_rate,
                    "success_count": success_count,
                    "demo_count": demo_count,
                    "demo_results": demo_result.get('demo_results', [])
                })
                
                await ctx.send_log(
                    f"🎬 演示完成: 轮次 {episode_idx}, 平均奖励: {avg_reward:.3f}, 成功率: {success_count}(成功数)/{demo_count}(演示数)={success_rate:.1%}",
                    "info"
                )
            elif status == 'skipped':
                print(f"⏭️ 演示跳过: {demo_result.get('reason', 'unknown')}")
            else:
                print(f"⚠️ 演示状态异常: {status}")
        
        # 开始训练阶段
        log_interval = ctx.config.get("log_interval", 10) if ctx.config else 10
        await ctx.send_log(f"🚀 开始模块化DDPG训练 ({episodes} 回合, 模式: {training_mode})", "info")
        await ctx.send_log(f"📊 指标口径: 成功率 = 成功Worker数/活跃Worker数（最近{log_interval}轮平均）", "info")
        
        # 使用新的模块化API运行训练（传递沙箱创建回调、episode进度回调和演示回调）
        preserve_sandbox = ctx.config.get("preserve_sandbox_after_training", False) if ctx.config else False
        result = await sandbox_runner.run_training(
            mode=training_mode,
            custom_config={
                "episodes": episodes,
                "eval_freq": eval_freq,
                "parallel_workers": parallel_workers,  # 并行沙箱数量
                "force_cleanup": not preserve_sandbox
            },
            on_sandbox_created=on_sandbox_created,
            on_episode_complete=on_episode_complete,
            on_demo_complete=on_demo_complete
        )
        
        # 构建结果数据
        result_data = {
            "sandbox_used": True,
            "total_episodes": episodes,
            "algorithm": "ddpg",
            "training_mode": training_mode,
            "parameters": {
                "episodes": episodes,
                "eval_freq": eval_freq,
                "mode": training_mode
            },
            "result": result
        }
        
        await ctx.send_log("🎉 模块化DDPG机械臂控制教学演示完成！", "success")
        
        return result_data
        
    except Exception as e:
        logger.error(f"DDPG教学演示错误: {e}")
        await ctx.send_log(f"教学演示出错: {str(e)}", "error")
        if ctx.on_stage_change:
            await ctx.on_stage_change("error")
        await ctx.broadcast_message("training_error", {"error": str(e)})
        raise
