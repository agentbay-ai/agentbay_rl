#!/usr/bin/env python3
"""
强化学习教学平台 - Web 后端服务

提供基于 AgentBay 沙箱的强化学习算法教学和可视化界面。
支持从基础到高级的 RL 算法演示和并行训练。

Features:
- 🎓 循序渐进的 RL 算法教学
- 🚀 沙箱并行执行加速训练
- 📊 实时训练进度可视化
- 🖥️ 沙箱环境流化界面展示
- 🧠 算法原理详解和执行过程展示
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "agentbay_samples"))

# 环境变量
from dotenv import load_dotenv
load_dotenv()

# Web 框架
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 沙箱管理导入
from common.simple_sandbox_manager import SimpleSandboxManager
from common.teaching_materials import teaching_manager

# 算法导入
from algorithms.bandit.sandbox_bandit import SandboxBanditRunner
from algorithms.ddpg.sandbox_ddpg import SandboxDDPG

# 教学模块导入
from algorithms.bandit.teaching import (
    run_bandit_teaching as _run_bandit_teaching,
    TeachingContext as BanditTeachingContext
)
from algorithms.ddpg.teaching import (
    run_ddpg_teaching as _run_ddpg_teaching,
    TeachingContext as DDPGTeachingContext
)

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 关闭高频日志输出
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("common.simple_sandbox_manager").setLevel(logging.WARNING)

# ============================================================================
# 全局状态管理
# ============================================================================

class RLAppState:
    """强化学习应用状态"""
    def __init__(self):
        self.is_training = False
        self.current_algorithm = ""
        self.current_stage = "idle"  # idle, initializing, training, completed
        self.episode_count = 0
        self.total_episodes = 0
        self.current_reward = 0.0
        self.avg_reward = 0.0
        self.win_rate = 0.0
        self.training_progress = 0.0  # 0-100%
        
        # 沙箱相关信息
        self.sandbox_sessions: List[Dict] = []
        self.active_sandboxes = 0
        self.sandbox_initialized = False
        self.sandbox_manager: Optional[SimpleSandboxManager] = None
        
        # 训练数据
        self.rewards_history: List[float] = []
        self.episode_lengths: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        
        # WebSocket 连接
        self.websockets: List[WebSocket] = []
        
        # 算法配置
        self.algorithm_configs: Dict[str, Dict] = {
            "bandit": {
                "name": "多臂老虎机",
                "description": "强化学习入门：探索与利用的平衡",
                "difficulty": "入门",
                "estimated_time": "5分钟",
                "module": "bandit"
            },
            "dqn": {
                "name": "深度Q网络 (DQN)",
                "description": "值函数方法：学习最优动作价值函数",
                "difficulty": "初级",
                "estimated_time": "15分钟",
                "module": "dqn"
            },
            "ppo": {
                "name": "近端策略优化 (PPO)",
                "description": "策略梯度方法：直接优化策略函数",
                "difficulty": "中级",
                "estimated_time": "20分钟",
                "module": "ppo"
            },
            "sac": {
                "name": "软 Actor-Critic (SAC)",
                "description": "最大熵强化学习：平衡收益与探索",
                "difficulty": "高级",
                "estimated_time": "25分钟",
                "module": "sac"
            },
            "ddpg": {
                "name": "深度确定性策略梯度 (DDPG)",
                "description": "连续控制方法：确定性策略与价值函数结合",
                "difficulty": "高级",
                "estimated_time": "30分钟",
                "module": "ddpg"
            }
        }
        
        # 算法实例
        self.algorithm_instances: Dict[str, Any] = {}
        
        # 当前教学演示
        self.current_demo = None

app_state = RLAppState()

# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    "max_parallel_sandboxes": 4,
    "default_episodes": 100,
    "log_interval": 10,
    "data_dir": str(Path(__file__).parent / "data"),
    "output_dir": str(Path(__file__).parent / "outputs"),
    "preserve_sandbox_after_training": False  # 训练完成后是否保留沙箱（供手动查看）
}

# 确保目录存在
Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

# ============================================================================
# WebSocket 管理
# ============================================================================

async def broadcast_message(msg_type: str, data: dict):
    """广播消息给所有连接的 WebSocket 客户端"""
    message = json.dumps({
        "type": msg_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    })
    
    # 调试日志：显示当前连接数
    ws_count = len(app_state.websockets)
    if msg_type == "progress_update":
        print(f"📡 广播 {msg_type} 消息到 {ws_count} 个客户端")
    
    disconnected = []
    for ws in app_state.websockets:
        try:
            await ws.send_text(message)
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            disconnected.append(ws)
    
    # 清理断开的连接
    for ws in disconnected:
        if ws in app_state.websockets:
            app_state.websockets.remove(ws)

async def send_log(message: str, level: str = "info"):
    """发送日志消息到前端"""
    logger.info(f"[{level.upper()}] {message}")
    await broadcast_message("log", {"message": message, "level": level})

# ============================================================================
# 数据模型
# ============================================================================

class AlgorithmRequest(BaseModel):
    algorithm: str
    episodes: int = 100
    parallel_sandboxes: int = 4
    config: Optional[Dict[str, Any]] = None

class TrainingStatus(BaseModel):
    is_training: bool
    current_algorithm: str
    current_stage: str
    episode_count: int
    total_episodes: int
    current_reward: float
    avg_reward: float
    win_rate: float
    training_progress: float
    active_sandboxes: int

# ============================================================================
# 应用生命周期管理
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 强化学习教学平台启动中...")
    
    # 初始化简化版沙箱管理器
    try:
        api_key = os.getenv("AGENTBAY_API_KEY")
        logger.info(f"API Key 存在: {bool(api_key)}")
        if api_key:
            app_state.sandbox_manager = SimpleSandboxManager(api_key)
            logger.info(f"创建沙箱管理器实例: {app_state.sandbox_manager is not None}")
            success = await app_state.sandbox_manager.initialize()
            logger.info(f"初始化返回: {success}")
            if success:
                app_state.sandbox_initialized = True
                await send_log("沙箱管理器初始化成功", "success")
            else:
                await send_log("沙箱管理器初始化失败", "error")
        else:
            await send_log("警告: 未设置 AGENTBAY_API_KEY，沙箱功能将不可用", "warning")
    except Exception as e:
        logger.error(f"沙箱初始化失败: {e}")
        await send_log(f"沙箱初始化失败: {str(e)}", "error")
    
    # 启动时初始化
    await send_log("系统初始化完成", "success")
    
    yield
    
    # 关闭时清理
    logger.info("🛑 强化学习教学平台关闭中...")
    if app_state.sandbox_manager:
        try:
            await app_state.sandbox_manager.close()
            await send_log("沙箱管理器已关闭", "info")
        except Exception as e:
            logger.error(f"关闭沙箱管理器失败: {e}")
    await send_log("系统正在关闭...", "warning")

# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="RL Teaching Platform",
    description="基于 AgentBay 沙箱的强化学习教学平台",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# ============================================================================
# API 路由
# ============================================================================

@app.get("/")
async def index():
    """主页"""
    return FileResponse(str(Path(__file__).parent / "templates" / "index.html"))

@app.get("/api/tutorial/{tutorial_name}")
async def get_tutorial_content(tutorial_name: str):
    """获取教程Markdown内容"""
    tutorial_path = os.path.join(CONFIG["data_dir"], f"{tutorial_name}.md")
    
    if not os.path.exists(tutorial_path):
        return {"error": "教程文件不存在"}
    
    try:
        with open(tutorial_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        return {"error": f"读取教程文件失败: {str(e)}"}

@app.get("/api/algorithms")
async def get_algorithms():
    """获取可用算法列表"""
    return {
        "algorithms": app_state.algorithm_configs,
        "current_algorithm": app_state.current_algorithm
    }

@app.get("/api/status")
async def get_status():
    """获取当前训练状态"""
    return TrainingStatus(
        is_training=app_state.is_training,
        current_algorithm=app_state.current_algorithm,
        current_stage=app_state.current_stage,
        episode_count=app_state.episode_count,
        total_episodes=app_state.total_episodes,
        current_reward=app_state.current_reward,
        avg_reward=app_state.avg_reward,
        win_rate=app_state.win_rate,
        training_progress=app_state.training_progress,
        active_sandboxes=app_state.active_sandboxes
    )

@app.post("/api/train")
async def start_training(request: AlgorithmRequest):
    """开始训练指定算法"""
    if app_state.is_training:
        raise HTTPException(status_code=400, detail="训练已在进行中")
    
    if request.algorithm not in app_state.algorithm_configs:
        raise HTTPException(status_code=400, detail="不支持的算法")
    
    # 更新状态
    app_state.is_training = True
    app_state.current_algorithm = request.algorithm
    app_state.current_stage = "initializing"
    app_state.total_episodes = request.episodes
    app_state.episode_count = 0
    app_state.active_sandboxes = min(request.parallel_sandboxes, CONFIG["max_parallel_sandboxes"])
    
    # 重置统计数据
    app_state.rewards_history.clear()
    app_state.episode_lengths.clear()
    app_state.policy_losses.clear()
    app_state.value_losses.clear()
    
    await send_log(f"开始训练算法: {request.algorithm}", "info")
    await broadcast_message("training_started", {
        "algorithm": request.algorithm,
        "total_episodes": request.episodes,
        "parallel_sandboxes": app_state.active_sandboxes
    })
    
    # 根据算法类型启动相应的训练任务
    if request.algorithm == "bandit":
        asyncio.create_task(run_bandit_teaching(request))
    elif request.algorithm == "ddpg":
        asyncio.create_task(run_ddpg_teaching(request))
    else:
        # 不支持的算法
        app_state.is_training = False
        raise HTTPException(status_code=400, detail=f"算法 '{request.algorithm}' 暂未实现")
    
    return {"status": "started", "algorithm": request.algorithm}

@app.get("/api/config/sandbox-preserve")
async def get_sandbox_preserve_setting():
    """获取沙箱保留设置"""
    return {
        "preserve_sandbox_after_training": CONFIG.get("preserve_sandbox_after_training", False),
        "description": "训练完成后是否保留沙箱供手动查看"
    }

@app.post("/api/config/sandbox-preserve/{setting}")
async def set_sandbox_preserve_setting(setting: str):
    """设置沙箱保留选项
    
    Args:
        setting: 'enable' 或 'disable'
    """
    if setting.lower() == "enable":
        CONFIG["preserve_sandbox_after_training"] = True
        message = "已启用沙箱保留模式：训练完成后沙箱将保留供手动查看"
    elif setting.lower() == "disable":
        CONFIG["preserve_sandbox_after_training"] = False
        message = "已禁用沙箱保留模式：训练完成后沙箱将自动清理"
    else:
        raise HTTPException(status_code=400, detail="参数必须是 'enable' 或 'disable'")
    
    await send_log(message, "info")
    return {"status": "success", "message": message, "setting": CONFIG["preserve_sandbox_after_training"]}

@app.post("/api/stop")
async def stop_training():
    """停止当前训练并清理资源"""
    if not app_state.is_training:
        raise HTTPException(status_code=400, detail="没有正在进行的训练")
    
    app_state.is_training = False
    app_state.current_stage = "stopped"
    
    # 清理沙箱资源（注意：不要在这里调用 close()，避免跨任务上下文问题）
    # MCP 客户端的 cancel scope 不支持跨任务关闭
    if app_state.sandbox_manager:
        try:
            # 只清理沙箱会话，不关闭 MCP 连接
            # MCP 连接会在下次训练时重用或在应用关闭时清理
            await app_state.sandbox_manager.cleanup_all()
            app_state.sandbox_sessions.clear()
            app_state.active_sandboxes = 0
            await send_log("沙箱资源已清理", "info")
        except Exception as e:
            logger.error(f"停止训练时清理沙箱失败：{e}")
            await send_log(f"清理沙箱时出现警告：{e}", "warning")
            # 即使清理失败也继续，重置状态
            app_state.sandbox_sessions.clear()
            app_state.active_sandboxes = 0
    
    await send_log("训练已停止", "warning")
    await broadcast_message("training_stopped", {})
    
    return {"status": "stopped"}

@app.get("/api/training-data")
async def get_training_data():
    """获取训练历史数据"""
    return {
        "rewards": app_state.rewards_history,
        "episode_lengths": app_state.episode_lengths,
        "policy_losses": app_state.policy_losses,
        "value_losses": app_state.value_losses,
        "timestamps": [datetime.now().isoformat()] * len(app_state.rewards_history)
    }

@app.post("/api/create-sandbox")
async def create_sandbox():
    """创建新的沙箱会话"""
    logger.info(f"检查沙箱初始化状态: sandbox_initialized={app_state.sandbox_initialized}, sandbox_manager={app_state.sandbox_manager is not None}")
    if not app_state.sandbox_initialized or not app_state.sandbox_manager:
        logger.error("沙箱管理器未初始化")
        raise HTTPException(status_code=400, detail="沙箱管理器未初始化")
    
    try:
        # 创建沙箱会话
        session = await app_state.sandbox_manager.create_sandbox()
        if not session:
            raise HTTPException(status_code=500, detail="创建沙箱会话失败")
        
        # 获取会话信息
        sandbox_info = await app_state.sandbox_manager.get_sandbox_info(session.session_id)
        if not sandbox_info:
            raise HTTPException(status_code=500, detail="获取会话信息失败")
        
        # 获取真实的流化页面URL
        stream_url = await app_state.sandbox_manager.get_sandbox_url(session.sandbox_id)
        if stream_url:
            sandbox_info["stream_url"] = stream_url
            await send_log(f"🚨 沙箱流化页面URL: {stream_url}", "success")
            print(f"\n{'='*80}")
            print(f"🚨 沙箱流化页面URL: {stream_url}")
            print(f"{'='*80}\n")
        else:
            await send_log("⚠️  获取流化页面URL失败，使用默认URL", "warning")
            print(f"\n{'='*80}")
            print(f"⚠️  获取流化页面URL失败，使用默认URL: {sandbox_info['resource_url']}")
            print(f"{'='*80}\n")
        
        # 存储会话信息
        app_state.sandbox_sessions.append(sandbox_info)
        app_state.active_sandboxes = len(app_state.sandbox_sessions)
        
        await send_log(f"沙箱会话创建成功: {session.session_id}", "success")
        await broadcast_message("sandbox_created", sandbox_info)
        
        return sandbox_info
        
    except Exception as e:
        logger.error(f"创建沙箱失败: {e}")
        await send_log(f"创建沙箱失败: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sandboxes")
async def get_sandboxes():
    """获取所有沙箱会话信息"""
    return {
        "sandboxes": app_state.sandbox_sessions,
        "active_count": app_state.active_sandboxes
    }

@app.post("/api/clear-sandboxes")
async def clear_sandboxes():
    """清理所有沙箱会话"""
    if app_state.sandbox_manager:
        try:
            await app_state.sandbox_manager.cleanup_all()
            await app_state.sandbox_manager.close()  # 关闭 MCP 连接
            app_state.sandbox_sessions.clear()
            app_state.active_sandboxes = 0
            await send_log("所有沙箱会话已清理", "info")
            await broadcast_message("sandboxes_cleared", {})
            return {"status": "success", "message": "沙箱会话已清理"}
        except Exception as e:
            logger.error(f"清理沙箱失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="沙箱管理器未初始化")

# ============================================================================
# 教学内容 API
# ============================================================================

@app.get("/api/courses")
async def get_courses():
    """获取所有课程列表"""
    courses = teaching_manager.list_courses()
    return {
        "courses": [{
            "algorithm_key": course.algorithm_key,
            "algorithm_name": course.algorithm_name,
            "difficulty": course.difficulty.value,
            "estimated_time": course.estimated_time,
            "icon": course.icon,
            "color": course.color,
            "learning_objectives": course.learning_objectives,
            "prerequisites": course.prerequisites
        } for course in courses]
    }

@app.get("/api/courses/{algorithm_key}")
async def get_course_detail(algorithm_key: str):
    """获取特定课程的详细信息"""
    course = teaching_manager.get_course(algorithm_key)
    if not course:
        raise HTTPException(status_code=404, detail="课程不存在")
    
    return {
        "algorithm_key": course.algorithm_key,
        "algorithm_name": course.algorithm_name,
        "difficulty": course.difficulty.value,
        "estimated_time": course.estimated_time,
        "icon": course.icon,
        "color": course.color,
        "learning_objectives": course.learning_objectives,
        "prerequisites": course.prerequisites,
        "assessment_criteria": course.assessment_criteria,
        "theory_concepts": [{
            "title": concept.title,
            "description": concept.description,
            "explanation": concept.explanation,
            "examples": concept.examples,
            "key_points": concept.key_points
        } for concept in course.theory_concepts],
        "practical_examples": [{
            "name": demo.name,
            "description": demo.description,
            "parameters": demo.parameters,
            "visualization_type": demo.visualization_type,
            "explanation": demo.explanation
        } for demo in course.practical_examples]
    }

@app.get("/api/learning-path")
async def get_learning_path():
    """获取推荐学习路径"""
    path = teaching_manager.get_learning_path()
    return {
        "learning_path": [{
            "algorithm_key": course.algorithm_key,
            "algorithm_name": course.algorithm_name,
            "difficulty": course.difficulty.value,
            "estimated_time": course.estimated_time,
            "order": idx + 1
        } for idx, course in enumerate(path)]
    }

# ============================================================================
# WebSocket 端点
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 连接端点"""
    await websocket.accept()
    app_state.websockets.append(websocket)
    
    try:
        # 发送初始状态
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": {
                "algorithms": app_state.algorithm_configs,
                "status": TrainingStatus(
                    is_training=app_state.is_training,
                    current_algorithm=app_state.current_algorithm,
                    current_stage=app_state.current_stage,
                    episode_count=app_state.episode_count,
                    total_episodes=app_state.total_episodes,
                    current_reward=app_state.current_reward,
                    avg_reward=app_state.avg_reward,
                    win_rate=app_state.win_rate,
                    training_progress=app_state.training_progress,
                    active_sandboxes=app_state.active_sandboxes
                ).dict()
            }
        }))
        
        # 保持连接
        while True:
            data = await websocket.receive_text()
            # 可以在这里处理来自前端的消息
            logger.info(f"Received WebSocket message: {data}")
            
    except WebSocketDisconnect:
        app_state.websockets.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in app_state.websockets:
            app_state.websockets.remove(websocket)

# ============================================================================
# 教学内容 API
# ============================================================================
async def run_bandit_teaching(request: AlgorithmRequest):
    """运行多臂老虎机教学演示（委托给算法模块）"""
    try:
        # 创建状态更新回调
        async def on_stage_change(stage: str):
            app_state.current_stage = stage
        
        # 创建教学上下文
        ctx = BanditTeachingContext(
            sandbox_manager=app_state.sandbox_manager,
            send_log=send_log,
            broadcast_message=broadcast_message,
            on_stage_change=on_stage_change,
            config=CONFIG
        )
        
        # 调用模块中的教学函数
        result_data = await _run_bandit_teaching(
            request_episodes=request.episodes,
            request_config=request.config,
            ctx=ctx
        )
        
        # 训练完成
        if app_state.is_training:  # 只有正常完成才标记为 completed
            app_state.current_stage = "completed"
            await broadcast_message("training_completed", {"final_stats": result_data})
        
    except Exception as e:
        logger.error(f"老虎机教学演示错误: {e}")
        app_state.current_stage = "error"
    finally:
        app_state.is_training = False
        app_state.current_demo = None

async def run_ddpg_teaching(request: AlgorithmRequest):
    """运行DDPG教学演示（委托给算法模块）"""
    try:
        # 创建状态更新回调
        async def on_stage_change(stage: str):
            app_state.current_stage = stage
        
        # 创建教学上下文
        ctx = DDPGTeachingContext(
            sandbox_manager=app_state.sandbox_manager,
            send_log=send_log,
            broadcast_message=broadcast_message,
            on_stage_change=on_stage_change,
            config=CONFIG
        )
        
        # 调用模块中的教学函数
        result_data = await _run_ddpg_teaching(
            request_episodes=request.episodes,
            request_parallel_sandboxes=request.parallel_sandboxes,
            request_config=request.config,
            ctx=ctx
        )
        
        # 训练完成
        if app_state.is_training:  # 只有正常完成才标记为 completed
            app_state.current_stage = "completed"
            await broadcast_message("training_completed", {"final_stats": result_data})
        
    except Exception as e:
        logger.error(f"DDPG教学演示错误: {e}")
        app_state.current_stage = "error"
    finally:
        app_state.is_training = False
        app_state.current_demo = None


# ============================================================================
# 主函数
# ============================================================================

def main():
    """启动服务器"""
    import webbrowser
    import threading
    import time
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    # 构建访问地址（如果是 0.0.0.0，浏览器使用 localhost）
    display_host = "localhost" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{port}"
    
    logger.info(f"🎯 启动强化学习教学平台")
    logger.info(f"📍 地址: {url}")
    logger.info(f"📁 数据目录: {CONFIG['data_dir']}")
    logger.info(f"📤 输出目录: {CONFIG['output_dir']}")
    
    # 延迟打开浏览器，等待服务器启动
    def open_browser():
        time.sleep(2)  # 等待服务器启动
        logger.info(f"🌐 正在打开浏览器...")
        webbrowser.open(url)
    
    # 在新线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()