"""训练协调器 - 协调整个训练流程"""

import asyncio
from typing import Dict, Any, Optional, List
from .sandbox_ddpg_base import SandboxDDPGBase
from .config_manager import ConfigManager, TrainingConfig
from ..trainers.trainer_factory import TrainerFactory
from ..communication.env_proxy import EnvProxy
from ..sandbox_components.sandbox_setup import SandboxSetup
from common.logger import setup_training_logger, DualLogger


class TrainingCoordinator(SandboxDDPGBase):
    """训练协调器"""
    
    def __init__(self, sandbox_manager):
        super().__init__(sandbox_manager)
        self.config_manager = ConfigManager()
        self.sandbox_setup = SandboxSetup(sandbox_manager)
        self.current_trainer = None
        self.env_proxy = None
        # 并行训练相关
        self.parallel_sessions = []  # 多个沙箱会话
        self.parallel_env_proxies = []  # 多个环境代理
        # SB3 多沙箱训练器
        self.sb3_sandbox_trainer = None
    
    async def run_training(self, mode: str, custom_config: Optional[Dict[str, Any]] = None,
                           on_sandbox_created: callable = None,
                           on_episode_complete: callable = None,
                           on_demo_complete: callable = None) -> Dict[str, Any]:
        """运行训练流程
        
        Args:
            mode: 训练模式 ('parallel', 'custom_batch', 'stable_baselines3')
            custom_config: 自定义配置参数
            on_sandbox_created: 沙箱创建后的回调函数，接收沙箱信息dict作为参数
            on_episode_complete: episode完成后的回调函数，接收进度信息dict作为参数
            on_demo_complete: 演示完成后的回调函数，接收演示结果dict作为参数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        try:
            # 检查是否为并行训练模式
            if mode == "parallel":
                return await self._run_parallel_training_flow(
                    custom_config, on_sandbox_created, on_episode_complete, on_demo_complete
                )
            
            # 检查是否为 SB3 多沙箱训练模式
            if mode == "stable_baselines3":
                return await self._run_sb3_sandbox_training(
                    custom_config, on_sandbox_created, on_episode_complete
                )
            
            # 1. 解析训练模式
            from ..trainers.trainer_factory import TrainingMode
            try:
                training_mode = TrainingMode(mode)
            except ValueError:
                return {
                    "status": "error",
                    "error": f"不支持的训练模式: {mode}",
                    "available_modes": TrainerFactory.get_available_modes() + ["parallel"]
                }
            
            # 2. 获取配置
            config = self.config_manager.get_config(training_mode, custom_config)
            if not self.config_manager.validate_config(config):
                return {"status": "error", "error": "配置验证失败"}
            
            print(f"🎯 开始 {mode} 模式训练")
            print(f"   Episodes: {config.episodes}")
            print(f"   Eval Frequency: {config.eval_freq}")
            
            # 3. 创建训练沙箱
            if not await self.create_training_sandbox():
                return {"status": "error", "error": "创建训练沙箱失败"}
            
            # 3.1 回调通知沙箱已创建
            if on_sandbox_created and callable(on_sandbox_created):
                sandbox_info = self.get_training_sandbox_info()
                try:
                    # 支持同步和异步回调
                    result = on_sandbox_created(sandbox_info)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    print(f"⚠️ 沙箱创建回调执行失败: {e}")
            
            # 4. 初始化日志记录器
            self.logger = setup_training_logger(self.training_session.session_id)
            self.logger.info(f"🚀 开始 {mode} 模式DDPG训练")
            self.logger.info(f"   模式描述: {TrainerFactory.get_mode_description(training_mode)}")
            
            # 5. 安装依赖
            self.logger.info("🔧 安装环境依赖")
            if not await self.sandbox_setup.install_dependencies(self.training_session.session_id):
                self.logger.error("❌ 依赖安装失败")
                return {"status": "error", "error": "依赖安装失败"}
            
            # 6. 设置环境执行器
            self.logger.info("⚙️  设置环境执行器")
            if not await self.sandbox_setup.setup_env_executor(self.training_session.session_id):
                self.logger.error("❌ 环境执行器设置失败")
                return {"status": "error", "error": "环境执行器设置失败"}
            
            # 7. 创建环境代理
            self.env_proxy = EnvProxy(self.sandbox_manager, self.training_session.session_id)
            
            # 8. 创建训练器
            self.logger.info("🧠 创建训练器")
            try:
                self.current_trainer = TrainerFactory.create_trainer(config)
            except Exception as e:
                self.logger.error(f"❌ 训练器创建失败: {e}")
                return {"status": "error", "error": f"训练器创建失败: {e}"}
            
            # 9. 执行训练
            self.logger.info("🏃 开始训练")
            if training_mode == TrainingMode.STABLE_BASELINES3:
                result = await self._run_sb3_training(config)
            else:
                result = await self._run_custom_training(config, on_episode_complete)
            
            # 10. 清理资源
            await self.cleanup_sandboxes(config.force_cleanup)
            
            return result
            
        except Exception as e:
            print(f"❌ 训练流程出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保清理资源
            try:
                await self.cleanup_sandboxes(True)
            except:
                pass
                
            return {"status": "error", "error": str(e)}
    
    async def _run_sb3_sandbox_training(
        self,
        custom_config: Optional[Dict[str, Any]] = None,
        on_sandbox_created: callable = None,
        on_episode_complete: callable = None
    ) -> Dict[str, Any]:
        """运行 SB3 多沙箱并行训练
        
        使用 Stable-Baselines3 DDPG + HER，配合多个沙箱并行执行环境交互。
        
        Args:
            custom_config: 自定义配置参数
            on_sandbox_created: 沙箱创建后的回调函数
            on_episode_complete: episode完成后的回调函数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        from ..trainers.sb3_trainer import SB3SandboxTrainer
        
        # 初始化 logger（使用默认 session_id）
        if self.logger is None:
            self.logger = setup_training_logger("sb3_sandbox")
        
        self.logger.info("🔬 使用 Stable-Baselines3 多沙箱并行训练")
        
        try:
            # 解析配置（兼容 parallel_sandboxes 和 parallel_workers 两种参数名）
            num_sandboxes = custom_config.get('parallel_sandboxes', 
                            custom_config.get('parallel_workers', 4)) if custom_config else 4
            episodes = custom_config.get('episodes', 100000) if custom_config else 100000
            eval_freq = custom_config.get('eval_freq', 5000) if custom_config else 5000
            learning_rate = custom_config.get('learning_rate', 1e-3) if custom_config else 1e-3
            
            self.logger.info(f"   并行沙箱数: {num_sandboxes}")
            self.logger.info(f"   总步数: {episodes}")
            self.logger.info(f"   评估频率: {eval_freq}")
            
            # 创建 SB3 多沙箱训练器（传递沙箱创建回调）
            self.sb3_sandbox_trainer = SB3SandboxTrainer(
                sandbox_manager=self.sandbox_manager,
                num_sandboxes=num_sandboxes,
                total_timesteps=episodes,
                eval_freq=eval_freq,
                learning_rate=learning_rate,
                on_sandbox_created=on_sandbox_created  # 沙箱创建时实时通知
            )
            
            # 定义进度回调
            # 注意：此回调在工作线程（model.learn() 所在线程）中被调用
            # 必须使用 run_coroutine_threadsafe 将协程提交到主事件循环
            main_loop = asyncio.get_running_loop()
            
            def progress_callback(progress_info):
                if on_episode_complete and callable(on_episode_complete):
                    try:
                        result = on_episode_complete({
                            "type": "sb3_progress",
                            "timesteps": progress_info.get('timesteps', 0),
                            "total_timesteps": progress_info.get('total_timesteps', 0),
                            "progress_percent": progress_info.get('progress_percent', 0)
                        })
                        if asyncio.iscoroutine(result):
                            # 从工作线程提交协程到主事件循环（不等待结果）
                            asyncio.run_coroutine_threadsafe(result, main_loop)
                    except Exception as e:
                        self.logger.warning(f"进度回调执行失败: {e}")
            
            # 执行训练（异步方法）
            result = await self.sb3_sandbox_trainer.train(
                log_dir=f"data/sb3_sandbox_ddpg",
                on_progress=progress_callback
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"SB3 多沙箱训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
        
        finally:
            # 清理资源
            if self.sb3_sandbox_trainer:
                try:
                    await self.sb3_sandbox_trainer.cleanup()
                except:
                    pass
                self.sb3_sandbox_trainer = None
    
    async def _run_custom_training(self, config: TrainingConfig, 
                                    on_episode_complete: callable = None) -> Dict[str, Any]:
        """运行自定义训练
        
        Args:
            config: 训练配置
            on_episode_complete: episode完成后的回调函数
        """
        self.logger.info("🔧 使用自定义网络+批量交互模式进行训练")
        
        try:
            # FetchReachDense-v4 环境：observation 维度为 10，action 维度为 4
            state_dim = 10
            action_dim = 4
            
            # 执行训练（传递episode回调）
            result = await self.current_trainer.train(
                self.env_proxy, 
                state_dim, 
                action_dim,
                on_episode_complete=on_episode_complete
            )
            result["mode"] = config.mode.value
            
            return result
            
        except Exception as e:
            self.logger.error(f"自定义训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    async def run_testing(self, model_path: str) -> Dict[str, Any]:
        """运行模型测试
        
        Args:
            model_path: 模型路径
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        try:
            # 创建测试沙箱
            if not await self.create_testing_sandbox():
                return {"status": "error", "error": "创建测试沙箱失败"}
            
            # 安装依赖
            if not await self.sandbox_setup.install_dependencies(self.testing_session.session_id):
                return {"status": "error", "error": "测试沙箱依赖安装失败"}
            
            # 这里实现测试逻辑
            result = {
                "status": "completed",
                "model_path": model_path,
                "message": "模型测试完成（占位实现）"
            }
            
            # 清理测试沙箱
            await self.cleanup_sandboxes(force_cleanup=True)
            
            return result
            
        except Exception as e:
            print(f"❌ 测试流程出错: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _run_parallel_training_flow(self, custom_config: Optional[Dict[str, Any]] = None,
                                          on_sandbox_created: callable = None,
                                          on_episode_complete: callable = None,
                                          on_demo_complete: callable = None) -> Dict[str, Any]:
        """并行训练流程
        
        创建多个沙箱，并行收集数据，统一更新模型
        
        Args:
            custom_config: 自定义配置参数
            on_sandbox_created: 沙箱创建后的回调函数
            on_episode_complete: episode完成后的回调函数
            on_demo_complete: 演示完成后的回调函数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        from ..trainers.parallel_trainer import ParallelTrainer
        
        # 获取并行参数
        num_workers = (custom_config or {}).get('parallel_workers', 5)
        episodes = (custom_config or {}).get('episodes', 500)
        eval_freq = (custom_config or {}).get('eval_freq', 50)
        
        print(f"🚀 启动并行DDPG训练")
        print(f"   并行沙箱数: {num_workers}")
        print(f"   总Episodes: {episodes}")
        print("=" * 60)
        
        # 测试沙箱相关
        test_sandbox_session = None
        test_env_proxy = None
        
        try:
            # 1. 创建多个沙箱
            self.parallel_sessions = []
            self.parallel_env_proxies = []
            
            print(f"📦 创建 {num_workers} 个并行训练沙箱...")
            
            for i in range(num_workers):
                print(f"   创建训练沙箱 {i+1}/{num_workers}...")
                session = await self.sandbox_manager.create_sandbox()
                if not session:
                    print(f"   ❌ 沙箱 {i+1} 创建失败")
                    continue
                
                self.parallel_sessions.append(session)
                
                # 获取流化URL
                stream_url = await self.sandbox_manager.get_sandbox_url(session.sandbox_id)
                if stream_url:
                    session.stream_url = stream_url
                
                print(f"   ✅ 训练沙箱 {i+1} 创建成功: {session.sandbox_id[:8]}...")
            
            if not self.parallel_sessions:
                return {"status": "error", "error": "所有沙箱创建失败"}
            
            print(f"✅ 成功创建 {len(self.parallel_sessions)} 个训练沙箱")
            
            # 1.5 创建测试沙箱（用于阶段性演示）
            print(f"\n🧪 创建测试沙箱（用于模型效果演示）...")
            test_sandbox_session = await self.sandbox_manager.create_sandbox()
            if test_sandbox_session:
                # 获取流化URL
                test_stream_url = await self.sandbox_manager.get_sandbox_url(test_sandbox_session.sandbox_id)
                if test_stream_url:
                    test_sandbox_session.stream_url = test_stream_url
                print(f"   ✅ 测试沙箱创建成功: {test_sandbox_session.sandbox_id[:8]}...")
            else:
                print(f"   ⚠️ 测试沙箱创建失败，将跳过演示功能")
            
            # 2. 回调通知沙箱已创建
            if on_sandbox_created and callable(on_sandbox_created):
                # 先通知训练沙箱
                for i, session in enumerate(self.parallel_sessions):
                    sandbox_info = {
                        "sandbox_id": session.sandbox_id,
                        "session_id": session.session_id,
                        "resource_url": getattr(session, 'resource_url', None),
                        "stream_url": getattr(session, 'stream_url', None),
                        "sandbox_type": "training",
                        "worker_id": i,
                        "total_workers": len(self.parallel_sessions)
                    }
                    try:
                        result = on_sandbox_created(sandbox_info)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        print(f"⚠️ 训练沙箱 {i} 创建回调执行失败: {e}")
                
                # 通知测试沙箱
                if test_sandbox_session:
                    test_sandbox_info = {
                        "sandbox_id": test_sandbox_session.sandbox_id,
                        "session_id": test_sandbox_session.session_id,
                        "resource_url": getattr(test_sandbox_session, 'resource_url', None),
                        "stream_url": getattr(test_sandbox_session, 'stream_url', None),
                        "sandbox_type": "testing",
                        "worker_id": -1,  # 特殊标识
                        "total_workers": 1
                    }
                    try:
                        result = on_sandbox_created(test_sandbox_info)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        print(f"⚠️ 测试沙箱创建回调执行失败: {e}")
            
            # 3. 并行安装依赖和设置环境执行器
            print(f"🔧 并行安装依赖和设置环境...")
            
            setup_tasks = []
            for session in self.parallel_sessions:
                setup_tasks.append(self._setup_single_sandbox(session.session_id))
            
            # 同时设置测试沙箱（如果存在）
            if test_sandbox_session:
                setup_tasks.append(self._setup_single_sandbox(test_sandbox_session.session_id))
            
            setup_results = await asyncio.gather(*setup_tasks, return_exceptions=True)
            
            # 过滤成功的训练沙箱
            num_training_sandboxes = len(self.parallel_sessions)
            successful_sessions = []
            for i, (session, result) in enumerate(zip(self.parallel_sessions, setup_results[:num_training_sandboxes])):
                if result is True:
                    successful_sessions.append(session)
                else:
                    print(f"   ⚠️ 训练沙箱 {session.sandbox_id[:8]}... 设置失败")
            
            if not successful_sessions:
                return {"status": "error", "error": "所有沙箱设置失败"}
            
            print(f"✅ {len(successful_sessions)} 个训练沙箱设置成功")
            
            # 检查测试沙箱设置结果
            if test_sandbox_session:
                test_setup_result = setup_results[num_training_sandboxes] if len(setup_results) > num_training_sandboxes else False
                if test_setup_result is True:
                    print(f"✅ 测试沙箱设置成功")
                    test_env_proxy = EnvProxy(self.sandbox_manager, test_sandbox_session.session_id)
                else:
                    print(f"⚠️ 测试沙箱设置失败，将跳过演示功能")
                    test_sandbox_session = None
                    test_env_proxy = None
            
            # 4. 创建环境代理
            self.parallel_env_proxies = []
            for session in successful_sessions:
                env_proxy = EnvProxy(self.sandbox_manager, session.session_id)
                self.parallel_env_proxies.append(env_proxy)
            
            # 5. 创建并行训练器（对齐 local_parallel_trainer 的有效超参数）
            parallel_trainer = ParallelTrainer(
                num_workers=len(self.parallel_env_proxies),
                episodes=episodes,
                eval_freq=eval_freq,
                updates_per_episode=100,   # 每轮 100 次梯度更新
                n_sampled_goal=8,          # HER 重标记目标数
                warmup_rounds=10,          # 预热轮数（降低以便快速验证演示功能）
                noise_start=0.3,
                noise_end=0.02,
                buffer_size=1_000_000,
                log_interval=10,           # 每 10 轮输出一次日志和演示（便于调试）
            )
            
            # 5.5 设置测试沙箱（用于阶段性演示）
            if test_env_proxy:
                # 定义演示完成回调（广播 WebSocket 消息）
                async def demo_complete_callback(demo_result):
                    """演示完成后的回调"""
                    if on_demo_complete and callable(on_demo_complete):
                        try:
                            result = on_demo_complete(demo_result)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            print(f"⚠️ 演示完成回调执行失败: {e}")
                
                parallel_trainer.set_test_sandbox(test_env_proxy, demo_complete_callback)
                print(f"✅ 测试沙箱已关联到训练器，将每隔 {parallel_trainer.log_interval} 轮进行一次演示")
            
            # 6. 执行并行训练
            # FetchReachDense-v4：state = concat([observation(10d), desired_goal(3d)]) = 13d
            state_dim  = 13   # obs(10) + goal(3)
            action_dim = 4
            
            result = await parallel_trainer.train(
                self.parallel_env_proxies,
                state_dim,
                action_dim,
                on_episode_complete=on_episode_complete
            )
            
            result["mode"] = "parallel"
            result["num_workers"] = len(self.parallel_env_proxies)
            
            # 7. 清理所有沙箱（包括测试沙箱）
            await self._cleanup_parallel_sandboxes(test_sandbox_session)
            
            return result
            
        except Exception as e:
            print(f"❌ 并行训练流程出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保清理资源（包括测试沙箱）
            await self._cleanup_parallel_sandboxes(test_sandbox_session)
            
            return {"status": "error", "error": str(e)}
    
    async def _setup_single_sandbox(self, session_id: str) -> bool:
        """设置单个沙箱（安装依赖 + 设置环境执行器）"""
        try:
            # 安装依赖
            if not await self.sandbox_setup.install_dependencies(session_id):
                return False
            
            # 设置环境执行器
            if not await self.sandbox_setup.setup_env_executor(session_id):
                return False
            
            return True
        except Exception as e:
            print(f"设置沙箱失败: {e}")
            return False
    
    async def _cleanup_parallel_sandboxes(self, test_sandbox_session=None):
        """清理所有并行沙箱（包括测试沙箱）"""
        print("🧹 清理沙箱...")
        
        # 清理训练沙箱
        for session in self.parallel_sessions:
            try:
                await self.sandbox_manager.release_sandbox(session.sandbox_id)
                print(f"   ✅ 释放训练沙箱: {session.sandbox_id[:8]}...")
            except Exception as e:
                print(f"   ⚠️ 释放训练沙箱失败: {e}")
        
        # 清理测试沙箱
        if test_sandbox_session:
            try:
                await self.sandbox_manager.release_sandbox(test_sandbox_session.sandbox_id)
                print(f"   ✅ 释放测试沙箱: {test_sandbox_session.sandbox_id[:8]}...")
            except Exception as e:
                print(f"   ⚠️ 释放测试沙箱失败: {e}")
        
        self.parallel_sessions = []
        self.parallel_env_proxies = []