"""在沙箱中运行的DDPG机械臂控制实现（带可视化）

这是模块化架构的主入口文件，整合了所有子模块。
"""

import json
import asyncio
from typing import Dict, Any, Optional
from common.simple_sandbox_manager import SimpleSandboxManager

# 导入模块化组件
from .core.training_coordinator import TrainingCoordinator
from .core.config_manager import TrainingMode, ConfigManager
from .trainers.trainer_factory import TrainerFactory
from .utils.validation_utils import ValidationUtils

# 导入日志模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from common.logger import setup_training_logger, DualLogger

class SandboxDDPG:
    """沙箱DDPG运行器 - 模块化架构主接口"""
    
    def __init__(self, sandbox_manager: SimpleSandboxManager):
        self.sandbox_manager = sandbox_manager
        self.training_coordinator = TrainingCoordinator(sandbox_manager)
        self.config_manager = ConfigManager()
        self.validator = ValidationUtils()
        
    async def run_training(self, mode: str = "custom_batch", 
                          custom_config: Optional[Dict[str, Any]] = None,
                          on_sandbox_created: callable = None,
                          on_episode_complete: callable = None,
                          on_demo_complete: callable = None) -> Dict[str, Any]:
        """运行训练流程（主入口方法）
        
        Args:
            mode: 训练模式 ('parallel', 'custom_batch', 'stable_baselines3')
            custom_config: 自定义配置参数
            on_sandbox_created: 沙箱创建后的回调函数，接收沙箱信息dict作为参数
            on_episode_complete: episode完成后的回调函数，接收进度信息dict作为参数
            on_demo_complete: 演示完成后的回调函数，接收演示结果dict作为参数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        print(f"🎯 启动DDPG训练 - 模式: {mode}")
        print("=" * 60)
        
        # 验证训练模式（parallel 模式由 training_coordinator 单独处理）
        available_modes = TrainerFactory.get_available_modes() + ["parallel"]
        if mode not in available_modes:
            return {
                "status": "error",
                "error": f"不支持的训练模式: {mode}",
                "available_modes": available_modes,
                "message": "请使用支持的训练模式之一"
            }
        
        # 执行训练（传递沙箱创建回调、episode回调和演示回调）
        result = await self.training_coordinator.run_training(
            mode, custom_config, on_sandbox_created, on_episode_complete, on_demo_complete
        )
        
        # 格式化结果
        formatted_result = self.validator.format_training_result(result)
        
        return formatted_result
    
    async def run_testing(self, model_path: str) -> Dict[str, Any]:
        """运行模型测试
        
        Args:
            model_path: 模型路径
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        print(f"🧪 启动模型测试 - 路径: {model_path}")
        print("=" * 60)
        
        result = await self.training_coordinator.run_testing(model_path)
        return self.validator.format_training_result(result)
    
    def get_available_modes(self) -> list:
        """获取可用的训练模式
        
        Returns:
            list: 可用训练模式列表
        """
        return TrainerFactory.get_available_modes() + ["parallel"]
    
    def get_mode_description(self, mode: str) -> str:
        """获取训练模式描述
        
        Args:
            mode: 训练模式
            
        Returns:
            str: 模式描述
        """
        if mode == "parallel":
            return "多沙箱并行数据收集，统一更新模型，大幅提升训练速度"
        try:
            training_mode = TrainingMode(mode)
            return TrainerFactory.get_mode_description(training_mode)
        except ValueError:
            return "未知模式"
    
    async def cleanup_sandboxes(self, force_cleanup: bool = True):
        """清理沙箱资源
        
        Args:
            force_cleanup: 是否强制清理
        """
        await self.training_coordinator.cleanup_sandboxes(force_cleanup)
    
    async def create_training_sandbox(self) -> bool:
        """创建训练沙箱会话"""
        try:
            # 创建训练沙箱
            self.training_session = await self.sandbox_manager.create_sandbox()
            if not self.training_session:
                return False
            
            training_sandbox_id = self.training_session.sandbox_id
            
            # 获取流化URL并打印
            stream_url = await self.sandbox_manager.get_sandbox_url(training_sandbox_id)
            if stream_url:
                print(f"\n{'='*80}")
                print(f"🚨 DDPG训练沙箱流化页面URL: {stream_url}")
                print(f"{'='*80}\n")
            
            return True
        except Exception as e:
            print(f"创建训练沙箱会话失败: {e}")
            return False
    
    async def create_testing_sandbox(self) -> bool:
        """创建测试沙箱会话"""
        try:
            # 创建测试沙箱
            self.testing_session = await self.sandbox_manager.create_sandbox()
            if not self.testing_session:
                return False
            
            testing_sandbox_id = self.testing_session.sandbox_id
            
            # 获取流化URL并打印
            stream_url = await self.sandbox_manager.get_sandbox_url(testing_sandbox_id)
            if stream_url:
                print(f"\n{'='*80}")
                print(f"🚨 DDPG测试沙箱流化页面URL: {stream_url}")
                print(f"{'='*80}\n")
            
            return True
        except Exception as e:
            print(f"创建测试沙箱会话失败: {e}")
            return False
    
    async def install_dependencies(self, session_id: str) -> bool:
        """在指定沙箱中安装依赖包"""
        try:
            print("🔧 正在安装沙箱依赖包...")
            
            # 只安装gymnasium-robotics（沙箱中只需要环境）
            print("  -> 安装gymnasium-robotics...")
            install_result = await self.sandbox_manager.execute_command(
                session_id,
                "pip3 install -i https://mirrors.aliyun.com/pypi/simple/ gymnasium-robotics",
                timeout_ms=600000  # 10分钟超时
            )
            
            if install_result and install_result.get('exit_code') == 0:
                print("✅ gymnasium-robotics安装成功")
                return True
            else:
                print("❌ 依赖包安装失败")
                if install_result:
                    stderr = install_result.get('stderr', '')
                    if stderr:
                        print(f"安装错误信息: {stderr}")
                return False
                
        except Exception as e:
            print(f"安装依赖时出错: {e}")
            return False
    
    async def setup_env_executor(self, session_id: str) -> bool:
        """在训练沙箱中设置环境执行器"""
        try:
            # 直接读取环境执行器脚本文件的完整内容
            import os
            import base64
            executor_path = os.path.join(os.path.dirname(__file__), "sandbox_env_executor.py")
            
            with open(executor_path, "r", encoding="utf-8") as f:
                full_script = f.read()
            
            print(f"✅ 成功读取环境执行器脚本，共 {len(full_script)} 字符")
            
            # 将脚本写入沙箱
            encoded_content = base64.b64encode(full_script.encode('utf-8')).decode('utf-8')
            write_result = await self.sandbox_manager.execute_command(
                session_id,
                f"echo '{encoded_content}' | base64 -d > /tmp/env_executor.py"
            )
            
            if not write_result:
                print("❌ 写入环境执行器失败: 无返回结果")
                return False
            
            exit_code = write_result.get('exit_code')
            if exit_code is not None and exit_code != 0:
                print(f"❌ 写入环境执行器失败: exit_code = {exit_code}")
                stderr = write_result.get('stderr', '')
                if stderr:
                    print(f"错误详情: {stderr}")
                return False
            
            # 验证环境执行器是否写入成功
            verify_result = await self.sandbox_manager.execute_command(
                session_id,
                "test -f /tmp/env_executor.py && echo '环境执行器存在' || echo '环境执行器不存在'"
            )
            
            if verify_result and verify_result.get('exit_code') == 0:
                output = verify_result.get('stdout', '')
                if '环境执行器存在' in output:
                    print("✅ 环境执行器写入验证成功")
                    return True
                else:
                    print("❌ 环境执行器文件不存在")
                    return False
            else:
                print("❌ 无法验证环境执行器文件状态")
                return False
                
        except Exception as e:
            print(f"设置环境执行器时出错: {e}")
            return False
    
    async def setup_training_script(self, session_id: str) -> bool:
        """在训练沙箱中设置DDPG训练脚本"""
        try:
            # 将训练脚本写入沙箱 - 使用base64编码避免特殊字符问题
            import base64
            encoded_content = base64.b64encode(DDPG_TRAINING_SCRIPT.encode('utf-8')).decode('utf-8')
            write_result = await self.sandbox_manager.execute_command(
                session_id,
                f"echo '{encoded_content}' | base64 -d > /tmp/ddpg_train.py"
            )
            
            if not write_result:
                print("❌ 写入训练脚本失败: 无返回结果")
                return False
            
            exit_code = write_result.get('exit_code')
            if exit_code is not None and exit_code != 0:
                print(f"❌ 写入训练脚本失败: exit_code = {exit_code}")
                stderr = write_result.get('stderr', '')
                if stderr:
                    print(f"错误详情: {stderr}")
                return False
            
            # 验证训练脚本是否写入成功
            verify_result = await self.sandbox_manager.execute_command(
                session_id,
                "test -f /tmp/ddpg_train.py && echo '训练脚本存在' || echo '训练脚本不存在'"
            )
            
            if verify_result and verify_result.get('exit_code') == 0:
                output = verify_result.get('stdout', '')
                if '训练脚本存在' in output:
                    print("✅ 训练脚本写入验证成功")
                    return True
                else:
                    print("❌ 训练脚本文件不存在")
                    return False
            else:
                print("❌ 无法验证训练脚本文件状态")
                return False
                
        except Exception as e:
            print(f"设置训练脚本时出错: {e}")
            return False
    
    async def run_ddpg_training(self, episodes: int = 1000000, eval_freq: int = 25000):
        """在沙箱中运行DDPG训练（本地训练 + 沙箱环境）"""
        
        if not self.training_session:
            raise RuntimeError("请先创建训练沙箱会话")
        
        # 初始化日志记录器
        self.logger = setup_training_logger(self.training_session.session_id)
        self.logger.info("🚀 开始DDPG机械臂训练")
        self.logger.info(f"   训练会话ID: {self.training_session.session_id}")
        self.logger.info(f"   训练回合数: {episodes}")
        self.logger.info(f"   评估频率: 每{eval_freq}回合")
        
        try:
            # 1. 安装环境依赖到沙箱
            self.logger.info("1️⃣ 安装环境依赖到沙箱")
            if not await self.install_dependencies(self.training_session.session_id):
                self.logger.error("❌ 环境依赖安装失败，无法继续训练")
                return None
                        
            # 2. 设置环境执行器到沙箱
            self.logger.info("2️⃣ 设置环境执行器到沙箱")
            if not await self.setup_env_executor(self.training_session.session_id):
                self.logger.error("❌ 环境执行器设置失败，无法继续训练")
                return None
                        
            # 3. 启动沙箱中的环境执行器
            self.logger.info("\n🚀 启动沙箱环境执行器...")
                        
            # 先测试沙箱Python环境
            self.logger.info("🧪 测试沙箱Python环境...")
            test_result = await self.sandbox_manager.execute_command(
                self.training_session.session_id,
                "python3 -c \"import sys; print('Python version:', sys.version); import json; print('JSON module OK')\"",
                timeout_ms=5000
            )
            self.logger.info(f"📋 Python环境测试结果: {test_result}")
            
            # 使用nohup启动环境执行器进程
            self.logger.info("📝 使用nohup启动环境执行器...")
            env_start_result = await self.sandbox_manager.execute_command(
                self.training_session.session_id,
                "nohup python3 /tmp/env_executor.py > /tmp/env_executor.log 2>&1 & echo $!",
                timeout_ms=10000
            )
            
            self.logger.info(f"📋 环境执行器启动结果: {env_start_result}")
            
            if not env_start_result:
                self.logger.error("❌ 环境执行器启动失败：无返回结果")
                return None
            
            exit_code = env_start_result.get('exit_code')
            stdout = env_start_result.get('stdout', '')
            stderr = env_start_result.get('stderr', '')
            system_error = env_start_result.get('system_error', '')
            
            self.logger.info(f"   exit_code: {exit_code}")
            self.logger.info(f"   stdout: {repr(stdout)}")
            self.logger.info(f"   stderr: {repr(stderr)}")
            self.logger.info(f"   system_error: {repr(system_error)}")
            
            # exit_code 124 表示命令超时，对于守护进程这是正常的
            if exit_code != 0 and exit_code != 124:
                self.logger.error("❌ 环境执行器启动失败")
                if stderr:
                    self.logger.error(f"   错误信息: {stderr}")
                if system_error:
                    self.logger.error(f"   系统错误: {system_error}")
                return None
            elif exit_code == 124:
                self.logger.info("✅ 环境执行器启动成功（命令超时是正常的守护进程行为）")
            
            # 获取环境执行器PID
            pid_str = env_start_result.get('stdout', '').strip()
            
            # 验证环境执行器是否真正启动成功
            self.logger.info("🔍 验证环境执行器启动状态...")
            await asyncio.sleep(5)  # 等待执行器初始化
            
            # 检查输出文件和日志文件（显示完整内容）
            self.logger.info("📂 收集环境执行器状态信息...")
            
            # 检查正常输出文件（环境状态结果）
            out_result = await self.sandbox_manager.execute_command(
                self.training_session.session_id,
                "cat /tmp/env_executor.out 2>/dev/null || echo 'NO_OUTPUT_FILE'",
                timeout_ms=5000
            )
            
            # 检查日志文件（执行日志）
            log_result = await self.sandbox_manager.execute_command(
                self.training_session.session_id,
                "cat /tmp/env_executor.log 2>/dev/null || echo 'NO_LOG_FILE'",
                timeout_ms=5000
            )
            
            print(f"📋 环境执行器状态检查详情:")
            
            # 处理正常输出文件
            print(f"📂 正常输出文件 (/tmp/env_executor.out):")
            if out_result:
                exit_code = out_result.get('exit_code', 'N/A')
                stdout = out_result.get('stdout', '')
                stderr = out_result.get('stderr', '')
                system_error = out_result.get('system_error', '')
                
                print(f"   exit_code: {exit_code}")
                print(f"   stdout长度: {len(stdout)} 字符")
                print(f"   stdout完整内容: {repr(stdout)}")
                print(f"   stderr: {repr(stderr)}")
                print(f"   system_error: {repr(system_error)}")
                
                output_content = stdout.strip()
                if output_content != 'NO_OUTPUT_FILE' and output_content:
                    print("✅ 正常输出文件有内容")
                    print(f"   内容: {repr(output_content)}")
                else:
                    print("⚠️  正常输出文件为空或不存在")
            else:
                print("⚠️  无法读取正常输出文件")
            
            # 处理日志文件
            print(f"\n📝 日志文件 (/tmp/env_executor.log):")
            if log_result:
                exit_code = log_result.get('exit_code', 'N/A')
                stdout = log_result.get('stdout', '')
                stderr = log_result.get('stderr', '')
                system_error = log_result.get('system_error', '')
                
                print(f"   exit_code: {exit_code}")
                print(f"   stdout长度: {len(stdout)} 字符")
                print(f"   stdout完整内容: {repr(stdout)}")
                print(f"   stderr: {repr(stderr)}")
                print(f"   system_error: {repr(system_error)}")
                
                log_content = stdout.strip()
                if log_content != 'NO_LOG_FILE' and log_content:
                    print("✅ 日志文件有内容")
                    print(f"   启动日志: {repr(log_content)}")
                    # 检查是否包含启动成功的标志
                    if '🚀 沙箱环境执行器已启动' in log_content or '等待初始化命令' in log_content:
                        print("✅ 环境执行器启动验证成功")
                    else:
                        print("⚠️  环境执行器启动状态不确定")
                else:
                    print("⚠️  日志文件为空或不存在")
            else:
                print("⚠️  无法读取日志文件")
            if pid_str.isdigit():
                env_pid = pid_str
                print(f"✅ 沙箱环境执行器已启动，PID: {env_pid}")
            else:
                print("⚠️  无法获取环境执行器PID")
                env_pid = None
            
            # 4. 初始化沙箱环境（带渲染）
            print("\n🔧 初始化沙箱环境（带渲染展示）...")
            init_cmd = {
                "command": "init",
                "env_name": "FetchReachDense-v4",
                "render_mode": "human"
            }
            
            # 发送初始化命令到环境执行器
            send_init_result = await self.sandbox_manager.execute_command(
                self.training_session.session_id,
                f"echo '{json.dumps(init_cmd)}' > /tmp/cmd_input"
            )
            
            # 简单等待环境初始化
            await asyncio.sleep(2)
            
            # 简单等待环境初始化
            await asyncio.sleep(2)
            
            # 5. 与沙箱环境交互的DDPG训练
            print("\n🎯 开始与沙箱环境交互的DDPG训练...")
            
            # 创建一个与沙箱环境交互的训练循环
            result = await self._run_ddpg_with_sandbox_env(episodes, eval_freq)
            
            # 6. 清理环境执行器进程
            if env_pid:
                await self.sandbox_manager.execute_command(
                    self.training_session.session_id,
                    f"kill {env_pid} 2>/dev/null || true"
                )
            
            return result
                
        except Exception as e:
            print(f"沙箱执行出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _run_ddpg_with_sandbox_env(self, episodes: int, eval_freq: int):
        """与沙箱环境交互的DDPG训练循环"""
        import asyncio
        import numpy as np
        import json
        import time
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from collections import deque
        import random
        
        # 创建一个本地代理环境类来与沙箱环境通信
        class SandboxEnvironmentProxy:
            def __init__(self, sandbox_manager, session_id):
                self.sandbox_manager = sandbox_manager
                self.session_id = session_id
                self.current_obs = None
                self.current_info = None
                
            async def reset(self):
                """重置环境"""
                print("🔄 调用 env.reset()")
                cmd = {"command": "reset"}
                result = await self.sandbox_manager.execute_command(
                    self.session_id,
                    f"echo '{json.dumps(cmd)}' > /tmp/cmd_input"
                )
                print(f"   发送reset命令结果: {result}")
                # 等待环境响应
                await asyncio.sleep(1.0)
                # 读取结果
                read_result = await self.sandbox_manager.execute_command(
                    self.session_id,
                    "cat /tmp/env_output 2>/dev/null || echo '{}'"
                )
                print(f"   读取reset结果: {read_result}")
                output = read_result.get('stdout', '{}')
                try:
                    data = json.loads(output)
                    print(f"   解析的reset数据: {data}")
                    
                    # 处理复杂观测结构
                    obs_data = data.get('observation', [])
                    if isinstance(obs_data, dict):
                        # 这是复杂观测结构，如 {'observation': [...], 'achieved_goal': [...], 'desired_goal': [...]}
                        if 'observation' in obs_data:
                            actual_obs = obs_data['observation']
                            self.current_obs = np.array(actual_obs)
                            print(f"   从复合字典中提取观测值: shape={self.current_obs.shape}")
                        else:
                            # 如果没有内部observation键，尝试其他可能的键
                            if 'achieved_goal' in obs_data:
                                self.current_obs = np.array(obs_data['achieved_goal'])
                                print(f"   从achieved_goal提取观测值: shape={self.current_obs.shape}")
                            else:
                                self.current_obs = np.array([])
                                print(f"   无法找到合适的观测值，使用空数组")
                    else:
                        # 简单数组或列表
                        self.current_obs = np.array(obs_data)
                        print(f"   使用简单观测值: shape={self.current_obs.shape}")
                    
                    self.current_info = data.get('info', {})
                    return self.current_obs, self.current_info
                except json.JSONDecodeError as e:
                    print(f"   JSON解析错误: {e}, output: {output}")
                    return np.array([]), {}
                except Exception as e:
                    print(f"   处理reset返回数据时出错: {e}, data: {data}")
                    return np.array([]), {}
            
            async def step(self, action):
                """执行单步（保持向后兼容）"""
                print(f"🕹️  调用 env.step(), action: {action}")
                cmd = {
                    "command": "step",
                    "action": action.tolist() if hasattr(action, 'tolist') else list(action)
                }
                result = await self.sandbox_manager.execute_command(
                    self.session_id,
                    f"echo '{json.dumps(cmd)}' > /tmp/cmd_input"
                )
                print(f"   发送step命令结果: {result}")
                # 等待环境响应
                await asyncio.sleep(1.0)
                # 读取结果
                read_result = await self.sandbox_manager.execute_command(
                    self.session_id,
                    "cat /tmp/env_output 2>/dev/null || echo '{}'"
                )
                print(f"   读取step结果: {read_result}")
                output = read_result.get('stdout', '{}')
                try:
                    data = json.loads(output)
                    print(f"   解析的step数据: {data}")
                                        
                    # 处理观测值 - 可能是列表、数组或其他格式
                    obs_data = data.get('observation', [])
                    if isinstance(obs_data, list):
                        observation = np.array(obs_data)
                    elif isinstance(obs_data, (int, float)):
                        # 标量值
                        observation = np.array([obs_data])
                    elif isinstance(obs_data, dict):
                        # 复杂观测结构，如 {'observation': [...], 'achieved_goal': [...], 'desired_goal': [...]}
                        if 'observation' in obs_data:
                            inner_obs = obs_data['observation']
                            if isinstance(inner_obs, (list, np.ndarray)):
                                observation = np.array(inner_obs)
                            elif isinstance(inner_obs, dict):
                                # 更深一层嵌套
                                if 'observation' in inner_obs:
                                    actual_obs = inner_obs['observation']
                                    if isinstance(actual_obs, (list, np.ndarray)):
                                        observation = np.array(actual_obs)
                                    else:
                                        observation = np.array([])
                                else:
                                    observation = np.array([])
                            else:
                                observation = np.array([])
                        else:
                            observation = np.array([])
                    elif hasattr(obs_data, '__iter__'):
                        # 其他可迭代对象
                        observation = np.array(list(obs_data))
                    else:
                        # 默认为空数组
                        observation = np.array([])
                                        
                    reward = data.get('reward', 0)
                    terminated = data.get('terminated', False)
                    truncated = data.get('truncated', False)
                    info = data.get('info', {})
                    print(f"   返回: obs={observation.shape}, reward={reward}, terminated={terminated}")
                    return observation, reward, terminated, truncated, info
                except json.JSONDecodeError as e:
                    print(f"   JSON解析错误: {e}, output: {output}")
                    return self.current_obs, 0, True, False, {}
                except Exception as e:
                    print(f"   处理step返回数据时出错: {e}, data: {data}")
                    return self.current_obs, 0, True, False, {}
            
            async def batch_step(self, actions):
                """批量执行多个动作"""
                print(f"🔁 调用 batch_step, 动作数量: {len(actions)}")
                print(f"   第一个动作类型: {type(actions[0])}")
                print(f"   第一个动作内容: {actions[0]}")
                
                # 确保动作格式正确
                formatted_actions = []
                for i, action in enumerate(actions):
                    if hasattr(action, 'tolist'):  # numpy数组
                        formatted_action = action.tolist()
                        formatted_actions.append(formatted_action)
                    elif isinstance(action, (list, tuple)):  # 列表或元组
                        formatted_action = list(action)
                        formatted_actions.append(formatted_action)
                    else:  # 其他类型
                        try:
                            formatted_action = [float(x) for x in action]
                            formatted_actions.append(formatted_action)
                        except Exception as e:
                            print(f"   ⚠️ 动作 {i} 格式转换失败: {e}")
                            formatted_actions.append(list(action))  # 尝试直接转换
                
                print(f"   格式化后第一个动作: {formatted_actions[0]}")
                print(f"   格式化后动作类型: {type(formatted_actions[0])}")
                
                cmd = {
                    "command": "batch_step",
                    "actions": formatted_actions
                }
                result = await self.sandbox_manager.execute_command(
                    self.session_id,
                    f"echo '{json.dumps(cmd)}' > /tmp/cmd_input"
                )
                print(f"   发送batch_step命令结果: {result}")
                # 等待环境响应（批量执行需要更多时间）
                await asyncio.sleep(2.0)
                # 读取结果
                read_result = await self.sandbox_manager.execute_command(
                    self.session_id,
                    "cat /tmp/env_output 2>/dev/null || echo '{}'"
                )
                print(f"   读取batch_step结果: {read_result}")
                output = read_result.get('stdout', '{}')
                try:
                    data = json.loads(output)
                    print(f"   解析的batch_step数据: {data}")
                    
                    if data.get('status') == 'success':
                        batch_results = data.get('batch_results', [])
                        executed_steps = data.get('executed_steps', 0)
                        print(f"   批量执行完成: {executed_steps}/{len(actions)} 步")
                        
                        # 转换为标准格式
                        results = []
                        for step_result in batch_results:
                            # 处理观测值 - 可能是列表、数组或其他格式
                            obs_data = step_result.get('observation', [])
                            if isinstance(obs_data, list):
                                observation = np.array(obs_data)
                            elif isinstance(obs_data, (int, float)):
                                # 标量值
                                observation = np.array([obs_data])
                            elif isinstance(obs_data, dict):
                                # 复杂观测结构，如 {'observation': [...], 'achieved_goal': [...], 'desired_goal': [...]}
                                if 'observation' in obs_data:
                                    inner_obs = obs_data['observation']
                                    if isinstance(inner_obs, (list, np.ndarray)):
                                        observation = np.array(inner_obs)
                                    elif isinstance(inner_obs, dict):
                                        # 更深一层嵌套
                                        if 'observation' in inner_obs:
                                            actual_obs = inner_obs['observation']
                                            if isinstance(actual_obs, (list, np.ndarray)):
                                                observation = np.array(actual_obs)
                                            else:
                                                observation = np.array([])
                                        else:
                                            observation = np.array([])
                                    else:
                                        observation = np.array([])
                                else:
                                    observation = np.array([])
                            elif hasattr(obs_data, '__iter__'):
                                # 其他可迭代对象
                                observation = np.array(list(obs_data))
                            else:
                                # 默认为空数组
                                observation = np.array([])
                            
                            reward = step_result.get('reward', 0)
                            terminated = step_result.get('terminated', False)
                            truncated = step_result.get('truncated', False)
                            info = step_result.get('info', {})
                            results.append((observation, reward, terminated, truncated, info))
                        
                        return results
                    else:
                        print(f"   批量执行失败: {data.get('message', 'Unknown error')}")
                        return []
                        
                except json.JSONDecodeError as e:
                    print(f"   JSON解析错误: {e}, output: {output}")
                    return []
        
        # 创建沙箱环境代理
        env_proxy = SandboxEnvironmentProxy(self.sandbox_manager, self.training_session.session_id)
        
        # 实现基础的DDPG算法组件
        class Actor(nn.Module):
            def __init__(self, state_dim, action_dim, max_action):
                super(Actor, self).__init__()
                
                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, action_dim)
                
                self.max_action = max_action

            def forward(self, state):
                a = torch.relu(self.l1(state))
                a = torch.relu(self.l2(a))
                return self.max_action * torch.tanh(self.l3(a))

        class Critic(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(Critic, self).__init__()

                # Q1 architecture
                self.l1 = nn.Linear(state_dim + action_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, 1)

                # Q2 architecture
                self.l4 = nn.Linear(state_dim + action_dim, 256)
                self.l5 = nn.Linear(256, 256)
                self.l6 = nn.Linear(256, 1)

            def forward(self, state, action):
                sa = torch.cat([state, action], 1)

                q1 = self.l3(torch.relu(self.l2(torch.relu(self.l1(sa)))))
                q2 = self.l6(torch.relu(self.l5(torch.relu(self.l4(sa)))))

                return q1, q2

            def Q1_forward(self, state, action):
                sa = torch.cat([state, action], 1)
                q1 = self.l3(torch.relu(self.l2(torch.relu(self.l1(sa)))))
                return q1

        class DDPGAgent:
            def __init__(self, state_dim, action_dim, max_action):
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.max_action = max_action
                
                self.actor = Actor(state_dim, action_dim, max_action)
                self.actor_target = Actor(state_dim, action_dim, max_action)
                self.critic = Critic(state_dim, action_dim)
                self.critic_target = Critic(state_dim, action_dim)
                
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.critic_target.load_state_dict(self.critic.state_dict())
                
                self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
                
                self.replay_buffer = deque(maxlen=100000)
                self.batch_size = 64
                
                self.gamma = 0.99  # 折扣因子
                self.tau = 0.005   # 软更新参数
                
            def select_action(self, state, noise_std=0.1):
                state = torch.FloatTensor(state.reshape(1, -1))
                action = self.actor(state).cpu().data.numpy().flatten()
                
                # 添加噪声
                noise = np.random.normal(0, noise_std, size=self.action_dim)
                action = action + noise
                action = np.clip(action, -self.max_action, self.max_action)
                
                return action
            
            def add_to_replay_buffer(self, state, action, reward, next_state, done):
                self.replay_buffer.append((state, action, reward, next_state, done))
            
            def soft_update(self, target_net, source_net):
                for target_param, param in zip(target_net.parameters(), source_net.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            def train(self):
                if len(self.replay_buffer) < self.batch_size:
                    return
                
                batch = random.sample(self.replay_buffer, self.batch_size)
                state, action, reward, next_state, done = map(np.stack, zip(*batch))
                
                state = torch.FloatTensor(state)
                action = torch.FloatTensor(action)
                reward = torch.FloatTensor(reward).unsqueeze(1)
                next_state = torch.FloatTensor(next_state)
                done = torch.BoolTensor(done).unsqueeze(1)
                
                # 计算目标Q值
                with torch.no_grad():
                    next_action = self.actor_target(next_state)
                    target_q1, target_q2 = self.critic_target(next_state, next_action)
                    target_q = torch.min(target_q1, target_q2)
                    target_q = reward + (~done) * self.gamma * target_q
                
                # 更新Critic
                current_q1, current_q2 = self.critic(state, action)
                critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # 更新Actor
                actor_loss = -self.critic.Q1_forward(state, self.actor(state)).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # 软更新目标网络
                self.soft_update(self.actor_target, self.actor)
                self.soft_update(self.critic_target, self.critic)
        
        # 初始化环境和代理
        obs, _ = await env_proxy.reset()
        # 处理观测值类型和维度
        if isinstance(obs, (list, np.ndarray)):
            state_dim = len(obs)
            print(f"✅ 检测到观测值维度: {state_dim}")
        elif isinstance(obs, (int, float)):
            # 单个数值，假设为1维
            state_dim = 1
            print(f"⚠️ 检测到标量观测值: {obs}, 假设维度为1")
        elif isinstance(obs, dict):
            # 如果是字典，检查是否是复杂观测结构
            if 'observation' in obs:
                # 这是复杂观测结构，如 {'observation': [...], 'achieved_goal': [...], 'desired_goal': [...]}
                inner_obs = obs['observation']
                if isinstance(inner_obs, (list, np.ndarray)):
                    state_dim = len(inner_obs)
                    print(f"✅ 从复合字典中检测到观测值维度: {state_dim}")
                    # 更新obs为实际观测值
                    obs = inner_obs
                elif isinstance(inner_obs, dict):
                    # 如果内部观测本身也是字典，则使用其所有值的长度
                    if 'observation' in inner_obs:
                        actual_obs = inner_obs['observation']
                        if isinstance(actual_obs, (list, np.ndarray)):
                            state_dim = len(actual_obs)
                            print(f"✅ 从深层字典中检测到观测值维度: {state_dim}")
                            obs = actual_obs
                        else:
                            state_dim = 4  # 默认维度
                            print(f"⚠️ 深层字典中观测值格式未知，默认维度: {state_dim}")
                    else:
                        state_dim = 4  # 默认维度
                        print(f"⚠️ 复合字典中无内部observation键，默认维度: {state_dim}")
                else:
                    state_dim = 4  # 默认维度
                    print(f"⚠️ 复合字典中观测值格式未知，默认维度: {state_dim}")
            else:
                state_dim = 4  # 默认维度
                print(f"⚠️ 字典中无observation键，默认维度: {state_dim}")
        else:
            # 默认维度
            state_dim = 4
            print(f"⚠️ 未知观测值类型: {type(obs)}，使用默认维度: {state_dim}")
        action_dim = 4  # FetchReach-v4 有4维动作空间
        max_action = 1.0  # 标准化的最大动作值
        
        agent = DDPGAgent(state_dim, action_dim, max_action)
        
        # 训练循环（使用批量执行优化）
        episode_count = 0
        total_steps = 0
        batch_size = 10  # 每次批量执行10步
        
        self.logger.info(f"🚀 开始DDPG训练（批量执行优化），总episodes: {episodes}")
        self.logger.info(f"   批量大小: {batch_size} 步/批次")
                
        for episode in range(episodes):
            self.logger.info(f"\n=== Episode {episode + 1} 开始 ===")
            obs, _ = await env_proxy.reset()
            self.logger.info(f"   reset返回: obs.shape={obs.shape if hasattr(obs, 'shape') else 'unknown'}, obs.type={type(obs)}")
                    
            # 确保观测值是有效数组
            if obs.size == 0:
                self.logger.warning("   ❌ 重置返回空观测值，跳过此episode")
                continue
            
            state = np.array(obs)
            episode_reward = 0
            episode_steps = 0
            
            while episode_steps < 100:  # 每个episode最多100步
                # 检查当前状态是否有效
                if state.size == 0:
                    self.logger.warning("   ❌ 当前状态为空，终止episode")
                    break
                
                # 预计算一批动作
                actions_batch = []
                states_batch = []
                current_state = state.copy()
                
                self.logger.info(f"   --- 准备批量执行 {batch_size} 步 ---")
                
                # 生成动作序列
                for i in range(min(batch_size, 100 - episode_steps)):
                    print(f"   当前状态形状: {current_state.shape if hasattr(current_state, 'shape') else 'unknown'}")
                    print(f"   当前状态大小: {current_state.size}")
                    if current_state.size == 0:
                        print("   ❌ 状态为空，跳过动作选择")
                        break
                    action = agent.select_action(current_state, noise_std=max(0.1, 0.1 - episode / episodes))
                    print(f"   生成动作 {i}: {action}, 类型: {type(action)}")
                    actions_batch.append(action)
                    states_batch.append(current_state.copy())
                    # 注意：这里只是预估下一个状态，实际状态会在批量执行后更新
                    
                if not actions_batch:
                    print("   ❌ 没有生成任何动作，终止episode")
                    break
                    
                print(f"   生成动作序列完成，共 {len(actions_batch)} 个动作")
                
                # 批量执行动作
                batch_results = await env_proxy.batch_step(actions_batch)
                
                if not batch_results:
                    print("   ⚠️ 批量执行失败，回退到单步执行")
                    # 回退到单步执行
                    next_obs, reward, terminated, truncated, info = await env_proxy.step(actions_batch[0])
                    done = terminated or truncated
                    
                    # 检查单步结果
                    if next_obs.size == 0:
                        print("   ❌ 单步执行返回空观测值，终止episode")
                        break
                        
                    next_state = np.array(next_obs)
                    agent.add_to_replay_buffer(state, actions_batch[0], reward, next_state, float(done))
                    agent.train()
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1
                    if done:
                        break
                    continue
                
                print(f"   批量执行返回 {len(batch_results)} 个结果")
                
                # 处理批量结果
                for i, (next_obs, reward, terminated, truncated, info) in enumerate(batch_results):
                    # 检查返回的观测值是否有效
                    if next_obs.size == 0:
                        print(f"   ❌ 第{i+1}步返回空观测值，终止episode")
                        done = True
                        break
                        
                    done = terminated or truncated
                    next_state = np.array(next_obs)
                    
                    # 存储到回放缓冲区
                    agent.add_to_replay_buffer(states_batch[i], actions_batch[i], reward, next_state, float(done))
                    
                    # 更新状态
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1
                    
                    print(f"   Step {episode_steps}: reward={reward:.3f}, terminated={terminated}")
                    
                    if done:
                        print(f"   Episode结束: done=True @ step {episode_steps}")
                        break
                
                # 批量训练（每次批量执行后训练一次）
                agent.train()
                
                if len(batch_results) < len(actions_batch):
                    # 如果批量执行提前结束（episode完成），跳出循环
                    break
            
            episode_count += 1
            
            # 打印进度
            if episode_count % 1 == 0:  # 每个episode都打印
                print(f"📊 Episode {episode_count} 完成, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                print(f"   通信次数: {episode_steps//batch_size + (1 if episode_steps % batch_size > 0 else 0)} 次批量调用")
            
            # 定期评估
            if episode_count % eval_freq == 0:
                print(f"🔄 评估模型性能 @ Episode {episode_count}")
                
                # 保存模型
                torch.save(agent.actor.state_dict(), f"/tmp/ddpg_actor_ep_{episode_count}.pth")
                torch.save(agent.critic.state_dict(), f"/tmp/ddpg_critic_ep_{episode_count}.pth")
                print(f"💾 模型已保存 @ Episode {episode_count}")
                
                # 如果达到期望的成功率，提前退出
                if episode_reward > -5:  # 如果奖励高于某个阈值
                    print(f"🎉 训练提前完成，达到目标性能")
                    break
        
        print(f"✅ DDPG训练完成，总共训练了 {episode_count} 个episodes")
        print(f"📈 优化效果: 从原来的 ~{total_steps} 次通信减少到 ~{total_steps//batch_size} 次批量通信")
        
        # 保存最终模型
        final_model_path = f"/tmp/ddpg_final_model.pth"
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'episode_count': episode_count
        }, final_model_path)
        
        return {"status": "completed", "final_episode": episode_count, "model_path": final_model_path}
    
    async def run_model_testing(self, model_path: str):
        """在测试沙箱中测试训练好的模型"""
        if not self.testing_session:
            raise RuntimeError("请先创建测试沙箱会话")
        
        try:
            # 安装依赖
            if not await self.install_dependencies(self.testing_session.session_id):
                print("❌ 依赖安装失败，无法继续测试")
                return None
            
            # 创建测试脚本
            test_script = f'''
#!/usr/bin/env python3
"""DDPG模型测试脚本"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import DDPG
import os
import json

# 注册环境
gym.register_envs(gymnasium_robotics)

def test_model(model_path, num_episodes=10):
    """测试模型性能"""
    print("🧪 开始测试模型...")
    
    # 创建环境
    env = gym.make("FetchReachDense-v4", render_mode='human')
    
    # 加载模型
    if os.path.exists(model_path + ".zip"):
        model = DDPG.load(model_path, env=env)
        print(f"✅ 成功加载模型: {model_path}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 测试结果统计
    total_rewards = []
    success_counts = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        success_count = 0
        
        print(f"\\n🚀 测试 Episode {episode + 1}/{num_episodes}")
        
        for step in range(100):  # 最大100步
            # 获取动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 累积奖励
            total_reward += reward
            
            # 检查成功
            if 'is_success' in info and info['is_success']:
                success_count += 1
                print(f"   ✅ Step {step}: 成功到达目标!")
            
            # 显示进展
            if step % 20 == 0:
                print(f"   Step {step}: 奖励={reward:.2f}")
            
            if done:
                break
        
        # 记录结果
        total_rewards.append(total_reward)
        success_counts.append(success_count)
        
        print(f"   📊 Episode {episode + 1} 结果:")
        print(f"      总奖励: {total_reward:.2f}")
        print(f"      成功次数: {success_count}")
    
    # 计算总体统计
    if total_rewards:
        avg_total_reward = np.mean(total_rewards)
        avg_success_rate = np.mean(success_counts)
        overall_success_rate = np.mean([1 if sc > 0 else 0 for sc in success_counts])
        
        results = {{
            "avg_total_reward": avg_total_reward,
            "avg_success_rate": avg_success_rate,
            "overall_success_rate": overall_success_rate,
            "best_episode_reward": np.max(total_rewards),
            "num_episodes": num_episodes
        }}
        
        # 保存结果
        with open('/tmp/test_results.json', 'w') as f:
            json.dump(results, f)
        
        print(f"\\n🎯 测试完成! 性能汇总:")
        print(f"   平均总奖励: {avg_total_reward:.2f}")
        print(f"   平均成功次数: {avg_success_rate:.2f}")
        print(f"   整体成功率: {overall_success_rate:.2%}")
        print(f"   最佳单次奖励: {np.max(total_rewards):.2f}")
        
        env.close()
        return results

if __name__ == "__main__":
    test_model("{model_path}")
'''
            
            # 将测试脚本写入沙箱
            import base64
            encoded_content = base64.b64encode(test_script.encode('utf-8')).decode('utf-8')
            write_result = await self.sandbox_manager.execute_command(
                self.testing_session.session_id,
                f"echo '{encoded_content}' | base64 -d > /tmp/ddpg_test.py"
            )
            
            if not write_result or write_result.get('exit_code') != 0:
                print("❌ 写入测试脚本失败")
                return None
            
            # 验证测试脚本
            verify_result = await self.sandbox_manager.execute_command(
                self.testing_session.session_id,
                "test -f /tmp/ddpg_test.py && echo '存在' || echo '不存在'"
            )
            
            if verify_result.get('stdout', '').strip() != '存在':
                print("❌ 测试脚本验证失败")
                return None
            
            print("🔧 启动DDPG模型测试...")
            test_start_result = await self.sandbox_manager.execute_command(
                self.testing_session.session_id,
                "python3 /tmp/ddpg_test.py > /tmp/ddpg_test.log 2>&1 &",
                timeout_ms=10000
            )
            
            if test_start_result and test_start_result.get('exit_code') == 0:
                print("✅ 测试进程已启动")
            else:
                print("❌ 测试进程启动失败")
                return None
            
            # 等待测试完成
            max_wait_time = 1800  # 最大等待30分钟
            check_interval = 10
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                # 检查测试结果文件是否存在
                result_check = await self.sandbox_manager.execute_command(
                    self.testing_session.session_id,
                    "test -f /tmp/test_results.json && echo 'completed' || echo 'running'",
                    timeout_ms=5000
                )
                
                if result_check and result_check.get('stdout', '').strip() == 'completed':
                    # 读取测试结果
                    result_read = await self.sandbox_manager.execute_command(
                        self.testing_session.session_id,
                        "cat /tmp/test_results.json",
                        timeout_ms=10000
                    )
                    
                    if result_read and result_read.get('exit_code') == 0:
                        try:
                            results = json.loads(result_read.get('stdout', ''))
                            print("✅ 测试完成，结果:")
                            print(f"   平均奖励: {results.get('avg_total_reward', 0):.2f}")
                            print(f"   整体成功率: {results.get('overall_success_rate', 0):.2%}")
                            return results
                        except json.JSONDecodeError:
                            print("❌ 无法解析测试结果")
                            return None
                
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval
            
            print("❌ 测试超时")
            return None
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"模型测试出错: {e}")
                import traceback
                self.logger.error(f"详细错误信息:\n{traceback.format_exc()}")
            else:
                print(f"模型测试出错: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    async def cleanup(self, force_cleanup: bool = True):
        """清理沙箱资源
        
        Args:
            force_cleanup: 是否强制清理。如果为False，则保留沙箱供手动查看
        """
        if self.training_session and force_cleanup:
            try:
                await self.sandbox_manager.cleanup_sandbox(self.training_session.session_id)
                print("DDPG训练沙箱已清理")
            except Exception as e:
                print(f"清理训练沙箱失败: {e}")
        
        if self.testing_session and force_cleanup:
            try:
                await self.sandbox_manager.cleanup_sandbox(self.testing_session.session_id)
                print("DDPG测试沙箱已清理")
            except Exception as e:
                print(f"清理测试沙箱失败: {e}")

# 使用示例
async def demo_sandbox_ddpg():
    """演示沙箱DDPG使用"""
    # 这里需要传入已初始化的sandbox_manager实例
    pass

if __name__ == "__main__":
    # 独立测试用
    asyncio.run(demo_sandbox_ddpg())