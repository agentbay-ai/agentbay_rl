"""在沙箱中运行的多臂老虎机实现（带可视化）"""

import json
import asyncio
from typing import Dict, Any
from common.simple_sandbox_manager import SimpleSandboxManager
from .local_training_codes import BANDIT_TRAINING_SCRIPT
from .local_monitor_codes import MONITOR_GUI_SCRIPT
from .local_monitor import TrainingMonitor

class SandboxBanditRunner:
    """沙箱老虎机运行器"""
    
    def __init__(self, sandbox_manager: SimpleSandboxManager):
        self.sandbox_manager = sandbox_manager
        self.session = None
        self.sandbox_id = None
    
    async def create_sandbox_session(self) -> bool:
        """创建沙箱会话"""
        try:
            # 创建沙箱
            self.session = await self.sandbox_manager.create_sandbox()
            if not self.session:
                return False
            
            self.sandbox_id = self.session.sandbox_id
            
            # 获取流化URL并打印
            stream_url = await self.sandbox_manager.get_sandbox_url(self.sandbox_id)
            if stream_url:
                print(f"\n{'='*80}")
                print(f"🚨 老虎机沙箱流化页面URL: {stream_url}")
                print(f"{'='*80}\n")
            
            return True
        except Exception as e:
            print(f"创建沙箱会话失败: {e}")
            return False
    
    async def run_epsilon_greedy_demo(self, n_arms: int = 10, n_episodes: int = 100, epsilon: float = 0.1):
        """在沙箱中运行ε-贪婪老虎机演示（实时可视化版本）"""
        
        if not self.session:
            raise RuntimeError("请先创建沙箱会话")
        
        try:
            # 1. 在训练开始前安装必要的依赖包
            print("🔧 正在安装沙箱依赖包...")
            
            # 首先安装系统级的tkinter支持
            print("  -> 安装python3-tk (tkinter系统支持)...")
            tkinter_install = await self.sandbox_manager.execute_command(
                self.session.session_id,
                "sudo apt-get update && sudo apt-get install -y python3-tk",
                timeout_ms=600000  # 10分钟超时
            )
            
            if tkinter_install and tkinter_install.get('exit_code') == 0:
                print("✅ python3-tk安装成功")
            else:
                print("⚠️  python3-tk安装可能失败，但继续执行...")
                if tkinter_install:
                    stderr = tkinter_install.get('stderr', '')
                    if stderr:
                        print(f"安装错误信息: {stderr}")
            
            # 安装Python包依赖
            print("  -> 安装Python依赖包（使用阿里云源）...")
            pip_install_result = await self.sandbox_manager.execute_command(
                self.session.session_id,
                "pip3 install -i https://mirrors.aliyun.com/pypi/simple/ numpy pandas matplotlib -q",
                timeout_ms=300000  # 5分钟超时
            )
            
            if pip_install_result and pip_install_result.get('exit_code') == 0:
                print("✅ Python依赖包安装成功")
            else:
                print("⚠️  Python依赖包安装可能失败，但继续执行训练...")
                if pip_install_result:
                    stderr = pip_install_result.get('stderr', '')
                    if stderr:
                        print(f"安装错误信息: {stderr}")
            
            # 检查整体安装状态
            tkinter_success = tkinter_install and tkinter_install.get('exit_code') == 0
            pip_success = pip_install_result and pip_install_result.get('exit_code') == 0
            
            if tkinter_success and pip_success:
                print("✅ 所有依赖包安装成功")
            else:
                print("⚠️  部分依赖包安装可能失败，但继续执行训练...")
                if not tkinter_success:
                    print("  -> tkinter系统支持安装失败")
                if not pip_success:
                    print("  -> Python包依赖安装失败")
            
            # 2. 先在沙箱中创建训练脚本和监控脚本文件
            # 从local_training_codes.py导入预定义的脚本内容
            script_content = BANDIT_TRAINING_SCRIPT
            
            # 将训练脚本写入沙箱 - 使用base64编码避免特殊字符问题
            import base64
            encoded_content = base64.b64encode(script_content.encode('utf-8')).decode('utf-8')
            write_result = await self.sandbox_manager.execute_command(
                self.session.session_id,
                f"echo '{encoded_content}' | base64 -d > /tmp/bandit_train.py"
            )
            
            if not write_result:
                print("❌ 写入训练脚本失败: 无返回结果")
                return None
            
            exit_code = write_result.get('exit_code')
            if exit_code is not None and exit_code != 0:
                print(f"❌ 写入训练脚本失败: exit_code = {exit_code}")
                stderr = write_result.get('stderr', '')
                if stderr:
                    print(f"错误详情: {stderr}")
                return None
            
            # 验证训练脚本是否写入成功
            try:
                verify_result = await self.sandbox_manager.execute_command(
                    self.session.session_id,
                    "test -f /tmp/bandit_train.py && echo '训练脚本存在' || echo '训练脚本不存在'"
                )
                
                if verify_result and verify_result.get('exit_code') == 0:
                    output = verify_result.get('stdout', '')
                    if '训练脚本存在' in output:
                        print("✅ 训练脚本写入验证成功")
                    else:
                        print("⚠️  警告: 训练脚本文件可能不存在")
                        return None
                else:
                    print("⚠️  警告: 无法验证训练脚本文件状态")
                    return None
            except Exception as verify_error:
                print(f"⚠️  警告: 训练脚本验证时出错: {verify_error}")
                return None
            
            # 将监控脚本写入沙箱
            monitor_script_content = MONITOR_GUI_SCRIPT
            encoded_monitor_content = base64.b64encode(monitor_script_content.encode('utf-8')).decode('utf-8')
            monitor_write_result = await self.sandbox_manager.execute_command(
                self.session.session_id,
                f"echo '{encoded_monitor_content}' | base64 -d > /tmp/monitor_gui.py"
            )
            
            if not monitor_write_result:
                print("❌ 写入监控脚本失败: 无返回结果")
                return None
            
            monitor_exit_code = monitor_write_result.get('exit_code')
            if monitor_exit_code is not None and monitor_exit_code != 0:
                print(f"❌ 写入监控脚本失败: exit_code = {monitor_exit_code}")
                stderr = monitor_write_result.get('stderr', '')
                if stderr:
                    print(f"错误详情: {stderr}")
                return None
            
            # 验证监控脚本是否写入成功
            try:
                monitor_verify_result = await self.sandbox_manager.execute_command(
                    self.session.session_id,
                    "test -f /tmp/monitor_gui.py && echo '监控脚本存在' || echo '监控脚本不存在'"
                )
                
                if monitor_verify_result and monitor_verify_result.get('exit_code') == 0:
                    output = monitor_verify_result.get('stdout', '')
                    if '监控脚本存在' in output:
                        print("✅ 监控脚本写入验证成功")
                    else:
                        print("⚠️  警告: 监控脚本文件可能不存在")
                        return None
                else:
                    print("⚠️  警告: 无法验证监控脚本文件状态")
                    return None
            except Exception as verify_error:
                print(f"⚠️  警告: 监控脚本验证时出错: {verify_error}")
                return None
            
            print("\n🚀 训练脚本已写入沙箱，正在启动实时训练...")
            print("🎯 使用原生Python文本界面展示训练进度！")
            print("📊 请在流化界面中观察实时训练进展！\n")
            
            # 3. 启动分离式训练架构
            print("🚀 启动分离式训练架构（训练进程 + 监控进程）...")
            
            # 3.1 启动纯后台训练进程（专注训练，不启动GUI）
            # 通过环境变量传递训练参数
            print("🔧 启动训练进程（后台模式）...")
            print(f"   参数: 臂数={n_arms}, 回合数={n_episodes}, 探索率={epsilon}")
            train_start_result = await self.sandbox_manager.execute_command(
                self.session.session_id,
                f"BACKGROUND_MODE=true N_ARMS={n_arms} N_EPISODES={n_episodes} EPSILON={epsilon} nohup python3 /tmp/bandit_train.py > /tmp/training.log 2>&1 & echo $!",
                timeout_ms=30000  # 30秒超时，足够启动进程
            )
            
            if train_start_result and train_start_result.get('exit_code') == 0:
                train_pid_output = train_start_result.get('stdout', '').strip()
                if train_pid_output.isdigit():
                    training_pid = train_pid_output
                    print(f"✅ 训练进程已启动，PID: {training_pid}")
                else:
                    print("⚠️  无法获取训练进程PID，将继续轮询检查")
                    training_pid = None
            else:
                print("❌ 训练进程启动失败")
                return None
            
            # 3.2 启动监控进程（负责GUI显示）
            print("🔧 启动监控进程（GUI界面）...")
            #"nohup python3 /tmp/monitor_gui.py > /tmp/monitor.log 2>&1 & echo $!",
            monitor_start_result = await self.sandbox_manager.execute_command(
                self.session.session_id,
                "nohup python3 /tmp/monitor_gui.py > /tmp/monitor.log 2>&1 & echo $!",
                timeout_ms=15000  # 较短超时，GUI启动应该很快
            )
            
            if monitor_start_result and monitor_start_result.get('exit_code') == 0:
                monitor_pid_output = monitor_start_result.get('stdout', '').strip()
                if monitor_pid_output.isdigit():
                    monitor_pid = monitor_pid_output
                    print(f"✅ 监控进程已启动，PID: {monitor_pid}")
                else:
                    print("⚠️  无法获取监控进程PID")
                    monitor_pid = None
            else:
                print("⚠️  监控进程启动失败（不影响训练）")
                monitor_pid = None
            
            # 3.3 创建训练监控器用于状态轮询
            monitor = TrainingMonitor(self.sandbox_manager, self.session.session_id)
            monitor_task = asyncio.create_task(monitor.monitor_training_progress())
            
            # 轮询检查训练状态
            max_wait_time = 3600  # 最大等待1小时
            check_interval = 10   # 每10秒检查一次
            elapsed_time = 0
            
            print("🔍 开始轮询检查训练状态...")
            
            while elapsed_time < max_wait_time:
                # 检查进程是否仍在运行
                ps_result = await self.sandbox_manager.execute_command(
                    self.session.session_id,
                    "ps aux | grep bandit_train.py | grep -v grep",
                    timeout_ms=10000
                )
                
                process_running = False
                if ps_result and ps_result.get('exit_code') == 0:
                    ps_output = ps_result.get('stdout', '').strip()
                    if ps_output:
                        process_running = True
                        print(f"⏱️  训练进程仍在运行 (运行时间: {elapsed_time}秒)")
                
                # 检查是否有最终结果文件
                result_check = await self.sandbox_manager.execute_command(
                    self.session.session_id,
                    "test -f /tmp/training_result.json && echo 'completed' || echo 'running'",
                    timeout_ms=5000
                )
                
                training_completed = False
                if result_check and result_check.get('exit_code') == 0:
                    status = result_check.get('stdout', '').strip()
                    if status == 'completed':
                        training_completed = True
                        print("🎉 检测到训练完成!")
                
                # 如果进程已结束且没有完成标记，检查日志
                if not process_running and not training_completed:
                    print("⚠️  训练进程已结束，检查完成状态...")
                    # 给一点时间让最终结果文件写入
                    await asyncio.sleep(2)
                    
                    # 再次检查结果文件
                    final_check = await self.sandbox_manager.execute_command(
                        self.session.session_id,
                        "test -f /tmp/training_result.json && echo 'completed' || echo 'failed'",
                        timeout_ms=5000
                    )
                    
                    if final_check and final_check.get('exit_code') == 0:
                        final_status = final_check.get('stdout', '').strip()
                        if final_status == 'completed':
                            training_completed = True
                            print("✅ 训练正常完成")
                        else:
                            print("❌ 训练异常终止")
                            # 查看错误日志
                            log_result = await self.sandbox_manager.execute_command(
                                self.session.session_id,
                                "tail -20 /tmp/training.log",
                                timeout_ms=10000
                            )
                            if log_result and log_result.get('stdout', '').strip():
                                print("📋 训练日志最后20行:")
                                print(log_result.get('stdout', ''))
                
                # 如果训练完成，退出循环
                if training_completed:
                    break
                
                # 如果进程还在运行，继续等待
                if process_running:
                    await asyncio.sleep(check_interval)
                    elapsed_time += check_interval
                else:
                    # 进程已结束，快速检查
                    await asyncio.sleep(2)
                    elapsed_time += 2
            
            # 停止监控
            monitor.stop_monitoring()
            final_result = await monitor_task
            
            print(f"🎯 沙箱训练监控结束 (总耗时: {elapsed_time}秒)")
            
            # 获取最终训练日志
            try:
                log_result = await self.sandbox_manager.execute_command(
                    self.session.session_id,
                    "cat /tmp/training.log",
                    timeout_ms=30000
                )
                
                if log_result and log_result.get('stdout', '').strip():
                    print("\n📋 完整训练日志:")
                    print("=" * 50)
                    print(log_result.get('stdout', ''))
                    print("=" * 50)
            except Exception as e:
                print(f"⚠️  获取训练日志失败: {e}")
            
            # 返回最终结果
            return final_result
                
        except Exception as e:
            print(f"沙箱执行出错: {e}")
            return None
    
    async def cleanup(self, force_cleanup: bool = True):
        """清理沙箱资源
        
        Args:
            force_cleanup: 是否强制清理。如果为False，则保留沙箱供手动查看
        """
        if self.session and force_cleanup:
            try:
                await self.sandbox_manager.cleanup_sandbox(self.session.session_id)
                print("老虎机沙箱已清理")
            except Exception as e:
                print(f"清理沙箱失败: {e}")
        elif self.session and not force_cleanup:
            print(f"\n{'='*80}")
            print(f"🚨 注意：沙箱会话已保留，不会自动清理！")
            print(f"   沙箱ID: {self.sandbox_id}")
            print(f"   会话ID: {self.session.session_id}")
            print(f"   您可以在其他浏览器中访问以下URL查看沙箱内容：")
            stream_url = await self.sandbox_manager.get_sandbox_url(self.sandbox_id)
            if stream_url:
                print(f"   流化页面URL: {stream_url}")
            print(f"{'='*80}\n")

# 使用示例
async def demo_sandbox_bandit():
    """演示沙箱老虎机使用"""
    # 这里需要传入已初始化的sandbox_manager实例
    pass

if __name__ == "__main__":
    # 独立测试用
    asyncio.run(demo_sandbox_bandit())