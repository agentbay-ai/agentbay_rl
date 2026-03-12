"""沙箱设置组件 - 管理沙箱环境的初始化和配置"""

import os
import asyncio
from typing import List, Dict, Any


class SandboxSetup:
    """沙箱设置管理器"""
    
    def __init__(self, sandbox_manager):
        self.sandbox_manager = sandbox_manager
        self.dependencies_installed = set()
    
    async def install_dependencies(self, session_id: str) -> bool:
        """安装环境依赖
        
        Args:
            session_id: 沙箱会话ID
            
        Returns:
            bool: 安装是否成功
        """
        try:
            print("🔧 开始安装环境依赖...")
            
            # 检查是否已安装
            if session_id in self.dependencies_installed:
                print("✅ 依赖已安装，跳过重复安装")
                return True
            
            # 安装基础依赖（使用阿里云镜像源）
            deps_commands = [
                "pip install -i https://mirrors.aliyun.com/pypi/simple/ gymnasium-robotics",
                "pip install -i https://mirrors.aliyun.com/pypi/simple/ numpy",
            ]
            
            for i, cmd in enumerate(deps_commands, 1):
                print(f"📦 安装依赖 ({i}/{len(deps_commands)}): {cmd}")
                result = await self.sandbox_manager.execute_command(session_id, cmd)
                
                if not result or result.get('exit_code') != 0:
                    stderr = result.get('stderr', '') if result else ''
                    print(f"❌ 依赖安装失败: {stderr}")
                    return False
            
            # 验证安装
            verify_cmd = "python3 -c 'import gymnasium_robotics, numpy; print(\"Dependencies OK\")'"
            result = await self.sandbox_manager.execute_command(session_id, verify_cmd)
            
            if result and result.get('exit_code') == 0:
                print("✅ 环境依赖安装验证成功")
                self.dependencies_installed.add(session_id)
                return True
            else:
                print("❌ 依赖验证失败")
                return False
                
        except Exception as e:
            print(f"❌ 依赖安装过程中出错: {e}")
            return False
    
    async def setup_env_executor(self, session_id: str) -> bool:
        """设置环境执行器
        
        Args:
            session_id: 沙箱会话ID
            
        Returns:
            bool: 设置是否成功
        """
        try:
            print("⚙️  开始设置环境执行器...")
            
            # 读取环境执行器脚本文件
            executor_script_path = os.path.join(
                os.path.dirname(__file__), 
                'sandbox_env_executor.py'
            )
            
            if not os.path.exists(executor_script_path):
                print(f"❌ 环境执行器脚本文件不存在: {executor_script_path}")
                return False
            
            with open(executor_script_path, 'r', encoding='utf-8') as f:
                executor_script = f.read()
            
            # 将执行器脚本写入沙箱
            encoded_content = executor_script.encode('utf-8').hex()
            write_result = await self.sandbox_manager.execute_command(
                session_id,
                f"echo '{encoded_content}' | xxd -r -p > /tmp/env_executor.py"
            )
            
            if not write_result or write_result.get('exit_code') != 0:
                print("❌ 写入环境执行器脚本失败")
                return False
            
            # 启动环境执行器
            print("🚀 启动环境执行器...")
            start_cmd = "nohup python3 /tmp/env_executor.py > /tmp/env_executor.log 2>&1 &"
            result = await self.sandbox_manager.execute_command(session_id, start_cmd)
            
            if not result or result.get('exit_code') != 0:
                print("❌ 启动环境执行器失败")
                return False
            
            # 验证执行器是否启动成功
            await asyncio.sleep(2)  # 等待执行器启动
            verify_cmd = "ps aux | grep env_executor.py | grep -v grep"
            result = await self.sandbox_manager.execute_command(session_id, verify_cmd)
            
            if result and result.get('stdout', '').strip():
                print("✅ 环境执行器启动成功")
                return True
            else:
                print("❌ 环境执行器启动验证失败")
                # 查看日志
                log_cmd = "cat /tmp/env_executor.log"
                log_result = await self.sandbox_manager.execute_command(session_id, log_cmd)
                if log_result and log_result.get('stdout'):
                    print(f"执行器日志: {log_result['stdout']}")
                return False
                
        except Exception as e:
            print(f"❌ 环境执行器设置失败: {e}")
            return False