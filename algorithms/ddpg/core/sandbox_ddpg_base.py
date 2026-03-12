"""沙箱DDPG基础类 - 核心沙箱管理功能"""

import json
import asyncio
from typing import Dict, Any, Optional
from common.simple_sandbox_manager import SimpleSandboxManager
from common.logger import setup_training_logger, DualLogger


class SandboxDDPGBase:
    """沙箱DDPG基础管理类"""
    
    def __init__(self, sandbox_manager: SimpleSandboxManager):
        self.sandbox_manager = sandbox_manager
        self.training_session = None
        self.testing_session = None
        self.logger: Optional[DualLogger] = None
    
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
                
                # 保存流化URL到session信息中
                self.training_session.stream_url = stream_url
            
            return True
        except Exception as e:
            print(f"创建训练沙箱会话失败: {e}")
            return False
    
    def get_training_sandbox_info(self) -> dict:
        """获取训练沙箱信息（用于广播给前端）"""
        if not self.training_session:
            return {}
        
        return {
            "session_id": self.training_session.session_id,
            "sandbox_id": self.training_session.sandbox_id,
            "resource_url": getattr(self.training_session, 'stream_url', None) or self.training_session.resource_url,
            "resource_id": self.training_session.resource_id,
            "auth_code": self.training_session.auth_code
        }
    
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
                print(f"🔍 DDPG测试沙箱流化页面URL: {stream_url}")
                print(f"{'='*80}\n")
            
            return True
        except Exception as e:
            print(f"创建测试沙箱会话失败: {e}")
            return False
    
    async def cleanup_sandboxes(self, force_cleanup: bool = True):
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