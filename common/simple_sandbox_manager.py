#!/usr/bin/env python3
"""
基于 MCP 协议的简化版 AgentBay 沙箱管理器
使用 AgentBayClient 与云环境交互，获取流化界面 URL
"""

import os
import json
import uuid
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

@dataclass
class SandboxSession:
    """沙箱会话信息"""
    session_id: str
    resource_url: str  # 流化界面URL
    auth_code: str
    resource_id: str
    created_at: datetime
    sandbox_id: str  # AgentBay 沙箱 ID

class SimpleSandboxManager:
    """简化版沙箱管理器"""
    
    BASE_URL = "https://agentbay.wuying.aliyuncs.com/v2/sse"
    
    # 重试配置
    MAX_RETRIES = 3           # 最大重试次数
    RETRY_BASE_DELAY = 2.0    # 基础重试延迟（秒）
    RETRY_MAX_DELAY = 30.0    # 最大重试延迟（秒）
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sessions: Dict[str, SandboxSession] = {}
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
    
    def _is_retryable_error(self, error_text: str) -> bool:
        """判断错误是否可重试
        
        Args:
            error_text: 错误信息文本
            
        Returns:
            bool: 是否可重试
        """
        retryable_keywords = [
            'InvalidSession.NotFound',    # 会话未找到，临时问题
            'exceed limit',               # 并发限制
            'concurrency limit',          # 并发限制
            'timeout',                    # 超时
            'Expecting value',            # JSON 解析失败（可能是临时错误响应）
            'connection',                 # 连接问题
            'temporarily',                # 临时不可用
            'retry',                      # 提示重试
        ]
        error_lower = error_text.lower()
        return any(keyword.lower() in error_lower for keyword in retryable_keywords)
    
    async def _retry_with_backoff(self, operation_name: str, operation_func, *args, **kwargs):
        """带指数退避的重试包装器
        
        Args:
            operation_name: 操作名称（用于日志）
            operation_func: 要执行的异步函数
            *args, **kwargs: 传递给函数的参数
            
        Returns:
            操作结果，或 None 如果所有重试都失败
        """
        last_error = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                result = await operation_func(*args, **kwargs)
                if result is not None:
                    if attempt > 0:
                        logger.info(f"✅ {operation_name} 在第 {attempt + 1} 次尝试成功")
                    return result
                    
            except Exception as e:
                last_error = str(e)
                error_msg = str(e)
                
                # 检查是否可重试
                if not self._is_retryable_error(error_msg):
                    logger.error(f"❌ {operation_name} 遇到不可重试错误: {error_msg}")
                    return None
            
            # 计算重试延迟（指数退避）
            if attempt < self.MAX_RETRIES:
                delay = min(self.RETRY_BASE_DELAY * (2 ** attempt), self.RETRY_MAX_DELAY)
                logger.warning(f"⚠️ {operation_name} 第 {attempt + 1} 次尝试失败，{delay:.1f} 秒后重试...")
                await asyncio.sleep(delay)
        
        logger.error(f"❌ {operation_name} 在 {self.MAX_RETRIES + 1} 次尝试后仍然失败，最后错误: {last_error}")
        return None
        
    @property
    def sse_url(self) -> str:
        return f"{self.BASE_URL}?APIKEY={self.api_key}&IMAGEID=linux_latest"
    
    async def initialize(self) -> bool:
        """初始化 - 建立 MCP 连接"""
        if not self.api_key:
            logger.error("AGENTBAY_API_KEY 未设置")
            return False
            
        try:
            logger.info("连接到 AgentBay MCP 服务器...")
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()
            
            streams = await self._exit_stack.enter_async_context(sse_client(self.sse_url))
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )
            await self._session.initialize()
            logger.info("MCP 连接建立成功")
            return True
        except Exception as e:
            logger.error(f"MCP 连接失败: {e}")
            return False
    
    async def _call_tool(self, tool_name: str, arguments: dict = None) -> any:
        """调用 MCP 工具"""
        if not self._session:
            raise RuntimeError("MCP 会话未初始化")
        
        result = await self._session.call_tool(tool_name, arguments or {})
        if result.content:
            for content in result.content:
                if hasattr(content, "text"):
                    return content.text
        return result
    
    async def _create_sandbox_api_call(self) -> Optional[str]:
        """调用 API 创建沙箱（单次尝试）
        
        Returns:
            沙箱ID字符串，失败返回 None
        """
        result = await self._session.call_tool("create_sandbox", {})
        
        # 提取文本内容
        sandbox_id_text = ""
        if hasattr(result, 'content'):
            for content in result.content:
                if hasattr(content, 'text'):
                    sandbox_id_text = content.text
                    break
        
        logger.info(f"create_sandbox API 返回: {sandbox_id_text[:100] if sandbox_id_text else '空'}...")
        
        if sandbox_id_text:
            sandbox_id = sandbox_id_text.strip()
            # 检查是否返回了错误信息而非沙箱ID
            if sandbox_id.startswith('Code:') or 'error' in sandbox_id.lower() or 'limit' in sandbox_id.lower():
                # 这是错误响应，抛出异常以便重试
                raise RuntimeError(f"API 返回错误: {sandbox_id}")
            if sandbox_id:
                return sandbox_id
        
        return None
    
    async def create_sandbox(self) -> Optional[SandboxSession]:
        """创建新的沙箱会话（带重试机制）"""
        last_error = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # 步骤1：调用 API 创建沙箱
                sandbox_id = await self._create_sandbox_api_call()
                
                if not sandbox_id:
                    logger.warning(f"创建沙箱第 {attempt + 1} 次尝试返回空 ID")
                    if attempt < self.MAX_RETRIES:
                        delay = min(self.RETRY_BASE_DELAY * (2 ** attempt), self.RETRY_MAX_DELAY)
                        logger.warning(f"⚠️ {delay:.1f} 秒后重试...")
                        await asyncio.sleep(delay)
                    continue
                
                logger.info(f"✅ 沙箱创建成功: {sandbox_id[:16]}...")
                
                # 步骤2：获取流化URL（此方法自带重试）
                resource_url = await self.get_sandbox_url(sandbox_id)
                if not resource_url:
                    logger.error(f"无法获取沙箱 {sandbox_id} 的流化URL")
                    # URL获取失败但沙箱已创建，不再重试创建（避免创建过多沙箱）
                    return None
                
                # 步骤3：从URL中提取参数并创建会话对象
                from urllib.parse import urlparse, parse_qs
                parsed_url = urlparse(resource_url)
                query_params = parse_qs(parsed_url.query)
                
                auth_code = query_params.get('authcode', [''])[0]
                resource_id = query_params.get('resourceId', [''])[0]
                
                session = SandboxSession(
                    session_id=str(uuid.uuid4()),
                    resource_url=resource_url,
                    auth_code=auth_code,
                    resource_id=resource_id,
                    created_at=datetime.now(),
                    sandbox_id=sandbox_id
                )
                
                self.sessions[session.session_id] = session
                if attempt > 0:
                    logger.info(f"✅ 沙箱会话创建成功（第 {attempt + 1} 次尝试）: {session.session_id}")
                else:
                    logger.info(f"✅ 沙箱会话创建成功: {session.session_id}")
                return session
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"创建沙箱第 {attempt + 1} 次尝试异常: {e}")
                
                # 检查是否可重试
                if not self._is_retryable_error(last_error):
                    logger.error(f"❌ 遇到不可重试错误，停止重试")
                    return None
                
                if attempt < self.MAX_RETRIES:
                    delay = min(self.RETRY_BASE_DELAY * (2 ** attempt), self.RETRY_MAX_DELAY)
                    logger.warning(f"⚠️ {delay:.1f} 秒后重试...")
                    await asyncio.sleep(delay)
        
        logger.error(f"❌ 创建沙箱在 {self.MAX_RETRIES + 1} 次尝试后失败，最后错误: {last_error}")
        return None
    
    async def execute_command(self, session_id: str, command: str, timeout_ms: int = 3600000) -> Optional[Dict]:
        """在指定沙箱中执行命令
        
        Args:
            session_id: 沙箱会话ID
            command: 要执行的命令
            timeout_ms: 超时时间（毫秒），默认1小时(3600000ms)
        
        Returns:
            执行结果字典，包含stdout、stderr、exit_code等信息
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                logger.error(f"会话不存在: {session_id}")
                return None
            
            # 直接调用工具，支持超时参数
            result = await self._session.call_tool("shell", {
                "sandbox_id": session.sandbox_id,
                "command": command,
                "timeout_ms": timeout_ms
            })
            
            # 提取文本内容
            result_text = ""
            if hasattr(result, 'content'):
                for content in result.content:
                    if hasattr(content, 'text'):
                        result_text = content.text
                        break
            
            # 尝试解析JSON结果以提取详细信息
            try:
                if result_text:
                    import json
                    parsed_result = json.loads(result_text)
                    # 返回标准化的结果格式
                    return {
                        "stdout": parsed_result.get("stdout", ""),
                        "stderr": parsed_result.get("stderr", ""),
                        "exit_code": parsed_result.get("exit_code"),
                        "result": result_text,
                        "raw_result": parsed_result
                    }
            except (json.JSONDecodeError, ValueError):
                # 如果解析失败，返回原始结果
                pass
            
            return {"result": result_text if result_text else str(result)}
            
        except Exception as e:
            logger.error(f"执行命令异常: {e}")
            return None
    
    async def _get_sandbox_url_single_attempt(self, sandbox_id: str) -> Optional[str]:
        """获取沙箱流化URL（单次尝试）"""
        result = await self._session.call_tool("get_sandbox_url", {
            "sandbox_id": sandbox_id
        })
        
        # 提取文本内容
        url_text = ""
        if hasattr(result, 'content'):
            for content in result.content:
                if hasattr(content, 'text'):
                    url_text = content.text
                    break
        
        if url_text:
            # 检查是否返回了错误信息
            if url_text.startswith('Code:') or 'error' in url_text.lower():
                raise RuntimeError(f"API 返回错误: {url_text}")
            
            # 检查是否是直接的URL字符串
            if url_text.startswith('http'):
                return url_text
            else:
                # 尝试解析JSON格式
                url_data = json.loads(url_text)
                if isinstance(url_data, dict) and "url" in url_data:
                    return url_data["url"]
                else:
                    raise RuntimeError(f"URL数据格式不正确: {url_data}")
        
        return None
    
    async def get_sandbox_url(self, sandbox_id: str) -> Optional[str]:
        """获取沙箱的流化界面URL（带重试机制）
        
        Args:
            sandbox_id: 沙箱ID，来自create_sandbox工具的返回值
            
        Returns:
            流化界面URL字符串，或None如果失败
        """
        last_error = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                logger.info(f"获取沙箱 {sandbox_id[:16]}... 的流化URL（第 {attempt + 1} 次尝试）")
                
                url = await self._get_sandbox_url_single_attempt(sandbox_id)
                
                if url:
                    if attempt > 0:
                        logger.info(f"✅ 成功获取流化URL（第 {attempt + 1} 次尝试）")
                    else:
                        logger.info(f"✅ 成功获取流化URL")
                    return url
                else:
                    logger.warning(f"获取流化URL第 {attempt + 1} 次尝试返回空")
                    
            except json.JSONDecodeError as e:
                last_error = f"JSON解析失败: {e}"
                logger.warning(f"获取流化URL第 {attempt + 1} 次尝试失败: {last_error}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"获取流化URL第 {attempt + 1} 次尝试异常: {e}")
                
                # 检查是否可重试
                if not self._is_retryable_error(last_error):
                    logger.error(f"❌ 遇到不可重试错误，停止重试")
                    return None
            
            # 计算重试延迟
            if attempt < self.MAX_RETRIES:
                delay = min(self.RETRY_BASE_DELAY * (2 ** attempt), self.RETRY_MAX_DELAY)
                logger.warning(f"⚠️ {delay:.1f} 秒后重试...")
                await asyncio.sleep(delay)
        
        logger.error(f"❌ 获取流化URL在 {self.MAX_RETRIES + 1} 次尝试后失败，最后错误: {last_error}")
        return None
    
    async def get_sandbox_info(self, session_id: str) -> Optional[Dict]:
        """获取沙箱信息"""
        session = self.sessions.get(session_id)
        if not session:
            return None
            
        return {
            "session_id": session.session_id,
            "resource_url": session.resource_url,
            "auth_code": session.auth_code,
            "resource_id": session.resource_id,
            "sandbox_id": session.sandbox_id,
            "created_at": session.created_at.isoformat()
        }
    
    async def write_file(self, session_id: str, file_path: str, content: str) -> bool:
        """写入文件到沙箱（通过shell命令实现）
        
        Args:
            session_id: 沙箱会话ID
            file_path: 文件路径
            content: 文件内容
            
        Returns:
            bool: 写入是否成功
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                logger.error(f"会话不存在: {session_id}")
                return False
            
            # 使用 python 命令写入文件（更安全地处理特殊字符）
            import base64
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
            command = f"python3 -c \"import base64; open('{file_path}', 'w').write(base64.b64decode('{encoded_content}').decode('utf-8'))\""
            
            result = await self.execute_command(session_id, command)
            
            if result:
                exit_code = result.get('exit_code')
                if exit_code == 0 or exit_code is None:
                    logger.debug(f"文件写入成功: {file_path}")
                    return True
                else:
                    logger.error(f"文件写入失败: {file_path}, exit_code={exit_code}")
                    return False
            else:
                logger.error(f"文件写入失败: {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"写入文件异常: {e}")
            return False
    
    async def read_file(self, session_id: str, file_path: str) -> Optional[str]:
        """从沙箱读取文件（通过shell命令实现）
        
        Args:
            session_id: 沙箱会话ID
            file_path: 文件路径
            
        Returns:
            str: 文件内容，失败返回None
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                logger.error(f"会话不存在: {session_id}")
                return None
            
            # 使用 cat 命令读取文件
            command = f"cat {file_path}"
            result = await self.execute_command(session_id, command)
            
            if result:
                # 从执行结果中提取stdout
                stdout = result.get('stdout', '')
                if stdout:
                    return stdout
                # 如果没有stdout字段，尝试从result字段获取
                return result.get('result', '')
            
            return None
            
        except Exception as e:
            logger.error(f"读取文件异常: {e}")
            return None
    
    def list_sandboxes(self) -> List[Dict]:
        """列出所有活跃沙箱"""
        return [self.get_sandbox_info(sid) for sid in self.sessions.keys() if self.get_sandbox_info(sid)]
    
    async def cleanup_sandbox(self, session_id: str) -> bool:
        """清理指定沙箱"""
        try:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # 调用 kill_sandbox 工具
                await self._call_tool("kill_sandbox", {"sandbox_id": session.sandbox_id})
                
                # 从本地缓存中移除
                del self.sessions[session_id]
                logger.info(f"沙箱已清理: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"清理沙箱异常: {e}")
            # 即使 API 调用失败，也从本地缓存中移除
            if session_id in self.sessions:
                del self.sessions[session_id]
            return True
    
    async def cleanup_all(self):
        """清理所有沙箱（容错处理，单个失败不影响整体）"""
        session_ids = list(self.sessions.keys())
        success_count = 0
        fail_count = 0
        
        for session_id in session_ids:
            try:
                if await self.cleanup_sandbox(session_id):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.warning(f"清理沙箱 {session_id[:8]}... 时出错: {e}")
                fail_count += 1
                # 确保从本地缓存中移除
                if session_id in self.sessions:
                    del self.sessions[session_id]
        
        logger.info(f"沙箱清理完成: 成功 {success_count}, 失败 {fail_count}")
    
    async def close(self):
        """关闭连接"""
        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)
            self._session = None
            self._exit_stack = None