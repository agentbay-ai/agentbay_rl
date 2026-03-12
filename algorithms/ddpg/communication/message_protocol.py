"""消息协议 - 定义沙箱与本地通信的消息格式"""

from typing import Dict, Any, Union
from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    """消息类型枚举"""
    RESET = "reset"
    STEP = "step"
    BATCH_STEP = "batch_step"
    CLOSE = "close"
    STATUS = "status"


@dataclass
class Message:
    """消息数据类"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息对象"""
        return cls(
            type=MessageType(data["type"]),
            data=data["data"],
            timestamp=data["timestamp"]
        )


class MessageProtocol:
    """消息协议处理器"""
    
    @staticmethod
    def create_reset_message() -> Message:
        """创建重置消息"""
        import time
        return Message(
            type=MessageType.RESET,
            data={},
            timestamp=time.time()
        )
    
    @staticmethod
    def create_step_message(action: Union[list, tuple]) -> Message:
        """创建单步执行消息
        
        Args:
            action: 动作数组
            
        Returns:
            Message: 消息对象
        """
        import time
        return Message(
            type=MessageType.STEP,
            data={"action": list(action)},
            timestamp=time.time()
        )
    
    @staticmethod
    def create_batch_step_message(actions: list) -> Message:
        """创建批量执行消息
        
        Args:
            actions: 动作序列列表
            
        Returns:
            Message: 消息对象
        """
        import time
        return Message(
            type=MessageType.BATCH_STEP,
            data={"actions": [list(action) for action in actions]},
            timestamp=time.time()
        )
    
    @staticmethod
    def create_close_message() -> Message:
        """创建关闭消息"""
        import time
        return Message(
            type=MessageType.CLOSE,
            data={},
            timestamp=time.time()
        )
    
    @staticmethod
    def create_status_message(status: str, message: str = "") -> Message:
        """创建状态消息
        
        Args:
            status: 状态
            message: 消息内容
            
        Returns:
            Message: 消息对象
        """
        import time
        return Message(
            type=MessageType.STATUS,
            data={"status": status, "message": message},
            timestamp=time.time()
        )
    
    @staticmethod
    def validate_message(message: Message) -> bool:
        """验证消息格式
        
        Args:
            message: 消息对象
            
        Returns:
            bool: 消息是否有效
        """
        if not isinstance(message.type, MessageType):
            return False
        
        if not isinstance(message.data, dict):
            return False
            
        if not isinstance(message.timestamp, (int, float)):
            return False
            
        return True