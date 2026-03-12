"""日志记录模块 - 将控制台输出同时写入文件"""
import os
import sys
import logging
from datetime import datetime
from typing import Optional

class DualLogger:
    """双重日志记录器 - 同时输出到控制台和文件"""
    
    def __init__(self, log_dir: str = "logs", log_name: Optional[str] = None):
        """
        初始化双重日志记录器
        
        Args:
            log_dir: 日志文件目录
            log_name: 日志文件名（不包含扩展名）
        """
        # 直接使用相对于agentbay_rl根目录的logs路径
        if not os.path.isabs(log_dir):
            # 从common目录向上一级到达agentbay_rl根目录，然后进入logs目录
            agentbay_rl_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log_dir = os.path.join(agentbay_rl_root, 'logs')
        else:
            self.log_dir = log_dir
            
        self.log_name = log_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = os.path.join(self.log_dir, f"{self.log_name}.log")
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(f'dual_logger_{self.log_name}')
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 控制台handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # 设置格式
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加handler
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """记录信息级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误级别日志"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """记录调试级别日志"""
        self.logger.debug(message)
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return self.log_file

# 全局日志记录器实例
_global_logger: Optional[DualLogger] = None

def get_logger(log_dir: str = "logs", log_name: Optional[str] = None) -> DualLogger:
    """
    获取全局日志记录器实例
    
    Args:
        log_dir: 日志文件目录
        log_name: 日志文件名
        
    Returns:
        DualLogger实例
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = DualLogger(log_dir, log_name)
    return _global_logger

def setup_training_logger(session_id: str = None) -> DualLogger:
    """
    为训练会话设置专门的日志记录器
    
    Args:
        session_id: 训练会话ID
        
    Returns:
        配置好的日志记录器
    """
    log_name = f"ddpg_training"
    if session_id:
        log_name += f"_{session_id}"
    
    return get_logger("agentbay_rl/logs", log_name)