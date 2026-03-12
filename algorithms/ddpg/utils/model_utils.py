"""模型工具 - 模型保存、加载和转换工具"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def save_model(model, path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """保存模型
        
        Args:
            model: 要保存的模型
            path: 保存路径
            metadata: 元数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 创建保存目录
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 准备保存数据
            save_data = {
                'model_state_dict': model.state_dict(),
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # 保存模型
            torch.save(save_data, path)
            print(f"✅ 模型已保存到: {path}")
            return True
            
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
            return False
    
    @staticmethod
    def load_model(model_class, path: str, device: str = 'cpu') -> Optional[Any]:
        """加载模型
        
        Args:
            model_class: 模型类
            path: 模型路径
            device: 设备
            
        Returns:
            Optional[Any]: 加载的模型，失败返回None
        """
        try:
            if not os.path.exists(path):
                print(f"❌ 模型文件不存在: {path}")
                return None
            
            # 加载数据
            checkpoint = torch.load(path, map_location=device)
            
            # 创建模型实例
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            print(f"✅ 模型已从 {path} 加载")
            print(f"   保存时间: {checkpoint.get('timestamp', 'Unknown')}")
            
            return model
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    
    @staticmethod
    def convert_array_to_list(obj) -> Any:
        """将numpy数组转换为列表，便于JSON序列化
        
        Args:
            obj: 要转换的对象
            
        Returns:
            Any: 转换后的对象
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ModelUtils.convert_array_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ModelUtils.convert_array_to_list(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    @staticmethod
    def save_training_history(history: list, path: str) -> bool:
        """保存训练历史
        
        Args:
            history: 训练历史数据
            path: 保存路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 转换为可序列化格式
            serializable_history = [ModelUtils.convert_array_to_list(item) for item in history]
            
            # 保存为JSON
            with open(path, 'w') as f:
                json.dump({
                    'history': serializable_history,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"✅ 训练历史已保存到: {path}")
            return True
            
        except Exception as e:
            print(f"❌ 训练历史保存失败: {e}")
            return False
    
    @staticmethod
    def load_training_history(path: str) -> Optional[list]:
        """加载训练历史
        
        Args:
            path: 历史文件路径
            
        Returns:
            Optional[list]: 训练历史，失败返回None
        """
        try:
            if not os.path.exists(path):
                print(f"❌ 历史文件不存在: {path}")
                return None
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ 训练历史已从 {path} 加载")
            return data.get('history', [])
            
        except Exception as e:
            print(f"❌ 训练历史加载失败: {e}")
            return None