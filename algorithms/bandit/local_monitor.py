"""本地训练进度监控器 - 轮询读取沙箱状态文件"""
import time
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

class TrainingMonitor:
    """训练进度监控器"""
    
    def __init__(self, sandbox_manager, session_id: str):
        self.sandbox_manager = sandbox_manager
        self.session_id = session_id
        self.last_status_time = 0
        self.monitoring = False
        
    async def read_status_file(self) -> Optional[Dict[str, Any]]:
        """从沙箱读取训练状态文件"""
        try:
            # 读取状态文件
            result = await self.sandbox_manager.execute_command(
                self.session_id,
                "cat /tmp/training_status.json 2>/dev/null || echo '{}'",
                timeout_ms=5000  # 5秒超时
            )
            
            if result and result.get('exit_code') == 0:
                output = result.get('stdout', '').strip()
                if output and output != '{}':
                    try:
                        status_data = json.loads(output)
                        # 检查是否是新的状态更新
                        timestamp = status_data.get('timestamp', 0)
                        if timestamp > self.last_status_time:
                            self.last_status_time = timestamp
                            return status_data
                    except json.JSONDecodeError:
                        pass
            return None
            
        except Exception as e:
            print(f"读取状态文件失败: {e}")
            return None
    
    async def read_result_file(self) -> Optional[Dict[str, Any]]:
        """从沙箱读取最终结果文件"""
        try:
            result = await self.sandbox_manager.execute_command(
                self.session_id,
                "cat /tmp/training_result.json 2>/dev/null || echo '{}'",
                timeout_ms=5000
            )
            
            if result and result.get('exit_code') == 0:
                output = result.get('stdout', '').strip()
                if output and output != '{}':
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        pass
            return None
            
        except Exception as e:
            print(f"读取结果文件失败: {e}")
            return None
    
    async def monitor_training_progress(self, callback=None):
        """监控训练进度"""
        print("🔍 开始监控沙箱训练进度...")
        self.monitoring = True
        training_completed = False
        
        while self.monitoring and not training_completed:
            try:
                # 读取训练状态
                status = await self.read_status_file()
                if status:
                    total_episodes = status.get('total_episodes', 30)
                    print(f"📊 训练进度更新 - 回合: {status['episode']}/{total_episodes}, "
                          f"进度: {status['progress']:.1f}%, "
                          f"奖励: {status['current_reward']:.2f}")
                    
                    # 调用回调函数（如果提供）
                    if callback:
                        callback(status)
                
                # 检查是否完成
                result = await self.read_result_file()
                if result and result.get('completed', False):
                    print("🎉 检测到训练完成!")
                    training_completed = True
                    if callback:
                        callback(result, is_final=True)
                
                # 等待下一次轮询
                await asyncio.sleep(2)  # 每2秒轮询一次
                
            except KeyboardInterrupt:
                print("\n🛑 监控被用户中断")
                break
            except Exception as e:
                print(f"监控过程中出现错误: {e}")
                await asyncio.sleep(5)  # 出错后等待5秒再重试
        
        self.monitoring = False
        print("🔍 训练监控结束")
        
        # 返回最终结果
        if training_completed:
            return await self.read_result_file()
        return None
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        print("🛑 停止训练监控")

# 使用示例
async def demo_monitor():
    """演示如何使用监控器"""
    # 这里需要传入实际的sandbox_manager和session_id
    # monitor = TrainingMonitor(sandbox_manager, session_id)
    # await monitor.monitor_training_progress()
    pass

if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_monitor())