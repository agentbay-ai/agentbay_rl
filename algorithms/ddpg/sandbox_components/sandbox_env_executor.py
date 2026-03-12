#!/usr/bin/env python3
"""沙箱环境执行器 - 在沙箱中运行FetchReach环境"""

import os
import sys
import json
import time
import signal
import traceback
import gymnasium as gym
import gymnasium_robotics
import numpy as np

# 注册环境
gym.register_envs(gymnasium_robotics)


class SandboxEnvExecutor:
    def __init__(self):
        self.env = None
        self.running = True
        self.cmd_file = "/tmp/cmd_input"
        self.output_file = "/tmp/env_output"
        
        # 设置信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # 初始化环境
        self._init_env()
    
    def _signal_handler(self, signum, frame):
        print(f"收到信号 {signum}，准备退出...")
        self.running = False
        self._cleanup()
        sys.exit(0)
    
    def _convert_to_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def _init_env(self):
        """初始化环境"""
        try:
            print("初始化FetchReach环境...")
            self.env = gym.make("FetchReachDense-v4", render_mode="human")
            print("✅ 环境初始化成功")
            
            # 创建通信文件
            os.makedirs(os.path.dirname(self.cmd_file), exist_ok=True)
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            # 初始化命令文件
            with open(self.cmd_file, 'w') as f:
                f.write('')
            
            print(f"✅ 通信文件已创建: {self.cmd_file}, {self.output_file}")
            
        except Exception as e:
            print(f"❌ 环境初始化失败: {e}")
            traceback.print_exc()
            self.running = False
    
    def _handle_reset(self, data):
        """处理重置命令"""
        try:
            obs, info = self.env.reset()
            # 处理复杂观测结构（FetchReach返回dict含observation/achieved_goal/desired_goal）
            if isinstance(obs, dict):
                actual_obs = obs.get('observation', obs)
                achieved_goal = obs.get('achieved_goal', None)
                desired_goal = obs.get('desired_goal', None)
            else:
                actual_obs = obs
                achieved_goal = None
                desired_goal = None
            
            response = {
                "type": "reset_result",
                "observation": actual_obs.tolist() if hasattr(actual_obs, 'tolist') else list(actual_obs),
                "achieved_goal": achieved_goal.tolist() if hasattr(achieved_goal, 'tolist') else (list(achieved_goal) if achieved_goal is not None else None),
                "desired_goal": desired_goal.tolist() if hasattr(desired_goal, 'tolist') else (list(desired_goal) if desired_goal is not None else None),
                "info": self._convert_to_serializable(info)
            }
            return response
        except Exception as e:
            print(f"Reset执行失败: {e}")
            traceback.print_exc()
            return {"type": "error", "message": str(e)}
    
    def _handle_step(self, data):
        """处理单步执行命令"""
        try:
            action = np.array(data['action'])
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 处理复杂观测结构（FetchReach返回dict含observation/achieved_goal/desired_goal）
            if isinstance(obs, dict):
                actual_obs = obs.get('observation', obs)
                achieved_goal = obs.get('achieved_goal', None)
                desired_goal = obs.get('desired_goal', None)
            else:
                actual_obs = obs
                achieved_goal = None
                desired_goal = None
            
            response = {
                "type": "step_result",
                "observation": actual_obs.tolist() if hasattr(actual_obs, 'tolist') else list(actual_obs),
                "achieved_goal": achieved_goal.tolist() if hasattr(achieved_goal, 'tolist') else (list(achieved_goal) if achieved_goal is not None else None),
                "desired_goal": desired_goal.tolist() if hasattr(desired_goal, 'tolist') else (list(desired_goal) if desired_goal is not None else None),
                "reward": float(reward),
                "done": bool(done),
                "info": self._convert_to_serializable(info)
            }
            return response
        except Exception as e:
            print(f"Step执行失败: {e}")
            traceback.print_exc()
            return {"type": "error", "message": str(e)}
    
    def _handle_batch_step(self, data):
        """处理批量执行命令（支持早停：遇到done=True时停止执行后续动作）"""
        try:
            actions = data['actions']
            results = []
            
            print(f"🔁 开始批量执行 {len(actions)} 个动作...")
            
            for i, action_list in enumerate(actions):
                action = np.array(action_list)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 处理复杂观测结构（FetchReach返回dict含observation/achieved_goal/desired_goal）
                if isinstance(obs, dict):
                    actual_obs = obs.get('observation', obs)
                    achieved_goal = obs.get('achieved_goal', None)
                    desired_goal = obs.get('desired_goal', None)
                else:
                    actual_obs = obs
                    achieved_goal = None
                    desired_goal = None
                
                results.append({
                    "observation": actual_obs.tolist() if hasattr(actual_obs, 'tolist') else list(actual_obs),
                    "achieved_goal": achieved_goal.tolist() if hasattr(achieved_goal, 'tolist') else (list(achieved_goal) if achieved_goal is not None else None),
                    "desired_goal": desired_goal.tolist() if hasattr(desired_goal, 'tolist') else (list(desired_goal) if desired_goal is not None else None),
                    "reward": float(reward),
                    "done": bool(done),
                    "info": self._convert_to_serializable(info)
                })
                
                # 早停：如果 done=True，停止执行后续动作
                if done:
                    print(f"   🏁 Episode在第 {i+1} 步结束 (done=True)，停止执行剩余 {len(actions)-i-1} 个动作")
                    break
            
            print(f"   ✅ 批量执行完成，共执行 {len(results)} 步")
            
            response = {
                "type": "batch_result",
                "results": results,
                "executed_steps": len(results),
                "total_requested": len(actions)
            }
            return response
        except Exception as e:
            print(f"批量执行失败: {e}")
            traceback.print_exc()
            return {"type": "error", "message": str(e)}
    
    def _handle_close(self, data):
        """处理关闭命令"""
        self.running = False
        self._cleanup()
        return {"type": "close_ack"}
    
    def _cleanup(self):
        """清理资源"""
        if self.env:
            try:
                self.env.close()
                print("环境已关闭")
            except:
                pass
    
    def run(self):
        """主运行循环"""
        print("🚀 环境执行器开始运行...")
        last_cmd = ""
        
        while self.running:
            try:
                # 读取命令
                if os.path.exists(self.cmd_file):
                    with open(self.cmd_file, 'r') as f:
                        cmd_content = f.read().strip()
                    
                    if cmd_content and cmd_content != last_cmd:
                        last_cmd = cmd_content
                        print(f"收到命令: {cmd_content[:100]}...")
                        
                        try:
                            cmd_data = json.loads(cmd_content)
                            cmd_type = cmd_data.get('type')
                            
                            # 处理不同类型的命令
                            if cmd_type == 'reset':
                                response = self._handle_reset(cmd_data)
                            elif cmd_type == 'step':
                                response = self._handle_step(cmd_data)
                            elif cmd_type == 'batch_step':
                                response = self._handle_batch_step(cmd_data)
                            elif cmd_type == 'close':
                                response = self._handle_close(cmd_data)
                            else:
                                response = {"type": "error", "message": f"未知命令类型: {cmd_type}"}
                            
                            # 写入响应（确保完整写入）
                            response_json = json.dumps(response)
                            with open(self.output_file, 'w') as f:
                                f.write(response_json)
                                f.flush()
                                os.fsync(f.fileno())
                            
                            print(f"响应已写入: {response.get('type', 'unknown')}, 长度: {len(response_json)}")
                            
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
                            traceback.print_exc()
                        except Exception as e:
                            print(f"命令处理错误: {e}")
                            traceback.print_exc()
                
                # 短暂休眠
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("收到键盘中断")
                break
            except Exception as e:
                print(f"运行时错误: {e}")
                traceback.print_exc()
                time.sleep(1)
        
        self._cleanup()
        print("环境执行器已停止")


if __name__ == "__main__":
    executor = SandboxEnvExecutor()
    executor.run()
