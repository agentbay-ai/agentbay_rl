"""环境代理 - 连接本地训练器与沙箱环境"""

import json
import asyncio
from typing import Tuple, Dict, Any, List, Union, Optional
from .message_protocol import MessageProtocol, MessageType


class EnvProxy:
    """环境代理类"""
    
    def __init__(self, sandbox_manager, session_id: str):
        self.sandbox_manager = sandbox_manager
        self.session_id = session_id
        self.message_protocol = MessageProtocol()
        # 缓存当前episode的goal（HER需要）
        self.current_desired_goal = None
    
    async def reset(self) -> Tuple[Any, Dict]:
        """重置环境
        
        Returns:
            Tuple[Any, Dict]: (观测值, info字典，包含achieved_goal和desired_goal)
        """
        try:
            # 先清空输出文件（避免读取到旧数据）
            await self.sandbox_manager.write_file(
                self.session_id, 
                "/tmp/env_output", 
                ""
            )
            
            # 发送重置命令到沙箱
            cmd = '{"type": "reset"}'
            await self.sandbox_manager.write_file(
                self.session_id, 
                "/tmp/cmd_input", 
                cmd
            )
            
            # 轮询等待环境响应（最多等待 30 秒）
            max_wait = 30
            poll_interval = 0.5
            waited = 0
            result = None
            
            while waited < max_wait:
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                
                result = await self.sandbox_manager.read_file(self.session_id, "/tmp/env_output")
                if result and result.strip():
                    # 检查是否是 reset 的响应
                    try:
                        data = json.loads(result)
                        if data.get('type') == 'reset_result':
                            break
                    except json.JSONDecodeError:
                        pass
            
            if result:
                data = json.loads(result)
                
                # 验证响应类型
                if data.get('type') != 'reset_result':
                    print(f"[DEBUG reset] ⚠️ 响应类型不匹配: {data.get('type')}")
                
                obs = data.get('observation', [])
                # 提取goal信息（HER需要）
                achieved_goal = data.get('achieved_goal', None)
                desired_goal = data.get('desired_goal', None)
                info = data.get('info', {})
                
                # 将goal信息添加到info中
                info['achieved_goal'] = achieved_goal
                info['desired_goal'] = desired_goal
                
                # 缓存当前desired_goal
                self.current_desired_goal = desired_goal
                
                return obs, info
            else:
                return [], {}
                
        except Exception as e:
            print(f"环境重置失败: {e}")
            return [], {}
    
    async def step(self, action: Union[List, Tuple]) -> Tuple[Any, float, bool, Dict]:
        """执行一步动作
        
        Args:
            action: 动作数组
            
        Returns:
            Tuple[Any, float, bool, Dict]: (观测值, 奖励, 完成标志, info字典，包含goal信息)
        """
        result = None
        try:
            # 先清空输出文件
            await self.sandbox_manager.write_file(
                self.session_id, 
                "/tmp/env_output", 
                ""
            )
            
            # 发送动作到沙箱
            cmd = json.dumps({
                "type": "step",
                "action": list(action)
            })
            await self.sandbox_manager.write_file(
                self.session_id, 
                "/tmp/cmd_input", 
                cmd
            )
            
            # 轮询等待环境响应（最多等待 10 秒）
            max_wait = 10
            poll_interval = 0.1
            waited = 0
            result = None
            
            while waited < max_wait:
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                
                result = await self.sandbox_manager.read_file(self.session_id, "/tmp/env_output")
                if result and result.strip():
                    try:
                        data = json.loads(result)
                        if data.get('type') == 'step_result':
                            break
                    except json.JSONDecodeError:
                        pass
            
            if result:
                data = json.loads(result)
                obs = data.get('observation', [])
                reward = data.get('reward', 0.0)
                done = data.get('done', False)
                info = data.get('info', {})
                
                # 提取goal信息（HER需要）
                achieved_goal = data.get('achieved_goal', None)
                desired_goal = data.get('desired_goal', self.current_desired_goal)
                info['achieved_goal'] = achieved_goal
                info['desired_goal'] = desired_goal
                
                return obs, reward, done, info
            else:
                return [], 0.0, True, {}
                
        except Exception as e:
            print(f"环境执行失败: {e}")
            return [], 0.0, True, {}
    
    async def batch_step(self, actions: List[Union[List, Tuple]]) -> List[Tuple]:
        """批量执行动作（一次沙箱通信执行整个episode的动作）
        
        Args:
            actions: 动作序列列表（最多 MAX_STEPS_PER_EPISODE 个动作）
            
        Returns:
            List[Tuple]: 批量执行结果列表，每个元素为 (observation, reward, done, info)
                        其中info包含achieved_goal和desired_goal用于HER
        """
        try:
            # 格式化动作列表（确保可JSON序列化）
            formatted_actions = []
            for action in actions:
                if hasattr(action, 'tolist'):
                    formatted_actions.append(action.tolist())
                else:
                    formatted_actions.append(list(action))
            
            # 发送批量动作到沙箱
            cmd = json.dumps({
                "type": "batch_step",
                "actions": formatted_actions
            })
            await self.sandbox_manager.write_file(
                self.session_id, 
                "/tmp/cmd_input", 
                cmd
            )
            
            # 根据动作数量动态等待（每个动作约0.05秒）
            wait_time = max(0.5, len(actions) * 0.05)
            await asyncio.sleep(wait_time)
            
            # 读取批量执行结果
            result = await self.sandbox_manager.read_file(self.session_id, "/tmp/env_output")
            
            if result:
                data = json.loads(result)
                
                if data.get('type') == 'batch_result':
                    raw_results = data.get('results', [])
                    executed_steps = data.get('executed_steps', len(raw_results))
                    
                    # 将字典格式转换为元组格式，包含goal信息
                    tuple_results = []
                    for r in raw_results:
                        obs = r.get('observation', [])
                        reward = r.get('reward', 0.0)
                        done = r.get('done', False)
                        info = r.get('info', {})
                        
                        # 提取goal信息（HER需要）
                        achieved_goal = r.get('achieved_goal', None)
                        desired_goal = r.get('desired_goal', self.current_desired_goal)
                        info['achieved_goal'] = achieved_goal
                        info['desired_goal'] = desired_goal
                        
                        tuple_results.append((obs, reward, done, info))
                    
                    return tuple_results
                    
                elif data.get('type') == 'error':
                    pass  # 错误处理静默
            
            # 如果没有收到有效批量结果，回退到逐个执行
            return await self._fallback_sequential_step(actions)
            
        except json.JSONDecodeError as e:
            return await self._fallback_sequential_step(actions)
        except Exception as e:
            return await self._fallback_sequential_step(actions)
    
    async def _fallback_sequential_step(self, actions: List) -> List[Tuple]:
        """回退到顺序单步执行
        
        Args:
            actions: 动作列表
            
        Returns:
            List[Tuple]: 结果列表，每个元素为 (observation, reward, done, info)
        """
        results = []
        for i, action in enumerate(actions):
            obs, reward, done, info = await self.step(action)
            results.append((obs, reward, done, info))
            
            if done:
                break
                
        return results
    
    async def close(self):
        """关闭环境"""
        try:
            cmd = '{"type": "close"}'
            await self.sandbox_manager.write_file(
                self.session_id, 
                "/tmp/cmd_input", 
                cmd
            )
        except Exception as e:
            pass