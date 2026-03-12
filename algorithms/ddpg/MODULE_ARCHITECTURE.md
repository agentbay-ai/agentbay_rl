# DDPG模块化架构说明

## 架构概览

本项目实现了模块化的DDPG机械臂训练架构，支持多种训练模式的灵活切换。

### 核心特性
- **模块化设计**: 代码拆分为多个独立的子模块
- **多模式支持**: 支持自定义PyTorch、Stable-Baselines3、批量优化三种训练模式
- **沙箱隔离**: 环境执行与训练算法分离，提高安全性
- **配置灵活**: 通过配置管理器统一管理训练参数

## 目录结构

```
ddpg/
├── core/                    # 核心组件
│   ├── sandbox_ddpg_base.py     # 基础沙箱管理
│   ├── training_coordinator.py  # 训练协调器
│   └── config_manager.py        # 配置管理器
├── trainers/                # 训练器实现
│   ├── trainer_factory.py       # 训练器工厂
│   ├── sb3_trainer.py           # SB3训练器
│   └── custom_trainer.py        # 自定义训练器
├── communication/           # 通信组件
│   ├── env_proxy.py             # 环境代理
│   ├── sb3_env_adapter.py       # SB3环境适配器
│   └── message_protocol.py      # 消息协议
├── sandbox_components/      # 沙箱组件
│   ├── env_executor.py          # 环境执行器
│   └── sandbox_setup.py         # 沙箱设置
├── utils/                   # 工具组件
│   ├── model_utils.py           # 模型工具
│   └── validation_utils.py      # 验证工具
├── sandbox_ddpg.py          # 主入口文件
└── example_usage.py         # 使用示例
```

## 使用方法

### 基本使用

```python
import asyncio
from common.simple_sandbox_manager import SimpleSandboxManager
from .sandbox_ddpg import SandboxDDPG

async def main():
    # 创建沙箱管理器
    sandbox_manager = SimpleSandboxManager()
    
    # 创建DDPG运行器
    ddpg_runner = SandboxDDPG(sandbox_manager)
    
    # 运行训练
    result = await ddpg_runner.run_training(
        mode="custom_pytorch",  # 或 "stable_baselines3", "batch_optimized"
        custom_config={
            "episodes": 10000,
            "eval_freq": 1000
        }
    )
    
    print(f"训练结果: {result}")

# 执行
asyncio.run(main())
```

### 训练模式说明

1. **custom_pytorch**: 自定义PyTorch实现
   - 完全控制训练过程
   - 适合研究和定制化需求
   - 支持批量执行优化

2. **stable_baselines3**: Stable-Baselines3实现
   - 使用成熟的RL框架
   - 包含HER等高级功能
   - 训练稳定性和可靠性高

3. **batch_optimized**: 批量执行优化模式
   - 减少网络通信开销
   - 适合大规模训练场景
   - 预计算动作序列

## 配置参数

### 通用参数
- `episodes`: 训练回合数
- `eval_freq`: 评估频率
- `force_cleanup`: 是否强制清理沙箱

### 模式特定参数
- **custom_pytorch**: `batch_size`
- **stable_baselines3**: `learning_rate`, `buffer_size`
- **batch_optimized**: `batch_size` (通常更大)

## 主要优势

### 1. 模块化架构
- 每个组件职责单一，易于维护
- 支持独立测试和调试
- 便于扩展新功能

### 2. 灵活的模式切换
- 运行时切换不同训练模式
- 统一的API接口
- 配置驱动的行为控制

### 3. 沙箱安全保障
- 环境执行与训练算法隔离
- 资源使用可控
- 故障隔离能力强

### 4. 高效通信
- 优化的批量执行机制
- 标准化的消息协议
- 异步通信支持

## 开发指南

### 添加新的训练模式
1. 在`trainers/`目录下创建新的训练器类
2. 在`trainer_factory.py`中注册新训练器
3. 更新`config_manager.py`中的默认配置
4. 在`sandbox_ddpg.py`中添加相应的处理逻辑

### 扩展通信协议
1. 修改`message_protocol.py`添加新的消息类型
2. 更新`env_proxy.py`支持新协议
3. 在沙箱环境中实现对应的消息处理

### 自定义验证规则
1. 在`validation_utils.py`中添加新的验证方法
2. 在相应位置调用验证函数
3. 确保数据质量和一致性