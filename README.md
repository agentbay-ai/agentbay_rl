# RL Teaching Platform - 强化学习教学平台

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.104%2B-green" alt="FastAPI Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/Platform-AgentBay-orange" alt="Platform">
</p>

一个基于 AgentBay 云沙箱的开源强化学习教学平台，通过直观的 Web 界面帮助学习者循序渐进地掌握强化学习算法。

## 🎯 项目特色

### 🎓 循序渐进的学习路径
- **深度确定性策略梯度 (DDPG)**: 连续控制方法，机械臂操控任务

### 🚀 沙箱并行加速
- 基于 AgentBay 云沙箱的并行训练
- 实时可视化训练进度和环境状态
- 相比本地训练显著提升效率

### 📊 直观的可视化界面
- 实时训练进度条和统计信息
- 交互式算法选择和参数配置
- 训练日志和性能图表展示
- 沙箱环境的流化界面展示

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端界面 (Web UI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ 算法选择区   │  │ 控制面板     │  │ 实时可视化展示           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      后端服务 (FastAPI)                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    强化学习训练管理器                        │ │
│  │                                                             │ │
│  │  ├── 算法调度器 (Algorithm Scheduler)                      │ │
│  │  ├── 沙箱管理器 (Sandbox Manager)                          │ │
│  │  ├── 训练监控器 (Training Monitor)                         │ │
│  │  └── 数据处理器 (Data Processor)                           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AgentBay 云沙箱集群                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ 沙箱 1   │  │ 沙箱 2   │  │ 沙箱 3   │  │ 沙箱 4   │          │
│  │ RL环境   │  │ RL环境   │  │ RL环境   │  │ RL环境   │          │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd agentbay_rl
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 设置环境变量

#### AgentBay API Key 获取步骤

1. 访问 [AgentBay 控制台](https://agentbay.console.aliyun.com)
2. 注册或登录阿里云账号
3. 进入服务管理页面
4. 创建新的 API KEY 或选择已有 KEY
5. 复制 API Key 并设置为环境变量

#### 设置环境变量

```bash
# 临时设置（当前终端会话有效）
export AGENTBAY_API_KEY="akm-XXXX"

# 或永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export AGENTBAY_API_KEY="akm-XXXX"' >> ~/.bashrc
source ~/.bashrc
```

**注意**: AgentBay 账号需要 Pro 或更高级别订阅才能使用完整功能。

### 5. 启动服务

```bash
# 方法1: 使用启动脚本（推荐）
chmod +x start.sh
./start.sh

# 方法2: 直接运行
python app.py
```

启动后浏览器将自动打开 `http://localhost:8000`

## 📖 使用指南

### 基本操作流程

1. **选择算法**: 选择 DDPG 算法开始学习
2. **配置参数**: 设置并行沙箱数量（默认5个）
3. **开始训练**: 点击"开始训练"按钮启动训练过程
4. **观察进度**: 实时查看训练进度、奖励曲线和日志信息
5. **分析结果**: 训练完成后查看性能统计和学习效果

### 算法介绍

#### 🦾 深度确定性策略梯度 (DDPG)
- **难度**: 中级
- **核心概念**: 确定性策略与价值函数结合
- **学习目标**: 掌握连续控制任务的强化学习方法
- **预计时间**: 30分钟
- **环境**: FetchReach-v4（机械臂抓取任务）
- **特点**: 使用 HER（ hindsight experience replay）提升样本效率

## 🛠️ 开发指南

### 项目结构

```
agentbay_rl/
├── app.py                      # 主应用入口
├── start.sh                    # 启动脚本
├── requirements.txt            # 依赖包列表
├── .env.example                # 环境变量模板
├── README.md                   # 本文档
├── data/                       # 教程数据目录
│   ├── bandit_tutorial.md
│   └── ddpg_tutorial.md
├── algorithms/                 # 算法实现目录
│   ├── bandit/                 # 多臂老虎机（后台保留）
│   │   ├── teaching.py
│   │   └── sandbox_bandit.py
│   └── ddpg/                   # DDPG 算法
│       ├── teaching.py
│       ├── sandbox_ddpg.py
│       ├── trainers/           # 训练器模块
│       ├── core/               # 核心组件
│       └── communication/      # 通信模块
├── templates/                  # HTML模板目录
│   └── index.html              # 主页面
├── static/                     # 静态资源目录
│   ├── css/
│   └── js/
└── outputs/                    # 模型输出目录
```

### 核心组件

#### 后端 (app.py)
- **FastAPI 应用**: 提供 REST API 和 WebSocket 服务
- **RLAppState**: 全局状态管理器
- **训练管道**: 算法调度和进度监控
- **WebSocket 管理**: 实时通信和状态同步

#### 算法模块 (algorithms/)
- **teaching.py**: 教学逻辑与 Web 框架解耦
- **sandbox_*.py**: 沙箱执行管理
- **trainers/**: 具体训练实现

#### 前端 (templates/index.html, static/)
- **响应式设计**: 支持桌面和移动端
- **实时更新**: 通过 WebSocket 接收训练状态
- **交互式控件**: 算法选择、参数配置、训练控制
- **数据可视化**: 进度条、统计卡片、训练图表

### 扩展开发

#### 添加新算法
1. 在 `algorithms/` 下创建新目录
2. 实现 `teaching.py` 和沙箱执行逻辑
3. 在 `config.js` 中注册新算法
4. 在 `experiments.js` 中添加实验界面模板

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 提交规范
- 使用 conventional commits 格式
- 功能分支开发
- 通过 CI 测试后再合并

## 📚 学习资源

### 官方文档
- [AgentBay Documentation](https://www.aliyun.com/product/agentbay)

### 推荐书籍
- 《强化学习》- Sutton & Barto
- 《深度强化学习实战》- Maxim Lapan
- 《动手学强化学习》- 张伟楠等

### 在线课程
- DeepMind x UCL RL Course
- UC Berkeley CS285
- Stanford CS234

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

Copyright 2025 AgentBay SDK Contributors

## 🙏 致谢

- 感谢 AgentBay 提供强大的云沙箱平台
- 感谢所有贡献者的努力和支持

---

<p align="center">
  Made with ❤️ for RL Education
</p>