# 前端模块化结构说明

## 📁 目录结构

```
agentbay_rl/
├── static/
│   ├── css/
│   │   └── styles.css           # 全局样式文件
│   └── js/
│       ├── main.js              # 主入口文件
│       ├── config.js            # 配置和常量
│       ├── websocket.js         # WebSocket 通信模块
│       ├── ui.js                # UI 工具模块
│       ├── training.js          # 训练控制模块
│       ├── sandbox.js           # 沙箱管理模块
│       ├── algorithm.js         # 算法选择和教程模块
│       └── experiments.js       # 实验界面渲染模块
└── templates/
    ├── index.html               # 精简的 HTML 模板（117 行）
    └── index.html.backup        # 原始大文件备份（1863 行）
```

## 🎯 模块职责划分

### 1. **main.js** - 主入口
- 应用初始化
- 协调各模块启动

### 2. **config.js** - 配置中心
- 算法配置（algorithmConfigs）
- 训练模式描述（trainingModeDescriptions）
- 状态映射（statusMap）

### 3. **websocket.js** - 通信层
- WebSocket 连接管理
- 消息分发处理
- 自动重连机制

### 4. **ui.js** - UI 工具库
- DOM 元素管理
- 状态指示器更新
- 日志添加
- 进度条更新
- 统计数据更新

### 5. **training.js** - 训练控制
- 训练开始/停止
- 训练数据管理
- 进度事件处理
- 训练状态回调

### 6. **sandbox.js** - 沙箱管理
- 沙箱创建/清理
- 沙箱查看器渲染
- 沙箱事件处理

### 7. **algorithm.js** - 算法管理
- 算法卡片渲染
- 算法选择处理
- 教程内容加载
- Markdown 解析

### 8. **experiments.js** - 实验界面
- 实验界面渲染
- 事件绑定管理
- 模板生成（Bandit, DQN, PPO, SAC, DDPG）

## 🔄 数据流

```
用户交互 → UI 事件 → 对应模块处理 → API 请求 → 后端
                                            ↓
WebSocket ← 消息分发 ← websocket.js ← 后端推送
    ↓
特定处理模块（training.js / sandbox.js）
    ↓
UI 更新（ui.js）
```

## 📊 与原版对比

| 指标 | 原版 index.html | 模块化版本 |
|------|----------------|-----------|
| **行数** | 1863 行 | 117 行 HTML + 8 个模块 |
| **可维护性** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **可扩展性** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **代码复用** | ⭐ | ⭐⭐⭐⭐⭐ |
| **测试友好** | ⭐ | ⭐⭐⭐⭐ |

## ✅ 优势

1. **职责分离**：每个模块职责单一，易于理解和维护
2. **代码复用**：公共函数抽取到工具模块，避免重复
3. **易于测试**：每个模块可以独立测试
4. **协作友好**：不同开发者可以并行开发不同模块
5. **版本控制**：Git diff 更清晰，冲突更少
6. **性能优化**：可按需加载模块（未来可引入）
7. **样式隔离**：CSS 独立文件，便于主题切换

## 🔧 使用方法

### 开发新功能
1. 确定功能属于哪个模块
2. 在对应模块中添加函数
3. 在 main.js 或其他模块中调用
4. 使用 ES6 模块语法导入导出

### 添加新算法
1. 在 `config.js` 中添加算法配置
2. 在 `experiments.js` 中添加对应实验模板
3. 无需修改其他文件

### 修改样式
直接编辑 `static/css/styles.css`

### 调试
浏览器控制台查看各模块的 console.log 输出

## 🚀 未来优化方向

1. **TypeScript 迁移**：增强类型安全
2. **构建工具**：Vite/Webpack 打包优化
3. **状态管理**：考虑引入 Redux/Zustand
4. **组件化**：Web Components 或轻量框架
5. **国际化**：i18n 支持
6. **主题切换**：暗黑/明亮模式
7. **单元测试**：Jest/Vitest 覆盖

## 📝 注意事项

- 所有 JavaScript 文件使用 ES6 模块（`type="module"`）
- 模块间使用 `import/export` 语法
- 保持原始文件 `index.html.backup` 作为参考
- 静态文件路径使用 `/static/` 前缀
