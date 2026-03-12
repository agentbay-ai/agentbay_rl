#!/bin/bash

# RL Teaching Platform 启动脚本
# 强化学习教学平台一键启动脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Python 版本
check_python_version() {
    print_info "检查 Python 版本..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "未找到 Python3，请先安装 Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python 版本: $PYTHON_VERSION"
}

# 检查虚拟环境
check_virtualenv() {
    print_info "检查虚拟环境..."
    
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "未检测到虚拟环境，建议创建并激活虚拟环境"
        print_info "创建虚拟环境: python3 -m venv venv"
        print_info "激活虚拟环境: source venv/bin/activate"
    else
        print_success "当前虚拟环境: $VIRTUAL_ENV"
    fi
}

# 安装依赖
install_dependencies() {
    print_info "安装依赖包..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "未找到 requirements.txt 文件"
        exit 1
    fi
    
    pip install -r requirements.txt
    
    print_success "依赖包安装完成"
}

# 检查环境变量
check_env_vars() {
    print_info "检查环境变量..."
    
    if [ -f ".env" ]; then
        print_info "加载 .env 文件"
        export $(cat .env | xargs)
    fi
    
    # 检查必要环境变量
    if [ -z "$AGENTBAY_API_KEY" ]; then
        print_warning "AGENTBAY_API_KEY 未设置，沙箱功能将不可用"
        print_info "请在 .env 文件中设置: AGENTBAY_API_KEY=your-api-key"
    else
        print_success "AGENTBAY_API_KEY 已设置"
    fi
    
    # 设置默认端口和主机
    export PORT=${PORT:-8000}
    export HOST=${HOST:-0.0.0.0}
}

# 创建必要目录
create_directories() {
    print_info "创建数据目录..."
    
    mkdir -p data
    mkdir -p outputs
    mkdir -p templates
    mkdir -p static
    
    print_success "目录创建完成"
}

# 启动服务
start_server() {
    print_info "启动 RL 教学平台..."
    print_info "访问地址: http://localhost:$PORT"
    print_info "按 Ctrl+C 停止服务"
    print_success "=============================="
    
    # 启动 FastAPI 应用
    python app.py
}

# 主函数
main() {
    print_success "🚀 启动 RL Teaching Platform"
    print_success "=============================="
    
    # 检查工作目录
    if [ ! -f "app.py" ]; then
        print_error "请在项目根目录运行此脚本"
        exit 1
    fi
    
    # 执行初始化步骤
    check_python_version
    check_virtualenv
    install_dependencies
    check_env_vars
    create_directories
    
    print_success "初始化完成"
    print_success "=============================="
    
    # 启动服务
    start_server
}

# 优雅退出处理
trap 'print_info "正在停止服务..."; exit 0' INT TERM

# 运行主函数
main "$@"