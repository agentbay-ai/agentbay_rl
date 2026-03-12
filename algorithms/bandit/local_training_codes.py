"""本地训练代码定义文件 - 包含要在沙箱中执行的Python脚本"""

import os


def load_bandit_training_script():
    """加载外部的多臂老虎机训练脚本"""
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建外部脚本文件的路径
    script_path = os.path.join(current_dir, 'bandit_training_script.py')
    
    # 读取外部脚本文件的内容
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    return script_content


# 加载外部脚本内容
BANDIT_TRAINING_SCRIPT = load_bandit_training_script()