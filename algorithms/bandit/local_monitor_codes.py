"""监控GUI脚本内容定义"""

MONITOR_GUI_SCRIPT = '''#!/usr/bin/env python3
"""
独立的GUI监控脚本
负责轮询状态文件并更新GUI界面
"""

import tkinter as tk
from tkinter import ttk
import json
import time
import sys
import os

class TrainingMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("老虎机训练监控器")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # GUI组件
        self.setup_ui()
        
        # 状态变量
        self.is_running = True
        self.last_timestamp = 0
        self.episode_count = 0
        
        print("🎯 训练监控GUI已初始化")
        
    def setup_ui(self):
        """设置GUI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 进度条
        ttk.Label(main_frame, text="训练进度:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var, 
            maximum=100,
            length=700
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="等待训练开始...", font=("Arial", 10))
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # 统计信息框架
        stats_frame = ttk.LabelFrame(main_frame, text="训练统计", padding="10")
        stats_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # 统计标签
        self.stats_labels = {}
        stats_info = [
            ("当前回合:", "episode"),
            ("平均奖励:", "avg_reward"),
            ("最优选择率:", "opt_rate"),
            ("当前奖励:", "current_reward")
        ]
        
        for i, (label_text, key) in enumerate(stats_info):
            ttk.Label(stats_frame, text=label_text, font=("Arial", 9)).grid(row=i, column=0, sticky=tk.W, padx=5)
            self.stats_labels[key] = ttk.Label(stats_frame, text="--", font=("Arial", 9, "bold"))
            self.stats_labels[key].grid(row=i, column=1, sticky=tk.W, padx=10)
        
        # 日志文本框
        ttk.Label(main_frame, text="训练日志:", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(10,5))
        self.log_text = tk.Text(main_frame, height=15, width=80)
        self.log_text.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=5, column=2, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        self.pause_button = ttk.Button(button_frame, text="暂停更新", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="清空日志", command=self.clear_log)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # 状态变量
        self.paused = False
        
    def toggle_pause(self):
        """切换暂停状态"""
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="继续更新")
            self.status_label.config(text="监控已暂停")
        else:
            self.pause_button.config(text="暂停更新")
            self.status_label.config(text="监控运行中...")
            
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        
    def update_display(self, data):
        """更新GUI显示"""
        try:
            # 更新进度条
            self.progress_var.set(data.get('progress', 0))
            
            # 更新状态标签
            episode = data.get('episode', 0)
            self.status_label.config(text=f"正在执行回合 {episode}/30")
            
            # 更新统计信息
            self.stats_labels['episode'].config(text=str(episode))
            self.stats_labels['avg_reward'].config(text=f"{data.get('avg_reward', 0):.3f}")
            self.stats_labels['opt_rate'].config(text=f"{data.get('opt_rate', 0):.1%}")
            self.stats_labels['current_reward'].config(text=f"{data.get('current_reward', 0):.2f}")
            
            # 更新日志（只添加新的日志条目）
            recent_logs = data.get('recent_logs', [])
            if recent_logs:
                for log_entry in recent_logs[-3:]:  # 只显示最新的3条
                    self.log_text.insert(tk.END, f"{log_entry}\\n")
                self.log_text.see(tk.END)
                
                # 限制日志行数
                lines = self.log_text.get(1.0, tk.END).split('\\n')
                if len(lines) > 100:
                    self.log_text.delete(1.0, f"{len(lines)-95}.0")
            
            # 强制更新GUI
            self.root.update_idletasks()
            self.root.update()
            
        except Exception as e:
            print(f"GUI更新错误: {e}")
            
    def poll_status(self):
        """轮询状态文件"""
        while self.is_running:
            if not self.paused:
                try:
                    # 读取状态文件
                    if os.path.exists('/tmp/training_status.json'):
                        with open('/tmp/training_status.json', 'r') as f:
                            data = json.load(f)
                        
                        timestamp = data.get('timestamp', 0)
                        if timestamp > self.last_timestamp:
                            self.last_timestamp = timestamp
                            self.update_display(data)
                            
                            # 检查是否完成
                            if data.get('progress', 0) >= 100:
                                self.status_label.config(text="🎉 训练完成!")
                                self.is_running = False
                                break
                                
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"状态轮询错误: {e}")
            
            time.sleep(0.5)  # 500ms轮询间隔
            
    def on_closing(self):
        """窗口关闭事件"""
        self.is_running = False
        self.root.destroy()
        
    def run(self):
        """运行监控器"""
        print("🚀 训练监控GUI启动")
        self.log_text.insert(tk.END, "🔍 开始监控训练进程...\\n")
        
        try:
            # 在主线程中运行轮询
            self.poll_status()
        except KeyboardInterrupt:
            print("收到中断信号")
        except Exception as e:
            print(f"监控器运行错误: {e}")
        finally:
            self.is_running = False
            print("监控器已停止")

def main():
    """主函数"""
    try:
        monitor = TrainingMonitorGUI()
        monitor.run()
    except Exception as e:
        print(f"监控器启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''