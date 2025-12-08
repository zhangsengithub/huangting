import matplotlib
matplotlib.use('TkAgg')  # 明确设置为非交互后端
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

# 强制使用英文字体避免乱码
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

def quick_comparison():
    """快速对比函数 - 完全避免中文"""
    
    # 示例数据 - 您可以用真实数据替换
    sample_results = [
        {
            'question': 'What is AI?',
            'before': 'AI is a field of computer science.',
            'after': 'Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.'
        },
        {
            'question': 'Write about spring',
            'before': 'Spring is nice.',
            'after': 'Spring brings new life, with flowers blooming and birds singing after winter.'
        },
        {
            'question': 'Calculate 2+3*4',
            'before': 'The answer is 20',
            'after': 'Following order of operations: 3*4=12, then 2+12=14. Answer is 14.'
        }
    ]
    
    # 创建简单的对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    for result in sample_results:
        table_data.append([
            result['question'],
            result['before'][:80] + '...' if len(result['before']) > 80 else result['before'],
            result['after'][:80] + '...' if len(result['after']) > 80 else result['after']
        ])
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=['Question', 'Before Training', 'After Training'],
        cellLoc='left',
        loc='center',
        colWidths=[0.25, 0.35, 0.4]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置样式
    for i in range(3):  # 标题行
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Training Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('simple_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 同时保存文本版本
    with open('simple_comparison.txt', 'w') as f:
        f.write("Training Comparison Results\n")
        f.write("="*50 + "\n")
        for result in sample_results:
            f.write(f"\nQuestion: {result['question']}\n")
            f.write(f"Before: {result['before']}\n")
            f.write(f"After: {result['after']}\n")
            f.write("-"*30 + "\n")
    
    print("✅ 对比图已生成: simple_comparison.png")
    print("✅ 文本报告已生成: simple_comparison.txt")

if __name__ == "__main__":
    quick_comparison()
