import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def setup_environment():
    """设置环境"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


def create_results_directory(config):
    """创建结果目录"""
    import os
    directories = [
        config.MODEL_SAVE_PATH,
        config.LOG_PATH,
        config.RESULTS_PATH
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")


def plot_feature_importance(feature_names, model, X_sample):
    """绘制特征重要性"""
    import torch

    # 获取梯度
    X_tensor = torch.FloatTensor(X_sample).requires_grad_(True)

    # 计算梯度
    model.eval()
    output = model(X_tensor)
    output.mean().backward()

    # 计算特征重要性
    gradients = X_tensor.grad.abs().mean(dim=0).mean(dim=0).cpu().numpy()

    # 排序
    indices = np.argsort(gradients)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(min(20, len(indices))), gradients[indices[:20]])
    plt.xticks(range(min(20, len(indices))),
               [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
    plt.title('Top 20 特征重要性（基于梯度）')
    plt.xlabel('特征')
    plt.ylabel('平均梯度绝对值')
    plt.tight_layout()
    plt.savefig(f"results/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(data, feature_names):
    """绘制特征相关性矩阵"""
    import seaborn as sns

    plt.figure(figsize=(15, 12))
    correlation_matrix = pd.DataFrame(data, columns=feature_names).corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})

    plt.title('特征相关性矩阵', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()


def analyze_prediction_errors(y_true, y_pred):
    """分析预测误差"""
    errors = y_true - y_pred
    error_stats = {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'skewness': pd.Series(errors).skew(),
        'kurtosis': pd.Series(errors).kurtosis(),
        'mape': np.mean(np.abs(errors / y_true)) * 100
    }

    # 误差分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--')
    axes[0].set_title('误差分布')
    axes[0].set_xlabel('误差')
    axes[0].set_ylabel('频率')

    import scipy.stats as stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('QQ图')

    plt.tight_layout()
    plt.savefig(f"results/error_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    return error_stats


def generate_report(results, config):
    """生成详细报告"""
    report = f"""
    {'=' * 60}
    股票预测系统报告
    {'=' * 60}

    股票代码: {config.SYMBOL}
    模型类型: {config.MODEL_TYPE}
    序列长度: {config.SEQUENCE_LENGTH}
    预测周期: {config.PREDICTION_HORIZON}天

    {'-' * 60}
    性能指标:
    {'-' * 60}
    """

    metrics = results['metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            report += f"    {key}: {value:.4f}\n"
        else:
            report += f"    {key}: {value}\n"

    report += f"""
    {'-' * 60}
    训练配置:
    {'-' * 60}
    """

    for key, value in vars(config).items():
        if not key.startswith('_'):
            report += f"    {key}: {value}\n"

    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{config.RESULTS_PATH}/report_{timestamp}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存到: {report_file}")

    return report