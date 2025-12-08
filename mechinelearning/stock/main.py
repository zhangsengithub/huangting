import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from config import Config
from data_loader import StockDataLoader
from models import create_model
from trainer import StockTrainer
from predictor import StockPredictor
from utils import (
    setup_environment,
    create_results_directory,
    plot_feature_importance,
    plot_correlation_matrix,
    analyze_prediction_errors,
    generate_report
)


def main():
    """主函数"""
    print("=" * 60)
    print("基于深度学习的股票价格预测系统")
    print("=" * 60)

    # 1. 设置环境
    setup_environment()

    # 2. 加载配置
    config = Config()
    create_results_directory(config)

    # 3. 加载数据
    print("\n1. 加载数据...")
    data_loader = StockDataLoader(config)
    data_dict = data_loader.prepare_data()

    # 4. 创建模型
    print(f"\n2. 创建 {config.MODEL_TYPE} 模型...")
    model = create_model(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 训练模型
    print("\n3. 训练模型...")
    trainer = StockTrainer(model, config)

    train_data = (data_dict['X_train'], data_dict['y_train'])
    val_data = (data_dict['X_val'], data_dict['y_val'])

    history = trainer.train(train_data, val_data)

    # 6. 评估模型
    print("\n4. 评估模型...")
    predictor = StockPredictor(model, config, data_dict['scalers'])

    results = predictor.evaluate(
        data_dict['X_test'],
        data_dict['y_test']
    )

    # 7. 可视化结果
    print("\n5. 可视化结果...")
    predictor.plot_predictions(
        results['true_values'],
        results['predictions']
    )

    # 8. 置信区间分析
    coverage = predictor.plot_confidence_intervals(
        results['true_values'],
        results['predictions']
    )
    print(f"置信区间覆盖率: {coverage:.2%}")

    # 9. 回测交易策略
    print("\n6. 回测交易策略...")
    backtest_results = predictor.backtest_strategy(
        results['true_values'],
        results['predictions'],
        initial_capital=10000
    )

    # 10. 特征分析
    print("\n7. 特征分析...")
    plot_correlation_matrix(
        data_dict['X_train'][:1000].reshape(-1, data_dict['X_train'].shape[-1]),
        data_dict['feature_names']
    )

    # 使用样本数据计算特征重要性
    sample_size = min(100, len(data_dict['X_test']))
    X_sample = data_dict['X_test'][:sample_size]
    plot_feature_importance(
        data_dict['feature_names'],
        model,
        X_sample
    )

    # 11. 误差分析
    print("\n8. 误差分析...")
    error_stats = analyze_prediction_errors(
        results['true_values'],
        results['predictions']
    )

    # 12. 生成报告
    print("\n9. 生成报告...")
    report = generate_report(results, config)

    print("\n" + "=" * 60)
    print("模型训练和评估完成！")
    print("=" * 60)

    return {
        'model': model,
        'trainer': trainer,
        'predictor': predictor,
        'results': results,
        'backtest_results': backtest_results,
        'error_stats': error_stats,
        'report': report
    }


def predict_future(model_path="models/best_model.pth", days=5):
    """预测未来价格"""
    print("\n预测未来价格...")

    # 加载配置和模型
    config = Config()

    # 获取最新数据
    data_loader = StockDataLoader(config)
    data_dict = data_loader.prepare_data()

    # 加载训练好的模型
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 使用最新的序列进行预测
    latest_sequence = data_dict['X_test'][-1:]  # 取最后一个序列

    predictor = StockPredictor(model, config, data_dict['scalers'])

    # 递归预测未来多天
    current_seq = latest_sequence.copy()
    future_predictions = []

    for day in range(days):
        # 预测下一天
        prediction = predictor.predict(current_seq)
        future_predictions.append(prediction[0, 0])

        # 更新序列（这里简化处理，实际应用中需要更复杂的逻辑）
        # 注意：实际应用中需要考虑所有特征的更新

    # 反归一化
    price_scaler = data_dict['scalers']['price']
    future_prices = price_scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()

    print(f"\n未来 {days} 天预测价格:")
    for i, price in enumerate(future_prices, 1):
        print(f"  第 {i} 天: ${price:.2f}")

    return future_prices


if __name__ == "__main__":
    try:
        # 训练和评估模型
        results = main()

        # 预测未来（可选）
        # future_prices = predict_future(days=5)

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()