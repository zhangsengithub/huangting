import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class StockPredictor:
    def __init__(self, model, config, scalers):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.scalers = scalers
        self.model.eval()

    def predict(self, X):
        """批量预测"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        # 预测
        predictions = self.predict(X_test)

        # 反归一化
        print(self.scalers)
        price_scaler = self.scalers['price']
        y_test_original = price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        predictions_original = price_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # 计算指标
        mse = mean_squared_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, predictions_original)
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        r2 = r2_score(y_test_original, predictions_original)

        # 方向准确率
        direction_true = np.sign(np.diff(y_test_original))
        direction_pred = np.sign(np.diff(predictions_original))
        direction_accuracy = np.mean(direction_true == direction_pred)

        # 夏普比率（假设策略）
        returns_true = np.diff(y_test_original) / y_test_original[:-1]
        returns_pred = np.diff(predictions_original) / predictions_original[:-1]

        # 计算信号（基于预测）
        signals = np.where(predictions_original[1:] > predictions_original[:-1], 1, -1)
        strategy_returns = returns_true * signals[:-1]

        sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Direction_Accuracy': direction_accuracy,
            'Sharpe_Ratio': sharpe_ratio,
            '预测数量': len(predictions_original)
        }

        # 打印结果
        print("\n" + "=" * 50)
        print("模型评估结果")
        print("=" * 50)
        for key, value in metrics.items():
            if key != '预测数量':
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        return {
            'predictions': predictions_original,
            'true_values': y_test_original,
            'metrics': metrics
        }

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """计算夏普比率"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252  # 年化无风险利率
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def plot_predictions(self, y_true, y_pred, title="预测结果对比"):
        """绘制预测对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 时间序列对比
        axes[0, 0].plot(y_true, label='真实值', alpha=0.7)
        axes[0, 0].plot(y_pred, label='预测值', alpha=0.7)
        axes[0, 0].set_title('价格预测对比')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('价格')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 散点图
        axes[0, 1].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0, 1].set_title('真实值 vs 预测值')
        axes[0, 1].set_xlabel('真实价格')
        axes[0, 1].set_ylabel('预测价格')
        axes[0, 1].grid(True)

        # 残差分布
        residuals = y_true - y_pred
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_title('残差分布')
        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('频率')
        axes[1, 0].grid(True)

        # 误差累积分布
        error_percentage = np.abs(residuals / y_true) * 100
        axes[1, 1].hist(error_percentage, bins=50, alpha=0.7, edgecolor='black', cumulative=True)
        axes[1, 1].set_title('误差累积分布')
        axes[1, 1].set_xlabel('百分比误差 (%)')
        axes[1, 1].set_ylabel('累积频率')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.config.RESULTS_PATH}/predictions_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confidence_intervals(self, y_true, y_pred, confidence=0.95):
        """绘制置信区间"""
        residuals = y_true - y_pred
        std_residual = np.std(residuals)

        # 计算置信区间
        z_score = stats.norm.ppf((1 + confidence) / 2)
        upper_bound = y_pred + z_score * std_residual
        lower_bound = y_pred - z_score * std_residual

        # 计算覆盖率
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))

        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='真实值', color='blue', alpha=0.7)
        plt.plot(y_pred, label='预测值', color='red', alpha=0.7)
        plt.fill_between(range(len(y_pred)), lower_bound, upper_bound,
                         alpha=0.3, color='gray', label=f'{confidence * 100:.0f}% 置信区间')
        plt.title(f'预测结果与置信区间（覆盖率: {coverage:.2%}）')
        plt.xlabel('时间步')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.RESULTS_PATH}/confidence_intervals.png", dpi=300, bbox_inches='tight')
        plt.show()

        return coverage

    def generate_trading_signals(self, predictions, threshold=0.001):
        """生成交易信号"""
        signals = []

        for i in range(1, len(predictions)):
            # 基于预测价格变化生成信号
            predicted_return = (predictions[i] - predictions[i - 1]) / predictions[i - 1]

            if predicted_return > threshold:
                signals.append(1)  # 买入
            elif predicted_return < -threshold:
                signals.append(-1)  # 卖出
            else:
                signals.append(0)  # 持有

        return signals

    def backtest_strategy(self, true_prices, predictions, initial_capital=10000):
        """回测交易策略"""
        signals = self.generate_trading_signals(predictions)

        capital = initial_capital
        position = 0
        trades = []
        equity_curve = [initial_capital]

        # 交易成本
        transaction_cost = 0.001  # 0.1%

        for i in range(1, len(true_prices)):
            current_price = true_prices[i]
            signal = signals[i - 1]

            if signal == 1 and position == 0:  # 买入
                # 计算可买数量
                position = (capital * (1 - transaction_cost)) / current_price
                capital = 0
                trades.append({
                    'day': i,
                    'action': 'BUY',
                    'price': current_price,
                    'position': position
                })

            elif signal == -1 and position > 0:  # 卖出
                capital = position * current_price * (1 - transaction_cost)
                position = 0
                trades.append({
                    'day': i,
                    'action': 'SELL',
                    'price': current_price,
                    'capital': capital
                })

            # 计算当前权益
            current_equity = capital + position * current_price
            equity_curve.append(current_equity)

        # 计算最终结果
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_capital) / initial_capital * 100
        annual_return = ((1 + total_return / 100) ** (252 / len(equity_curve)) - 1) * 100

        # 计算最大回撤
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak * 100
        max_drawdown = np.min(drawdown)

        # 计算波动率
        returns = np.diff(equity_array) / equity_array[:-1]
        volatility = np.std(returns) * np.sqrt(252) * 100

        results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': len([t for t in trades if t['action'] in ['BUY', 'SELL']]),
            'trades': trades,
            'equity_curve': equity_curve,
            'drawdown': drawdown
        }

        self.plot_backtest_results(results, true_prices)

        return results

    def plot_backtest_results(self, results, true_prices):
        """绘制回测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 权益曲线
        axes[0, 0].plot(results['equity_curve'])
        axes[0, 0].set_title('权益曲线')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('权益')
        axes[0, 0].grid(True)

        # 回撤曲线
        axes[0, 1].plot(results['drawdown'])
        axes[0, 1].fill_between(range(len(results['drawdown'])), results['drawdown'], 0, alpha=0.3)
        axes[0, 1].set_title('回撤曲线')
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('回撤 (%)')
        axes[0, 1].grid(True)

        # 价格与交易点
        axes[1, 0].plot(true_prices, label='价格', alpha=0.7)

        # 标记买卖点
        buy_points = [t['day'] for t in results['trades'] if t['action'] == 'BUY']
        sell_points = [t['day'] for t in results['trades'] if t['action'] == 'SELL']

        if buy_points:
            axes[1, 0].scatter(buy_points, [true_prices[i] for i in buy_points],
                               color='green', s=50, label='买入', zorder=5)
        if sell_points:
            axes[1, 0].scatter(sell_points, [true_prices[i] for i in sell_points],
                               color='red', s=50, label='卖出', zorder=5)

        axes[1, 0].set_title('价格与交易信号')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('价格')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 交易统计
        axes[1, 1].axis('off')
        stats_text = f"""
        初始资金: ${results['initial_capital']:,.2f}
        最终权益: ${results['final_equity']:,.2f}
        总收益率: {results['total_return']:.2f}%
        年化收益率: {results['annual_return']:.2f}%
        最大回撤: {results['max_drawdown']:.2f}%
        波动率: {results['volatility']:.2f}%
        交易次数: {results['num_trades']}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12,
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{self.config.RESULTS_PATH}/backtest_results.png", dpi=300, bbox_inches='tight')
        plt.show()