import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import talib
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class StockDataLoader:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.target_feature_count = 16  # 目标特征数量

    def fetch_data(self) -> pd.DataFrame:
        """获取股票数据"""
        print(f"正在下载 {self.config.SYMBOL} 数据...")
        stock = yf.download(
            self.config.SYMBOL,
            start=self.config.START_DATE,
            end=self.config.END_DATE,
            progress=False
        )

        if stock.empty:
            raise ValueError(f"无法获取 {self.config.SYMBOL} 数据")
        print(stock.head())
        print(f"数据形状: {stock.shape}")
        return stock

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标 - 确保最终有16个特征"""
        data = df.copy()

        print("=== 开始计算技术指标 ===")
        print(f"输入数据形状: {data.shape}")
        print(f"输入数据列: {data.columns.tolist()}")

        # === 1. 检查必要列是否存在 ===
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        available_cols = {}

        for col in required_cols:
            if col in data.columns:
                available_cols[col] = col
                print(f"✓ 找到列: {col}")
            else:
                print(f"✗ 未找到列: {col}")
                # 尝试查找相似的列
                for actual_col in data.columns:
                    if col.lower() in actual_col.lower():
                        available_cols[col] = actual_col
                        print(f"  使用替代列: {actual_col}")
                        break

        # 如果Close列不存在，无法继续
        if 'Close' not in available_cols:
            print("错误: 没有找到收盘价列，无法计算技术指标")
            return data

        # 简化列名引用
        close_col = available_cols.get('Close', 'Close')
        high_col = available_cols.get('High', 'High')
        low_col = available_cols.get('Low', 'Low')
        open_col = available_cols.get('Open', 'Open')
        volume_col = available_cols.get('Volume', 'Volume')

        # === 2. 计算技术指标 ===
        print("\n=== 计算技术指标 ===")

        # 确保我们有足够的特征来达到16个
        technical_features = []

        # 2.1 基础价格特征 (5个)
        if close_col in data.columns:
            data['Close'] = data[close_col]
            technical_features.append('Close')
            print("✓ 添加 Close")

        if high_col in data.columns:
            data['High'] = data[high_col]
            technical_features.append('High')
            print("✓ 添加 High")

        if low_col in data.columns:
            data['Low'] = data[low_col]
            technical_features.append('Low')
            print("✓ 添加 Low")

        if open_col in data.columns:
            data['Open'] = data[open_col]
            technical_features.append('Open')
            print("✓ 添加 Open")

        if volume_col in data.columns:
            data['Volume'] = data[volume_col]
            technical_features.append('Volume')
            print("✓ 添加 Volume")

        # 2.2 移动平均线 (3个)
        if close_col in data.columns:
            ma_windows = [5, 10, 20]
            for window in ma_windows:
                ma_col = f'MA_{window}'
                data[ma_col] = data[close_col].rolling(window=window, min_periods=1).mean()
                technical_features.append(ma_col)
                print(f"✓ 计算 {ma_col}")

        # 2.3 价格动量指标 (5个)
        if close_col in data.columns:
            # 相对强弱指数 (RSI)
            try:
                data['RSI'] = talib.RSI(data[close_col], timeperiod=14)
                technical_features.append('RSI')
                print("✓ 计算 RSI")
            except:
                data['RSI'] = 50.0  # 默认值

            # 移动平均收敛散度 (MACD)
            try:
                macd, macdsignal, macdhist = talib.MACD(data[close_col])
                data['MACD'] = macd
                data['MACD_Signal'] = macdsignal
                data['MACD_Hist'] = macdhist
                technical_features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
                print("✓ 计算 MACD 指标")
            except:
                data['MACD'] = 0
                data['MACD_Signal'] = 0
                data['MACD_Hist'] = 0
                technical_features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])

        # 2.4 波动率指标 (2个)
        if high_col in data.columns and low_col in data.columns and close_col in data.columns:
            # 布林带
            try:
                upper, middle, lower = talib.BBANDS(data[close_col])
                data['BB_Upper'] = upper
                data['BB_Lower'] = lower
                technical_features.extend(['BB_Upper', 'BB_Lower'])
                print("✓ 计算布林带")
            except:
                data['BB_Upper'] = data[close_col]
                data['BB_Lower'] = data[close_col]
                technical_features.extend(['BB_Upper', 'BB_Lower'])

        # 2.5 成交量指标 (1个)
        if volume_col in data.columns:
            data['Volume_MA5'] = data[volume_col].rolling(window=5, min_periods=1).mean()
            technical_features.append('Volume_MA5')
            print("✓ 计算 Volume_MA5")

        # 检查当前特征数量
        current_features = len(technical_features)
        print(f"\n当前特征数量: {current_features}")

        # 如果特征不足16个，添加一些衍生特征
        if current_features < self.target_feature_count:
            print(f"特征不足 {self.target_feature_count} 个，添加衍生特征...")

            # 价格变化率
            if close_col in data.columns:
                for period in [1, 3, 5]:
                    feature_name = f'Return_{period}d'
                    data[feature_name] = data[close_col].pct_change(periods=period)
                    technical_features.append(feature_name)
                    print(f"✓ 添加 {feature_name}")

            # 价格范围
            if high_col in data.columns and low_col in data.columns:
                data['Price_Range'] = (data[high_col] - data[low_col]) / data[low_col]
                technical_features.append('Price_Range')
                print("✓ 添加 Price_Range")

            # 收盘-开盘关系
            if close_col in data.columns and open_col in data.columns:
                data['Close_Open_Ratio'] = data[close_col] / data[open_col]
                technical_features.append('Close_Open_Ratio')
                print("✓ 添加 Close_Open_Ratio")

        # 再次检查特征数量
        current_features = len(technical_features)
        print(f"添加衍生特征后数量: {current_features}")

        # 如果仍然不足，用随机噪声填充（作为最后的手段）
        if current_features < self.target_feature_count:
            additional_needed = self.target_feature_count - current_features
            print(f"仍然需要 {additional_needed} 个特征，使用随机特征填充")

            for i in range(additional_needed):
                feature_name = f'Feature_{i + 1}'
                data[feature_name] = np.random.randn(len(data))
                technical_features.append(feature_name)
                print(f"✓ 添加随机特征 {feature_name}")

        # 如果特征超过16个，只选择前16个
        if len(technical_features) > self.target_feature_count:
            print(f"特征过多 ({len(technical_features)}个)，只保留前{self.target_feature_count}个")
            technical_features = technical_features[:self.target_feature_count]

        # === 3. 选择最终的特征 ===
        final_features = technical_features
        data = data[final_features]

        # === 4. 处理NaN值 ===
        print("\n=== 处理NaN值 ===")
        nan_before = data.isna().sum().sum()
        if nan_before > 0:
            print(f"处理前有 {nan_before} 个NaN值")

        # 填充NaN值
        data = data.fillna(method='ffill')
        data = data.fillna(method='bfill')
        data = data.fillna(0)

        nan_after = data.isna().sum().sum()
        print(f"处理后剩余 {nan_after} 个NaN值")

        # === 5. 输出结果 ===
        print(f"\n=== 技术指标计算完成 ===")
        print(f"输出数据形状: {data.shape}")
        print(f"特征数量: {len(data.columns)}")
        print(f"目标特征数: {self.target_feature_count}")

        print("\n所有特征列:")
        for i, col in enumerate(data.columns):
            print(f"  {i + 1:2d}. {col}")

        return data

    def create_sequences(self, data: np.ndarray, seq_length: int, prediction_horizon: int = 1):
        """
        创建时间序列数据

        Args:
            data: 输入数据，形状为 (n_samples, n_features)
            seq_length: 序列长度（时间步数）
            prediction_horizon: 预测步长（未来多少步）

        Returns:
            X: 输入序列，形状为 (n_sequences, seq_length, n_features)
            y: 目标值，形状为 (n_sequences, prediction_horizon)
        """
        X, y = [], []

        n_samples = len(data)

        # 检查数据长度是否足够
        if n_samples < seq_length + prediction_horizon:
            print(f"警告: 数据长度({n_samples})小于序列长度({seq_length}) + 预测步长({prediction_horizon})")
            return np.array(X), np.array(y)

        # 创建序列
        for i in range(n_samples - seq_length - prediction_horizon + 1):
            # 输入序列：从i到i+seq_length-1
            seq_x = data[i:(i + seq_length), :]

            # 目标值：从i+seq_length到i+seq_length+prediction_horizon-1
            # 假设最后一列是收盘价
            seq_y = data[(i + seq_length):(i + seq_length + prediction_horizon), -1]

            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)

        print(f"创建序列完成: X.shape={X.shape}, y.shape={y.shape}")
        print(f"序列数: {len(X)}, 序列长度: {seq_length}, 特征数: {X.shape[2]}")
        print(f"预测步长: {prediction_horizon}")

        return X, y

    def prepare_data(self) -> Dict:
        """准备训练、验证、测试数据 - 确保最终有16个特征"""
        # 获取原始数据
        raw_data = self.fetch_data()

        print(f"原始数据形状: {raw_data.shape}")

        # 扁平化多级列索引
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = [col[0] for col in raw_data.columns]

        print(f"扁平化后列名: {raw_data.columns.tolist()}")

        # 计算技术指标
        data_with_features = self.calculate_technical_indicators(raw_data)

        print(f"\n技术指标处理后:")
        print(f"数据形状: {data_with_features.shape}")
        print(f"数据列: {data_with_features.columns.tolist()}")

        # 确保特征数量为16
        if data_with_features.shape[1] != self.target_feature_count:
            print(f"警告: 特征数量不是{self.target_feature_count}，当前为{data_with_features.shape[1]}")
            # 强制调整
            if data_with_features.shape[1] > self.target_feature_count:
                data_with_features = data_with_features.iloc[:, :self.target_feature_count]
                print(f"截取前{self.target_feature_count}个特征")
            else:
                # 添加随机特征补足
                needed = self.target_feature_count - data_with_features.shape[1]
                for i in range(needed):
                    data_with_features[f'Padding_{i}'] = 0
                print(f"添加{needed}个零填充特征")

        # 确认最终特征数
        final_feature_count = data_with_features.shape[1]
        print(f"最终特征数量: {final_feature_count}")

        # 提取特征数据
        feature_data = data_with_features.values

        print(f"\n特征数据形状: {feature_data.shape}")

        # 标准化特征
        feature_scaler = StandardScaler()
        scaled_data = feature_scaler.fit_transform(feature_data)

        print(f"标准化后数据形状: {scaled_data.shape}")

        # 关键：更新配置中的输入维度
        self.config.INPUT_SIZE = final_feature_count
        print(f"更新 config.INPUT_SIZE = {final_feature_count}")

        # 创建序列
        X, y = self.create_sequences(
            scaled_data,
            self.config.SEQUENCE_LENGTH,
            self.config.PREDICTION_HORIZON
        )

        # 检查序列是否创建成功
        if len(X) == 0 or len(y) == 0:
            print("错误: 无法创建有效的序列")
            return {}

        # 划分数据集
        train_size = int(len(X) * (1 - self.config.TEST_SIZE - self.config.VAL_SIZE))
        val_size = int(len(X) * self.config.VAL_SIZE)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

        # 保存scaler
        self.scalers = {
            'feature': feature_scaler
        }

        print(f"\n最终数据集:")
        print(f"训练集: {X_train.shape}, {y_train.shape}")
        print(f"验证集: {X_val.shape}, {y_val.shape}")
        print(f"测试集: {X_test.shape}, {y_test.shape}")
        print(f"实际输入维度: {final_feature_count}")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'scalers': self.scalers,
            'feature_names': data_with_features.columns.tolist(),
            'actual_input_size': final_feature_count
        }