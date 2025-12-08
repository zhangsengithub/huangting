import torch


class Config:
    # 数据配置
    SYMBOL = "AAPL"  # 股票代码
    START_DATE = "2020-11-04"
    END_DATE = "2025-12-05"
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    # 特征配置
    SEQUENCE_LENGTH = 60  # 时间序列长度
    PREDICTION_HORIZON = 1  # 预测未来1天
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    # 技术指标参数
    MA_WINDOWS = [5, 10, 20, 50]
    RSI_WINDOW = 14
    MACD_SHORT = 12
    MACD_LONG = 26
    MACD_SIGNAL = 9
    BB_WINDOW = 20
    BB_STD = 2

    # 模型配置
    MODEL_TYPE = "transformer"  # "lstm", "gru", "transformer", "cnn_lstm"
    INPUT_SIZE = 16  # 自动计算
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DROPOUT = 0.2
    USE_BATCH_NORM = True

    # 训练配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15  # 早停耐心
    GRAD_CLIP = 1.0

    # 损失函数权重
    LOSS_WEIGHTS = {
        'mse': 1.0,
        'mae': 0.5,
        'direction': 0.3
    }

    # 文件路径
    MODEL_SAVE_PATH = "models/"
    LOG_PATH = "logs/"
    RESULTS_PATH = "results/"