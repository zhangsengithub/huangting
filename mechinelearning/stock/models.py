import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionLayer(nn.Module):
    """多头注意力层"""

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape

        # 线性变换
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # 加权和
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, input_dim)
        out = self.fc_out(out)

        # 残差连接和层归一化
        out = self.dropout(out)
        out = self.layer_norm(x + out)

        return out


class TemporalBlock(nn.Module):
    """时间块（CNN + Attention）"""

    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.2, use_batch_norm=True):
        super(TemporalBlock, self).__init__()

        # 因果卷积
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=2,
            dilation=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        # 注意力
        self.attention = AttentionLayer(hidden_dim, num_heads, dropout)

        # 归一化
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 激活和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 残差连接
        self.downsample = nn.Conv1d(input_dim, hidden_dim, 1) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        residual = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        residual = self.downsample(residual).transpose(1, 2)  # [batch_size, seq_len, hidden_dim]

        # 因果卷积
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]

        # 注意力
        x = self.attention(x)

        # 残差连接
        x = self.layer_norm(x + residual)

        return x


class LSTMModel(nn.Module):
    """LSTM模型"""

    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_SIZE * 2,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT,
            batch_first=True
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE),
            nn.BatchNorm1d(config.HIDDEN_SIZE) if config.USE_BATCH_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.BatchNorm1d(config.HIDDEN_SIZE // 2) if config.USE_BATCH_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, config.PREDICTION_HORIZON)
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # 注意力机制
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )

        # 取最后一个时间步
        last_out = attn_out[:, -1, :]

        # 全连接层
        output = self.fc_layers(last_out)

        return output


class TransformerModel(nn.Module):
    """Transformer模型"""

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config

        # 位置编码
        self.pos_encoder = PositionalEncoding(config.INPUT_SIZE, config.DROPOUT)

        # 编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.INPUT_SIZE,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.HIDDEN_SIZE,
            dropout=config.DROPOUT,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config.NUM_LAYERS
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(config.INPUT_SIZE * config.SEQUENCE_LENGTH, config.HIDDEN_SIZE),
            nn.BatchNorm1d(config.HIDDEN_SIZE) if config.USE_BATCH_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.BatchNorm1d(config.HIDDEN_SIZE // 2) if config.USE_BATCH_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, config.PREDICTION_HORIZON)
        )

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码器
        encoded = self.transformer_encoder(x)

        # 展平
        batch_size = encoded.size(0)
        flattened = encoded.reshape(batch_size, -1)

        # 解码
        output = self.decoder(flattened)

        return output


class HybridModel(nn.Module):
    """混合模型（CNN + LSTM + Attention）"""

    def __init__(self, config):
        super(HybridModel, self).__init__()
        self.config = config

        # CNN层提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(config.INPUT_SIZE, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

        # LSTM层捕获时间依赖
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )

        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(config.HIDDEN_SIZE, 1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE),
            nn.BatchNorm1d(config.HIDDEN_SIZE) if config.USE_BATCH_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, config.PREDICTION_HORIZON)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size = x.size(0)

        # CNN处理
        cnn_input = x.transpose(1, 2)  # [batch, features, seq_len]
        cnn_out = self.cnn(cnn_input)  # [batch, 128, seq_len]
        cnn_out = cnn_out.transpose(1, 2)  # [batch, seq_len, 128]

        # LSTM处理
        lstm_out, _ = self.lstm(cnn_out)  # [batch, seq_len, hidden_size*2]

        # 注意力机制
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden_size*2]

        # 最终预测
        output = self.fc(context_vector)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # 如果维度是奇数，最后一维只填充0
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # 形状: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # 确保位置编码维度与输入匹配
        if x.size(2) != self.pe.size(2):
            print(f"警告: 位置编码维度({self.pe.size(2)})与输入维度({x.size(2)})不匹配")
            print(f"调整位置编码维度从 {self.pe.size(2)} 到 {x.size(2)}")

            # 如果输入维度较小，截取位置编码
            if x.size(2) < self.pe.size(2):
                pe_adjusted = self.pe[:, :, :x.size(2)]
            # 如果输入维度较大，扩展位置编码
            else:
                pe_adjusted = torch.zeros(1, self.pe.size(1), x.size(2), device=x.device)
                pe_adjusted[:, :, :self.pe.size(2)] = self.pe

            x = x + pe_adjusted[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)


def create_model(config):
    """创建模型并确保参数兼容"""
    print(f"\n=== 创建模型 ===")

    # 获取实际的特征维度
    if hasattr(config, 'INPUT_SIZE') and config.INPUT_SIZE:
        input_size = config.INPUT_SIZE
    else:
        # 默认值或从数据中获取
        input_size = 6  # 你的特征数
        config.INPUT_SIZE = input_size

    # 确保注意力头数有效
    if not hasattr(config, 'NUM_HEADS') or not config.NUM_HEADS:
        config.NUM_HEADS = 8  # 默认值

    print(f"输入维度: {input_size}, 头数: {config.NUM_HEADS}")

    # 检查并调整维度兼容性
    if input_size % config.NUM_HEADS != 0:
        print(f"维度不兼容: {input_size} % {config.NUM_HEADS} != 0")

        # 选择调整策略
        adjustment_strategy = getattr(config, 'DIM_ADJUSTMENT', 'input')  # 'input' 或 'heads'

        if adjustment_strategy == 'input':
            # 调整输入维度
            original_input = input_size
            input_size = ((input_size + config.NUM_HEADS - 1) // config.NUM_HEADS) * config.NUM_HEADS
            config.INPUT_SIZE = input_size
            print(f"调整输入维度从 {original_input} 到 {input_size}")
        else:
            # 调整头数
            original_heads = config.NUM_HEADS
            # 寻找能整除 input_size 的最大头数
            for i in range(min(config.NUM_HEADS, input_size), 0, -1):
                if input_size % i == 0:
                    config.NUM_HEADS = i
                    break
            print(f"调整头数从 {original_heads} 到 {config.NUM_HEADS}")

    # 确保其他参数存在
    if not hasattr(config, 'FFN_DIM'):
        config.FFN_DIM = 2048

    if not hasattr(config, 'NUM_LAYERS'):
        config.NUM_LAYERS = 3

    if not hasattr(config, 'DROPOUT'):
        config.DROPOUT = 0.1

    if not hasattr(config, 'ACTIVATION'):
        config.ACTIVATION = 'relu'

    if not hasattr(config, 'OUTPUT_SIZE'):
        config.OUTPUT_SIZE = 1

    print(f"最终参数: dim={config.INPUT_SIZE}, heads={config.NUM_HEADS}, layers={config.NUM_LAYERS}")

    # 创建模型
    model_class = TransformerModel
    return model_class(config).to(config.DEVICE)