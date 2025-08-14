import torch
import torch.nn as nn


class Conv1dGRU(nn.Module):
    def __init__(self,
                 input_features,    # 输入特征数（如7）
                 pred_len,          # 预测长度（如2）
                 conv_out_channels=64,  # 卷积输出通道数（超参数）
                 conv_kernel_size=3,    # 卷积核大小（超参数）
                 gru_hidden_sizes=[128, 128, 256],  # GRU隐藏层维度（超参数）
                 dense1_units=256,    # 全连接层单元数（超参数）
                 dropout_rate=0.5):   # Dropout比率（超参数）
        super(Conv1dGRU, self).__init__()

        # 保存关键参数
        self.input_features = input_features
        self.pred_len = pred_len

        # 1. 卷积特征提取模块（依赖input_features）
        self.conv_block = nn.Sequential(
            # 卷积层输入通道数 = 输入特征数
            nn.Conv1d(in_channels=input_features,
                      out_channels=conv_out_channels,
                      kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=conv_out_channels),
            nn.Dropout(p=dropout_rate)
        )

        # 2. GRU时序处理模块（依赖卷积输出通道和GRU隐藏维度）
        self.gru_layers = nn.ModuleList()
        # 第一层GRU输入维度 = 卷积输出通道数
        input_size = conv_out_channels
        for hidden_size in gru_hidden_sizes:
            self.gru_layers.append(
                nn.GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=1,
                       batch_first=True)
            )
            input_size = hidden_size  # 下一层GRU输入 = 当前层输出

        # 对应GRU层的LayerNorm
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for hidden_size in gru_hidden_sizes
        ])

        # 3. 输出预测模块（依赖pred_len和input_features）
        self.output_block = nn.Sequential(
            nn.Linear(in_features=gru_hidden_sizes[-1],
                      out_features=dense1_units),
            nn.ReLU(),
            nn.LayerNorm(dense1_units),
            nn.Dropout(p=dropout_rate),
            # 输出维度 = 预测长度 × 特征数（用于reshape为[pred_len, input_features]）
            nn.Linear(in_features=dense1_units,
                      out_features=pred_len * input_features),
            nn.Tanh()
        )

        # 激活函数和dropout
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # 输入形状：[batch_size, seq_len, input_features]
        batch_size, seq_len, _ = x.shape

        # 1. 卷积模块处理
        x = x.permute(0, 2, 1)  # [batch_size, input_features, seq_len]
        x = self.conv_block(x)  # [batch_size, conv_out_channels, conv_seq_len]
        # 卷积后序列长度 = seq_len - conv_kernel_size + 1
        x = x.permute(0, 2, 1)  # [batch_size, conv_seq_len, conv_out_channels]

        # 2. GRU模块处理（带LayerNorm）
        for i, (gru, ln) in enumerate(zip(self.gru_layers, self.layer_norms)):
            x, _ = gru(x)  # [batch_size, current_seq_len, hidden_size]
            x = self.tanh(x)
            x = ln(x)      # LayerNorm：对每个样本的hidden_size维度归一化
            if i != len(self.gru_layers) - 1:  # 最后一层GRU后可保留dropout
                x = self.dropout(x)

        # 3. 输出模块处理
        x = x[:, -1, :]  # 取最后一个时间步特征 [batch_size, last_hidden_size]
        x = self.output_block(x)  # [batch_size, pred_len * input_features]
        # 重塑为 [batch_size, pred_len, input_features]
        x = x.view(batch_size, self.pred_len, self.input_features)

        return x


# 测试模型输入输出形状
if __name__ == "__main__":
    model = Conv1dGRU(
        input_features=7,    # 特征数
        pred_len=2
    )
    input_data = torch.randn(256, 10, 7)  # 模拟单批次输入
    output = model(input_data)
    print(f"输入形状: {input_data.shape}")    # 应输出：torch.Size([256, 10, 7])
    print(f"输出形状: {output.shape}")      # 应输出：torch.Size([256, 2, 7])
