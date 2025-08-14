import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TimeSeriesDataset(Dataset):
    """通用时间序列数据集，用于多步预测任务"""

    def __init__(
        self,
        data,  # 输入数据（可以是numpy数组、CSV文件路径或pandas DataFrame）
        seq_len=10,  # 输入序列长度（用前seq_len个时间步作为特征）
        pred_len=1,  # 预测序列长度（需要预测未来pred_len个时间步）
        feature_col=None  # 特征列名（仅当data为CSV路径时需要，指定要预测的列）
    ):
        # 加载数据并转换为numpy数组
        if isinstance(data, str):  # 从CSV文件加载
            df = pd.read_csv(data)
            if feature_col is None:
                raise ValueError("当data为文件路径时，必须指定feature_col")
            self.data = df[feature_col].values.astype(np.float64)
        elif isinstance(data, pd.DataFrame):  # 从DataFrame加载
            self.data = data.values.astype(np.float64)
        elif isinstance(data, np.ndarray):  # 从numpy数组加载
            self.data = data.astype(np.float64)
        else:
            raise TypeError("data必须是numpy数组、CSV路径或pandas DataFrame")

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # 可生成的样本数 = 总长度 - 输入序列长度 - 预测序列长度 + 1
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 输入序列：[idx, idx+seq_len)
        x = self.data[idx: idx + self.seq_len]
        # 目标序列：[idx+seq_len, idx+seq_len+pred_len)
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

        # 转换为PyTorch Tensor，时间序列数据在 PyTorch 中的标准输入形状为：(batch_size, seq_len, num_features)
        return torch.tensor(x), torch.tensor(y)
