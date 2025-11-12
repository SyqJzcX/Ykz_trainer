import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TextFormerDataset(Dataset):
    def __init__(self, csv_path, tokenizer, label2id, max_seq_length=48):
        """
        Args:
            csv_path: 数据集CSV文件路径（train.csv/dev.csv）
            tokenizer: BertTokenizer实例
            label2id: 标签→ID映射字典
            max_seq_length: 文本最大截断/补全长度（按你的分析设为48）
        """
        # 读取CSV文件（制表符分隔，第一行为表头）
        self.df = pd.read_csv(csv_path, sep='\t')
        # 过滤无效数据（确保text_a和label列存在且非空）
        self.df = self.df.dropna(
            subset=['text_a', 'label']).reset_index(drop=True)

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length

    def __len__(self):
        """返回数据集总条数"""
        return len(self.df)

    def __getitem__(self, idx):
        """按索引获取单条数据，完成Tokenize和标签转换"""
        # 提取文本和标签
        text_a = self.df.iloc[idx]['text_a'].strip()
        label = self.df.iloc[idx]['label'].strip()

        # Tokenize处理（生成input_ids、attention_mask、token_type_ids）
        encoding = self.tokenizer(
            text_a,
            max_length=self.max_seq_length,
            padding=False,  # 暂不padding，在collate_fn中统一处理（更高效）
            truncation=True,
            return_tensors=None  # 返回列表格式，方便后续pad
        )

        # 标签转换为ID（数值型）
        label_id = self.label2id[label]

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(encoding['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }


def collate_fn(batch_samples, pad_token_id=0):
    """
    批次数据处理：将单个样本拼接成批次，统一补全padding
    对应Paddle中的batchify_fn功能
    """
    # 提取批次中各字段的列表
    input_ids_list = [sample['input_ids'] for sample in batch_samples]
    attention_mask_list = [sample['attention_mask']
                           for sample in batch_samples]
    token_type_ids_list = [sample['token_type_ids']
                           for sample in batch_samples]
    labels_list = [sample['labels'] for sample in batch_samples]

    # 对序列进行padding（按批次中最长序列对齐）
    padded_input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    padded_attention_mask = pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0  # attention_mask pad 0
    )
    padded_token_type_ids = pad_sequence(
        token_type_ids_list, batch_first=True, padding_value=0  # token_type_ids pad 0
    )
    # 标签拼接成批次
    batch_labels = torch.stack(labels_list, dim=0)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'token_type_ids': padded_token_type_ids,
        'labels': batch_labels
    }


def create_dataloader(
    csv_path, tokenizer, label2id, max_seq_length=48, batch_size=256, mode='train'
):
    """
    构建DataLoader迭代器
    Args:
        mode: 'train'（训练集，打乱）/'dev'（验证集，不打乱）
    """
    # 构建数据集实例
    dataset = NewsDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_length=max_seq_length
    )

    # 构建Sampler（训练集打乱，验证集不打乱）
    shuffle = True if mode == 'train' else False

    # 构建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,  # 自定义批次处理函数
        num_workers=2,  # 多线程加载（根据CPU核心数调整，0为单线程）
        pin_memory=True  # 加速GPU训练（如果使用GPU）
    )

    return dataloader
