from sklearn.model_selection import train_test_split


def split_three_ways(
    features,  # 数据
    train_ratio=0.7,  # 训练集比例
    val_ratio=0.15,  # 验证集比例
    test_ratio=0.15,  # 测试集比例
    shuffle=False  # 是否打乱
):
    """划分训练集、验证集、测试集"""
    # 检查比例是否合法
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为1"

    # 划分训练集和临时集（验证+测试）
    train_feats, temp_feats = train_test_split(
        features,
        train_size=train_ratio,
        shuffle=False  # 关键：保持时间顺序
    )

    # 若验证集或测试集比例为 0，直接返回训练集与验证集/测试集
    if val_ratio == 0 or test_ratio == 0:
        return train_feats, temp_feats

    else:
        # 从临时集中划分验证集和测试集
        # 临时集中验证集占比 = val_ratio / (val_ratio + test_ratio)
        val_feats, test_feats = train_test_split(
            temp_feats,
            train_size=val_ratio / (val_ratio + test_ratio),
            shuffle=False  # 再次保持时间顺序
        )

        return train_feats, val_feats, test_feats
