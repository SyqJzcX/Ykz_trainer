import numpy as np
import pandas as pd
from learner.datasets.time_series_datasets import TimeSeriesDataset
from torch.utils.data import DataLoader
from learner.trainers.time_series_trainer import TimeSeriesTrainer
from learner.utils.plot import plot_ts
from learner.utils.split import split_three_ways
from sklearn.preprocessing import MinMaxScaler
from learner.models.conv1d_gru import Conv1dGRU
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch import nn
from learner.utils.lr import WarmUpCosineAnnealingLR
from learner.utils.plot import draw_loss, conf_matrix, draw_lr


def data():
    # 超参数指定
    data_path = './data/household_power_consumption.txt'
    seq_len = 5
    pred_len = 1
    batch_size = 3

    # 读取CSV数据
    df = pd.read_csv(
        data_path,
        sep=";",          # 指定分隔符为;
        header=0,      # 第一行为列名
        na_values=['?'],  # 缺失数据形式
    )

    # 合并日期和时间为时间戳（可选，仅用于查看，不影响模型输入）
    df["Timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format=r"%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"])
    # 将Timestamp设置为索引 - 这是关键步骤！
    df.set_index('Timestamp', inplace=True)

    # 缺失值处理
    # 1. Global_active_power 插值 (时间加权线性插值)
    df['Global_active_power'] = df['Global_active_power'].interpolate(
        method='time')

    # 2. Global_reactive_power 插值 (时间加权线性插值)
    df['Global_reactive_power'] = df['Global_reactive_power'].interpolate(
        method='time')

    # 3. Voltage 插值 - 多策略组合
    df['Voltage'] = (
        df['Voltage']
        .interpolate(method='time')  # 时间加权插值
        .fillna(df['Voltage'].rolling(window=30, min_periods=1).median())  # 滚动中位数
        .ffill()  # 前向填充
        .bfill()  # 后向填充
    )

    # 4. Global_intensity 插值 (基于物理关系计算)
    # 先尝试物理关系计算，再使用线性插值
    # df['Global_intensity_calc'] = df['Global_active_power'] * \
    #     1000 / (df['Voltage'] * np.sqrt(3))
    # df['Global_intensity'] = df['Global_intensity'].combine_first(
    #     df['Global_intensity_calc'])
    # df['Global_intensity'] = df['Global_intensity'].interpolate(method='time')
    # df = df.drop(columns=['Global_intensity_calc'])
    df['Global_intensity'] = df['Global_intensity'].interpolate(method='time')

    # 5-7. Sub_metering 系列插值 (前向填充 + 众数填充)
    for col in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
        # 前向填充
        # df[col] = df[col].fillna(method='ffill')
        df[col] = df[col].ffill()

        # 剩余缺失值使用众数填充
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            # df[col] = df[col].fillna(mode_val)
            df[col] = df[col].fillna(mode_val)

        # 转换为整数 (因为这些值本质上是离散的)
        df[col] = df[col].astype(int)

    print(df.dtypes)
    """
    Global_active_power      float64
    Global_reactive_power    float64
    Voltage                  float64
    Global_intensity         float64
    Sub_metering_1             int32
    Sub_metering_2             int32
    Sub_metering_3             int32
    dtype: object
    """
    print(df)

    # 检查缺失值
    print("处理后缺失值统计:")
    print(df.isnull().sum())
    """
    Global_active_power      0
    Global_reactive_power    0
    Voltage                  0
    Global_intensity         0
    Sub_metering_1           0
    Sub_metering_2           0
    Sub_metering_3           0
    dtype: int64
    """

    # 降采样为日粒度数据
    # 根据不同指标的特性选择合适的聚合函数
    df_daily = df.resample('D').agg({
        'Global_active_power': 'mean',         # 平均有功功率
        'Global_reactive_power': 'mean',       # 平均无功功率
        'Voltage': 'mean',                     # 平均电压
        'Global_intensity': 'mean',            # 平均电流强度
        'Sub_metering_1': 'sum',               # 子表1总消耗
        'Sub_metering_2': 'sum',               # 子表2总消耗
        'Sub_metering_3': 'sum'                # 子表3总消耗
    })

    # 检查降采样后的数据
    print("\n降采样后的日粒度数据:")
    print(df_daily.head())
    print(f"\n降采样前数据量: {len(df)} 条")
    print(f"降采样后数据量: {len(df_daily)} 条")

    # 数据归一化到[0,1]区间
    # 初始化MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 对所有特征列进行归一化
    # 注意：需要先将DataFrame转换为numpy数组，拟合后再转换回来
    scaled_data = scaler.fit_transform(df_daily)
    df_scaled = pd.DataFrame(
        scaled_data,
        columns=df_daily.columns,
        index=df_daily.index  # 保留原始索引
    )

    # 保存处理后的日粒度数据
    df_scaled.to_csv('./data/cleaned_daily_power_data.csv')

    # 提取特征列（假设我们用所有特征预测f1）
    # 输入特征：f1-f7（7个特征）
    # 目标特征：f1（第1个特征）
    features = df_scaled[["Global_active_power", "Global_reactive_power", "Voltage",
                         "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].values
    target = df_scaled["Global_active_power"].values  # 也可以用全部特征作为目标，这里以f1为例

    # 用自定义Dataset封装（假设用前5个时间步预测未来2个时间步的f1）
    dataset = TimeSeriesDataset(
        data=features,    # 输入所有特征
        seq_len=seq_len,        # 输入序列长度：用前5行数据
        pred_len=pred_len        # 预测长度：未来2行的f1
    )

    # 用DataLoader加载
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 查看第一个批次的结构
    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"第{batch_idx+1}个批次：")
        print("输入序列X的形状：", x.shape)  # (batch_size, seq_len, 特征数)
        print("目标序列y的形状：", y.shape)  # (batch_size, pred_len, 1)，因为只预测f1
        print("\n输入序列X（前2个样本，每个样本5个时间步，7个特征）：")
        print(x)
        print("\n目标序列y（前2个样本，每个样本2个时间步的f1）：")
        print(y)
        break  # 只看第一个批次

    plot_ts(df_scaled)


def train():
    # 超参数指定
    data_path = './data/cleaned_daily_power_data.csv'
    columns = ["Global_active_power", "Global_reactive_power", "Voltage",
               "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    seq_len = 150
    pred_len = 1
    batch_size = 64
    input_features = 7  # 特征数
    epoch_num = 50  # 总训练轮次
    learning_rate = 0.03
    warmup_epochs = 5
    warmup_start_factor = 0.01
    warmup_end_factor = 1.0
    cos_end_factor = 0.03

    # 读取CSV数据
    df = pd.read_csv(
        data_path,
        sep=",",          # 指定分隔符为;
        header=0,      # 第一行为列名
        na_values=['?'],  # 缺失数据形式
    )
    df.set_index('Timestamp', inplace=True)
    # 获取指定列名的索引
    index_label = df.columns.get_loc('Global_active_power')
    # print(index_label)

    # plot_ts(df, feature_cols=["Global_active_power"], max_samples=30)

    train_feats, val_feats = split_three_ways(
        df, train_ratio=0.8, val_ratio=0.2, test_ratio=0)
    
    plot_ts(val_feats, feature_cols=["Global_active_power"])
    print(len(val_feats))

    train_data = TimeSeriesDataset(
        data=train_feats[columns].values,  # 传入numpy数组
        seq_len=seq_len,  # 输入序列长度为12（1年）
        pred_len=pred_len  # 预测未来3个月
    )

    val_data = TimeSeriesDataset(
        data=val_feats[columns].values,  # 传入numpy数组
        seq_len=seq_len,  # 输入序列长度为12（1年）
        pred_len=pred_len  # 预测未来3个月
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False)

    # # 测试：打印一个批次的形状
    # for x, y in train_loader:
    #     print("输入序列形状:", x.shape)  # (batch_size, seq_len, 1)
    #     print("目标序列形状:", y.shape)  # (batch_size, pred_len, 1)
    #     print(x[0][0][0])
    #     break

    model = Conv1dGRU(input_features=input_features,  pred_len=pred_len)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    scheduler = WarmUpCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epoch_num,
        warmup_start_factor=warmup_start_factor,
        warmup_end_factor=warmup_end_factor,
        cos_end_factor=cos_end_factor
    )

    trainer = TimeSeriesTrainer(
        model=model,
        train_dataloader=train_loader,
        dev_dataloader=val_loader,  # 使用测试集作为发展集
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        model_path="./models/checkpoints/",  # 模型保存路径
        label_idx=index_label
    )

    trainer.fit(
        epoch_num=epoch_num,  # 训练轮次
        pretrain=0,  # 是否使用预训练参数
    )

    draw_loss("./models/checkpoints/")  # 绘制训练损失和准确率曲线
    
    eva = trainer.eval(val_loader)
    print(f"mean_squared_error: {eva['mean_squared_error']}")
    print(f"root_mean_squared_error: {eva['root_mean_squared_error']}")
    print(f"mean_absolute_error: {eva['mean_absolute_error']}")
    print(f"mean_absolute_percentage_error(%): {eva['mean_absolute_percentage_error(%)']}")
    print(f"r2_score: {eva['r2_score']}")


if __name__ == "__main__":
    # data()
    train()
