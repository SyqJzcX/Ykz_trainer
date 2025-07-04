# 基于 Pytorch 的深度学习模型训练工具

## 1. Trainer 类

**功能：**

1. 定义模型训练所需工具：
   1. 优化器
   2. 学习率调度器
   3. 梯度缩放器
   4. . . .
2. 定义模型训练所需超参数
   1. epoch
   2. batch_size
   3. learning_rate
   4. . . .
3. 加载模型训练所需数据
   1. 训练集
   2. 发展集（可选，不改变使用方法）
4. 保存模型检查点，包含：
   1. 模型参数
   2. 优化器状态
   3. 训练的轮数
   4. 训练集损失
   5. 发展集损失（没有发展集则没有）
   6. 发展及准确率（没有发展集则没有）
5. 模型训练方法
   1. 可分段训练，只需指明加载模型的编号即可
      1. 检查点命名规则：`checkpoint_*.pth`，其中`*`为加载的模型的训练总轮数
      2. 训练后按规则保存新检查点
   2. 通过继承模型自定义模型训练与验证方法
6. 训练数据图像绘制
   1. 训练集损失曲线
   2. 发展集损失曲线（没有发展集则没有）
   3. 发展集准确率曲线（没有发展集则没有）
7. 模型评估
8. 模型预测

**用法演示：**

```Python
# 定义模型参数
vocab_size = 21128  # 词汇表大小
hidden_size = 768  # 隐藏层大小
max_position_embeddings = 512  # 序列的最大长度
num_hidden_layers = 4  # Transformer 层的数量
num_attention_heads = 8  # 注意力头的数量
num_labels = len(tag_set)  # 标签数量
batch_size = 256  # 样本批量

# 初始化EncoderModel+CRF模型对象
model = EncoderModelWithCRF(
        vocab_size, 
        max_position_embeddings,
        num_hidden_layers, 
        hidden_size, 
        num_attention_heads, 
        num_labels
)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)  # 验证时不需要shuffle

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()
# 学习率调度器：验证集上的性能指标停止提升时，减小学习率来帮助模型跳出局部最优，继续优化
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5
)

# 训练器
trainer = Trainer(
    model=model,  # 模型
    train_dataloader=train_loader,  # 训练集加载器
    dev_dataloader=dev_loader,  # 验证集加载器
    scaler=scaler,  # 梯度缩放器
    optimizer=optimizer,  # 优化器：带动量的随机梯度下降 
    scheduler=scheduler,  # 学习率调度器：验证集上的性能指标停止提升时，减小学习率来帮助模型跳出局部最优，继续优化
    batch_size=batch_size,  # 样本批量
    model_path='./model/checkpoint/',  # 模型检查点保存路径
)
```
