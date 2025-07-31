import torch
from torch import nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from glob import glob


class Trainer:
    """模型训练类"""

    def __init__(
        self,
        model,  # 模型
        train_dataloader,  # 训练集加载器
        dev_dataloader=None,  # 发展集加载器：有无发展集均可
        scaler=GradScaler(),  # 梯度缩放器
        optimizer=optim.AdamW,  # 优化器
        # 学习率调度器：验证集上的性能指标停止提升时，减小学习率来帮助模型跳出局部最优，继续优化
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        batch_size=512,  # 样本批量
        model_path='./model/checkpoint/',  # 模型检查点保存路径
    ):
        self.batch_size = batch_size
        self.model_path = model_path

        # 检查 GPU 是否可用
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # 如果有可用的 GPU，则使用 GPU
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")  # 如果没有可用的 GPU，则使用 CPU
            print("使用 CPU")

        # 定义属性
        self.model = model.to(self.device)

        self.optimizer = optimizer(model.parameters(), lr=5e-5)
        self.scheduler = scheduler(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader

        if self.dev_dataloader is None and self.scheduler is optim.lr_scheduler.ReduceLROnPlateau:
            print("没有发展集，学习率调度器自动替换为 StepLR")
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1,
                gamma=0.1
            )

        self.scaler = scaler

    def fit(
        self,
        epoch_num,  # 训练轮次数
        pretrain=0,  # 预训练模型编号（0 代表没有）
    ):
        """模型训练"""
        # 加载预训练模型
        if pretrain:
            checkpoint = torch.load(
                # 加载保存的检查点
                self.model_path + f'checkpoint_{pretrain}.pth', map_location=self.device)

            self.model.load_state_dict(
                checkpoint['model_state_dict'])  # 加载模型参数
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])  # 加载优化器状态
            # 获取之前训练的轮数（理论上 start_epoch == pretrain）
            start_epoch = checkpoint['epoch'][-1]
            pre_loss = checkpoint['train_loss'][-1]  # 获取之前训练的损失

            print(
                f'加载预训练模型: {pretrain}，已训练轮数: {start_epoch}，先前训练集损失: {pre_loss}')

            if self.dev_dataloader is not None:
                self.dev_loss = checkpoint['dev_loss'][-1]
                pre_acc = checkpoint['dev_acc'][-1]  # 获取之前训练的准确率
                print(f'先前发展集损失: {self.dev_loss}，先前发展集准确率: {pre_acc}')

        else:
            print('无预训练模型，从零开始训练 . . .')
            start_epoch = 0

        # 迭代训练
        epoch_list = []
        train_loss_list = []  # 训练集损失列表
        dev_loss_list = []  # 发展集损失列表
        dev_acc_list = []  # 发展集准确率列表
        for epoch in range(epoch_num):
            # 该epoch的
            train_loss = 0.0  # 训练集损失

            # 遍历训练集，训练模型参数
            self.model.train()  # 设置模型为训练模式
            # 获取每个 batch 的 loss
            for loss in train_model():
                train_loss += loss.data.item()  # 获取损失值并求和

            # 存在发展集
            if self.dev_dataloader is not None:
                # 计算发展集上的损失值
                self.model.eval()  # 设置模型为评估模式
                with torch.no_grad():  # 禁用梯度计算以提高性能
                    # 计算发展集上的损失值和准确度
                    dev_loss = 0.0  # 发展集损失
                    acc_list = []
                    for loss, acc in dev_model():
                        dev_loss += loss.data.item()
                        acc_list.append(acc)

                    # 更新学习率并监测验证集上的性能
                    self.scheduler.step(dev_loss)

            # 如果没有发展集，直接更新学习率
            else:
                self.scheduler.step()

            train_loss = train_loss / \
                len(self.train_dataloader) * self.batch_size  # 训练集每个批次的平均损失

            if self.dev_dataloader is not None:
                dev_loss = dev_loss / \
                    len(self.dev_dataloader) * self.batch_size  # 验证集每个批次的平均损失
                dev_acc = np.mean(np.array(acc_list))
                print(
                    f'第 {start_epoch + epoch + 1} 轮训练结束，训练集 loss 为 {train_loss}，发展集 loss 为 {dev_loss}，发展集准确率为 {dev_acc}')
                train_loss_list.append(train_loss)
                dev_loss_list.append(dev_loss)
                dev_acc_list.append(dev_acc)
                epoch_list.append(start_epoch + epoch + 1)

            else:
                print(
                    f'第 {start_epoch + epoch + 1} 轮训练结束，训练集 loss 为 {train_loss}')
                train_loss_list.append(train_loss)
                epoch_list.append(start_epoch + epoch + 1)

        if self.dev_dataloader is not None:
            # 保存模型和优化器状态
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 当前训练的轮数列表
                'epoch': list(range(pretrain + 1, pretrain + epoch_num + 1)),
                'train_loss': train_loss_list,
                'dev_loss': dev_loss_list,
                'dev_acc': dev_acc_list,
            }
        else:
            # 保存模型和优化器状态
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 当前训练的轮数列表
                'epoch': list(range(pretrain + 1, pretrain + epoch_num + 1)),
                'train_loss': train_loss_list,
            }

        # 保存模型检查点
        print(f'保存模型到 {self.model_path}checkpoint_{pretrain + epoch_num}.pth')
        torch.save(checkpoint, self.model_path +
                   f'checkpoint_{pretrain + epoch_num}.pth')  # 保存模型

    def train_model():
        """在训练集上个更新模型权重"""
        # TODO: 需要根据不同模型重新定义
        for batch in tqdm(self.train_dataloader, desc="模型训练"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()  # 将梯度置零放在循环开始处，以避免潜在的优化问题

            # 自动管理混合精度的上下文
            with autocast(device_type=self.device.type):
                logits = self.model(input_ids, attention_mask=attention_mask)

                # 在训练函数中计算损失
                log_likelihood = self.model.crf(
                    logits, labels, mask=attention_mask.byte(), reduction='mean')
                loss = -log_likelihood

            # 使用梯度缩放进行反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)  # 使用scaler来更新模型参数
            self.scaler.update()  # 更新缩放器

            yield loss

    def dev_model():
        """在发展集上验证模型，并更新学习率"""
        # TODO: 需要根据不同模型重新定义
        total_correct = 0
        total_samples = 0
        for batch_idx, batch in enumerate(tqdm(self.dev_dataloader, desc="模型验证")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)  # 仍需要labels来进行准确率计算

            logits = self.model(input_ids, attention_mask=attention_mask)

            # 计算损失
            with autocast(device_type=self.device.type):
                logits = self.model(input_ids, attention_mask=attention_mask)

                # 在训练函数中计算损失
                log_likelihood = self.model.crf(
                    logits, labels, mask=attention_mask.byte(), reduction='mean')
                loss = -log_likelihood

            # 计算预测序列
            preds = self.model.decode(logits, attention_mask)

            # 计算准确率
            acc = accuracy_score(labels, preds)

            yield loss, acc

    def train_plot(
        self,
        model_path
    ):
        '''绘制训练曲线（有无发展集数据均可）'''
        self.model_path = model_path

        # 查找目标目录下所有符合条件的.pth文件
        file_pattern = os.path.join(self.model_path, 'checkpoint_*.pth')
        pth_files = glob(file_pattern)

        # 初始化列表用于存储各指标数据
        all_epochs = []
        all_train_losses = []
        all_dev_losses = []
        all_dev_accuracies = []

        # 标志位：是否包含验证集指标（初始为None）
        has_dev_metrics = None

        # 遍历找到的文件
        for file_path in pth_files:
            try:
                checkpoint = torch.load(file_path)

                all_epochs.extend(checkpoint['epoch'])
                all_train_losses.extend(checkpoint['train_loss'])

                # 如果是第一个有效文件，确定是否包含验证集指标
                if has_dev_metrics is None:
                    has_dev_metrics = (
                        'dev_loss' in checkpoint and 'dev_acc' in checkpoint)
                    if has_dev_metrics:
                        print("检测到验证集指标，将绘制完整曲线")
                    else:
                        print("未检测到验证集指标，仅绘制训练损失曲线")

                # 如果包含验证集指标，收集验证数据
                if has_dev_metrics:
                    all_dev_losses.extend(checkpoint['dev_loss'])
                    all_dev_accuracies.extend(checkpoint['dev_acc'])

                print(file_path)

            except Exception as e:
                print(f"读取 {file_path} 时出错，错误信息: {e}")

        # 对数据按epoch排序（处理可能的乱序情况）
        if has_dev_metrics:
            # 对数据按epoch排序
            sorted_data = sorted(zip(all_epochs, all_train_losses,
                                     all_dev_losses, all_dev_accuracies), key=lambda x: x[0])
            epochs, train_losses, dev_losses, dev_accuracies = zip(
                *sorted_data)

        else:
            sorted_data = sorted(
                zip(all_epochs, all_train_losses), key=lambda x: x[0])
            epochs, train_losses = zip(*sorted_data)

        # 创建图表
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # 绘制损失曲线（左Y轴）
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, train_losses, marker='o',
                 color=color, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 如果有验证集指标
        if has_dev_metrics:
            # 绘制验证损失曲线（左Y轴）
            ax1.plot(epochs, dev_losses, marker='s',
                     color='tab:orange', label='Validation Loss')

            # 绘制准确率曲线（右Y轴）
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Accuracy (%)', color=color)
            ax2.plot(epochs, [acc * 100 for acc in dev_accuracies],
                     marker='^', color=color, label='Validation Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)

            # 确保准确率轴范围在0-100%之间，同时留出适当边距
            min_acc = min(dev_accuracies) * 100
            max_acc = max(dev_accuracies) * 100
            margin = max(5, (max_acc - min_acc) * 0.1)  # 边距至少为5%或数据范围的10%
            ax2.set_ylim(max(0, min_acc - margin), min(100, max_acc + margin))

            # 添加图例和标题
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

            plt.title('Model Training Metrics Over Epochs')

        else:
            # 仅显示训练损失
            ax1.legend(loc='best')
            plt.title('Training Loss')

        plt.tight_layout()  # 确保布局合理
        plt.show()

    def predict(test_dataloader):
        '''模型推理'''

    def eval():
        '''模型评估'''
