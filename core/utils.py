from torch.utils.tensorboard import SummaryWriter
from torch.utils.hooks import RemovableHandle
import torch
from typing import List
import torch.nn as nn
from typing import Dict, Optional
import os


class TorchTrainingVisualizer:

    def __init__(self,
                 log_dir: str = "./runs",
                 comment: str = "",
                 flush_secs: int = 10):
        """
        :param log_dir: TensorBoard 日志目录
        :param comment: 实验名（自动拼在 log_dir 后）
        :param flush_secs: 每隔多少秒刷盘（默认10秒，避免频繁IO）
        """
        self.log_dir = os.path.join(log_dir, comment) if comment else log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=flush_secs)
        self.step_count = 0  # 全局步数计数器（可选，用于自动递增）

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is None:
            step = self.step_count
            self.step_count += 1  # 自动递增，确保每个调用对应唯一 step

        for key, value in metrics.items():
            if not isinstance(value, float):
                value = float(value)
            self.writer.add_scalar(key, value, step)


    def log_gradients(self, model: torch.nn.Module, step: Optional[int] = None):
        """
        记录所有参数的梯度分布（用于检测梯度爆炸）
        """
        if step is None:
            step = self.step_count
            self.step_count += 1

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"gradients/{name}", param.grad, step)
                self.writer.add_histogram(f"weights/{name}", param.data, step)

    def log_hparams(self, hparams: Dict, metrics: Dict):
        """记录超参数 + 最终指标（仅在训练结束时调用一次）"""
        self.writer.add_hparams(hparams, metrics)

    def close(self):
        self.writer.close()
        print(f"TensorBoard 日志已保存至: {self.log_dir}")
        print("在终端运行: tensorboard --logdir=./runs  查看实时曲线")