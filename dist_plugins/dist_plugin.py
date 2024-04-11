from typing import List
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss


class DistPlugin:

    """
        分布式训练插件基类
    """
    def __init__(self,
                 total_step: int,
                 save_step: int,
                 model: Module,
                 dataloader: DataLoader,
                 optimizer: Optimizer,
                 loss: _Loss,
                 device_ids: List[int],
                 rank: int
                 ):
        self.total_step = total_step
        self.save_step = save_step
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.device_ids = device_ids
        self.rank = rank

    def train(self, input: dict):
        raise NotImplemented

    def save_weights(self,
                     output_dir: str,
                     save_adapter: bool = True
                     ):
        """
        :param output_dir:      模型保存地址
        :param save_adapter:    模型是否保存Adapter数据
        :return:
        """
        raise NotImplemented
