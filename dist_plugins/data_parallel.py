import torch
from typing import List
from dist_plugin import DistPlugin
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss


class DataParallel(DistPlugin):

    """
        数据并行插件
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
        super().__init__(total_step, save_step, model, dataloader,
                         optimizer, loss, device_ids, rank)
        self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    def train(self, input: dict):
        loss = self.model(**input)
        loss.backward()
        self.optimizer.step()



