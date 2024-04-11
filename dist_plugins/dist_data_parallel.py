import os
import torch
from typing import List
from dist_plugin import DistPlugin
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel as DDP


class DistDataParallel(DistPlugin):

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
                 rank: int,
                 master_addr: str,
                 master_port: str,
                 ):
        super().__init__(total_step, save_step, model, dataloader,
                         optimizer, loss, device_ids, rank)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        self.model = DDP(model, device_ids=[rank])

    def train(self, input: dict):
        loss = self.model(**input)
        loss.backward()
        self.optimizer.step()



