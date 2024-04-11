import os
import torch
import functools
from typing import List
from dist_plugin import DistPlugin
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)


class FsdpParallel(DistPlugin):

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

        # TODO: 优化FSDP分区逻辑
        self.model =  FSDP(model,
                 cpu_offload=CPUOffload(offload_params=True)
                 )

    def train(self, input: dict):
        self.optimizer.zero_grad()
        loss = self.model(**input)
        loss.backward()
        self.optimizer.step()



