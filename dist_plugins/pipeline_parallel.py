import os
import torch
import functools
from typing import List
from dist_plugin import DistPlugin
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from pippy import pipeline
from pippy import SplitPoint, annotate_split_points, PipelineStage


class PipelineParallel(DistPlugin):
    """
        流水线并行--Gpipe并行
        相关链接: https://zhuanlan.zhihu.com/p/648637061

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
                 args,
                 world_size: int
                 ):
        super().__init__(total_step, save_step, model, dataloader,
                         optimizer, loss, device_ids, rank)

        self.layers_per_rank = self.model.config.num_hidden_layers // world_size
        for i in range(1, world_size):
            annotate_split_points(self.model,
                                  {f"model.layers.{i * self.layers_per_rank}": SplitPoint.BEGINNING})
        self.model = pipeline(self.model, world_size)
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        self.stage = PipelineStage(self.model, rank, device=device)

    def train(self, input: dict):
        loss = self.stage(**input)
        loss.backward()
        self.optimizer.step()
