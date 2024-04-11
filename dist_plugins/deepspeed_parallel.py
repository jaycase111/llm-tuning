import os
import torch
import deepspeed
from tqdm import tqdm
from typing import List
from dist_plugin import DistPlugin
from torch.nn import Module
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel as DDP


class DeepSpeedParallel(DistPlugin):
    """
        deepspeed并行插件
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
                 args
                 ):
        super().__init__(total_step, save_step, model, dataloader,
                         optimizer, loss, device_ids, rank)
        deepspeed.init_distributed(dist_backend="nccl")
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(args=args,
                                                             model=model,
                                                             model_parameters=model.get_trainable_parameters(),
                                                             )

    def train(self,
              input: dict):
        loss = self.model_engine(**input)
        self.model_engine.backward(loss)
        self.model_engine.step()

    def save_weights(self,
                     output_dir: str,
                     save_adapter: bool = True
                     ):
        """
        :param output_dir:      模型保存地址
        :param save_adapter:    模型是否保存Adapter数据
        :return:
        """
        pass


