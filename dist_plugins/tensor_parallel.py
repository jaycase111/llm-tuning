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
import colossalai
from colossalai.cluster import DistCoordinator
from colossalai.shardformer import ShardConfig, ShardFormer


class TensorParallel(DistPlugin):
    """
        ColossalAI 张量并行:
        https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/examples/convergence_benchmark.py
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
        colossalai.launch_from_torch(config={}, seed=42)
        if dist.get_world_size() > 1:
            tp_group = dist.new_group(backend="nccl")
            shard_config = ShardConfig(
                tensor_parallel_process_group=tp_group, enable_tensor_parallelism=True, enable_all_optimization=True
            )
            shard_former = ShardFormer(shard_config=shard_config)
            self.model, _ = shard_former.optimize(self.model)
        else:
            pass

    def train(self, input: dict):
        loss = self.model(**input)
        loss.backward()
        self.optimizer.step()
