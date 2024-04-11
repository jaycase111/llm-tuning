"""
    自定义实现 AdapterFusion 部分功能
"""
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention
from transformers import AutoModel
from torch import Module
from peft_tuning.base_config import Config
from peft_tuning.adapter import AdapterLayer
from transformers.models.ctrl.modeling_ctrl import MultiHeadAttention
from adapters import BnConfig, AutoAdapterModel, Fuse


class AdapterFusion(nn.Module):

    def __init__(self,
                 attention: LlamaAttention,
                 input_size: int,
                 adapter_size: int,
                 adpter_nums: int
                 ):
        super().__init__()
        self.attention = attention
        self.adapter_list = [AdapterLayer(input_size,
                                          adapter_size) for i in range(adpter_nums)]
        self.weight_attention = MultiHeadAttention(
            input_size,
            num_heads=8
        )

    def forward(self, **kwargs):
        x = self.attention(**kwargs)  # batch_size * seq_len * hidden_nums
        adapter_outputs = torch.cat([adapter_layer(x) for adapter_layer in self.adapter_list],
                                    dim=1)  # batch_size * adpter_nums * seq_len * hidden_nums
        adapter_outputs = [self.weight_attention(v=adapter_output, q=x, k=adapter_output)
                           for adapter_output in adapter_outputs]
        return torch.mean(torch.tensor(adapter_outputs, x.dtype), dim=1)


def wrap_adapter_fusion(model_path: str,
                        config: Config,
                        model: Module = None,
                        ):
    if model is None:
        model = AutoModel.from_pretrained(model_path,
                                          trust_remote_code=True)
    model = AutoAdapterModel.from_pretrained(model,
                                             trust_remote_code=True)
    assert len(config.adapter_names) > 0
    for name in config.adapter_names:
        bn_config = BnConfig(
            mh_adapter=config.mh_adapter,
            output_adapter=config.output_adapter,
            reduction_factor=config.reduction_factor,
            non_linearity=config.non_linearity
        )
        model.add_adapter(name, config=bn_config)
    model.add_adapter_fusion(Fuse(config.adapter_names))
    model.set_active_adapters(Fuse(config.adapter_names))
    return model


