"""
    PEFT中已经废弃Adapter、现自定义实现Adapter
    现提供LlamaAttention 的 Adapter方法
"""
from torch import nn
from torch import Module
from transformers import AutoModel
from peft_tuning.base_config import Config
from adapters import BnConfig, AutoAdapterModel


class AdapterLayer(nn.Module):
    def __init__(self, input_size, adapter_size):
        super(AdapterLayer, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_size, adapter_size),
            nn.ReLU(),
            nn.Linear(adapter_size, input_size)
        )

    def forward(self, x):
        return x + self.adapter(x)


class Adapter(nn.Module):

    def __init__(self,
                 attention: nn.Module,
                 input_size: int,
                 adapter_size: int
                 ):
        super().__init__()
        self.attention = attention
        self.adapter = AdapterLayer(input_size, adapter_size)

    def forward(self,
                **kwargs):
        x = self.attention(**kwargs)
        return self.adapetr(x)


# def wrap_adapter(model: str,
#                  config: Config
#                  ):
#     """
#     :param model:     待接入Adapter结构的模型
#     :param config:
#     TODO: 完成真实的Adapter改造、现使用LlamaAttention 完成模型修改
#     :return:
#     """
#     attention: LlamaAttention = None
#     return Adapter(attention,
#                    config.adapter_input_size,
#                    config.adapter_hidden_size)


def wrap_adapter(model_path: str,
                 config: Config,
                 model: Module = None
                 ):
    """
    :param model:   模型名称
    :param config:  配置项
    :return:
    真实adapter-transformer中 bottleneck-adapter  加载过程
    """
    if model is None:
        model = AutoModel.from_pretrained(model_path,
                                          trust_remote_code=True)
    model = AutoAdapterModel.from_pretrained(model,
                                             trust_remote_code=True)
    bn_config = BnConfig(
        mh_adapter=config.mh_adapter,
        output_adapter=config.output_adapter,
        reduction_factor=config.reduction_factor,
        non_linearity=config.non_linearity
    )
    model.add_adapter(config.adapter_name, config=bn_config)
    return model



