from typing import List
from dataclasses import dataclass
from peft import TaskType


class Config:
    r: int                                      # Lora 维度
    lora_alpha: int                             # Lora alpha参数
    lora_dropout: float                         # lora dropout 参数
    target_modules: list                        # Lora 要改造的层数
    task_type: TaskType                         # Lora 任务类型
    bnb_4bit_quant_type: str                    # Qlora 量化参数
    adapter_input_size: int                     # adapter 输入维度
    adapter_hidden_size: int                    # adapter 中间维度
    adapter_num: int                            # adapterFusion adapter个数

    mh_adapter: bool                            # bottleneck-mh_adapter 配置参数
    output_adapter: bool                        # bottleneck-output_adapter 配置参数
    reduction_factor: int                       # bottleneck-reduction_factor 配置参数
    non_linearity: str                          # bottleneck-non_linearity      配置参数
    adapter_name: str                           # bottleneck_name
    adapter_names: List[str]                    # bottleneck—adapterFusion name 配置参数

    prompt_text: str                            # Prompt-Tuning 中的SoftTuning、若不存在则需要注入空字符串或者None
    num_virtual_tokens: int                     # Prompt-Tuning 中指定Soft-Prompt token数量

    prefix_projection: bool                     # prefix-tuning 是否启用、不启用: Embedding层 启用: Embedding层 + 双层Linear层
    encoder_hidden_size: int                    # 当启用 prefix_projection时启用的hidden_dim

    encoder_reprameterization_type: str # P-tuning Embedding编码器选项 可选项: [MLP、LSTM]
