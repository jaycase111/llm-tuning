from peft_tuning.base_config import Config
from transformers import AutoTokenizer, AutoModel
from peft import TaskType, get_peft_model, PrefixTuningConfig
from torch import Module


def wrap_prefix_tuning(model_path: str,
                       config: Config,
                       model: Module = None):
    """
    :param model:       待finetune模型名称
    :param config:
    :return:
    """
    if model is None:
        model = AutoModel.from_pretrained(model_path,
                                          trust_remote_code=True)
    if config.prefix_projection:
        prefix_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=config.num_virtual_tokens,
            prefix_projection=config.prefix_projection,
            # encoder_hidden_size=config.encoder_hidden_size
        )
    else:
        prefix_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=config.num_virtual_tokens
        )
    model = get_peft_model(model, prefix_config)
    return model