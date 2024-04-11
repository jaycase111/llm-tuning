from peft_tuning.base_config import Config
from transformers import AutoModel, AutoTokenizer
from peft import TaskType, get_peft_model,  PromptEncoderConfig
from torch import Module


def wrap_p_tuning(model_path: str,
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
    p_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=config.num_virtual_tokens,
        encoder_hidden_size=config.encoder_hidden_size,
        encoder_reparameterization_type=config.encoder_reprameterization_type
    )
    return get_peft_model(model, p_config)
