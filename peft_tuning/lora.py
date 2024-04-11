from torch import Module
from peft_tuning.base_config import Config
from transformers import AutoModel
from peft import LoraConfig, TaskType, get_peft_model


def wrap_lora(model_path: str,
              config: Config,
              model: Module = None):
    if model is None:
        model = AutoModel.from_pretrained(model_path,
                                          trust_remote_code=True)
    peft_config = LoraConfig(
        task_type=config.task_type,
        inference_mode=False,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules
    )
    model = get_peft_model(model, peft_config)
    return model

