from peft_tuning.base_config import Config
from transformers import AutoModelForCausalLM, AutoModel
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig
from torch import Module


def wrap_qlora(model_path: str,
               config: Config,
               model: Module = None):
    if model is None:
        model = AutoModel.from_pretrained(model_path,
                                      trust_remote_code=True)
    nf4_config = BitsAndBytesConfig(
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        task_type=config.task_type,
        inference_mode=False,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    return model
