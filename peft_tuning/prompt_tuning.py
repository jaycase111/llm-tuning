"""
    Prompt-Tuning可以用在Transformer中各种架构上、但是在Prefix-Encoder模型上应用更多、本项目为方便使用Causal架构选型
"""
from torch import Module
from peft_tuning.base_config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import TaskType, get_peft_model,  PromptTuningConfig, PromptTuningInit


def wrap_prompt_tuning(model_path: str,
                       config: Config,
                       model: Module = None):
    """
    :param model:       待finetune模型名称
    :param config:
    :return:
    存在两种finetune模式: (1) Hard-prompt (输入固定的Prompt)
                        (2) Soft-prompt (生成随机Prompt-Embedding)
    """
    if model is None:
        model = AutoModel.from_pretrained(model,
                                          trust_remote_code=True)
    soft_prompt = config.prompt_text
    if soft_prompt is not None\
        and len(soft_prompt) > 0:
        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=soft_prompt,
            num_virtual_tokens=config.num_virtual_tokens,
            inference_mode=False,
            tokenizer_name_or_path=model
        )
    else:
        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=config.num_virtual_tokens,
            inference_mode=False
        )


    model = get_peft_model(model, config)
    return model
