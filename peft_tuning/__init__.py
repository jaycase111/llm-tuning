from torch import Module
from util.io_util import read_json, read_yaml
from transformers import AutoModel
from peft_tuning.p_tuning import wrap_p_tuning
from peft_tuning.lora import wrap_lora
from peft_tuning.prefix_tuning import wrap_prefix_tuning
from peft_tuning.base_config import Config


__global_map__ = {
    "lora": wrap_lora,
    "p_tuning": wrap_p_tuning
}



def get_wrap_model(model_path: str,
                   peft_config_file: str,
                   model: Module = None
                   ):
    if peft_config_file is None:
        model = AutoModel.from_pretrained(model_path,
                                          trust_remote_code=True)
    else:
        config_proxy = Config()
        peft_config = read_yaml(peft_config_file)
        name = peft_config["name"]
        if name not in __global_map__.keys():
            suporrt_method = " | ".join(list(__global_map__.keys()))
            raise NotImplementedError(f"not support {name} peft method, now support {suporrt_method}")
        config = peft_config["config"]
        for key, value in config.items():
            config_proxy.__setattr__(key, value)
        model = __global_map__[name](model_path, config_proxy)
    return model
