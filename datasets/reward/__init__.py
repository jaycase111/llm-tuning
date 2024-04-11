from util.io_util import read_json
from transformers import AutoTokenizer
from datasets.reward_data import RewardData
from datasets.reward.glm import GlmRewardDataset
from datasets.reward.llama import LlamaRewardDataset
from datasets.reward.openchat import OpenchatRewardDataset

__global_dataset__ = {
    "glm": GlmRewardDataset,
    "llama": LlamaRewardDataset,
    "openchat": OpenchatRewardDataset
}


def get_dataset(model_type: str,
                model_path: str,
                train_file: str,
                max_token_nums: int):
    """
    :param model_type:  训练模型类型
    :param train_file:  训练文件
    :return:
    """
    if model_type not in __global_dataset__.keys():
        support_type = " | ".join(list(__global_dataset__.keys()))
        raise NotImplemented(f"Not support {model_type} model architecture,  now only support {support_type}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        sft_data_list = [RewardData.init_from_element(element) for element in read_json(train_file)]
        return __global_dataset__[model_type](sft_data_list, max_token_nums, tokenizer=tokenizer)
