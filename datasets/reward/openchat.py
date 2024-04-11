import torch
from copy import deepcopy
from datasets.reward_data import Dialogue, RewardData, List, \
    RewardDataset, PreTrainedTokenizer
from datasets.utils import get_pair_element


class OpenchatRewardDataset(RewardDataset):

    """
        TODO: 后续支持openchat数据集构建
    """

    def __init__(self,
                 reward_data_list: List[RewardData],
                 max_token_nums: int,
                 tokenizer: PreTrainedTokenizer = None,
                 debug: bool = False
                 ):
        super().__init__(
            reward_data_list, max_token_nums, tokenizer, debug
        )



