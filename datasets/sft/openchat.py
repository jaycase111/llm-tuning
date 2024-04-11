import copy
import torch
from datasets.sft_data import Dialogue, SftData, List, \
    SftDataset, PreTrainedTokenizer
from datasets.utils import get_pair_element


class OpenchatSftDataset(SftDataset):

    """
        TODO: 完成Openchat数据类定义
    """

    def __init__(self,
                 sft_data_list: List[SftData],
                 max_token_nums: int,
                 max_turn: int = 1,
                 tokenizer: PreTrainedTokenizer = None,
                 ):
        super().__init__(sft_data_list,
                         max_token_nums,
                         max_turn,
                         tokenizer)