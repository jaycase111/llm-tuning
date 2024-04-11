from typing import List
from torch.nn.modules import Module
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class BaseQuantization:

    def __init__(self,
                 model_str: str,
                 bit: str,
                 save_path: str,
                 examples: List = None,
                 dataloader: DataLoader = None
                 ):
        """
        :param model_str:       待量化的地址
        :param tokenizer:       待量化的分词器
        :param bit:             量化的位数
        :param save_path:       量化模型保存地址
        :param examples:        校准数据集
        :param dataloader:      squeezellm 使用校准数据集
        """
        self.model_str = model_str
        assert bit in ["4bit", "8bit"], "now only support 4bit or 8bit quantization"
        self.bit = bit
        self.save_path = save_path
        self.examples = examples
        self.dataloader = dataloader

    def quant_use_examples(self):
        """
        使用校准数据--校准量化数据
        :return:
        """
        raise NotImplemented

    def quant_not_use_examples(self):
        """
        不校准量化模型
        :return:
        """
        raise NotImplemented
