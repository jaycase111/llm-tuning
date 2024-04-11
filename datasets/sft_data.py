from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class Dialogue:

    def __init__(self,
                 text: str,
                 speaker: str):
        self.text = text
        self.speaker = speaker

    def get_text(self):
        return self.text

    def get_speaker(self):
        return self.speaker

    def set_text(self,
                 text: str):
        self.text = text

    def set_speaker(self, speaker: str):
        self.speaker = speaker

    @classmethod
    def init_from_element(cls, element: dict):
        text = element["text"]
        speaker = element["speaker"]
        if element["speaker"] not in ["user", "assistant"]:
            raise ValueError(f"{cls.__class__.__name__} init speaker value error, now input speaker: {speaker}")
        if len(text) == 0:
            return None
        return Dialogue(**element)


class SftData:
    """
        SFT 单条数据类
    """

    def __init__(self,
                 id: int,
                 instruction: str,
                 dialogue_list: List[Dialogue]
                 ):
        self.id = id
        self.instruction = instruction
        self.dialogue_list = dialogue_list

    def get_id(self) -> int:
        return self.id

    def get_instruction(self) -> str:
        return self.instruction

    def get_dilaogue_list(self) -> List[Dialogue]:
        return self.dialogue_list

    @classmethod
    def init_from_element(cls,
                          element: Dict):
        dialogue_list = [Dialogue.init_from_element(dialogue) for dialogue in element["dialog"]]
        dialogue_list = [dialogue for dialogue in dialogue_list if dialogue is not None]
        return cls(id=element["id"],
                   instruction=element["instruction"],
                   dialogue_list=dialogue_list)




class SftDataset(Dataset):

    MASK_ID = -100

    def __init__(self,
                 sft_data_list: List[SftData],
                 max_token_nums: int,
                 max_turn: int = 1,
                 tokenizer: PreTrainedTokenizer = None,
                 debug: bool = False
                 ):
        """
        
        :param sft_data_list:  Instruction-Sft 数据列表
        :param max_token_nums: go
        :param max_turn:       最大轮次数
        TODO: 后续支持当数据量较大时候、解决一次内存无法完全加载SFT数据的问题
        """
        self.sft_data_list = sft_data_list
        self.max_turn = max_turn
        self.tokenizer = tokenizer
        self.max_token_nums = max_token_nums
        self._check_debug(debug)

    @classmethod
    def verify_contain_history(cls,
                               dialogue_list: List[Dialogue]):
        return len(dialogue_list) > 0

    def _check_debug(self, debug: bool):
        if debug:
            print("Sft training in debug mode")
            self.sft_data_list = self.sft_data_list[:10]

    def __len__(self):
        return len(self.sft_data_list)

    def __getitem__(self, item):
        raise NotImplemented


