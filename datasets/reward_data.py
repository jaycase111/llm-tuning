from datasets.sft_data import SftData, List, Dialogue
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RewardData(SftData):

    def __init__(self,
                 id: int,
                 instruction: str,
                 dialogue_list: List[Dialogue],
                 score: int):
        """
        :param id:                  对话ID
        :param instruction:         对话Instruction
        :param dialogue_list:       对话列表、由于为评分模型所以数据只能是一问一答数据
        :param score:
        """
        super().__init__(id, instruction, dialogue_list)
        assert len(dialogue_list) == 2, "Reward dialogue Data must be 1 turn"
        self.score = score

    def get_score(self) -> int:
        return self.score

    @classmethod
    def init_from_element(cls,
                          element: Dict):
        dialogue_list = [Dialogue.init_from_element(dialogue) for dialogue in element["dialog"]]
        dialogue_list = [dialogue for dialogue in dialogue_list if dialogue is not None]
        return cls(id=element["id"],
                   instruction=element["instruction"],
                   dialogue_list=dialogue_list,
                   score=element["score"])


class RewardDataset(Dataset):

    def __init__(self,
                 reward_data_list: List[RewardData],
                 max_token_nums: int,
                 tokenizer: PreTrainedTokenizer = None,
                 debug: bool = False
                 ):
        self.reward_data_list = reward_data_list
        self.tokenizer = tokenizer
        self.max_token_nums = max_token_nums
        self._check_debug(debug)

    def _check_debug(self, debug: bool):
        if debug:
            print("Sft training in debug mode")
            self.reward_data_list = self.reward_data_list[:10]

    def __len__(self):
        return len(self.reward_data_list)

    def __getitem__(self, item):
        raise NotImplemented
