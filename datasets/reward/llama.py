import torch
from copy import deepcopy
from datasets.reward_data import Dialogue, RewardData, List, \
    RewardDataset, PreTrainedTokenizer
from datasets.utils import get_pair_element


class LlamaRewardDataset(RewardDataset):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{}\n\n### Response:"
        ),
    }

    user_input = "USER: "
    assist_input = "ASSISTANT: "

    def __init__(self,
                 reward_data_list: List[RewardData],
                 max_token_nums: int,
                 tokenizer: PreTrainedTokenizer = None,
                 debug: bool = False
                 ):
        super().__init__(
            reward_data_list, max_token_nums, tokenizer, debug
        )

    def _get_prompt_text(self,
                         dialogue_list: List[Dialogue]):
        prompt_text = ""
        for dialogue in dialogue_list:
            text, speaker = dialogue.get_text(), dialogue.get_speaker()
            if speaker == "user":
                prompt_text += self.user_input + text + "\n"
            else:
                prompt_text += self.assist_input + text + "\n"
        return prompt_text

    def __getitem__(self, item_index: int):
        """
        :param          index:  数据集索引
        :return:
        """
        item = self.reward_data_list[item_index]
        input_text = item.get_instruction()
        dialogue_list = item.get_dilaogue_list()
        score = item.get_score()
        assert len(dialogue_list) == 2 , \
            "Reward data rounds must is 1"
        prompt_text = self._get_prompt_text(dialogue_list)
        if len(input_text) > 0:
            prompt = self.PROMPT_DICT["prompt_input"].format(prompt_text, input_text)
        else:
            prompt = self.PROMPT_DICT["prompt_no_input"].format(prompt_text)
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        return {"input_ids": input_ids, "labels": score}