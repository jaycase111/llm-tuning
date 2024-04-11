from copy import deepcopy
from datasets.reward_data import Dialogue, RewardData, List, \
    RewardDataset, PreTrainedTokenizer
from datasets.utils import get_pair_element


class GlmRewardDataset(RewardDataset):

    round_prompt = "[Round {}]\n问：{}\n答：{}\n"
    single_prompt = "问:{}\n答: "

    def __init__(self,
                 reward_data_list: List[RewardData],
                 max_token_nums: int,
                 tokenizer: PreTrainedTokenizer = None,
                 debug: bool = False
                 ):
        super().__init__(
            reward_data_list, max_token_nums, tokenizer, debug
        )

    def __getitem__(self, item_index: int):
        """
        :param          index:  数据集索引
        :return:        glm 单条数据数据集 参考资料: https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
                        preprocess_function_train 函数
        """
        item = self.reward_data_list[item_index]
        instruction = item.get_instruction()
        dialogue_list = item.get_dilaogue_list()
        assert len(dialogue_list) == 2 , \
            "Reward data rounds must is 1"
        dialogue_one, dialogue_two = dialogue_list[0], dialogue_list[1]
        score = item.get_score()
        prompt = deepcopy(self.single_prompt).format(dialogue_one.get_text(),
                                                     dialogue_two.get_text())
        input_ids = self.tokenizer(prompt)
        if len(input_ids) > self.max_token_nums:
            input_ids = input_ids[:self.max_token_nums]
        else:
            pad_len = self.max_token_nums - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            
        return {"input_ids": input_ids,
                "labels": score}