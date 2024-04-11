import torch
from datasets.sft_data import Dialogue, SftData, List, \
    SftDataset, PreTrainedTokenizer
from datasets.utils import get_pair_element


class GlmSftDataset(SftDataset):

    """
        当前SFT数据集构建为V2版本
    """

    round_prompt = "[Round {}]\n问：{}\n答：{}\n"
    single_prompt = "问:{}\n答: "

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

    def __getitem__(self, item_index: int):
        """
        :param          index:  数据集索引
        :return:        glm 单条数据数据集 参考资料: https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
                        preprocess_function_train 函数
        """
        item = self.sft_data_list[item_index]
        instruction = item.get_instruction()
        dialogue_list = item.get_dilaogue_list()[:2 * self.max_turn]
        assert len(dialogue_list) > 0 and len(dialogue_list) % 2 == 0, \
            "Training data rounds must be even"
        prompt, answer = "", ""
        pair_dialogue_list = get_pair_element(dialogue_list)
        if self.verify_contain_history(dialogue_list):
            # 构造多伦数据集
            for index, pair_dialogue in enumerate(pair_dialogue_list):
                dialogue_one, dialogue_two = pair_dialogue
                if index < len(pair_dialogue_list) - 1:
                    prompt += self.round_prompt.format(index + 1, dialogue_one.text, dialogue_two.text)
                else:
                    prompt += self.single_prompt.format(index + 1, dialogue_one.text)
                    answer = dialogue_two.text
        else:
            # 单轮数据集
            dialogue_one, dialogue_two = pair_dialogue_list[-1]
            prompt = self.single_prompt.format(1, dialogue_one.text)
            answer = dialogue_two.text

        prompt_ids = self.tokenizer.encode(prompt)
        answer_ids = self.tokenizer.encode(answer)
        input_ids = prompt_ids + answer_ids + [self.tokenizer.eos_token_id]

        if len(input_ids) > self.max_token_nums:
            input_ids = input_ids[:self.max_token_nums]
            prompt_ids = prompt_ids[:self.max_token_nums]
        else:
            pad_len = self.max_token_nums - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        labels = (
                [-100] * (len(prompt_ids) - 1)
                + input_ids[(len(prompt_ids) - 1):]
        )

        return {"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(labels)}