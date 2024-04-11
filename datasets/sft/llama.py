import copy
import torch
from datasets.sft_data import Dialogue, SftData, List, \
    SftDataset, PreTrainedTokenizer
from datasets.utils import get_pair_element


class LlamaSftDataset(SftDataset):
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
                 sft_data_list: List[SftData],
                 max_token_nums: int,
                 max_turn: int = 1,
                 tokenizer: PreTrainedTokenizer = None,
                 ):
        super().__init__(sft_data_list,
                         max_token_nums,
                         max_turn,
                         tokenizer)

    def _get_preview_text(self, dialogue_list: List[Dialogue]):
        preview_text = ""
        dialogue_list = dialogue_list[:-1]
        for dialogue in dialogue_list:
            text, speaker = dialogue.get_text(), dialogue.get_speaker()
            if speaker == "user":
                preview_text += self.user_input + text + "\n"
            else:
                preview_text += self.assist_input + text + "\n"
        preview_text = preview_text.strip()
        return preview_text

    def _get_answer_text(self, dialogue_list: List[Dialogue]):
        dialogue = dialogue_list[-1]
        return self.assist_input + dialogue.get_text()

    def __getitem__(self, item_index: int):
        """
         :param          index:  数据集索引
         :return:        llama 单条数据数据集 参考资料: https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/alpaca_dataset.py
                         InstructionDataset 类
         TODO: 完成真实逻辑情况
         """
        item = self.sft_data_list[item_index]
        input_text = item.get_instruction()
        dialogue_list = item.get_dilaogue_list()[:2*self.max_turn]
        assert len(dialogue_list) > 0 and len(dialogue_list) % 2 == 0, \
            "Training data rounds must be even"
        preview_text = self._get_preview_text(dialogue_list)
        answer_text = self._get_answer_text(dialogue_list)
        if len(input_text) > 0:
            prompt = self.PROMPT_DICT["prompt_input"].format(preview_text, input_text)
        else:
            prompt = self.PROMPT_DICT["prompt_no_input"].format(preview_text)

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = prompt + answer_text
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_token_nums - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_token_nums]

        labels = copy.deepcopy(example)     # example: 完整Tensor
        labels[: len(prompt)] = -1          # 将prompt部分转换为-1
        example_mask = example.ge(0)        # 是否大于等于0 完成Mask pad位置填充的是-1
        label_mask = labels.ge(0)           # 是否大于等于0 完成Mask pad的位置填充的是-1
        example[~example_mask] = 0          # 将小于0的数字设置为0
        labels[~label_mask] = self.MASK_ID  # 将小于0的数字设置为 -100
        example_mask = example_mask.float()

        return {
            "input_ids": example,  # pad的位置设置为0
            "labels": labels,  # prompt 部分设置为 -100 | pad的位置设置为 -100
            "attention_mask": example_mask,  # 其实输入的是 input_ids pad attention_mask
        }



