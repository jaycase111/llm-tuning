from typing import List
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets.utils import read_json
from dschat.utils.data.data_utils import DataCollatorRLHF,pad_sequence


class RlhfData:

    def __init__(self,
                 id: int,
                 question: str):
        self.id = id
        self.question = question

    def get_id(self):
        return self.id

    def get_question(self):
        return self.question

    @classmethod
    def init_from_element(cls, element: dict):
        return cls(id=element["id"],
                   question=element["question"])


class RlhfTrainDataset(Dataset):

    def __init__(self,
                 rlhf_data_list: List[RlhfData],
                 tokenizer: AutoTokenizer,
                 max_prompt_seq_len: int,
                 ):
        super().__init__()
        self.rlhf_data_list = rlhf_data_list
        self.tokenizer = tokenizer
        self.max_prompt_seq_len = max_prompt_seq_len
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.rlhf_data_list)

    def __getitem__(self, index):
        item = self.rlhf_data_list[index]
        prompt_token = self.tokenizer(item.get_question(), return_tensors="pt")
        for key_word in ["input_ids", "attention_mask"]:
            prompt_token[key_word] = prompt_token[
                key_word].squeeze(0).flip(0)
        return prompt_token["input_ids"], prompt_token["attention_mask"], self.pad_token_id


class BatchCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        """
            max_token_len: 256
            inference_tp_size 设置为 1
        """
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size
        print("max_input_length ", self.max_token_len)

    def __call__(self, data):
        """
            将批次数据补成定长
        """
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = prompt[:self.max_token_len]
            batch["prompt_att_mask"] = prompt_mask[:self.max_token_len]
        else:
            batch["prompt"] = prompt[:self.max_token_len]
            batch["prompt_att_mask"] = prompt_mask[:self.max_token_len]
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch

def create_rlhf_dataset(
        rank: int,
        file: str,
        seed: int,
        tokenizer: AutoTokenizer,
        max_prompt_seq_len: int
):
    content = read_json(file)
    content = [RlhfData.init_from_element(element) for element in content]
    return RlhfTrainDataset(
        content,
        tokenizer,
        max_prompt_seq_len
    )
