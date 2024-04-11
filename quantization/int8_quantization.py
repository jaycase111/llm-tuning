import torch
from transformers import BitsAndBytesConfig, AutoModel
from quantization.base_quantization import \
    BaseQuantization, Module, PreTrainedTokenizer, List


class IntQuantization(BaseQuantization):
    """
        Transformer结合BitsAndBytesConfig实现数值量化、可直接使用Transformer加载时完成
    """

    def __init__(self,
                 model_str: str,
                 bit: str,
                 save_path: str,
                 examples: List = None
                 ):
        super().__init__(model_str, bit,
                         save_path, examples)

    def quant_not_use_examples(self):
        if self.bit == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                # bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,  # 可以再次节约显存
                bnb_4bit_compute_dtype=torch.float16
            )
        model = AutoModel.from_pretrained(self.model_str, quantization_config=quant_config)
        model.save_pretrained(self.save_path)
        print(f"Int quantization {self.bit} success")

