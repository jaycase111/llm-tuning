from awq import AutoAWQForCausalLM
from quantization.base_quantization import \
    BaseQuantization, Module, PreTrainedTokenizer, List
from transformers import AutoTokenizer, AutoModel


class AwqQuantization(BaseQuantization):

    """
        AWQ量化--不需要使用校准数据集
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
        model = AutoModel.from_pretrained(self.model_str)
        tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        if self.bit == "4bit":
            quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
        else:
            quant_config =  { "zero_point": True, "q_group_size": 128, "w_bit": 8, "version": "GEMM" }
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(self.save_path)
        tokenizer.save_pretrained(self.save_path)
        print(f"AWQ quantization {self.bit} success")



