from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from quantization.base_quantization import \
    BaseQuantization, Module, PreTrainedTokenizer, List


class GptqQuantization(BaseQuantization):
    """
        Gpt量化--不需要使用校准数据集
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
        self.examples = []
        self.quant_use_examples()

    def quant_use_examples(self):
        if self.bit == "4bit":
            bit = 4
        else:
            bit = 8
        quantize_config = BaseQuantizeConfig(
            bits=bit,  # 将模型量化为 4-bit 数值类型
            group_size=128,  # 一般推荐将此参数的值设置为 128
            desc_act=False,  # 设为 False 可以显著提升推理速度，但是 ppl 可能会轻微地变差
        )
        model = AutoGPTQForCausalLM.from_pretrained(self.save_path, quantize_config)
        model.quantize(self.examples)
        # model.save_quantized(self.save_path, use_safetensors=False) # TOOD: 提供 safetensor 两种保存模式
        print(f"GPTQ quantization {self.bit} success")
