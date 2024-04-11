import torch
import torch.optim as optim
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from tqdm import tqdm



from transformers import AutoModel
from quantization.base_quantization import \
    BaseQuantization, Module, PreTrainedTokenizer, List
from torch.utils.data import DataLoader


def get_modules(layer):
    # NOTE: This is llama-specific
    # For other models, replace this with proper names for all linear layers
    return[
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ]


class SequeezeLlmQuantization(BaseQuantization):
    """
        SqueezeLLM量化--需要使用校准数据集
    """

    def __init__(self,
                 model_str: str,
                 bit: str,
                 save_path: str,
                 dataloader: DataLoader
                 ):
        super().__init__(model_str, bit,
                         save_path, dataloader=dataloader)

    def quant_not_use_examples(self):
        self.examples = []
        self.quant_use_examples()

    def quant_use_examples(self):
        model = AutoModel.from_pretrained(self.model_str)
        model = model.bfloat16()
        _model = model.model
        _layers = _model.layers
        _model.set_devices()

        def square_grad_hook(grad):
            return grad.pow(2)

        for layer in _layers:
            for module in get_modules(layer):
                module.weight.register_hook(square_grad_hook)

        for data in tqdm(self.dataloader):
            data = data[0]
            x = data.cuda()
            outputs = model(input_ids=x, labels=x)
            loss = outputs.loss
            loss.backward()

        for layer in _layers:
            for module in get_modules(layer):
                module.weight.data = module.weight.grad

        model.save_pretrained(self.save_path)
        print(f"SqueezeLLM quantization {self.bit} success")

