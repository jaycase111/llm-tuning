import math
import argparse
import deepspeed
from functools import partial


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def create_lr_scheduler(args, train_total_steps):
    lr_scheduler = partial(
        cosine_schedule_with_warmup_lr_lambda,

        num_warmup_steps=round(args.lr_warmup_ratio * train_total_steps),
        num_training_steps=train_total_steps,
        min_ratio=args.lr_min_ratio
    )

    return lr_scheduler


def parse_args():
    """
    参数配置函数
    :return:
    """
    parser = argparse.ArgumentParser(description="OpenChat Sft Training.")
    parser.add_argument("--model", type=str, help="Model type. Leave empty to auto-detect.")
    parser.add_argument("--model_type", type=str, default="glm", help="traing model type")
    parser.add_argument("--peft_file", type=str, default=None, help="Peft config file")
    parser.add_argument("--train_file", type=str, help="Training file address")
    parser.add_argument("--epochs", type=int, default=10, help="Model training specifies epoch")
    parser.add_argument("--total_steps", type=int, default=10000, help="Maximum step for model training")
    parser.add_argument("--save_steps", type=int, default=1000, help="Model save steps")
    parser.add_argument("--save_path", type=str, help="save path")
    parser.add_argument("--max_len", default=128, type=int)
    parser.add_argument("--batch_max_len", type=int, default=128)
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher') # Rank值
    # 学习率相关和官方文档保持一致
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--lr_min_ratio",       type=float, default=0.1)
    parser.add_argument("--lr_warmup_ratio",    type=int,   default=0.05)
    parser.add_argument("--weight_decay",       type=float, default=0.1)

    parser.add_argument("--beta1",              type=float, default=0.9)
    parser.add_argument("--beta2",              type=float, default=0.95)
    parser.add_argument("--eps",                type=float, default=1e-5)

    parser = deepspeed.add_config_arguments(parser)

    args, unknown = parser.parse_known_args()
    return args