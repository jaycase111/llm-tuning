import os
import time
import random
import torch
import argparse
import deepspeed
from dschat.utils.utils import print_rank_0
from transformers import AutoTokenizer
from transformers import (
    SchedulerType,
    default_data_collator,
)
from dschat.utils.utils import set_random_seed
from deepspeed.accelerator import get_accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from dschat.rlhf.ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from dschat.utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from torch.utils.data.distributed import DistributedSampler
from dschat.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from dschat.utils.utils import get_optimizer_grouped_parameters
from dschat.utils.data.data_utils import DataCollatorRLHF,pad_sequence
from torch.utils.data import Dataset, DataLoader
from dschat.rlhf.rlhf_engine import DeepSpeedRLHFEngine
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer, \
    ExponentialMovingAverage
from dschat.utils.perf import print_throughput_step3
from dschat.rlhf.rlhf_engine import DeepSpeedRLHFEngine, log_init, make_model_gradient_checkpointing_compatible, \
    DeepSpeedCPUAdam, FusedAdam, get_scheduler
from train.rlhf_train.rlhf_reward import create_critic_model
from datasets.rlhf_data import RlhfTrainDataset, BatchCollatorRLHF, create_rlhf_dataset


class AutoDeepSpeedRLHFEngine(DeepSpeedRLHFEngine):

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters):
        super().__init__(actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters)

    def _init_critic(self, critic_model_name_or_path):
        stime = log_init("Critic")
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            dtype=self.args.dtype,
            stage=self.args.critic_zero_stage,
            enable_tensorboard=self.args.enable_tensorboard,
            tb_path=self.args.tensorboard_path,
            tb_name="step3_critic")
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        # TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
        ) * self.args.gradient_accumulation_steps
        # 设置Critc训练DeepSpeed配置

        ds_eval_config = get_eval_ds_config(offload=False,
                                            dtype=self.args.dtype,
                                            stage=self.args.critic_zero_stage)
        # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
        ds_eval_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_eval_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
        ) * self.args.gradient_accumulation_steps
        # 设置Critc验证DeepSpeed配置

        # Model
        critic_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            dropout=self.args.critic_dropout,
            zero_stage=self.args.critic_zero_stage)

        # LoRA
        if self.args.critic_lora_dim > 0:
            critic_model = convert_linear_layer_to_lora(
                critic_model, self.args.critic_lora_module_name,
                self.args.critic_lora_dim)
            if self.args.only_optimize_lora:
                critic_model = only_optimize_lora_parameters(critic_model)
                critic_model = make_model_gradient_checkpointing_compatible(
                    critic_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            critic_model, self.args.critic_weight_decay,
            self.args.critic_lora_learning_rate)
        optim = AdamOptimizer(optim_params,
                              lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config)

        log_init("Critic", stime=stime)
        return critic_engine

    def _init_reward(self, critic_model_name_or_path):
        """
            创建reward模型及model_engine
        """
        stime = log_init("Reward")
        # DS Config
        zero_stage = self.args.critic_zero_stage
        if zero_stage != 3:
            # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0

        ds_config = get_eval_ds_config(offload=self.args.offload,
                                       dtype=self.args.dtype,
                                       stage=zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
        ) * self.args.gradient_accumulation_steps

        ds_eval_config = get_eval_ds_config(offload=False,
                                            dtype=self.args.dtype,
                                            stage=zero_stage)

        # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
        ds_eval_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_eval_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
        ) * self.args.gradient_accumulation_steps

        # Model
        reward_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            dropout=self.args.critic_dropout,
            zero_stage=zero_stage)

        reward_model.config.model_type = "llama"

        print("reward_model ", reward_model.config.model_type)

        reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                                 config=ds_config)

        log_init("Reward", stime=stime)
        return reward_engine


def create_datasets(args,
                        tokenizer):
    """
    :param args:            参数
    :param tokenizer:       分词器
    :return:                这里不做 rank 分配
    """
    prompt_train_dataset = create_rlhf_dataset(args.local_rank,
                        args.data_file,
                        args.seed,
                        tokenizer,
                        args.max_prompt_seq_len)

    data_collator = BatchCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)

    prompt_train_sampler = DistributedSampler(prompt_train_dataset)

    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size)

    num_update_steps_per_epoch = len(prompt_train_dataloader) * \
                                 (args.per_device_generation_batch_size / args.per_device_training_batch_size) * \
                                 args.ppo_epochs / args.gradient_accumulation_steps  # 每个epoch中PPO迭代更新次数
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, num_total_iters

def parse_args():
    global writer
    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    # 训练文件
    parser.add_argument(
        '--data_file',
        type=str,
        help=
        'Path to the training dataset.'
    )

    # 决策模型文件地址
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)

    # reward模型文件地址
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)

    # padding数量
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )

    # 每张显卡生成的数量
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=1,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )

    # 每张显卡训练数量
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=1,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )

    # 生成批数
    parser.add_argument("--generation_batches",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")

    # ppo_batch 数量
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")

    # 最大Prompt Token输入长度
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=128,
                        help="The maximum sequence length.")

    # 最大Answer Token输入长度
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=128,
                        help="The maximum sequence length.")

    # 决策网络学习率
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )

    # reward网络学习率
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )

    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")

    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")

    # 总的训练Epochs
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")

    # 学习率调整器
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )

    # 梯度累计步数
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")

    # warmpup 步数
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")

    # 模型保存地址
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")

    # 随机数
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")

    # 总进程数
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    # rank 值
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed 相关配置
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )

    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')

    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')

    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')

    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,

        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')

    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')

    parser.add_argument(
        "--actor_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the actor model."
    )

    parser.add_argument(
        "--critic_dropout",
        type=float,
        default=None,
        help="If critic dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the critic model."
    )

    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")

    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")

    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")

    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")

    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial actor LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial critic LoRA learning rate (after the potential warmup period) to use."
    )
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')
    ## Mixed Precision ZeRO++
    parser.add_argument(
        '--enable_mixed_precision_lora',
        action='store_true',
        help='Enable Mixed Precision ZeRO++ for training and generation.')
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.'
        'This applies for both actor and critic models.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step3_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Actor/critic model overflow alignment
    parser.add_argument(
        '--align_overflow',
        action='store_true',
        help='Align loss scale overflow between actor and critic')
    ## Print actor model answers during training
    parser.add_argument('--print_answers',
                        action='store_true',
                        help='Print prompt and answers during training')
    parser.add_argument(
        "--print_answers_interval",
        type=int,
        default=1,
        help="If --print_answers enabled, controls the printing interval.")
    ## Testing
    parser.add_argument(
        '--enable_test_mode',
        action='store_true',
        help=
        'Enable a testing mode that terminates training based on args.test_stop_step'
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help=
        "Training non-overflow step at which to terminate training during testing."
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.enable_tensorboard:
        print(
            f"Tensorboard logs going to: {args.tensorboard_path}/step3_tensorboard_logs"
        )
        writer = SummaryWriter(
            f"{args.tensorboard_path}/step3_tensorboard_logs")

    # Validate settings
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if args.actor_zero_stage == 2 and args.critic_zero_stage == 2 and args.enable_hybrid_engine and args.offload and args.actor_lora_dim == 0:
        raise ValueError(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )

    return args




if __name__ == '__main__':
    args = parse_args()
    local_rank = args.local_rank

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed(dist_backend="nccl")                 # 设置分布式Device

    args.global_rank = torch.distributed.get_rank()

    args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps  # 设置actor梯度累计步数
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # 确定 当前模型是否支持 结束文本符
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    tokenizer = AutoTokenizer.from_pretrained(args.actor_model_name_or_path,
                                              fast_tokenizer=True,
                                              add_special_tokens=additional_special_tokens)  # 设置分词器
    tokenizer.add_special_tokens({"pad_token": '[PAD]'})

    print("tokenizer ", tokenizer)

    # 构建RLHF训练集
    prompt_train_dataloader, num_total_iters = create_datasets(args,
                                                               tokenizer)

    # DeepSpeed 训练引擎
    rlhf_engine = AutoDeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)  # 加载 actor | ref | reward | critic 网络 自动完成 lora 加载

    if args.enable_mixed_precision_lora:
        assert args.actor_lora_dim > 0, "Mixed Precision LoRA requires LoRA to be enabled"
        assert args.actor_zero_stage == 3, "Mixed Precision LoRA requires Zero stage 3"
        rlhf_engine.actor.optimizer.quantize_nontrainable_params()
        print_rank_0("Mixed Precision ZeRO++ enabled")

    ppo_trainer = DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)  # 构建训练器

    exp_mini_dataset = MiniDataset(args.generation_batches,  # 对应真实的max_size
                                   args.per_device_training_batch_size  # 对应真实的 small_batch_size

                                   )  # rlhf数据器 包括真实的生成 | 奖励 |

    print_rank_0(
        f"***** Running training (total_iters={num_total_iters}) *****",
        args.global_rank)

    non_overflow_step_count = 0
    step_average_reward = 0.

    ema_reward_score = ExponentialMovingAverage()  # ema 记录 rlhf 冲量

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Generation Batches {len(prompt_train_dataloader)}",
            args.global_rank)

        for step, batch_prompt in enumerate(
                prompt_train_dataloader):

            batch_prompt = to_device(batch_prompt, device)  # 输入问题

            # 多输入的Question生成答案
            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'],
                                              step)  # 结果生成

            training_start = time.time()
            exp_dataset = exp_mini_dataset.add(out)  # 取出一个子采样数据集

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for ppo_ep in range(args.ppo_epochs):
                    for i, exp_data in enumerate(exp_dataset):
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        # 并且内部完成数据训练、
                        # 输出actor_loss 和 critic_loss、方便打印
                        actor_loss_sum += actor_loss.item()  #
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()

                        inner_iter += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)

                end = time.time()
                training_time = end - training_start
                e2e_time = training_time + trainer.generate_time * args.generation_batches  # it is an approximation, we did not include, e.g., rw forward time etc

                print_rank_0(
                    f'Epoch: {epoch} | Step: {step} | PPO Epoch: {ppo_ep + 1} | Actor Loss: {actor_loss_sum / inner_iter} | Critic Loss: {critic_loss_sum / inner_iter} | Unsupervised Loss: {unsup_loss_sum / inner_iter}',
                    args.global_rank)

                average_reward = get_all_reduce_mean(average_reward).item()
                step_average_reward += average_reward / args.gradient_accumulation_steps_actor
                if (step + 1) % args.gradient_accumulation_steps_actor == 0:
                    ema_reward_score.update(step_average_reward)
                    step_average_reward = 0.

                print_rank_0(
                    f"Average reward score: {average_reward / inner_iter} | EMA reward score: {ema_reward_score.get()}",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)

                if args.enable_tensorboard and torch.distributed.get_rank(
                ) == 0:
                    writer.add_scalar('reward',
                                      average_reward / inner_iter,
                                      global_step=step)
                    writer.add_scalar('actor_loss',
                                      actor_loss.item(),
                                      global_step=step)
                    writer.add_scalar('actor_loss_sum',
                                      actor_loss_sum,
                                      global_step=step)
                    writer.add_scalar('critic_loss',
                                      critic_loss.item(),
                                      global_step=step)
                    writer.add_scalar('critic_loss_sum',
                                      critic_loss_sum,
                                      global_step=step)
                    writer.flush()

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()  # 禁用梯度检查点、减少梯度使用

            actor_overflow, critic_overflow = trainer.get_overflow()

            if not actor_overflow and not critic_overflow:
                non_overflow_step_count += 1

            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break

        if args.enable_test_mode:
            break

    if args.output_dir is not None:
        print_rank_0('saving model ...')
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)  # 将Lora模型转化为正常模型
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            # actor | critic | ema 肯定保存
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')
            save_hf_format(rlhf_engine.critic,
                           tokenizer,
                           args,
                           sub_folder='critic')
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')  # 将lora参数全部不保存

        # # zero3特有读取方案 保存在其他节点上运行的参数
        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=args.global_rank,
                                      save_dir=os.path.join(
                                          args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)
        if args.critic_zero_stage == 3:
            save_zero_three_model(rlhf_engine.critic,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)

        # 保存的是完整参数 | 除了Lora参数

        torch.distributed.barrier()
        print("save success")