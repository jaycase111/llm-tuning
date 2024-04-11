import os
import tqdm
import wandb
import argparse
import deepspeed
import torch.distributed as dist
from peft_tuning import get_wrap_model
from datasets.sft import get_dataset
from train import create_lr_scheduler, parse_args



if __name__ == '__main__':
    deepspeed.init_distributed(dist_backend="nccl")
    RANK = dist.get_rank()
    args = parse_args()
    model = get_wrap_model(args.model, args.peft_file)
    dataset = get_dataset(args.model_type, args.model, args.train_file, args.max_len)
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(), training_data=dataset)
    step, progress_bar = 0, None

    train_total_steps = args.epochs * len(trainloader)
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_total_steps)
        wandb.init(project=os.path.basename(args.model), config=args)
    lr_scheduler = create_lr_scheduler(args, train_total_steps)

    for epoch in range(args.epochs):
        print(f"[rank {RANK}]: Epoch {epoch}")
        model_engine.train()
        for i, data in enumerate(trainloader):
            data = {key:value.to(model_engine.local_rank) for key, value in data.items()}
            output = model_engine(**data)
            loss = output.loss
            model_engine.backward(loss)

            if model_engine.is_gradient_accumulation_boundary():
                lr_this_step = args.lr * lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            model_engine.step()

            step += 1

            if RANK == 0:
                wandb.log({
                    "train/loss": loss.item()
                }, step=step)
                progress_bar.update()

            if step == args.total_steps:
                # TODO 后续完成退出机制
                pass
            if step % args.save_steps:
                # TODO 后续支持权重保存机制
                pass
    dist.barrier()
    # TODO 后续支持完成整体训练权重保存机制





