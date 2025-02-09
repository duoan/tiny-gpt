import os
import pickle
import torch
import numpy as np
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
import time
from tqdm import tqdm
from datetime import datetime
from contextlib import nullcontext
from torch.utils.data import DataLoader, RandomSampler
from tiny_gpt.model.dataset import PretrainDataset
from tiny_gpt.model.config import TinyGPTConfig
from tiny_gpt.model.model import TinyGPTModel
import logging
import wandb
import sys


logger = logging.getLogger(__name__)

torch.manual_seed(1337)
np.random.seed(1337)

dataset = "openwebtext"

device_type = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

args = {
    "dtype": dtype,
}

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type)
)


def device_put(x, y):
    if device_type == "cuda":
        x = x.pin_memory().to(device_type, non_blocking=True)
        y = y.pin_memory().to(device_type, non_blocking=True)
    else:
        x = x.to(device_type, non_blocking=True)
        y = y.to(device_type, non_blocking=True)

    return x, y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.inference_mode()
def estimate_loss(
    model,
    loaders,
    config: TinyGPTConfig,
):
    model.eval()
    out = {}
    for split in loaders.keys():
        losses = torch.zeros(len(loaders[split]))
        for step, (x, y) in enumerate(loaders[split]):
            x, y = device_put(x, y)
            with ctx:
                _, loss = model(x, y)
            losses[step] = loss.item()

            if (step + 1) % config.log_interval == 0:
                logger.info(
                    f"Evaluation [{split}] step: {step}, loss: {losses[:step+1].mean()}"
                )

        out[split] = losses.mean()
    model.train()
    return out


best_val_loss = 1e9


def train_epoch(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    epoch: int,
    config: TinyGPTConfig,
):
    global best_val_loss

    iter_per_epoch = len(train_loader)
    start_time = time.time()
    for step, (x, y) in enumerate(train_loader):
        global_step = epoch * iter_per_epoch + step
        # Evaluation
        if (global_step + 1) % config.eval_interval == 0:
            losses = estimate_loss(
                model, {"train": train_loader, "val": val_loader}, config
            )
            log_data = {
                "train/loss": losses["train"],
                "val/loss": losses["val"],
            }
            logger.info(f"step: {global_step},{log_data}")
            wandb.log(log_data, global_step)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                checkpoint_path = os.path.join(
                    config.out_dir, f"pretrain_ckpt_{global_step}.pt"
                )
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)

        # Training
        x, y = device_put(x, y)
        with ctx:
            logits, last_loss = model(x, y)
            loss = last_loss / config.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            wandb.log(
                {
                    "grad_norm": torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.clip_grad_max_norm
                    )
                },
                global_step,
            )

            scaler.step(optimizer)
            scaler.update()

            # update learning rate
            scheduler.step()

            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        if step % config.log_interval == 0:
            spend_time = time.time() - start_time
            word_counts = np.histogram(torch.get_device(x), bins=50)
            log_data = {
                "loss": loss.item() * config.accumulation_steps,
                "lr": optimizer.param_groups[0]["lr"],
                "spend_time": spend_time,
                "word_distribution": wandb.Histogram(word_counts[0]),
            }

            for param_name, param in model.named_parameters():
                if param.requires_grad is None:
                    continue
                param_data = param.data.cpu().detach().numpy()

                # Check for NaN values
                if np.isnan(param_data).any():
                    logger.warning(
                        f"NaN detected in parameter {param_name} at step {global_step}"
                    )
                    continue

                # Only log if we have valid finite values
                if np.isfinite(param_data).any():
                    log_data[f"param_distribution/{param_name}"] = wandb.Histogram(
                        param_data
                    )

            wandb.log(log_data, global_step)
            logger.info(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} step_time:{}min:".format(
                    epoch,
                    config.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * config.accumulation_steps,
                    optimizer.param_groups[0]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )
            start_time = time.time()


def main(xargs):
    config = TinyGPTConfig(**args)

    # ensure the output dir is ready
    os.makedirs(config.out_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = f"pretrain_{run_id}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",  # Log format
        handlers=[
            logging.FileHandler(log_filename),  # Log to a file
            logging.StreamHandler(sys.stdout),  # Print to the console
        ],
    )
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"tiny-gpt-{device_type}",
        name=run_id,
        # track hyperparameters and run metadata
        config=config.to_dict(),
    )

    data_dir = os.path.join(xargs["data_dir"], dataset)

    train_dataset = PretrainDataset(os.path.join(data_dir, "train.bin"), config)
    val_dataset = PretrainDataset(os.path.join(data_dir, "val.bin"), config)
    # creat new dataloader each epoch, ensure the random sampler pick up differrent sets.
    train_loader = DataLoader(
        train_dataset,
        config.batch_size,
        pin_memory=True if device_type == "cuda" else False,
        num_workers=config.num_workers,
        sampler=RandomSampler(
            train_dataset,
            num_samples=int(len(train_dataset) * config.sample_rate),
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        config.batch_size,
        pin_memory=True if device_type == "cuda" else False,
        num_workers=config.num_workers,
        sampler=RandomSampler(
            val_dataset, num_samples=int(len(val_dataset) * config.sample_rate)
        ),
    )
    logger.info(
        f"TrainDataset: {len(train_dataset):,}, TrainDataloader: {len(train_loader):,} "
        f"ValDataset: {len(val_dataset):,}, ValDataloader: {len(val_loader):,}"
    )

    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        config.vocab_size = meta_vocab_size

    model = TinyGPTModel(config).to(device_type)
    if device_type == "cuda":
        model = torch.compile(model)

    scaler = torch.amp.GradScaler(enabled=(config.dtype in ["float16", "bfloat16"]))
    optimizer = model.configure_optimizer(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        device_type,
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.001, end_factor=1.0, total_iters=config.warmup_steps
    )

    total_steps = len(train_loader) * config.epochs
    effective_steps = (total_steps - config.warmup_steps) // config.accumulation_steps
    cosin_scheduler = CosineAnnealingLR(optimizer, T_max=effective_steps, eta_min=0.001)

    scheduler = ChainedScheduler([warmup_scheduler, cosin_scheduler])

    for epoch in range(config.epochs):
        train_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            scaler,
            epoch,
            config,
        )


if __name__ == "__main__":
    main(xargs={"data_dir": "data"})
