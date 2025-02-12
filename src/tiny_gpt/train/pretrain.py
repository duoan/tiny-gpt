import os
import torch
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import time
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
    val_loader,
    config: TinyGPTConfig,
):
    model.eval()

    iter_steps = len(val_loader)
    losses = torch.zeros(iter_steps)
    for step, (x, y) in enumerate(val_loader):
        x, y = device_put(x, y)
        with ctx:
            _, loss = model(x, y)
        losses[step] = loss.item()

        if (step + 1) % config.log_interval == 0:
            logger.info(
                f"Evaluation step: {step}/{iter_steps}, loss: {losses[:step+1].mean()}"
            )

    model.train()
    return losses.mean().item()


def check_invalid_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            valid = torch.isfinite(param.grad).all()
            if not valid:
                return False, name
    return True, None


best_val_loss = 1e9


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: TinyGPTConfig,
):
    global best_val_loss

    iter_per_epoch = len(train_loader)
    total_steps = iter_per_epoch * config.epochs

    train_start_time = time.time()
    train_epoch_losses = torch.zeros(iter_per_epoch)
    tokens_per_batch = config.batch_size * config.seq_len
    for step, (x, y) in enumerate(train_loader):
        global_step = epoch * iter_per_epoch + step + 1
        # Evaluation
        eval_time_span = 0
        if epoch > 0 and global_step % config.eval_interval == 0:
            eval_start_time = time.time()
            val_loss = estimate_loss(model, val_loader, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "epoch_step": step,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                checkpoint_path = os.path.join(
                    config.out_dir, f"pretrain_ckpt_{global_step}.pt"
                )
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)

            eval_time_span = time.time() - eval_start_time
            log_data = {
                "val/loss": val_loss,
                "val/time": eval_time_span,
            }
            logger.info(f"Evaluation {global_step},{log_data}")
            wandb.log(log_data, global_step)

        # Training
        x, y = device_put(x, y)
        with ctx:
            _, last_loss = model(x, y)
            train_epoch_losses[step] = last_loss.item()
            loss = last_loss / config.accumulation_steps

        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % config.accumulation_steps == 0:
            # reverse scale the gradient
            scaler.unscale_(optimizer)

            grads_valid, invalid_param = check_invalid_gradients(model)
            grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            grad_norm_post_clip = torch.stack(
                [p.grad.norm() for p in model.parameters() if p.grad is not None]
            ).norm()

            wandb.log(
                {
                    "train/grad_norm_pre_clip": grad_norm_pre_clip,
                    "train/grad_norm_post_clip": grad_norm_post_clip,
                    "train/grad_norm_clip_ratio": grad_norm_post_clip
                    / grad_norm_pre_clip,
                },
                global_step,
            )

            # update parameters if gradient in normal range
            if grads_valid:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                logger.warning(
                    f"Step {step}: Skip parameter update due to invalid gradients in {invalid_param}. "
                    f"Pre-clip grad norm: {grad_norm_pre_clip:.4f}, "
                    f"Post-clip grad norm: {grad_norm_post_clip:.4f}"
                )

            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        if (1 + step) % config.log_interval == 0:
            train_step_time = (
                time.time() - train_start_time + eval_time_span
            ) / config.log_interval
            estimated_remaining_time = (total_steps - global_step) * train_step_time
            train_grad_loss = loss.item() * config.accumulation_steps
            train_mean_loss = train_epoch_losses[: step + 1].mean()
            current_lr = optimizer.param_groups[0]["lr"]
            current_momentum = optimizer.param_groups[0]["betas"][0]
            tokens_per_second = tokens_per_batch / train_step_time

            logger.info(
                f"Epoch:[{ epoch + 1}/{config.epochs}]({step + 1}/{iter_per_epoch}) "
                f"train_losss grad:{train_grad_loss:.3f} mean:{train_mean_loss:.3f} "
                f"lr:{current_lr:.7f} momentum:{current_momentum:.7f} "
                f"step_time:{train_step_time:.4f} seconds "
                f"remain_time: {estimated_remaining_time//60} mins"
            )

            log_data = {
                "train/grad_loss": train_grad_loss,
                "train/mean_loss": train_mean_loss,
                "train/lr": current_lr,
                "train/momentum": current_momentum,
                "train/step_time": train_step_time,
                "train/tokens_per_second": tokens_per_second,
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
                    # the parameter distribution should be smooth and no extreme values
                    try:
                        # Use fewer bins and handle small ranges
                        num_bins = min(
                            64, int(np.sqrt(param_data.size))
                        )  # Reduce number of bins
                        hist = wandb.Histogram(param_data, num_bins=num_bins)
                        log_data[f"param/dist/{param_name}"] = hist
                    except ValueError as e:
                        # If histogram creation fails, log basic statistics instead
                        log_data[f"param/stats/{param_name}/mean"] = np.mean(param_data)
                        log_data[f"param/stats/{param_name}/std"] = np.std(param_data)
                        log_data[f"param/stats/{param_name}/min"] = np.min(param_data)
                        log_data[f"param/stats/{param_name}/max"] = np.max(param_data)
                        logger.warning(
                            f"Failed to create histogram for {param_name}: {str(e)}"
                        )

                if param.grad is not None:
                    # monitor the parameter level gradients
                    grad_norm = param.grad.norm().item()
                    param_norm = param.norm().item()
                    # gradient normalization
                    log_data[f"gradients/{param_name}/norm"] = grad_norm
                    # the parameter normalization
                    log_data[f"gradients/{param_name}/weight_norm"] = param_norm
                    # the ratio of gradient to parameter
                    grad_to_weight_ratio = grad_norm / (param_norm + 1e-8)
                    log_data[f"gradients/{param_name}/grad_to_weight_ratio"] = (
                        grad_to_weight_ratio
                    )
                    # parameter update ratio. when it is large, the learning is not stable.
                    update_ratio = current_lr * grad_to_weight_ratio
                    log_data[f"optim/{param_name}/update_ratio"] = update_ratio

            wandb.log(log_data, global_step)

            train_start_time = time.time()


def main(xargs):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = TinyGPTConfig(**args)
    config.out_dir = os.path.join(config.out_dir, run_id)

    # ensure the output dir is ready
    os.makedirs(config.out_dir, exist_ok=True)

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
            num_samples=int(len(train_dataset) * config.train_sample_rate),
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        config.batch_size,
        pin_memory=True if device_type == "cuda" else False,
        num_workers=config.num_workers,
        sampler=RandomSampler(
            val_dataset, num_samples=int(len(val_dataset) * config.val_sample_rate)
        ),
    )

    f"ValDataset: {len(val_dataset):,}, ValDataloader: {len(val_loader):,}"

    logger.info(
        f"TrainDataset: {len(train_dataset):,}, TrainDataloader: {len(train_loader):,}"
    )

    model = TinyGPTModel(config).to(device_type)
    if device_type == "cuda":
        model = torch.compile(model)

    scaler = torch.amp.GradScaler(
        enabled=(config.dtype in ["float16", "bfloat16"]),
        init_scale=2**14,
        growth_factor=1.2,
        growth_interval=4000,
        backoff_factor=0.5,
    )

    optimizer = model.configure_optimizer(device_type)

    scheduler_total_steps = (
        config.epochs * len(train_loader) // config.accumulation_steps
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=scheduler_total_steps,
        pct_start=config.scheduler_warmup_pct,  # 10% steps for warmup
        anneal_strategy=config.scheduler_anneal_strategy,
        div_factor=config.scheduler_div_factor,  # start lr = max_lr/25
        final_div_factor=config.scheduler_final_div_factor,  # final lr = max_lr/1e4
        base_momentum=config.scheduler_base_momentum,
        max_momentum=config.scheduler_max_momentum,
        cycle_momentum=config.scheduler_cycle_momentum,
    )

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
