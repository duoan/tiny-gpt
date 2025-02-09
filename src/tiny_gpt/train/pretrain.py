import os
import pickle
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tiny_gpt.model.dataset import PretrainDataset
from tiny_gpt.model.config import TinyGPTConfig
from tiny_gpt.model.model import TinyGPTModel
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

torch.manual_seed(1337)

dataset = "openwebtext"


config = TinyGPTConfig()


device_type = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type)
)


def train_epoch(model, train_loader, optimizer, scaler, epoch, config):
    iter_per_epoch = len(train_loader)
    start_time = time.time()
    for step, (x, y) in enumerate(train_loader):
        if device_type == "cuda":
            x = x.pin_memory().to(device_type, non_blocking=True)
            y = y.pin_memory().to(device_type, non_blocking=True)
        else:
            x = x.to(device_type, non_blocking=True)
            y = y.to(device_type, non_blocking=True)
        with ctx:
            logits, last_loss = model(x, y)
            loss = last_loss / config.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.clip_grad_max_norm
            )

            scaler.step(optimizer)
            scaler.update()

            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        if step % config.log_interval == 0:
            spend_time = time.time() - start_time
            logger.info(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:".format(
                    epoch,
                    config.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * config.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )


def main(xargs):
    data_dir = os.path.join(xargs["data_dir"], dataset)
    train_loader = DataLoader(
        PretrainDataset(os.path.join(data_dir, "train.bin"), config),
        config.batch_size,
        shuffle=True,
        pin_memory=True if device_type == "cuda" else False,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        PretrainDataset(os.path.join(data_dir, "val.bin"), config),
        config.batch_size,
        shuffle=False,
        pin_memory=True if device_type == "cuda" else False,
        num_workers=config.num_workers,
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

    scaler = torch.amp.GradScaler(enabled=(config.dtype in ["float16", "bfloat16"]))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if device_type == "cuda":
        model = torch.compile(model)

    for epoch in range(config.epochs):
        train_epoch(model, train_loader, optimizer, scaler, epoch, config)


if __name__ == "__main__":
    main(xargs={"data_dir": "data"})
