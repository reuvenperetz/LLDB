import argparse
import os
import sys
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger

import utils as util

from lightning.lightning_module import LLDBLightningModule
from lightning.lightning_data import LLDBDataModule


class PeriodicPTHCheckpoint(Callback):
    def __init__(self, models_dir: str, every_n_steps: int):
        super().__init__()
        self.models_dir = models_dir
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0:
            return
        global_step = trainer.global_step
        if global_step % self.every_n_steps != 0:
            return

        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, f"{global_step}_G.pth")
        ema_path = os.path.join(self.models_dir, f"{global_step}_EMA.pth")

        base_model = pl_module.model
        torch_state = base_model.state_dict()
        ema_state = pl_module.ema.ema_model.state_dict() if hasattr(pl_module, "ema") else None

        import torch
        torch.save(torch_state, model_path)
        if ema_state is not None:
            torch.save(ema_state, ema_path)


class LatestPTHCheckpoint(Callback):
    def __init__(self, models_dir: str):
        super().__init__()
        self.models_dir = models_dir

    def on_train_end(self, trainer, pl_module):
        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, "latest_G.pth")
        ema_path = os.path.join(self.models_dir, "latest_EMA.pth")

        import torch
        torch.save(pl_module.model.state_dict(), model_path)
        torch.save(pl_module.ema.ema_model.state_dict(), ema_path)


def run(opt: Dict[str, Any], num_devices: Optional[int] = None) -> None:
    resume_state = opt["path"].get("resume_state", None)
    if resume_state is None:
        util.mkdir_and_rename(opt["path"]["experiments_root"])
        util.mkdirs(
            (
                path
                for key, path in opt["path"].items()
                if key != "experiments_root"
                and "pretrain_model" not in key
                and "resume" not in key
            )
        )
    else:
        os.makedirs(opt["path"]["experiments_root"], exist_ok=True)
        os.makedirs(opt["path"]["models"], exist_ok=True)
        os.makedirs(opt["path"]["val_images"], exist_ok=True)

    use_tb = opt.get("use_tb_logger", False)
    logger = None
    if use_tb:
        logger = TensorBoardLogger(save_dir="log", name=opt["name"])

    callbacks = []

    save_freq = int(opt["logger"].get("save_checkpoint_freq", 0))
    if save_freq > 0:
        callbacks.append(PeriodicPTHCheckpoint(opt["path"]["models"], save_freq))

    callbacks.append(LatestPTHCheckpoint(opt["path"]["models"]))

    checkpoint_cb = ModelCheckpoint(
        dirpath=opt["path"]["models"],
        filename="{epoch}-{step}-{val/psnr:.4f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(checkpoint_cb)

    train_opt = opt["train"]
    max_steps = train_opt.get("niter")
    max_epochs = train_opt.get("epochs")
    if max_steps is None:
        max_steps = -1
    if max_epochs is None:
        max_epochs = -1

    val_epoch_freq = int(train_opt.get("val_epoch_freq", 0))
    check_val_every_n_epoch = val_epoch_freq if val_epoch_freq > 0 else 1

    val_check_interval = None
    if train_opt.get("val_freq") is not None:
        try:
            val_check_interval = int(train_opt["val_freq"])
            if val_check_interval <= 0:
                val_check_interval = None
        except Exception:
            val_check_interval = None

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_devices,
        strategy="ddp" if (num_devices and num_devices > 1) else "auto",
        max_steps=max_steps,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        log_every_n_steps=int(opt["logger"].get("print_freq", 100)),
        enable_progress_bar=True,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )

    data_module = LLDBDataModule(opt)
    module = LLDBLightningModule(opt)

    trainer.fit(module, datamodule=data_module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to option YAML file.")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices (GPUs).")
    args = parser.parse_args()

    # Options parsing is task-specific. This file expects the caller to parse options.
    # For direct use, fall back to lol-v1 options parser if present.
    opt_path = args.opt
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Try to load options parser from the task directory containing the opt file.
    opt_dir = os.path.dirname(opt_path)
    task_dir = os.path.dirname(opt_dir)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)

    try:
        import options as option
    except Exception as exc:
        raise RuntimeError(f"Failed to import options.py from {task_dir}: {exc}")

    opt = option.parse(opt_path, is_train=True)
    opt = option.dict_to_nonedict(opt)

    run(opt, num_devices=args.devices)


if __name__ == "__main__":
    main()
