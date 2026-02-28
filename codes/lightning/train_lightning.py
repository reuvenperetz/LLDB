import argparse
import os
import sys
from typing import Any, Dict, Optional
import math

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
try:
    from pytorch_lightning.loggers import MLFlowLogger
except Exception:
    MLFlowLogger = None

import utils as util

from lightning.lightning_module import LLDBLightningModule
from lightning.lightning_data import LLDBDataModule
from data import create_dataset


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


class MLFlowArtifactsCallback(Callback):
    def __init__(self, val_images_dir: str, models_dir: str, val_image_freq: int) -> None:
        super().__init__()
        self.val_images_dir = val_images_dir
        self.models_dir = models_dir
        self.val_image_freq = max(int(val_image_freq), 1)

    def _get_mlflow_logger(self, trainer):
        if MLFlowLogger is None:
            return None
        loggers = getattr(trainer, "loggers", None)
        if not loggers:
            single_logger = getattr(trainer, "logger", None)
            loggers = [single_logger] if single_logger else []
        for logger in loggers:
            if isinstance(logger, MLFlowLogger):
                return logger
        return None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        if trainer.current_epoch % self.val_image_freq != 0:
            return
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return
        try:
            client = mlflow_logger.experiment
            run_id = mlflow_logger.run_id
            epoch_tag = f"epoch_{trainer.current_epoch}"
            if os.path.isdir(self.val_images_dir):
                client.log_artifacts(
                    run_id,
                    self.val_images_dir,
                    artifact_path=os.path.join("val_images", epoch_tag),
                )
            if os.path.isdir(self.models_dir):
                client.log_artifacts(
                    run_id,
                    self.models_dir,
                    artifact_path=os.path.join("models", epoch_tag),
                )
        except Exception:
            pass


def run(opt: Dict[str, Any], num_devices: Optional[int] = None) -> None:
    resume_state = opt["path"].get("resume_state", None)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if resume_state is None and (world_size <= 1 or rank == 0):
        if os.environ.get("LLDB_SKIP_RENAME", "0") != "1":
            try:
                util.mkdir_and_rename(opt["path"]["experiments_root"])
            except OSError:
                pass
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
    loggers = []
    if use_tb:
        loggers.append(TensorBoardLogger(save_dir="log", name=opt["name"]))
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if MLFlowLogger is not None and mlflow_tracking_uri:
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT", opt.get("name", "lldb"))
        mlflow_run_name = os.environ.get("JOB_NAME")
        loggers.append(
            MLFlowLogger(
                experiment_name=mlflow_experiment,
                tracking_uri=mlflow_tracking_uri,
                run_name=mlflow_run_name,
                log_model=True,
            )
        )
    if len(loggers) == 1:
        logger = loggers[0]
    elif len(loggers) > 1:
        logger = loggers
    else:
        logger = None

    callbacks = []

    train_opt = opt["train"]

    checkpoint_cb = ModelCheckpoint(
        dirpath=opt["path"]["models"],
        filename="best",
        monitor="val/psnr",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    callbacks.append(checkpoint_cb)
    if mlflow_tracking_uri:
        val_image_freq = int(
            train_opt.get(
                "val_image_freq",
                train_opt.get("val_epoch_freq", 1),
            )
        )
        callbacks.append(
            MLFlowArtifactsCallback(
                opt["path"]["val_images"],
                opt["path"]["models"],
                val_image_freq,
            )
        )

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

    if val_check_interval is not None:
        train_set = create_dataset(opt["datasets"]["train"])
        max_train_images = opt["datasets"]["train"].get("max_train_images")
        if max_train_images is not None:
            train_set = list(range(min(len(train_set), max_train_images)))
        batch_size = opt["datasets"]["train"]["batch_size"]
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            if batch_size % world_size != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
                )
            batch_size = batch_size // world_size
        train_batches = int(math.ceil(len(train_set) / batch_size))
        if val_check_interval > train_batches:
            print(
                f"[warn] val_freq ({val_check_interval}) > train_batches ({train_batches}); "
                "falling back to once-per-epoch validation."
            )
            val_check_interval = None
            check_val_every_n_epoch = 1
        else:
            check_val_every_n_epoch = None

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_devices,
        strategy=(
            "ddp_find_unused_parameters_true"
            if (num_devices and num_devices > 1)
            else "auto"
        ),
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

    if opt.get("datasets", {}).get("test") is not None:
        trainer.test(module, datamodule=data_module)


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
