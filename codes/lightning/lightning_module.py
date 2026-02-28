import os
from typing import Any, Dict, Optional, List

import torch
import pytorch_lightning as pl

from ema_pytorch import EMA

from models import networks
from models.modules.loss import MatchingLoss
import utils as util


class LLDBLightningModule(pl.LightningModule):
    def __init__(self, opt: Dict[str, Any]):
        super().__init__()
        self.opt = opt
        self.train_opt = opt["train"]
        self.save_hyperparameters({"name": opt.get("name", "lldb")})

        self.model = networks.define_G(opt)

        is_weighted = self.train_opt.get("is_weighted", False)
        loss_type = self.train_opt.get("loss_type", "l1")
        self.loss_fn = MatchingLoss(loss_type, is_weighted)
        self.loss_weight = self.train_opt.get("weight", 1.0)

        self.ema = EMA(self.model, beta=0.995, update_every=10)
        self.sde = None

        self._val_preview_paths: List[str] = []
        self._val_num_preview = int(self.train_opt.get("val_num_preview", 2))
        if self._val_num_preview > 2:
            self._val_num_preview = 2
        self._val_image_freq = int(
            self.train_opt.get(
                "val_image_freq",
                self.train_opt.get("val_epoch_freq", 1),
            )
        )
        if self._val_image_freq <= 0:
            self._val_image_freq = 1

        self._psnr_fn = None
        self._ssim_fn = None
        self._lpips_fn = None
        self._niqe_fn = None

    def setup(self, stage: Optional[str] = None) -> None:
        device = self.device
        sde_opt = self.opt["sde"]
        self.sde = util.GOUB(
            lambda_square=sde_opt["lambda_square"],
            T=sde_opt["T"],
            schedule=sde_opt["schedule"],
            eps=sde_opt["eps"],
            device=device,
        )
        self.sde.set_model(self.model)
        self.ema = self.ema.to(device)

        metric_device = device
        try:
            import pyiqa
            self._psnr_fn = pyiqa.create_metric("psnr", device=metric_device)
        except Exception:
            self._psnr_fn = None
        try:
            import pyiqa
            self._ssim_fn = pyiqa.create_metric("ssim", device=metric_device)
        except Exception:
            self._ssim_fn = None
        try:
            import pyiqa
            self._lpips_fn = pyiqa.create_metric("lpips", device=metric_device)
        except Exception:
            self._lpips_fn = None
        try:
            import pyiqa
            self._niqe_fn = pyiqa.create_metric("niqe", device=metric_device)
        except Exception:
            self._niqe_fn = None

    def on_validation_epoch_start(self) -> None:
        self._val_preview_paths = []

    def forward(self, xt, cond, time):
        return self.model(xt, cond, time)

    def training_step(self, batch, batch_idx):
        LQ, GT = batch["LQ"].to(self.device), batch["GT"].to(self.device)

        timesteps, states = self.sde.generate_random_states(x0=GT, mu=LQ)
        self.sde.set_mu(LQ)

        timesteps = timesteps.to(self.device)
        states = states.to(self.device)

        noise = self.sde.noise_fn(states, timesteps.squeeze())
        score = self.sde.get_score_from_noise(noise, timesteps)

        xt_1_expectation = self.sde.reverse_sde_step_mean(states, score, timesteps)
        xt_1_optimum = self.sde.reverse_optimum_step(states, GT, timesteps)

        loss = self.loss_fn(xt_1_expectation, xt_1_optimum) * self.loss_weight

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer_closure()
        optimizer.step()
        self.ema.update()

    def validation_step(self, batch, batch_idx):
        LQ, GT = batch["LQ"].to(self.device), batch["GT"].to(self.device)

        self.sde.set_mu(LQ)
        with torch.no_grad():
            output = self.sde.reverse_mean_ode(LQ)

        sr = output.clamp(0, 1)
        gt = GT.clamp(0, 1)

        sr = sr.detach()
        gt = gt.detach()

        metrics = {}
        if self._psnr_fn is not None:
            metrics["val/psnr"] = self._psnr_fn(sr, gt).mean()
        if self._ssim_fn is not None:
            metrics["val/ssim"] = self._ssim_fn(sr, gt).mean()
        if self._lpips_fn is not None:
            metrics["val/lpips"] = self._lpips_fn(sr, gt).mean()
        if self._niqe_fn is not None:
            metrics["val/niqe"] = self._niqe_fn(sr).mean()

        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=(k == "val/psnr"), sync_dist=True)

        if (
            self._val_num_preview > 0
            and self.current_epoch % self._val_image_freq == 0
            and len(self._val_preview_paths) < self._val_num_preview
        ):
            img_path = batch["GT_path"][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            preview_path = os.path.join(self.opt["path"]["val_images"], "artifacts", f"{img_name}_compare_{self.current_epoch}.png")
            self._val_preview_paths.append(preview_path)

            lq_img = util.tensor2img(LQ.detach().cpu().squeeze())
            sr_img = util.tensor2img(sr.detach().cpu().squeeze())
            gt_img = util.tensor2img(gt.detach().cpu().squeeze())

            import numpy as np
            preview_img = np.concatenate([lq_img, sr_img, gt_img], axis=1)

            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            util.save_img(preview_img, preview_path)

        return metrics

    def test_step(self, batch, batch_idx):
        LQ, GT = batch["LQ"].to(self.device), batch["GT"].to(self.device)

        self.sde.set_mu(LQ)
        with torch.no_grad():
            output = self.sde.reverse_mean_ode(LQ)

        sr = output.clamp(0, 1)
        gt = GT.clamp(0, 1)

        sr = sr.detach()
        gt = gt.detach()

        metrics = {}
        if self._psnr_fn is not None:
            metrics["test/psnr"] = self._psnr_fn(sr, gt).mean()
        if self._ssim_fn is not None:
            metrics["test/ssim"] = self._ssim_fn(sr, gt).mean()
        if self._lpips_fn is not None:
            metrics["test/lpips"] = self._lpips_fn(sr, gt).mean()
        if self._niqe_fn is not None:
            metrics["test/niqe"] = self._niqe_fn(sr).mean()

        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return metrics

    def configure_optimizers(self):
        optim_name = self.train_opt.get("optimizer", "Adam")
        lr = self.train_opt.get("lr_G", 1e-4)
        wd = self.train_opt.get("weight_decay_G") or 0
        beta1 = self.train_opt.get("beta1", 0.9)
        beta2 = self.train_opt.get("beta2", 0.99)

        if optim_name == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
        elif optim_name == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
        elif optim_name == "Lion":
            from models.optimizer import Lion
            optimizer = Lion(self.model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))

        lr_scheme = self.train_opt.get("lr_scheme", "MultiStepLR")
        if lr_scheme == "MultiStepLR":
            lr_steps = self.train_opt.get("lr_steps", [])
            lr_gamma = self.train_opt.get("lr_gamma", 0.5)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        if lr_scheme == "TrueCosineAnnealingLR":
            eta_min = self.train_opt.get("eta_min", 1e-7)
            niter = self.train_opt.get("niter", 100000)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=eta_min)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer
