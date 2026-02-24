import argparse
import logging
import math
import os
import random
import sys
import copy
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed
import pyiqa
import mlflow
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

import options as option

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from models import create_model

import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        if torch.backends.mps.is_available():
            map_location = "mps"
        elif torch.cuda.is_available():
            map_location = "cuda"
        else:
            map_location = "cpu"
        resume_state = torch.load(opt["path"]["resume_state"], map_location=map_location)
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            max_train_images = dataset_opt.get("max_train_images")
            if max_train_images is not None:
                train_set = torch.utils.data.Subset(
                    train_set, list(range(min(len(train_set), max_train_images)))
                )
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            if opt["train"].get("epochs") is not None:
                total_epochs = int(opt["train"]["epochs"])
                total_iters = int(total_epochs * train_size)
            else:
                total_iters = int(opt["train"]["niter"])
                total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### mlflow setup (rank 0 only)
    def _flatten_dict(d, parent_key="", sep=".", out=None):
        if out is None:
            out = {}
        for k, v in d.items():
            key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                _flatten_dict(v, key, sep, out)
            else:
                out[key] = v
        return out

    if rank <= 0:
        mlruns_dir = os.path.abspath("mlruns")
        mlflow.set_tracking_uri(f"file:{mlruns_dir}")
        mlflow.set_experiment(opt["name"])
        mlflow.start_run()
        flat_opt = _flatten_dict(opt)
        for k, v in flat_opt.items():
            if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 250:
                mlflow.log_param(k, v)

    #### create model
    model = create_model(opt) 
    
    print(model)
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.GOUB(lambda_square=opt["sde"]["lambda_square"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)
    val_epoch_freq = int(opt["train"].get("val_epoch_freq", 0))
    val_num_preview = int(opt["train"].get("val_num_preview", 0))
    metric_device = device
    try:
        psnr_fn = pyiqa.create_metric("psnr", device=metric_device)
    except Exception as exc:
        psnr_fn = None
        logger.warning("PSNR metric init failed: %s", exc)
    try:
        ssim_fn = pyiqa.create_metric("ssim", device=metric_device)
    except Exception as exc:
        ssim_fn = None
        logger.warning("SSIM metric init failed: %s", exc)
    try:
        lpips_fn = pyiqa.create_metric("lpips", device=metric_device)
    except Exception as exc:
        lpips_fn = None
        logger.warning("LPIPS metric init failed: %s", exc)
    try:
        niqe_fn = pyiqa.create_metric("niqe", device=metric_device)
    except Exception as exc:
        niqe_fn = None
        logger.warning("NIQE metric init failed: %s", exc)

    artifacts_dir = os.path.join(opt["path"]["val_images"], "artifacts")
    util.mkdir(artifacts_dir)

    def _get_base_model(net):
        return net.module if isinstance(net, (DataParallel, DistributedDataParallel)) else net
    last_log_time = time.time()
    last_log_iter = 0

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            LQ, GT = train_data["LQ"], train_data["GT"]
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)

            model.feed_data(states, LQ, GT) # xt, mu, x0
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        mlflow.log_metric(k, v, step=current_step)
                if rank <= 0:
                    now = time.time()
                    elapsed = max(now - last_log_time, 1e-6)
                    iters = max(current_step - last_log_iter, 1)
                    time_per_iter = elapsed / iters
                    samples_per_sec = (iters * train_loader.batch_size) / elapsed
                    lr = model.get_current_learning_rate()
                    mlflow.log_metric("lr", lr, step=current_step)
                    mlflow.log_metric("time_per_iter_sec", time_per_iter, step=current_step)
                    mlflow.log_metric("samples_per_sec", samples_per_sec, step=current_step)
                    mlflow.log_metric("epoch", epoch, step=current_step)
                    if torch.cuda.is_available():
                        mlflow.log_metric(
                            "gpu_mem_alloc_mb",
                            torch.cuda.max_memory_allocated() / (1024 * 1024),
                            step=current_step,
                        )
                        torch.cuda.reset_peak_memory_stats()
                    last_log_time = now
                    last_log_iter = current_step
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):

                    LQ, GT = val_data["LQ"], val_data["GT"]
                    # valid Predictor
                    model.feed_data(LQ, LQ, GT)
                    model.test(sde)
                    visuals = model.get_current_visuals()

                    output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                    gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

                    # calculate PSNR
                    avg_psnr += util.calculate_psnr(output, gt_img)
                    idx += 1

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)

            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    # model.save_training_state(epoch, current_step)

        if val_epoch_freq > 0 and (epoch % val_epoch_freq == 0) and rank <= 0:
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_lpips = 0.0
            avg_niqe = 0.0
            psnr_count = 0
            ssim_count = 0
            lpips_count = 0
            niqe_count = 0
            idx = 0
            for i, val_data in enumerate(val_loader):
                LQ, GT = val_data["LQ"], val_data["GT"]
                model.feed_data(LQ, LQ, GT)
                model.test(sde)
                visuals = model.get_current_visuals()

                output = util.tensor2img(visuals["Output"].squeeze())
                gt_img = util.tensor2img(visuals["GT"].squeeze())
                lq_img = util.tensor2img(visuals["Input"].squeeze())

                sr_tensor = visuals["Output"].detach().to(metric_device).clamp(0, 1)
                gt_tensor = visuals["GT"].detach().to(metric_device).clamp(0, 1)

                sr_tensor = sr_tensor.unsqueeze(0)
                gt_tensor = gt_tensor.unsqueeze(0)

                if psnr_fn is not None:
                    try:
                        avg_psnr += psnr_fn(sr_tensor, gt_tensor).mean().item()
                        psnr_count += 1
                    except Exception as exc:
                        logger.warning("PSNR metric failed on batch: %s", exc)
                if ssim_fn is not None:
                    try:
                        avg_ssim += ssim_fn(sr_tensor, gt_tensor).mean().item()
                        ssim_count += 1
                    except Exception as exc:
                        logger.warning("SSIM metric failed on batch: %s", exc)
                if lpips_fn is not None:
                    try:
                        avg_lpips += lpips_fn(sr_tensor, gt_tensor).mean().item()
                        lpips_count += 1
                    except Exception as exc:
                        logger.warning("LPIPS metric failed on batch: %s", exc)
                if niqe_fn is not None:
                    try:
                        avg_niqe += niqe_fn(sr_tensor).mean().item()
                        niqe_count += 1
                    except Exception as exc:
                        logger.warning("NIQE metric failed on batch: %s", exc)

                if val_num_preview > 0 and i < val_num_preview:
                    img_path = val_data["GT_path"][0]
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    preview_path = os.path.join(artifacts_dir, "{}_compare_{}.png".format(img_name, epoch))
                    preview_img = np.concatenate([lq_img, output, gt_img], axis=1)
                    util.save_img(preview_img, preview_path)
                    logger.info("Saved preview image: %s", preview_path)
                    mlflow.log_artifact(preview_path, artifact_path="val_previews")

                idx += 1

            if psnr_count > 0:
                avg_psnr /= psnr_count
            if ssim_count > 0:
                avg_ssim /= ssim_count
            if lpips_count > 0:
                avg_lpips /= lpips_count
            if niqe_count > 0:
                avg_niqe /= niqe_count

            logger.info(
                "# Epoch Validation # epoch: {:d}, PSNR: {:.6f} (n={:d}), SSIM: {:.6f} (n={:d}), "
                "LPIPS: {:.6f} (n={:d}), NIQE: {:.6f} (n={:d})".format(
                    epoch,
                    avg_psnr, psnr_count,
                    avg_ssim, ssim_count,
                    avg_lpips, lpips_count,
                    avg_niqe, niqe_count,
                )
            )
            mlflow.log_metric("val_psnr", avg_psnr, step=epoch)
            mlflow.log_metric("val_ssim", avg_ssim, step=epoch)
            mlflow.log_metric("val_lpips", avg_lpips, step=epoch)
            mlflow.log_metric("val_niqe", avg_niqe, step=epoch)

            if psnr_count > 0 and avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_iter = current_step
                best_path = os.path.join(artifacts_dir, "best_G.pth")
                base_model = _get_base_model(model.model)
                torch.save(base_model.state_dict(), best_path)
                logger.info(
                    "Saved best model (PSNR %.6f) to %s at epoch %d, iter %d",
                    best_psnr,
                    best_path,
                    epoch,
                    current_step,
                )
                mlflow.log_artifact(best_path, artifact_path="models")

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()
    if rank <= 0:
        mlflow.log_artifacts(artifacts_dir, artifact_path="artifacts")
        mlflow.end_run()


if __name__ == "__main__":
    main()
