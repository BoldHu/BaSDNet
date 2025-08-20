# ------------------------------------------------------------------
# train_helper_ALTGVT.py  –  Pipeline 版本（含 SegLoss / Patch-Val）
# ------------------------------------------------------------------
import os, time, wandb
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh, CustomDataset
from losses.ot_loss import OT_Loss
from losses.seg_loss import SegLoss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils

from Networks.pipeline import CrowdPipelineNet   # ← 新网络


# -------------------------- Data Collate ------------------------ #
def train_collate(batch):
    transposed = list(zip(*batch))
    images = torch.stack(transposed[0], 0)
    points = transposed[1]              # list[Tensor]
    gt_discretes = torch.stack(transposed[2], 0)
    return images, points, gt_discretes


# ============================ Trainer ========================== #
class Trainer(object):
    def __init__(self, args):
        self.args = args

    # ------------------------- Setup --------------------------- #
    def setup(self):
        args = self.args
        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        sub_dir = (
            f"Pipeline/{args.run_name}_{time_str}_in-{args.crop_size}_wot-{args.wot}"
            f"_wtv-{args.wtv}_wseg-{args.wseg}"
            f"_iter-{args.num_of_iter_in_ot}_norm-{args.norm_cood}"
        )
        self.save_dir = os.path.join("ckpts", sub_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, f"train-{time_str}.log")
        )
        log_utils.print_config(vars(args), self.logger)

        # ----- device & dataset -----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Use device : {self.device}")

        downsample_ratio = 8
        if args.dataset.lower() == "qnrf":
            ds = lambda s: Crowd_qnrf(os.path.join(args.data_dir, s),
                                      args.crop_size, downsample_ratio, s)
            self.datasets = {x: ds(x) for x in ["train", "val"]}
        elif args.dataset.lower() == "nwpu":
            ds = lambda s: Crowd_nwpu(os.path.join(args.data_dir, s),
                                      args.crop_size, downsample_ratio, s)
            self.datasets = {x: ds(x) for x in ["train", "val"]}
        elif args.dataset.lower() in ["sha", "shb"]:
            self.datasets = {
                "train": Crowd_sh(os.path.join(args.data_dir, "train_data"),
                                  args.crop_size, downsample_ratio, "train"),
                "val": Crowd_sh(os.path.join(args.data_dir, "test_data"),
                                args.crop_size, downsample_ratio, "val")
            }
        elif args.dataset.lower() == "custom":
            self.datasets = {
                "train": CustomDataset(args.data_dir, args.crop_size,
                                       downsample_ratio, method="train"),
                "val"  : CustomDataset(args.data_dir, args.crop_size,
                                       downsample_ratio, method="valid"),
            }
        else:
            raise NotImplementedError

        self.dataloaders = {
            split: DataLoader(
                self.datasets[split],
                batch_size=(args.batch_size if split == "train" else 1),
                shuffle=(split == "train"),
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=(train_collate if split == "train" else default_collate),
            )
            for split in ["train", "val"]
        }

        # ----- model / opt / losses -----
        self.model = CrowdPipelineNet(img_size=args.crop_size).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=args.lr, weight_decay=args.weight_decay)
        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio,
                               args.norm_cood, self.device,
                               args.num_of_iter_in_ot, args.reg)
        self.tv_loss = nn.L1Loss(reduction="none").to(self.device)
        self.mae     = nn.L1Loss().to(self.device)

        self.seg_loss_fn = SegLoss(mode="bce_dice", alpha=0.5,
                                   stride=4, radius=2).to(self.device)
        self.wseg = getattr(args, "wseg", 1.0)

        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        if args.wandb: self.wandb_run = wandb.init(project="Pipeline", config=args)
        else: wandb.init(mode="disabled")

    # --------------------- Train / Val ------------------------- #
    def train(self):
        for epoch in range(self.args.max_epoch + 1):
            self.epoch = epoch
            self.logger.info(f"--- Epoch {epoch}/{self.args.max_epoch} ---")
            self.train_epoch()
            if epoch % self.args.val_epoch == 0 and epoch >= self.args.val_start:
                self.val_epoch()

    # -------------- Training 1 epoch (同前版，无变化) ------------ #
    def train_epoch(self):
        args = self.args
        meters = {
            "ot": AverageMeter(), "ot_obj": AverageMeter(), "wd": AverageMeter(),
            "count": AverageMeter(), "tv": AverageMeter(), "seg": AverageMeter(),
            "total": AverageMeter(), "mae": AverageMeter()
        }
        self.model.train()
        epoch_start = time.time()

        for step, (imgs, points, gt_discrete) in enumerate(self.dataloaders["train"]):
            imgs = imgs.to(self.device)
            points = [p.to(self.device) for p in points]
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            N = imgs.size(0)

            gt_discrete = gt_discrete.to(self.device)

            # ---------- forward ----------
            dens_map, mask_prob = self.model(imgs)              # (B,1,H/4,W/4)

            # OT-Loss 需要归一化密度
            normed = dens_map / (dens_map.sum([1, 2, 3], keepdim=True) + 1e-6)

            # ---------- OT Loss ----------
            ot_loss, wd, ot_obj = self.ot_loss(normed, dens_map, points)
            ot_loss = ot_loss * args.wot
            ot_obj  = ot_obj  * args.wot
            meters["ot"].update(ot_loss.item(), N)
            meters["ot_obj"].update(ot_obj.item(), N)
            meters["wd"].update(wd, N)

            # ---------- Count Loss ----------
            count_loss = self.mae(
                dens_map.view(N, -1).sum(1),
                torch.from_numpy(gd_count).float().to(self.device)
            )
            meters["count"].update(count_loss.item(), N)

            # ---------- TV Loss ----------
            gd_count_t = torch.from_numpy(gd_count).float().to(self.device)
            gt_discrete_norm = gt_discrete / (gd_count_t.view(-1, 1, 1, 1) + 1e-6)
            tv_loss = (
                self.tv_loss(normed, gt_discrete_norm).sum([1, 2, 3]) * gd_count_t
            ).mean(0) * args.wtv
            meters["tv"].update(tv_loss.item(), N)

            # ---------- Seg Loss ----------
            # SegLoss 需要 logits，mask_prob ∈ (0,1) → logits=log(p/(1-p))
            mask_logits = torch.log(mask_prob.clamp(1e-6, 1 - 1e-6) /
                                    (1 - mask_prob.clamp(1e-6, 1 - 1e-6)))
            seg_loss = self.seg_loss_fn(mask_logits, points) * self.wseg
            meters["seg"].update(seg_loss.item(), N)

            # ---------- Total Loss ----------
            total_loss = ot_loss + count_loss + tv_loss + seg_loss
            meters["total"].update(total_loss.item(), N)

            # ---------- backward ----------
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # ---------- err / metrics ----------
            pred_count = dens_map.view(N, -1).sum(1).detach().cpu().numpy()
            meters["mae"].update(np.mean(np.abs(pred_count - gd_count)), N)

            # wandb log (optional)
            wandb.log({
                "ot_loss": ot_loss, "tv_loss": tv_loss,
                "count_loss": count_loss, "seg_loss": seg_loss,
                "total_loss": total_loss
            })

        # ---------- epoch summary ----------
        self.logger.info(
            f"Epoch {self.epoch} | "
            f"Loss {meters['total'].get_avg():.3f} | "
            f"OT {meters['ot'].get_avg():.3e} | "
            f"TV {meters['tv'].get_avg():.3f} | "
            f"Seg {meters['seg'].get_avg():.3f} | "
            f"MAE {meters['mae'].get_avg():.2f} | "
            f"Time {time.time() - epoch_start:.1f}s"
        )

        # ---------- save ckpt ----------
        save_path = os.path.join(self.save_dir, f"{self.epoch}_ckpt.tar")
        torch.save({
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": self.model.state_dict(),
        }, save_path)
        self.save_list.append(save_path)

    # ---------------------------------------------------------- #

    def val_epoch(self):
        """
        * 滑窗裁成 crop_size×crop_size
        * 预测输出 stride = 8 → 上采 ×8 并 /64 保积分
        * 坐标贴回整幅图，重叠区取平均
        * 计算 MAE / RMSE
        """
        args = self.args
        self.model.eval()

        mae_meter, mse_meter = AverageMeter(), AverageMeter()
        epoch_start = time.time()

        with torch.no_grad():
            for img, gt_cnt, name in self.dataloaders["val"]:   # ← batch=1
                img     = img.to(self.device)                   # (1,3,H,W)
                gt_cnt  = gt_cnt.item()
                _, _, H, W = img.shape
                rh = rw = args.crop_size                       # patch size (256…)

                # ---------- 1. 切 patch 并记录坐标 ----------
                patches, coords = [], []
                for top in range(0, H, rh):
                    for left in range(0, W, rw):
                        gs = min(top,  H - rh); ge = gs + rh
                        ls = min(left, W - rw); le = ls + rw
                        patches.append(img[:, :, gs:ge, ls:le])       # (1,3,rh,rw)
                        coords.append((gs, ge, ls, le))

                patches = torch.cat(patches, dim=0)                   # (N,3,rh,rw)

                # ---------- 2. 批量推理 ----------
                preds = []
                bs = args.batch_size
                for idx in range(0, patches.size(0), bs):
                    pred_map, _ = self.model(patches[idx:idx + bs])   # (m,1,rh/8,rw/8)

                    # 固定倍率 ×8，并 /64 保积分
                    pred_up = F.interpolate(
                        pred_map,
                        scale_factor=8,               # stride=8 → ×8 上采
                        mode='bilinear',
                        align_corners=False
                    ) / 64.0                          # 8×8 = 64
                    preds.append(pred_up)

                preds = torch.cat(preds, dim=0)                      # (N,1,rh,rw)

                # ---------- 3. 贴回整幅图 ----------
                full_map = torch.zeros((1, 1, H, W), device=self.device)
                overlap  = torch.zeros((1, 1, H, W), device=self.device)

                for pred, (gs, ge, ls, le) in zip(preds, coords):
                    full_map[:, :, gs:ge, ls:le] += pred
                    overlap[:, :, gs:ge, ls:le]  += 1.0

                full_map = full_map / overlap.clamp(min=1.0)

                # ---------- 4. 误差统计 ----------
                pred_cnt = full_map.sum().item()
                err = pred_cnt - gt_cnt
                mae_meter.update(abs(err), 1)
                mse_meter.update(err ** 2, 1)

        mae  = mae_meter.get_avg()
        rmse = np.sqrt(mse_meter.get_avg())
        self.logger.info(
            f"[Val] MAE {mae:.2f} | RMSE {rmse:.2f} | "
            f"time {time.time() - epoch_start:.1f}s"
        )

        # ---------- Save best ----------
        if mae < self.best_mae:
            self.best_mae = mae
            best_path = os.path.join(self.save_dir, f"best_model_mae_{self.best_mae:.2f}.pth")
            torch.save(self.model.state_dict(), best_path)
            self.logger.info(f"New best saved to {best_path}")

        # ---------- wandb ----------
        wandb.log({"val/MAE": mae, "val/RMSE": rmse}, step=self.epoch)
