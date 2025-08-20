# seg_loss.py
# -------------------------------------------------------------
#   Binary segmentation loss for crowd-mask branch
#   Inputs :
#       pred_logits : (B,1,H',W')
#       points      : list[Tensor(#points_i, 2)] – original image coords
# -------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SegLoss(nn.Module):
    """
    Compute BCE / Dice / BCE+Dice loss.
    Internally build gt_mask from sparse point annotations.
    """

    def __init__(
        self,
        mode: str = "bce_dice",          # "bce", "dice", "bce_dice"
        alpha: float = 0.5,              # λ for BCE when mode="bce_dice"
        stride: int = 4,                 # down-sampling factor wrt original img
        radius: int = 2,                 # radius (in low-res px) for disk mask
        pos_weight: Optional[float] = None,
        smooth: float = 1e-6
    ):
        super().__init__()
        assert mode in ["bce", "dice", "bce_dice"]
        self.mode       = mode
        self.alpha      = alpha
        self.stride     = stride
        self.radius     = radius
        self.user_pw    = pos_weight
        self.smooth     = smooth

    # ---------- helpers -------------------------------------------------- #
    @staticmethod
    def _draw_disk(mask: torch.Tensor, cy: int, cx: int, r: int):
        """
        Draw filled disk of radius r on single-channel mask (H,W).
        """
        H, W = mask.shape
        y0, y1 = max(cy - r, 0), min(cy + r + 1, H)
        x0, x1 = max(cx - r, 0), min(cx + r + 1, W)
        if y1 <= y0 or x1 <= x0:
            return
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=mask.device),
            torch.arange(x0, x1, device=mask.device),
            indexing="ij",
        )
        cond = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        mask[yy[cond], xx[cond]] = 1.0

    def _points_to_mask(
        self, pts: torch.Tensor, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """
        Convert N×2 points (x,y) to binary mask (1,H,W)
        under down-sampling stride.
        """
        m = torch.zeros((H, W), dtype=torch.float32, device=device)
        if pts.numel() == 0:
            return m
        # scale coordinates
        xs = (pts[:, 0] / self.stride).long()
        ys = (pts[:, 1] / self.stride).long()
        for x, y in zip(xs, ys):
            if 0 <= x < W and 0 <= y < H:
                if self.radius == 0:
                    m[y, x] = 1.0
                else:
                    self._draw_disk(m, y.item(), x.item(), self.radius)
        return m

    # ---------- forward -------------------------------------------------- #
    def forward(
        self,
        pred_logits: torch.Tensor,
        points: List[torch.Tensor],
    ):
        """
        Args:
            pred_logits : (B,1,H',W') – raw network output (before sigmoid)
            points      : list of len B, each tensor (Ni,2) with (x,y) coords.
        Returns:
            scalar segmentation loss
        """
        B, _, H, W = pred_logits.shape
        assert len(points) == B, "points list length mismatch with batch"

        # build gt masks
        gt_masks = torch.stack(
            [
                self._points_to_mask(p, H, W, pred_logits.device)
                for p in points
            ],
            dim=0,
        ).unsqueeze(1)  # (B,1,H,W)

        prob = torch.sigmoid(pred_logits)

        # -------- Binary-Cross-Entropy -------- #
        if self.user_pw is None:
            # foreground / background ratio over batch
            pos_cnt = gt_masks.sum()
            neg_cnt = gt_masks.numel() - pos_cnt
            pos_weight = torch.tensor(
                (neg_cnt / (pos_cnt + self.smooth)).item(),
                device=pred_logits.device,
            )
        else:
            pos_weight = torch.tensor(self.user_pw, device=pred_logits.device)

        bce = F.binary_cross_entropy_with_logits(
            pred_logits, gt_masks, pos_weight=pos_weight, reduction="mean"
        )

        # -------- Dice Loss -------- #
        intersection = (prob * gt_masks).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2 * intersection + self.smooth) /
                      (union + self.smooth)).mean()

        # -------- Combine -------- #
        if self.mode == "bce":
            loss = bce
        elif self.mode == "dice":
            loss = dice
        else:  # "bce_dice"
            loss = self.alpha * bce + (1.0 - self.alpha) * dice

        return loss
    
if __name__ == "__main__":
    torch.manual_seed(0)

    # ---------------- 伪造网络输出 ---------------- #
    B, H_, W_ = 2, 56, 56          # 预测分辨率
    pred_logits = torch.randn(B, 1, H_, W_, requires_grad=True)

    # ---------------- 构造点标注 ---------------- #
    stride = 4                     # ↓ 与 SegLoss 实例化保持一致
    H_orig, W_orig = H_ * stride, W_ * stride

    points = []
    for b in range(B):
        # 随机 5~10 个点
        n = torch.randint(5, 11, (1,)).item()
        pt = torch.randint(
            low=0,
            high=min(H_orig, W_orig),
            size=(n, 2),
            dtype=torch.float32,
        )
        points.append(pt)

    # ---------------- SegLoss 实例 ---------------- #
    criterion = SegLoss(
        mode="bce_dice",      # 同论文设置
        alpha=0.5,
        stride=stride,
        radius=2,
    )

    # ---------------- 前向 & 反向 ---------------- #
    loss = criterion(pred_logits, points)
    print(f"Segmentation loss = {loss.item():.6f}")

    # 确认可反向传播
    loss.backward()
    print("Grad check passed:", pred_logits.grad is not None)
