# crowd_pipeline_model.py
# -------------------------------------------------------------
#  Crowd Density Estimation Network following the proposed
#  pipeline diagram (Encoder → Mask → Attention → Decoder → Head)
#  ------------------------------------------------------------
#
#  Requirements mapping (see README / paper figure):
#  1. Background-Aware Encoder :  ALT-GVT backbone  + v1/v2/v3 →  x
#  2. Density Decoder          :  stage1–4 (+ concat)          →  y
#  3. Predicted Density Head   :  res                          →  D
#  4. Extra blocks (Seg branch / Attention) implemented here.
#  5.  Optional Conditional Diffusion Module  (placeholder)
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from Networks.ALTGVT import CPVTV2
from Networks.ALTGVT import alt_gvt_large

# --------------------------- Encoder --------------------------- #
class BackgroundAwareEncoder(nn.Module):
    """
    ALT-GVT / PVT backbone             : 提取多尺度特征
    v1 / v2 / v3 (原 Regression 中)    : 对齐通道 & 上采样
    输出                                : 聚合特征 x  (B,256,H/4,W/4)
    """
    def __init__(self, img_size=224, in_chans=3, pretrained=True):
        super().__init__()
        self.backbone = alt_gvt_large(pretrained=pretrained)

        # ↓↓↓ 与 Regression.v1 / v2 / v3 完全保留参数一致 ↓↓↓
        self.v1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # ↑↑↑--------------------------------------------------↑↑↑

    def forward(self, img):
        """
        Args:
            img : (B,3,H,W)
        Returns:
            x   : aggregated feature  (B,256,H/4,W/4)
            feats: list of multi-scale features from backbone
        """
        feats = self.backbone.forward_features(img)      # 4 scales
        x1 = self.v1(feats[1])    # C=256, stride 4
        x2 = self.v2(feats[2])    # C=512 → 256, stride 8 → 4
        x3 = self.v3(feats[3])    # C=1024→256, stride16 → 4
        x  = x1 + x2 + x3    # 对应图中 x = x1+x2+x3
        return x, feats


# --------------------- Segmentation  Decoder ------------------ #
class SegmentationDecoder(nn.Module):
    """
    轻量级解码器，产生前景概率掩码 (M)
    """
    def __init__(self, in_channels=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, 1)                # 1-channel mask
        )

    def forward(self, x):
        mask_logits = self.decoder(x)
        mask_prob   = torch.sigmoid(mask_logits)
        return mask_prob


# ---------------------- Density   Decoder --------------------- #
class DensityDecoder(nn.Module):
    """
    stage1 / 2 / 3 / 4   →   concat & add  →   y
    （完全复刻 Regression 中相应部分）
    """
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y1 = self.stage1(x)
        y2 = self.stage2(x)
        y3 = self.stage3(x)
        y4 = self.stage4(x)
        y  = torch.cat((y1, y2, y3), dim=1) + y4   # (B,384,H/4,W/4)
        return y


# --------------------- Predicted Density  Head ---------------- #
class DensityHead(nn.Module):
    """
    res 模块： y → Density Map (D)
    """
    def __init__(self):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True)     # 保持与原实现一致
        )

    def forward(self, y):
        return self.res(y)            # (B,1,H/4,W/4)


# -------------------------  Full Model ------------------------ #
class CrowdPipelineNet(nn.Module):
    """
    Pipeline: Encoder → SegMask → Attention → DensityDecoder → DensityHead
    """
    def __init__(self, img_size=224, in_chans=3):
        super().__init__()
        self.encoder          = BackgroundAwareEncoder(img_size, in_chans)
        self.seg_decoder      = SegmentationDecoder(in_channels=256)
        self.density_decoder  = DensityDecoder()
        self.density_head     = DensityHead()
        # Optional Diffusion Module (placeholder)
        # self.diffusion      = ...

    def forward(self, img):
        # 1) Background-Aware Encoder
        x, _ = self.encoder(img)             # x : (B,256,H/4,W/4)

        # 2) Segmentation  & Attention Gate
        mask = self.seg_decoder(x)           # (B,1,H/4,W/4)
        x_att = x * mask                     # Attention Gate (M → D)

        # 3) Density  Decoder  &  Head
        y_feat      = self.density_decoder(x_att) #x_att
        density_map = self.density_head(y_feat)

        return density_map, mask             # D & M


# ----------------------- Quick Sanity Test -------------------- #
if __name__ == "__main__":
    model = CrowdPipelineNet(img_size=224)
    dummy = torch.randn(2, 3, 224, 224)
    D, M = model(dummy)
    print("Density map :", D.shape)  # torch.Size([2, 1, 56, 56])
    print("Mask        :", M.shape)  # torch.Size([2, 1, 56, 56])
