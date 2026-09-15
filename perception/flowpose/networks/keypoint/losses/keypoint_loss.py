import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KeypointLoss(nn.Module):
    def __init__(self, grid_h=16, grid_w=16,
                 focal_alpha=0.25, focal_gamma=2.0,
                 obj_weight=1.0, reg_weight=2.0,
                 use_gaussian=False, gaussian_sigma=1.0,
                 neighborhood_size=3):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.obj_weight = obj_weight
        self.reg_weight = reg_weight
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        self.neighborhood_size = neighborhood_size
        self.neighbor_radius = neighborhood_size // 2  # 5->2, 3->1, 7->3

    def build_targets(self, gt_coords, gt_vis):
        """
        Convert sparse GT coords to dense target maps.

        Args:
            gt_coords: [B, C, K, 2] normalised coordinates [0, 1]
            gt_vis:    [B, C, K]    visibility flags

        Returns:
            target_obj:              [B, C, H, W]    objectness targets (0-1 or gaussian)
            target_offsets_sigmoid:  [B, C, H, W, 2] offset targets in sigmoid space [0, 1]
            pos_mask:                [B, C, H, W]    positive-sample boolean mask
        """
        B, C, K, _ = gt_coords.shape
        device = gt_coords.device

        target_obj = torch.zeros(B, C, self.grid_h, self.grid_w, device=device)
        target_offsets_sigmoid = torch.zeros(B, C, self.grid_h, self.grid_w, 2, device=device)
        pos_mask = torch.zeros(B, C, self.grid_h, self.grid_w, dtype=torch.bool, device=device)

        grid_size = 1.0 / self.grid_w  # assumes H == W

        for b in range(B):
            for c in range(C):
                for k in range(K):
                    if gt_vis[b, c, k] < 0.5:
                        continue

                    gt_coord = gt_coords[b, c, k]  # [2]

                    gx = int(gt_coord[0].item() * self.grid_w)
                    gy = int(gt_coord[1].item() * self.grid_h)
                    gx = max(0, min(gx, self.grid_w - 1))
                    gy = max(0, min(gy, self.grid_h - 1))

                    y_start = max(0, gy - self.neighbor_radius)
                    y_end   = min(self.grid_h, gy + self.neighbor_radius + 1)
                    x_start = max(0, gx - self.neighbor_radius)
                    x_end   = min(self.grid_w, gx + self.neighbor_radius + 1)

                    for dy in range(y_start, y_end):
                        for dx in range(x_start, x_end):
                            # Grid centre coordinates (normalised)
                            grid_center = torch.tensor(
                                [(dx + 0.5) / self.grid_w, (dy + 0.5) / self.grid_h],
                                device=gt_coord.device, dtype=gt_coord.dtype
                            )

                            # Offset in sigmoid space: diff / grid_size + 0.5 → [0, 1]
                            diff = gt_coord - grid_center
                            offset_sig = torch.clamp(diff / grid_size + 0.5, 0.0, 1.0)

                            # Objectness weight
                            grid_dist = abs(dx - gx) + abs(dy - gy)
                            if self.use_gaussian:
                                euclidean_dist = ((dx - gx) ** 2 + (dy - gy) ** 2) ** 0.5
                                weight = np.exp(-(euclidean_dist ** 2) / (2 * self.gaussian_sigma ** 2))
                            else:
                                if grid_dist == 0:
                                    weight = 1.0
                                elif self.neighborhood_size == 3:
                                    weight = 0.5
                                elif self.neighborhood_size == 5:
                                    weight = 0.5 if grid_dist <= 2 else 0.2
                                elif self.neighborhood_size == 7:
                                    if grid_dist <= 2:
                                        weight = 0.5
                                    elif grid_dist <= 4:
                                        weight = 0.2
                                    else:
                                        weight = 0.1
                                else:
                                    max_dist = self.neighbor_radius * 2
                                    weight = max(0.1, 1.0 - (grid_dist / max_dist) * 0.9)

                            target_obj[b, c, dy, dx] = max(
                                target_obj[b, c, dy, dx].item(), weight
                            )

                            # Keep offset from the nearest GT (smaller norm wins)
                            current_norm = torch.norm(target_offsets_sigmoid[b, c, dy, dx]).item()
                            new_norm = torch.norm(offset_sig).item()
                            if current_norm == 0 or new_norm < current_norm:
                                target_offsets_sigmoid[b, c, dy, dx] = offset_sig

                            pos_mask[b, c, dy, dx] = True

        return target_obj, target_offsets_sigmoid, pos_mask

    def focal_loss(self, pred_logits, target):
        """Focal Loss for objectness. pred_logits/target: [B, C, H, W]"""
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pred_prob = torch.sigmoid(pred_logits)
        p_t = torch.where(target >= 0.5, pred_prob, 1 - pred_prob)
        alpha_t = torch.where(
            target >= 0.5,
            torch.tensor(self.focal_alpha, device=target.device),
            torch.tensor(1 - self.focal_alpha, device=target.device)
        )
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        return (focal_weight * bce_loss).mean()

    def regression_loss(self, pred_offsets, target_offsets_sigmoid, pos_mask):
        """Smooth L1 regression loss over positive cells only."""
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_offsets.device)

        pred_offsets_sigmoid = torch.sigmoid(pred_offsets)

        pred_flat   = pred_offsets_sigmoid.reshape(-1, 2)
        target_flat = target_offsets_sigmoid.reshape(-1, 2)
        mask_flat   = pos_mask.reshape(-1)

        return F.smooth_l1_loss(pred_flat[mask_flat], target_flat[mask_flat])

    def forward(self, pred_obj, pred_offsets, gt_coords, gt_vis):
        """
        Args:
            pred_obj:     [B, C, H, W]    objectness logits
            pred_offsets: [B, C, H, W, 2] offset logits
            gt_coords:    [B, C, K, 2]    normalised GT coordinates
            gt_vis:       [B, C, K]       GT visibility

        Returns:
            total_loss (Tensor), loss_dict (dict)
        """
        target_obj, target_offsets_sigmoid, pos_mask = self.build_targets(gt_coords, gt_vis)

        loss_obj = self.focal_loss(pred_obj, target_obj)
        loss_reg = self.regression_loss(pred_offsets, target_offsets_sigmoid, pos_mask)

        total_loss = self.obj_weight * loss_obj + self.reg_weight * loss_reg

        loss_dict = {
            'total':   total_loss.item(),
            'obj':     loss_obj.item(),
            'reg':     loss_reg.item(),
            'num_pos': pos_mask.sum().item()
        }

        return total_loss, loss_dict
