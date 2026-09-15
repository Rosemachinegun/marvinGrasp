import torch
import torch.nn as nn

class DenseKeypointHead(nn.Module):
    """
    密集关键点预测头（YOLO风格）
    
    每个特征图位置预测 num_classes 个可能的关键点
    输出形状: [B, H, W, num_classes * 3]
    
    对于每个类别c，预测：
    - dx, dy: 相对于网格中心的坐标偏移（归一化）
    - objectness: 该位置是否有该类别的关键点
    """
    
    def __init__(self, d_model=768, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征提取
        self.feature_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.GroupNorm(16, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 为每个类别预测：dx, dy, objectness
        # 拆分为两个head以便使用不同的学习率
        # 形状: [B, num_classes * 3, H, W]

        # Objectness head（分类）
        self.obj_head = nn.Conv2d(
            d_model // 4,
            num_classes,  # 每个类别: objectness
            kernel_size=1
        )

        # Regression head（回归）
        self.reg_head = nn.Conv2d(
            d_model // 4,
            num_classes * 2,  # 每个类别: dx, dy
            kernel_size=1
        )
        
    def forward(self, patch_features):
        """
        Args:
            patch_features: [B, N, D] (D=384 for vits14, 768 for vitb14，如 16x16=256个patch 或 37x37个patch)

        Returns:
            outputs: dict {
                'coords': [B, num_classes, H, W, 2]  # 绝对坐标（归一化）
                'objectness': [B, num_classes, H, W]  # 每个类别的置信度
                'offsets': [B, num_classes, H, W, 2]  # 相对偏移
            }
        """
        B, N, D = patch_features.shape
        H = W = int(N ** 0.5)  # e.g., 16 or 37
        
        # 重塑为特征图 [B, D, H, W]
        x = patch_features.transpose(1, 2).reshape(B, D, H, W)
        
        # 特征提取
        x = self.feature_conv(x)  # [B, D//4, H, W]

        # 分别预测objectness和offsets
        objectness = self.obj_head(x)  # [B, num_classes, H, W]
        offsets_xy = self.reg_head(x)  # [B, num_classes * 2, H, W]

        # 重塑offsets为 [B, num_classes, 2, H, W]
        offsets_xy = offsets_xy.view(B, self.num_classes, 2, H, W)

        # 转换为 [B, C, H, W, 2]
        offsets_xy = offsets_xy.permute(0, 1, 3, 4, 2)  # [B, C, H, W, 2]
        
        # ===== 计算绝对坐标 =====
        # 创建网格坐标 [H, W, 2]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]
        
        # 归一化到 [0, 1] - 这是网格左上角坐标
        grid_top_left = grid / torch.tensor([W, H], device=x.device).float()
        
        # 🔧 关键修复：计算网格中心坐标
        # 网格中心 = 左上角 + 半个网格大小
        grid_size = torch.tensor([1.0 / W, 1.0 / H], device=x.device).float()
        grid_center = grid_top_left + grid_size / 2  # [H, W, 2]
        
        # 广播到 [B, C, H, W, 2]
        grid_center = grid_center.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
        
        # 🔧 绝对坐标 = 网格中心 + (sigmoid(偏移) - 0.5) * 网格大小
        # sigmoid(offsets) ∈ [0, 1]
        # sigmoid(offsets) - 0.5 ∈ [-0.5, 0.5]
        # 因此最终坐标在网格中心±半个网格范围内
        offsets_sigmoid = torch.sigmoid(offsets_xy)  # [0, 1]
        offsets_normalized = (offsets_sigmoid - 0.5) * grid_size.view(1, 1, 1, 1, 2)  # [-0.5*grid_size, +0.5*grid_size]
        
        coords = grid_center + offsets_normalized
        coords = torch.clamp(coords, 0, 1)  # 确保在 [0, 1] 范围内
        
        return {
            'coords': coords,           # [B, C, H, W, 2] 绝对坐标（归一化）
            'objectness': objectness,   # [B, C, H, W] 置信度logits
            'offsets': offsets_xy,      # [B, C, H, W, 2] 原始偏移量
        }
    
    def _nms_grid(self, indices, confs, coords, radius=2):
        """网格空间的简单NMS"""
        # 按置信度排序
        sorted_idx = torch.argsort(confs, descending=True)
        
        keep_indices = []
        keep_confs = []
        keep_coords = []
        
        while len(sorted_idx) > 0:
            # 取置信度最高的
            idx = sorted_idx[0]
            keep_indices.append(indices[idx])
            keep_confs.append(confs[idx])
            keep_coords.append(coords[idx])
            
            if len(sorted_idx) == 1:
                break
            
            # 计算距离
            current_pos = indices[idx].float()
            other_pos = indices[sorted_idx[1:]].float()
            dists = torch.norm(other_pos - current_pos, dim=1)
            
            # 保留距离大于radius的
            far_mask = dists > radius
            sorted_idx = sorted_idx[1:][far_mask]
        
        if len(keep_indices) > 0:
            return (torch.stack(keep_indices), 
                    torch.stack(keep_confs),
                    torch.stack(keep_coords))
        else:
            return indices[:0], confs[:0], coords[:0]

    # Prediction decoding
    def extract_keypoints(self, outputs, conf_threshold=0.3, nms_radius=2):
        """
        从密集预测中提取关键点（推理时使用）
        
        Args:
            outputs: forward() 的输出
            conf_threshold: 置信度阈值
            nms_radius: NMS半径（网格单位）
        
        Returns:
            keypoints: List[List[Dict]] 
                外层list对应batch，内层list对应类别
                Dict包含 {'x', 'y', 'conf', 'class'}
        """
        coords = outputs['coords']  # [B, C, H, W, 2]
        objectness = outputs['objectness']  # [B, C, H, W]
        
        B, C, H, W, _ = coords.shape
        
        # Sigmoid获取置信度
        confidences = torch.sigmoid(objectness)  # [B, C, H, W]
        
        batch_keypoints = []
        
        for b in range(B):
            image_keypoints = []
            
            for c in range(C):
                conf_map = confidences[b, c]  # [H, W]
                coord_map = coords[b, c]      # [H, W, 2]
                
                # 1. 阈值过滤
                mask = conf_map > conf_threshold
                
                if mask.sum() == 0:
                    image_keypoints.append([])
                    continue
                
                # 2. 获取候选位置
                indices = torch.nonzero(mask)  # [N, 2] (y, x)
                confs = conf_map[mask]          # [N]
                coords_selected = coord_map[mask]  # [N, 2]
                
                # 3. 简单NMS（可选）
                if nms_radius > 0:
                    indices, confs, coords_selected = self._nms_grid(
                        indices, confs, coords_selected, radius=nms_radius
                    )
                
                # 4. 组装关键点
                class_keypoints = []
                for idx, conf, coord in zip(indices, confs, coords_selected):
                    class_keypoints.append({
                        'x': coord[0].item(),  # 归一化坐标
                        'y': coord[1].item(),
                        'conf': conf.item(),
                        'class': c,
                        'grid_x': idx[1].item(),
                        'grid_y': idx[0].item()
                    })
                
                image_keypoints.append(class_keypoints)
            
            batch_keypoints.append(image_keypoints)
        
        return batch_keypoints
    
    def load_ckpt(self, model_dir, load_model_only=True, strict=False):
        ckpt = torch.load(model_dir, map_location=self.args.device)
        state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        # Checkpoint may have been saved from DenseKeypointDetectionModel which wraps the
        # decoder under a "decoder." prefix.  Strip it so weights load into KeyPoint directly.
        if any(k.startswith('decoder.') for k in state_dict):
            state_dict = {k[len('decoder.'):]: v for k, v in state_dict.items() if k.startswith('decoder.')}
        self.load_state_dict(state_dict, strict=strict)