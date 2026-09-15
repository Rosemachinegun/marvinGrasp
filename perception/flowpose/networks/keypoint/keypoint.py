import torch
import torch.nn as nn
from networks.keypoint.losses.keypoint_loss import KeypointLoss
from networks.keypoint.decoder import DenseKeypointHead

class KeypointModel(nn.Module):
    def __init__(self,
                 args,
                 num_classes=1,
                 dino=None,
                 img_size=224
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.dino = dino
        self.img_size = img_size
        self.patch_size = 14
        self.grid_size = self.img_size // self.patch_size

        # Initialize loss function
        self.loss_fn = KeypointLoss(
            grid_h=16,
            grid_w=16,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            obj_weight=args.obj_weight,
            reg_weight=args.reg_weight,
            use_gaussian=args.use_gaussian,
            gaussian_sigma=args.gaussian_sigma,
            neighborhood_size=args.neighborhood_size
        )

        # Initialize decoder head
        self.decoder = DenseKeypointHead(
            d_model=384,
            num_classes=num_classes
        )

        # 创建优化器（分层学习率）
        print("\n配置优化器...")
        backbone_params = []
        obj_head_params = []
        reg_head_params = []
        other_head_params = []

        # 【修复】只收集 requires_grad=True 的参数
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # 跳过冻结的参数
            if 'dino' in name:
                backbone_params.append(param)
            elif 'decoder.obj_head' in name:
                obj_head_params.append(param)
            elif 'decoder.reg_head' in name:
                reg_head_params.append(param)
            else:
                other_head_params.append(param)

        # 打印参数组信息
        print(f"  Backbone参数 (可训练): {len(backbone_params)} 个 (lr={args.lr_backbone})")
        print(f"  Obj Head参数: {len(obj_head_params)} 个 (lr={args.lr})")
        print(f"  Reg Head参数: {len(reg_head_params)} 个 (lr={args.lr * 2.0})")
        if other_head_params:
            print(f"  其他Head参数: {len(other_head_params)} 个 (lr={args.lr})")

        # 创建优化器（回归头使用2倍学习率）
        param_groups = [
            {'params': backbone_params, 'lr': args.lr_backbone},
            {'params': obj_head_params, 'lr': args.lr},
            {'params': reg_head_params, 'lr': args.lr * 2.0}  # 回归头2倍学习率
        ]

        if other_head_params:
            param_groups.append({'params': other_head_params, 'lr': args.lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W], 值域 [0, 1]
        
        Returns:
            outputs: dict {
                'coords': [B, C, H, W, 2]  # 归一化坐标
                'objectness': [B, C, H, W]  # 置信度
            }
        """
        # 调整尺寸到 self.img_size x self.img_size
        if images.shape[-2:] != (self.img_size, self.img_size):
            images = torch.nn.functional.interpolate(
                images, size=(self.img_size, self.img_size), 
                mode='bilinear', align_corners=False
            )
        
        # 编码
        self.dino.extract_features(images)          # Extract DINO features and store in DinoLoader.feat
        features = self.dino.get_feature()          # get feature from dino
        
        # 解码
        outputs = self.decoder(features)
        
        return outputs

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

    def save_ckpt(self, save_dir, filename, epoch, global_step, best_val_loss, val_loss=None):
        """Save checkpoint. Returns updated best_val_loss."""
        from pathlib import Path
        save_dir = Path(save_dir)

        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'config': {
                'num_classes': self.num_classes,
                'grid_h': 16,
                'grid_w': 16,
            }
        }

        if val_loss is not None:
            checkpoint['val_loss'] = val_loss

        save_path = save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"✅ 检查点已保存: {save_path}")

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 最佳模型已保存: {best_path}")

        return best_val_loss

    def load_ckpt(self, path):
        """Load checkpoint. Returns (epoch, global_step, best_val_loss)."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ 检查点已加载: {path} (epoch={epoch})")
        return epoch, global_step, best_val_loss
