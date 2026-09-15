import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from utils.logs_kp import log_loss, save_loss_log, save_validation_predictions
from utils.visualizations_kp import plot_loss_curves, visualize_batch, visualize_validation

class KeypointTrainer:
    def __init__(self, args, model, train_loader, val_loader, optimizer, scheduler,
                 loss_fn, device, save_dir, max_grad_norm=1.0,
                 visualize_validation=True, num_val_vis_samples=None,
                 val_every_n_epochs=10):
        """
        Args:
            model: DenseKeypointDetectionModel
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_fn: DenseKeypointLoss
            device: 设备
            save_dir: 保存目录
            max_grad_norm: 梯度裁剪阈值
            visualize_validation: 是否保存验证可视化
            num_val_vis_samples: 验证可视化的样本数（None表示保存全部）
            val_every_n_epochs: 每N个epoch验证一次
        """
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.should_visualize_validation = visualize_validation
        self.num_val_vis_samples = num_val_vis_samples
        self.val_every_n_epochs = val_every_n_epochs

        # 训练统计
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Loss记录
        self.train_losses = []  # 每个epoch的训练loss
        self.val_losses = []    # 每次验证的loss
        self.val_epochs = []    # 验证发生的epoch编号

        # 可视化目录
        self.vis_dir = self.save_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)

        if hasattr(self.args, 'use_wandb') and self.args.use_wandb and HAS_WANDB:
            wandb.init(project=self.args.wandb_project, config=vars(self.args))

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_obj_loss = 0
        total_reg_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['image'].to(self.device)
            coords = batch['coords'].to(self.device)
            visibility = batch['visibility'].to(self.device)

            # 前向传播（AMP混合精度）
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                pred_obj = outputs['objectness']
                pred_offsets = outputs['offsets']

                # 计算损失
                loss, loss_dict = self.loss_fn(pred_obj, pred_offsets, coords, visibility)

            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 学习率调度（如果使用CosineAnnealingWarmRestarts）
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step(self.epoch + batch_idx / len(self.train_loader))

            # 统计
            total_loss += loss_dict['total']
            total_obj_loss += loss_dict['obj']
            total_reg_loss += loss_dict['reg']

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'obj': f"{loss_dict['obj']:.4f}",
                'reg': f"{loss_dict['reg']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # 记录到wandb
            if HAS_WANDB and wandb.run is not None:
                wandb.log({
                    'train/loss': loss_dict['total'],
                    'train/obj_loss': loss_dict['obj'],
                    'train/reg_loss': loss_dict['reg'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.global_step
                })

            # 定期可视化
            if self.global_step % 100 == 0:
                visualize_batch(self.vis_dir, self.epoch, self.model, images, coords, visibility, outputs, batch_idx)

            self.global_step += 1

        # 平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_obj_loss = total_obj_loss / len(self.train_loader)
        avg_reg_loss = total_reg_loss / len(self.train_loader)

        return {
            'total': avg_loss,
            'obj': avg_obj_loss,
            'reg': avg_reg_loss
        }

    @torch.no_grad()
    def validate(self, num_vis_samples=None):
        """
        验证

        Args:
            num_vis_samples: 可视化的样本数量（None表示保存全部）

        Returns:
            val_metrics: 验证指标字典
        """
        self.model.eval()
        total_loss = 0
        total_obj_loss = 0
        total_reg_loss = 0

        # 收集所有验证结果
        all_predictions = []

        # 收集用于可视化的样本（如果启用）
        vis_samples = []
        should_vis_all = num_vis_samples is None  # None表示保存全部

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
            images = batch['image'].to(self.device)
            coords = batch['coords'].to(self.device)
            visibility = batch['visibility'].to(self.device)
            image_ids = batch.get('image_id', [f"val_{batch_idx}_{i}" for i in range(len(images))])

            # 前向传播
            outputs = self.model(images)
            pred_obj = outputs['objectness']
            pred_offsets = outputs['offsets']

            # 计算损失
            loss, loss_dict = self.loss_fn(pred_obj, pred_offsets, coords, visibility)

            total_loss += loss_dict['total']
            total_obj_loss += loss_dict['obj']
            total_reg_loss += loss_dict['reg']

            # 收集所有预测结果
            batch_size = images.shape[0]
            for i in range(batch_size):
                # 使用模型的extract_keypoints方法提取关键点
                batch_outputs = {
                    'objectness': pred_obj[i:i+1],
                    'coords': outputs['coords'][i:i+1]
                }

                keypoints_list = self.model.extract_keypoints(
                    batch_outputs,
                    conf_threshold=0.3,
                    nms_radius=2
                )

                # 转换为简单的格式
                keypoints = []
                for class_kpts in keypoints_list[0]:
                    for kp in class_kpts:
                        # 只提取需要的字段，convert_to_serializable会处理类型转换
                        keypoints.append({
                            'class': kp['class'],
                            'x': kp['x'],
                            'y': kp['y'],
                            'confidence': kp['conf']
                        })

                # GT关键点
                gt_keypoints = []
                for c in range(coords.shape[1]):  # 遍历类别
                    for k in range(coords.shape[2]):  # 遍历每个关键点
                        if visibility[i, c, k] > 0.5:  # 访问具体的keypoint
                            gt_keypoints.append({
                                'class': int(c),
                                'x': float(coords[i, c, k, 0].item()),  # x坐标
                                'y': float(coords[i, c, k, 1].item()),  # y坐标
                                'visibility': float(visibility[i, c, k].item())
                            })

                all_predictions.append({
                    'image_id': image_ids[i] if isinstance(image_ids, list) else image_ids,
                    'gt_keypoints': gt_keypoints,
                    'pred_keypoints': keypoints
                })

            # 收集可视化样本（根据num_vis_samples决定是否保存全部）
            if self.should_visualize_validation:
                if should_vis_all:
                    # 保存全部样本
                    for i in range(images.shape[0]):
                        vis_samples.append({
                            'image': images[i].detach().cpu(),
                            'coords': coords[i].detach().cpu(),
                            'visibility': visibility[i].detach().cpu(),
                            'outputs': {
                                'objectness': outputs['objectness'][i].detach().cpu(),
                                'coords': outputs['coords'][i].detach().cpu()
                            }
                        })
                elif len(vis_samples) < num_vis_samples:
                    # 只保存前N个样本
                    for i in range(min(images.shape[0], num_vis_samples - len(vis_samples))):
                        vis_samples.append({
                            'image': images[i].detach().cpu(),
                            'coords': coords[i].detach().cpu(),
                            'visibility': visibility[i].detach().cpu(),
                            'outputs': {
                                'objectness': outputs['objectness'][i].detach().cpu(),
                                'coords': outputs['coords'][i].detach().cpu()
                            }
                        })

        # 平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_obj_loss = total_obj_loss / len(self.val_loader)
        avg_reg_loss = total_reg_loss / len(self.val_loader)

        val_metrics = {
            'epoch': self.epoch,
            'total': avg_loss,
            'obj': avg_obj_loss,
            'reg': avg_reg_loss
        }

        # 保存所有预测结果到JSON
        save_validation_predictions(self.save_dir, self.epoch, all_predictions, val_metrics)

        # 保存验证可视化
        if self.should_visualize_validation and len(vis_samples) > 0:
            visualize_validation(self.vis_dir, self.epoch, self.model, vis_samples, val_metrics)

        return val_metrics

    def train(self, num_epochs):
        """完整训练流程"""
        # 初始化GradScaler
        self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            self.epoch = epoch

            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # 训练
            train_metrics = self.train_one_epoch()

            print(f"\n训练 - Loss: {train_metrics['total']:.4f}, "
                  f"Obj: {train_metrics['obj']:.4f}, Reg: {train_metrics['reg']:.4f}")

            # 验证（每N个epoch验证一次）
            val_metrics = None
            should_validate = (epoch + 1) % self.val_every_n_epochs == 0

            if should_validate:
                val_metrics = self.validate(
                    num_vis_samples=self.num_val_vis_samples
                )

                print(f"验证 - Loss: {val_metrics['total']:.4f}, "
                      f"Obj: {val_metrics['obj']:.4f}, Reg: {val_metrics['reg']:.4f}")

                # 记录到wandb
                if HAS_WANDB and wandb.run is not None:
                    wandb.log({
                        'epoch': epoch,
                        'val/loss': val_metrics['total'],
                        'val/obj_loss': val_metrics['obj'],
                        'val/reg_loss': val_metrics['reg']
                    })
            else:
                print(f"跳过验证 (每{self.val_every_n_epochs}个epoch验证一次)")

            # 记录loss
            log_loss(self.train_losses, self.val_losses, self.val_epochs, self.epoch, train_metrics, val_metrics)

            # 学习率调度
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step()

            # 保存检查点
            if val_metrics is not None:
                # 每次验证时都保存checkpoint（epoch编号）
                self.best_val_loss = self.model.save_ckpt(
                    self.save_dir, f'checkpoint_epoch_{epoch+1}.pth',
                    self.epoch, self.global_step, self.best_val_loss, val_metrics['total']
                )

                # 始终保存latest
                self.model.save_ckpt(
                    self.save_dir, 'checkpoint_latest.pth',
                    self.epoch, self.global_step, self.best_val_loss, val_metrics['total']
                )

                # 保存loss log和更新曲线图
                save_loss_log(self.save_dir, self.train_losses, self.val_losses, self.val_epochs, self.best_val_loss)
                plot_loss_curves(self.save_dir, self.train_losses, self.val_losses, self.best_val_loss)

        print(f"\n{'='*60}")
        print("训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        # 训练结束后，保存最终的log和曲线图
        save_loss_log(self.save_dir, self.train_losses, self.val_losses, self.val_epochs, self.best_val_loss)
        plot_loss_curves(self.save_dir, self.train_losses, self.val_losses, self.best_val_loss)

        # Stop wandb run
        if HAS_WANDB and wandb.run is not None:
            wandb.finish()