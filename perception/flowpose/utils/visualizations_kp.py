import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def visualize_batch(vis_dir, epoch, model, images, gt_coords, gt_vis, outputs, batch_idx):
    """可视化一个batch的预测结果（只可视化第一张图）"""
    image = images[0].detach().cpu()
    gt_coord = gt_coords[0].detach().cpu()
    gt_visi = gt_vis[0].detach().cpu()

    pred_obj = torch.sigmoid(outputs['objectness'][0].detach()).cpu()  # [C, H, W]
    pred_coords = outputs['coords'][0].detach().cpu()  # [C, H, W, 2]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).numpy()

    num_classes = pred_obj.shape[0]
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))

    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for c in range(num_classes):
        ax1 = axes[c, 0]
        ax1.imshow(image_np)
        for k in range(gt_coord.shape[1]):
            if gt_visi[c, k] > 0.5:
                x, y = gt_coord[c, k].numpy()
                ax1.plot(x * image_np.shape[1], y * image_np.shape[0], 'go', markersize=10, markeredgewidth=2)
        ax1.set_title(f'Class {c} - Ground Truth')
        ax1.axis('off')

        ax2 = axes[c, 1]
        heatmap = pred_obj[c].numpy()
        im = ax2.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
        ax2.set_title(f'Class {c} - Predicted Heatmap')
        plt.colorbar(im, ax=ax2)

        ax3 = axes[c, 2]
        ax3.imshow(image_np)

        keypoints = model.extract_keypoints(
            {
                'coords': outputs['coords'][:1],
                'objectness': outputs['objectness'][:1]
            },
            conf_threshold=0.3,
            nms_radius=2
        )[0][c]

        for kp in keypoints:
            x, y = kp['x'], kp['y']
            conf = kp['conf']
            ax3.plot(x * image_np.shape[1], y * image_np.shape[0], 'ro', markersize=8, alpha=0.7)
            ax3.text(x * image_np.shape[1], y * image_np.shape[0], f'{conf:.2f}', color='white', fontsize=8,
                     bbox=dict(facecolor='red', alpha=0.5))

        ax3.set_title(f'Class {c} - Predictions')
        ax3.axis('off')

    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.05)

    vis_path = Path(vis_dir) / f'epoch_{epoch}_step_{batch_idx}.png'
    plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.05, dpi=150)
    plt.close(fig)


def visualize_validation(vis_dir, epoch, model, vis_samples, val_metrics):
    """
    可视化验证集样本

    Args:
        vis_dir: 可视化根目录
        epoch: 当前epoch编号
        model: 模型（用于extract_keypoints）
        vis_samples: 验证样本列表，每个样本包含 image, coords, visibility, outputs
        val_metrics: 验证指标字典，包含 total, obj, reg loss
    """
    num_samples = len(vis_samples)

    val_vis_dir = Path(vis_dir) / f'val_epoch_{epoch}'
    val_vis_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, sample in enumerate(vis_samples):
        image = sample['image']
        gt_coord = sample['coords']
        gt_visi = sample['visibility']
        outputs = sample['outputs']

        pred_obj = torch.sigmoid(outputs['objectness'])  # [C, H, W]
        pred_coords = outputs['coords']  # [C, H, W, 2]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
        image_denorm = image * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        image_np = image_denorm.permute(1, 2, 0).cpu().numpy()

        num_classes = pred_obj.shape[0]
        fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))

        if num_classes == 1:
            axes = axes.reshape(1, -1)

        total_loss = val_metrics['total']
        obj_loss = val_metrics['obj']
        reg_loss = val_metrics['reg']
        title_suffix = f' (Val Loss: {total_loss:.4f})'

        for c in range(num_classes):
            ax1 = axes[c, 0]
            ax1.imshow(image_np)
            for k in range(gt_coord.shape[1]):
                if gt_visi[c, k] > 0.5:
                    x, y = gt_coord[c, k].cpu().numpy()
                    ax1.plot(x * image_np.shape[1], y * image_np.shape[0], 'go', markersize=10, markeredgewidth=2)
            ax1.set_title(f'Sample {sample_idx} - Class {c} - Ground Truth\nLoss: Total {total_loss:.4f}, Obj {obj_loss:.4f}, Reg {reg_loss:.4f}')
            ax1.axis('off')

            ax2 = axes[c, 1]
            heatmap = pred_obj[c].cpu().numpy()
            im = ax2.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
            ax2.set_title(f'Sample {sample_idx} - Class {c} - Predicted Heatmap{title_suffix}')
            plt.colorbar(im, ax=ax2)

            ax3 = axes[c, 2]
            ax3.imshow(image_np)

            keypoints = model.extract_keypoints(
                {
                    'coords': outputs['coords'].unsqueeze(0),
                    'objectness': outputs['objectness'].unsqueeze(0)
                },
                conf_threshold=0.3,
                nms_radius=2
            )[0][c]

            for kp in keypoints:
                x, y = kp['x'], kp['y']
                conf = kp['conf']
                ax3.plot(x * image_np.shape[1], y * image_np.shape[0], 'ro', markersize=8, alpha=0.7)
                ax3.text(x * image_np.shape[1], y * image_np.shape[0], f'{conf:.2f}', color='white', fontsize=8,
                         bbox=dict(facecolor='red', alpha=0.5))

            ax3.set_title(f'Sample {sample_idx} - Class {c} - Predictions')
            ax3.axis('off')

        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.05)

        vis_path = val_vis_dir / f'sample_{sample_idx}.png'
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.05, dpi=150)
        plt.close(fig)

    print(f"✅ 验证可视化已保存: {val_vis_dir}/ - {num_samples} 个样本")


def plot_loss_curves(save_dir, train_losses, val_losses, best_val_loss):
    """绘制并保存loss曲线图"""
    epochs = [loss['epoch'] for loss in train_losses]
    train_total = [loss['total'] for loss in train_losses]
    train_obj = [loss['obj'] for loss in train_losses]
    train_reg = [loss['reg'] for loss in train_losses]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Total Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_total, 'b-', label='Train Total Loss', linewidth=2, alpha=0.7)
    if val_losses:
        val_epochs = [loss['epoch'] for loss in val_losses]
        val_total = [loss['total'] for loss in val_losses]
        ax1.plot(val_epochs, val_total, 'r-', label='Val Total Loss', linewidth=2, marker='o', markersize=6)
    ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # 2. Objectness Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_obj, 'g-', label='Train Obj Loss', linewidth=2, alpha=0.7)
    if val_losses:
        val_obj = [loss['obj'] for loss in val_losses]
        ax2.plot(val_epochs, val_obj, 'r-', label='Val Obj Loss', linewidth=2, marker='o', markersize=6)
    ax2.set_title('Objectness Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # 3. Regression Loss
    ax3 = axes[1, 0]
    ax3.plot(epochs, train_reg, 'm-', label='Train Reg Loss', linewidth=2, alpha=0.7)
    if val_losses:
        val_reg = [loss['reg'] for loss in val_losses]
        ax3.plot(val_epochs, val_reg, 'r-', label='Val Reg Loss', linewidth=2, marker='o', markersize=6)
    ax3.set_title('Regression Loss', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # 4. 损失比例分析
    ax4 = axes[1, 1]
    if val_losses:
        val_epochs = [loss['epoch'] for loss in val_losses]
        val_obj = [loss['obj'] for loss in val_losses]
        val_reg = [loss['reg'] for loss in val_losses]

        val_obj_weighted = [obj * 1.0 for obj in val_obj]
        val_reg_weighted = [reg * 2.0 for reg in val_reg]

        ax4.plot(val_epochs, val_obj_weighted, 'g-', label='Obj Loss × 1.0', linewidth=2, marker='s', markersize=5)
        ax4.plot(val_epochs, val_reg_weighted, 'm-', label='Reg Loss × 2.0', linewidth=2, marker='^', markersize=5)

        val_total_calc = [obj + reg for obj, reg in zip(val_obj_weighted, val_reg_weighted)]
        ax4.plot(val_epochs, val_total_calc, 'r--', label='Total (Sum)', linewidth=2, alpha=0.5)

    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Weighted Loss', fontsize=12)
    ax4.set_title('Loss Composition (Validation)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Training Loss Curves (Best Val Loss: {best_val_loss:.4f})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    plot_path = Path(save_dir) / 'loss_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Loss曲线图已保存: {plot_path}")
