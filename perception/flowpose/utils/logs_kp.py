import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path


def convert_to_serializable(obj):
    """递归转换tensor/numpy类型为Python原生类型"""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_validation_predictions(save_dir, epoch, all_predictions, val_metrics):
    """
    保存所有验证预测结果到JSON

    Args:
        save_dir: 保存目录
        epoch: 当前epoch编号
        all_predictions: 所有验证样本的预测结果列表
        val_metrics: 验证指标
    """
    val_results_dir = Path(save_dir) / 'val_results'
    val_results_dir.mkdir(parents=True, exist_ok=True)

    val_json_path = val_results_dir / f'val_epoch_{epoch}.json'

    save_data = {
        'epoch': epoch,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'val_metrics': {
            'total_loss': float(val_metrics['total']),
            'obj_loss': float(val_metrics['obj']),
            'reg_loss': float(val_metrics['reg'])
        },
        'predictions': all_predictions
    }

    save_data = convert_to_serializable(save_data)

    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 验证结果已保存到: {val_json_path} ({len(all_predictions)} 个样本)")


def log_loss(train_losses, val_losses, val_epochs, epoch, train_metrics, val_metrics=None):
    """
    记录loss到列表中

    Args:
        train_losses: 训练loss列表（就地修改）
        val_losses: 验证loss列表（就地修改）
        val_epochs: 验证epoch列表（就地修改）
        epoch: 当前epoch编号
        train_metrics: 训练指标字典
        val_metrics: 验证指标字典（可选）
    """
    train_losses.append({
        'epoch': epoch,
        'total': train_metrics['total'],
        'obj': train_metrics['obj'],
        'reg': train_metrics['reg']
    })

    if val_metrics is not None:
        val_losses.append({
            'epoch': val_metrics['epoch'],
            'total': val_metrics['total'],
            'obj': val_metrics['obj'],
            'reg': val_metrics['reg']
        })
        val_epochs.append(val_metrics['epoch'])


def save_loss_log(save_dir, train_losses, val_losses, val_epochs, best_val_loss):
    """保存loss log到JSON和CSV文件"""
    save_dir = Path(save_dir)
    log_path = save_dir / 'training_log.json'

    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_epochs': len(train_losses),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_epochs': val_epochs,
        'best_val_loss': best_val_loss
    }

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 训练log已保存: {log_path}")

    csv_path = save_dir / 'training_log.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_total_loss,train_obj_loss,train_reg_loss,val_total_loss,val_obj_loss,val_reg_loss\n')

        for train_loss in train_losses:
            ep = train_loss['epoch']

            val_total = val_obj = val_reg = ''
            if ep in val_epochs:
                val_idx = val_epochs.index(ep)
                val_total = val_losses[val_idx]['total']
                val_obj = val_losses[val_idx]['obj']
                val_reg = val_losses[val_idx]['reg']

            f.write(f"{ep},{train_loss['total']:.6f},{train_loss['obj']:.6f},{train_loss['reg']:.6f},"
                    f"{val_total},{val_obj},{val_reg}\n")

    print(f"✅ 训练log CSV已保存: {csv_path}")
