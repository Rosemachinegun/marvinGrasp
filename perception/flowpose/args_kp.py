import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='密集关键点检测训练')

    parser.add_argument('--is_train', action='store_true', default=False, 
                        help='Set this flag for training mode')

    # 数据参数
    parser.add_argument('--raw', type=str, default='?', dest='data_path',
                        help='数据集根目录（包含images和labels子目录）')
    parser.add_argument('--train-val-split', type=float, default=0.8,
                        help='训练集比例（0-1）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--img-size', type=int, default=224,
                        help='图像尺寸（实例级crop后resize的目标尺寸）')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')

    # 模型参数
    parser.add_argument('--num-classes', type=int, default=2,
                        help='类别数')
    parser.add_argument('--num-keypoints', type=int, default=4,
                        help='每类最大关键点数（用于数据集）')
    parser.add_argument('--dino-model', type=str, default='dinov2_vits14',
                        help='DINOv2模型 (torch.hub版本: dinov2_vits14/dinov2_vitb14/dinov2_vitl14)')
    parser.add_argument('--freeze-dino', type=lambda x: x.lower() == 'true', default=True,
                        help='是否冻结DINOv2 backbone (默认: True)')

    # 损失函数参数
    parser.add_argument('--focal-alpha', type=float, default=0.5,
                        help='Focal Loss alpha (默认0.5，减少对分类的过度关注)')
    parser.add_argument('--focal-gamma', type=float, default=1.0,
                        help='Focal Loss gamma (默认1.0，降低难样本权重)')
    parser.add_argument('--obj-weight', type=float, default=0.5,
                        help='Objectness损失权重 (默认0.5，降低分类权重)')
    parser.add_argument('--reg-weight', type=float, default=3.0,
                        help='回归损失权重 (默认3.0，增加定位权重)')
    parser.add_argument('--use-gaussian', action='store_true',
                        help='是否使用高斯衰减标记正样本')
    parser.add_argument('--gaussian-sigma', type=float, default=1.0,
                        help='高斯标准差')
    parser.add_argument('--neighborhood-size', type=int, default=3,
                        help='正样本邻域大小（3/5/7，默认3）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--lr-backbone', type=float, default=1e-5,
                        help='Backbone学习率')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='梯度裁剪阈值')

    # 保存和日志
    parser.add_argument('--save-dir', type=str, default='checkpoints_dense',
                        help='保存目录')
    parser.add_argument('--use-wandb', action='store_true',
                        help='是否使用wandb')
    parser.add_argument('--wandb-project', type=str, default='dense-keypoint-detector',
                        help='wandb项目名')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--val-vis', action='store_true', default=True,
                        help='是否保存验证可视化（默认启用）')
    parser.add_argument('--no-val-vis', action='store_true',
                        help='禁用验证可视化')
    parser.add_argument('--val-vis-samples', type=lambda x: None if x.lower() == 'all' else int(x),
                        default='all',
                        help='每个epoch保存的验证可视化样本数（默认all保存全部，或指定数字如4）')
    parser.add_argument('--val-every-n-epochs', type=int, default=10,
                        help='每N个epoch验证一次（默认10）')

    args = parser.parse_args()

    return args