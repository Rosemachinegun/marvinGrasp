from networks.pts_encoder.pointnet2 import Pointnet2ClsMSGFus
import torch
import torch.nn as nn
import sys
import os
import gc
import torch.optim as optim
import numpy as np
import networks.flow.rng as rng

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tensorboardX import SummaryWriter
from utils.networks_utils import zero_module, get_ckpt_and_writer_path, GaussianFourierProjection
from utils.transforms.rotation import matrix_to_quaternion, get_rot_matrix, normalize_rotation
from utils.clock import TrainClock
from networks.flow.ema import ExponentialMovingAverage

# also add dino
# shit mountain starts from here
class MeanFlow(nn.Module):
    dino_dim = 384
    embedding_dim = 60

    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.args = args
        self.clock = TrainClock()

        # get checkpoint and writer path
        self.model_dir, writer_path = get_ckpt_and_writer_path(self.args)

        if self.args.is_train:
            self.writer = SummaryWriter(writer_path)  

        # dino
        self.dino_dim = MeanFlow.dino_dim
        self.embedding_dim = MeanFlow.embedding_dim
        
        # point cloud encoder: 3 (xyz) + dino_dim (384) = 387 channels
        self.pts_encoder = Pointnet2ClsMSGFus(self.dino_dim)
        state_dict = torch.load("results/pretrained_ckpts/pts_encoder_stripped.pth")
        self.pts_encoder.load_state_dict(state_dict, strict=True)
        self.pts_encoder.requires_grad_(False)

        # pose encoder
        self.ang_encoder = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # time encoder
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # fusion tail - rotation x and y regress head 
        # xxxx
        self.fusion_tail_rot_x = nn.Sequential(
            nn.Linear(128+256+256+1024, 256),
            nn.ReLU(),
            zero_module(nn.Linear(256, 3)),
        )
        # yyyy
        self.fusion_tail_rot_y = nn.Sequential(
            nn.Linear(128+256+256+1024, 256),
            nn.ReLU(),
            zero_module(nn.Linear(256, 3)),
        )
            
        # translation regress head 
        self.fusion_tail_trans = nn.Sequential(
            nn.Linear(128+256+256+1024, 256),
            nn.ReLU(),
            zero_module(nn.Linear(256, 3)),
        )

        self.to(args.device)
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.ema = ExponentialMovingAverage(self.parameters(), 
                                            decay=0.999, 
                                            use_num_updates=True,
                                            period=1,  # periodic update
                                            use_double_precision=True # MeanFlow's numerical stability
                                            )  

    def extract_pts_feature(self, data):
        pts = data['pts']  # [bs, N, 3]

        with torch.no_grad():
            feat = data['dino_feat']  # [bs, 384]
            xs = data['roi_xs'] // 14
            ys = data['roi_ys'] // 14
            pos = xs * 16 + ys
            pos = torch.unsqueeze(pos, -1).expand(-1, -1, self.dino_dim)
            rgb_feat = torch.gather(feat, 1, pos)

            # freeze dino weights
            # rgb_feat.requires_grad_(False)
            data['dino_feat'] = feat.mean(dim=1, keepdim=True).squeeze(1)  # [bs, dino_dim]

        
        if not self.training:
            with torch.no_grad():
                pts_feat = self.pts_encoder(torch.concatenate([pts, rgb_feat], dim=-1))  # [bs, 1024]
        else:
            pts_feat = self.pts_encoder(torch.concatenate([pts, rgb_feat], dim=-1))  # [bs, 1024]

        return pts_feat  # Keep gradients attached

    
    def train_flow_one_step(self, data, compiled_train_step=None, teacher=None):
        gc.collect()
        self.optimizer.zero_grad()

        loss = rng.train_step_with_rng_control(compiled_train_step, self, self.clock.step, self.args.seed, data=data)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.ema.update(self.parameters())
        return loss
    

    def forward(self, data, time_cond, aug_cond=None):
        
        # TIME feat
        t, r = time_cond
        time_feat = self.t_encoder(t.view(-1, 1))  # [bs, 128]
        # PRE-EXTRACTED feat
        pts_feat = data['pts_feat']                       # [bs, 1024]
        sampled_pose = data['sampled_pose']               # [bs, pose_dim]
        ang_feat = self.ang_encoder(sampled_pose[:, :6])  # [bs, 256]
        pos_feat = self.pos_encoder(sampled_pose[:, 6:])  # [bs, 256]
        # # Concatenate all features [bs, 256+256+1024=1536]
        total_feat = torch.cat([time_feat, ang_feat, pos_feat, pts_feat], dim=-1)
        rot_x = self.fusion_tail_rot_x(total_feat)        # 通过X轴旋转回归头输出X轴旋转向量 [bs, 3]
        rot_y = self.fusion_tail_rot_y(total_feat)        # 通过Y轴旋转回归头输出Y轴旋转向量 [bs, 3]
        trans = self.fusion_tail_trans(total_feat)        # [bs, 3]

        return torch.cat([rot_x, rot_y, trans], dim=-1)   # [bs, 9.]


    def forward_with_loss(self, data, aug_cond=None):
        """
        Standard Flow Matching training (no JVP, no ODE)
        """
        gt_pose = data['zero_mean_gt_pose']        # x1  [B, D]
        pts_feat = data['pts_feat']
        dino_feat = data['dino_feat']
        device = gt_pose.device
        B = gt_pose.shape[0]

        # ------------------------------------------------
        # 2. sample t ~ U(0, 1)
        # ------------------------------------------------
        eps = 1e-3
        t_raw = torch.rand((B,)).to(device)  # 均匀分布的0-1采样
        t = eps + (1 - 2*eps) * t_raw  # 将采样结果映射到[eps, 1-eps]
        t_view = t.view(B, 1)


        # # option 2: reflected flow matching
        sigma_min = 0.1
        sigma_max = 2.0
        sigma = sigma_min * torch.exp(torch.log(torch.tensor(sigma_max / sigma_min, device=t.device)) * (1-t_view))
        # sigma = sigma_min * (sigma_max / sigma_min) ** t_view 

        z = torch.randn_like(gt_pose)
        xt = gt_pose + z * sigma  # 添加与时间相关的噪声，得到xt
        target_score = -z * sigma / (sigma**2)
        
        
        # ------------------------------------------------
        # 5. predict velocity field
        # ------------------------------------------------
        temp_data = {
            'sampled_pose': xt,
            'pts_feat': pts_feat,
            'dino_feat': dino_feat,
        }

        v_pred = self.forward(
            temp_data,
            (t,t),
            aug_cond
        )                    # [B, D]

        ##
        loss_weighting = sigma**2
        loss_r = 1.0 * loss_weighting * ((v_pred[:, :6]-target_score[:, :6])**2)
        loss_t = 1.0 * loss_weighting * ((v_pred[:, 6:]-target_score[:, 6:])**2)
        loss = (loss_r.sum(dim=1) + loss_t.sum(dim=1)).mean()
        ##
        if self.clock.step % 100 == 0:
            print(f"Step {self.clock.step}: rot: {loss_r.sum().item():.6f} trans: {loss_t.sum().item():.6f} mean: {loss.mean().item():.6f}")

        return loss


    def save_ckpt(self, name=None):
        if name is None:
            save_path = os.path.join(
                self.model_dir,
                f"ckpt_epoch{self.clock.epoch}.pth"
            )
            print(f"Saving checkpoint epoch {self.clock.epoch}...")
        else:
            save_path = os.path.join(self.model_dir, f"{name}.pth")

        ckpt = {
            "clock": self.clock.make_checkpoint(),
            "model_state_dict": self.state_dict(),          # 原模型
            "ema_state_dict": self.ema.state_dict(),        # ✅ EMA
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        torch.save(ckpt, save_path)
    

    def load_ckpt(self, model_dir, load_model_only=False):
        if not os.path.exists(model_dir):
            raise ValueError("Checkpoint {} not exists.".format(model_dir))
        
        ckpt = torch.load(model_dir)
        print("Loading checkpoint from {} ...".format(model_dir))

        # Load model state
        self.load_state_dict(ckpt['model_state_dict'])
            
        # Restore EMA state if available
        if 'ema_state_dict' in ckpt:
            self.ema.load_state_dict(ckpt['ema_state_dict'])
            self.ema.copy_to(self.parameters())  

    # called from trainer
    def encode_func(self, data):
        data['pts_feat'] = self.extract_pts_feature(data)

    def set_optimizer(self):
        if self.args.optimizer == 'Adam':
            return optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        # Add other optimizer types as needed
        
    def set_scheduler(self):
        return optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.lr_decay)
    
    def update_learning_rate(self):
        self.base_lr = self.args.lr
        if self.clock.step <= self.args.warmup:
            self.optimizer.param_groups[-1]['lr'] = self.base_lr / self.args.warmup * self.clock.step
        elif not self.optimizer.param_groups[-1]['lr'] < 1e-4:
            self.scheduler.step()
 
    def record_lr(self):
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.clock.step)

