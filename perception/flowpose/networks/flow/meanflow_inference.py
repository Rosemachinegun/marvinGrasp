from networks.pts_encoder.pointnet2 import Pointnet2ClsMSGFus
import torch
import torch.nn as nn
import sys
import os
import gc
import time
import torch.optim as optim
import numpy as np
import networks.flow.rng as rng

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tensorboardX import SummaryWriter
from utils.networks_utils import get_ckpt_and_writer_path, GaussianFourierProjection

from utils.transforms.rotation import matrix_to_quaternion, get_rot_matrix, normalize_rotation
from utils.clock import TrainClock
from utils.misc import get_pose_dim
from networks.flow.ema import ExponentialMovingAverage

sigma_min = 0.1
sigma_max = 2.0
eps = 1e-4

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

        self.dino_dim = MeanFlow.dino_dim
        self.embedding_dim = MeanFlow.embedding_dim

        self.pts_encoder = Pointnet2ClsMSGFus(self.dino_dim)
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
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # fusion tail - rotation x and y regress head 
        self.fusion_tail_rot_x = nn.Sequential(
            nn.Linear(128+256+256+1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.fusion_tail_rot_y = nn.Sequential(
            nn.Linear(128+256+256+1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.fusion_tail_trans = nn.Sequential(
            nn.Linear(128+256+256+1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
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

        # Only freeze DINO gradients
        with torch.no_grad():
            feat = data['dino_feat']
            xs = data['roi_xs'] // 14
            ys = data['roi_ys'] // 14
            pos = xs * 16 + ys
            pos = torch.unsqueeze(pos, -1).expand(-1, -1, self.dino_dim)
            rgb_feat = torch.gather(feat, 1, pos)

            # freeze dino weights
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
        pts_feat = data['pts_feat']  # [bs, 1024]
        sampled_pose = data['sampled_pose']  # [bs, pose_dim]
        ang_feat = self.ang_encoder(sampled_pose[:, :6])  # [bs, 256]
        pos_feat = self.pos_encoder(sampled_pose[:, 6:])  # [bs, 256]

        # # Concatenate all features [bs, 256+256+1024=1536]
        total_feat = torch.cat([time_feat, ang_feat, pos_feat, pts_feat], dim=-1)
        rot_x = self.fusion_tail_rot_x(total_feat)   # 通过X轴旋转回归头输出X轴旋转向量 [bs, 3]
        rot_y = self.fusion_tail_rot_y(total_feat)   # 通过Y轴旋转回归头输出Y轴旋转向量 [bs, 3]
        trans = self.fusion_tail_trans(total_feat)   # [bs, 3]

        return torch.cat([rot_x, rot_y, trans], dim=-1)  # [bs, 9]
    
    @torch.no_grad()
    def sample(self, data, num_samples=1, device=None, init_pose=None):
        device = self.args.device if device is None else device
        # ------------------------------------------------
        # 1. extract point features
        # ------------------------------------------------
        if 'pts_feat' not in data:
            data['pts_feat'] = self.extract_pts_feature(data)

        pts_feat = data['pts_feat']
        bs = pts_feat.shape[0]

        # ------------------------------------------------
        # 2. initial noise x0
        # ------------------------------------------------
        if init_pose is None:
            x = 0.5*torch.randn(bs, 9, device=device)
            x[:, 6:] *= 0.0
            num_steps = 10
            t_seq = torch.linspace(eps, 1.0-eps, num_steps + 1, device=device)
        else:
            x = init_pose.clone().to(device)
            num_steps = 10
            t_seq = torch.linspace(0.75, 1.0-eps, num_steps + 1, device=device)

        # ------------------------------------------------
        # 3. Heun (RK2) integration
        # ------------------------------------------------
        for i in range(num_steps):
            t = t_seq[i]
            t_next = t_seq[i + 1]
            dt = t_next - t 

            # ---- k1 ----
            data['sampled_pose'] = x
            sigma = sigma_min * torch.exp(torch.log(torch.tensor(sigma_max / sigma_min, device=t.device)) * (1-t))
            diffusion_coeff = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
            diffusion = diffusion_coeff.cpu().numpy()
            
            u1 = self.forward(
                data,
                (t.expand(bs), t.expand(bs)),
                aug_cond=None
            )
            
            # ---- predictor ----
            x_pred = x + 0.5 * (diffusion**2) * u1 * dt
            
            # ---- k2 ----
            data['sampled_pose'] = x_pred
            u2 = self.forward(
                data,
                (t_next.expand(bs), t_next.expand(bs)),
                aug_cond=None
            )

            # ---- corrector ----
            x = x + 0.25 * (diffusion**2) * dt * (u1 + u2)
            
        # ------------------------------------------------
        # 5. post-process pose
        # ------------------------------------------------
        x[:, -3:] += data['pts_center']
        x[:, :-3] = normalize_rotation(x[:, :-3], 'rot_matrix')
    
        return x.detach()

    def pred_func(self, data, device=None, init_pose=None, valid_prev_label=None):

        device = self.args.device if device is None else device

        # ------------------------------------------------
        # 1. extract point features
        # ------------------------------------------------
        if 'pts_feat' not in data:
            data['pts_feat'] = self.extract_pts_feature(data)

        pts_feat = data['pts_feat']
        bs = pts_feat.shape[0]

        # ------------------------------------------------
        # 2. Handle previous pose logic
        # ------------------------------------------------
        # valid_prev_label = None
        if valid_prev_label and init_pose is not None:
            # compute indices of items that have previous poses
            prev_indices = [i for i, lbl in enumerate(data.get('labels', [])) if lbl in init_pose]
            non_prev_indices = [i for i in range(bs) if i not in prev_indices]

            # build repeated_data for entire batch
            repeated_data = {}
            for key, val in data.items():
                if val is None:
                    continue
                if not isinstance(val, torch.Tensor):
                    repeated_data[key] = val
                    continue
                repeated_data[key] = val.repeat_interleave(self.args.repeat_num, dim=0)

            # prepare init_x tensors for previous items
            if len(prev_indices) > 0:
                init_x_tensors = torch.stack([init_pose[data['labels'][i]] for i in prev_indices], dim=0)
                repeated_init_x = init_x_tensors.repeat_interleave(self.args.repeat_num, dim=0)
            else:
                repeated_init_x = None

            # helper to build repeated index ranges
            def ranges_for_indices(indices):
                if len(indices) == 0:
                    return torch.empty((0,), device=device, dtype=torch.long)
                ranges = [torch.arange(i * self.args.repeat_num, (i + 1) * self.args.repeat_num, device=device) for i in indices]
                return torch.cat(ranges, dim=0)

            prev_repeat_idxs = ranges_for_indices(prev_indices)
            wo_repeat_idxs = ranges_for_indices(non_prev_indices)

            # run inference separately for prev and non-prev groups
            res_prev = None
            res_wo_prev = None
            if prev_repeat_idxs.numel() > 0:
                sub_prev = {k: (v.index_select(0, prev_repeat_idxs) if isinstance(v, torch.Tensor) else v) for k, v in repeated_data.items()}
                res_prev = self.sample(sub_prev, num_samples=1, device=device, init_pose=repeated_init_x)
            if wo_repeat_idxs.numel() > 0:
                sub_wo = {k: (v.index_select(0, wo_repeat_idxs) if isinstance(v, torch.Tensor) else v) for k, v in repeated_data.items()}
                res_wo_prev = self.sample(sub_wo, num_samples=1, device=device, init_pose=None)

            # reconstruct results into original repeated ordering
            pose_dim = (res_prev.shape[-1] if res_prev is not None else res_wo_prev.shape[-1])
            res = torch.empty((bs * self.args.repeat_num, pose_dim), device=device, dtype=(res_prev.dtype if res_prev is not None else res_wo_prev.dtype))

            # Fill blocks
            if res_prev is not None:
                for j, orig_i in enumerate(prev_indices):
                    src_start = j * self.args.repeat_num
                    src_end = (j + 1) * self.args.repeat_num
                    dst_start = orig_i * self.args.repeat_num
                    dst_end = (orig_i + 1) * self.args.repeat_num
                    res[dst_start:dst_end] = res_prev[src_start:src_end]
            if res_wo_prev is not None:
                for j, orig_i in enumerate(non_prev_indices):
                    src_start = j * self.args.repeat_num
                    src_end = (j + 1) * self.args.repeat_num
                    dst_start = orig_i * self.args.repeat_num
                    dst_end = (orig_i + 1) * self.args.repeat_num
                    res[dst_start:dst_end] = res_wo_prev[src_start:src_end]

        else:
            # ------------------------------------------------
            # 3. No previous pose - standard path
            # ------------------------------------------------
            repeated_data = {}
            for key, val in data.items():
                if val is None:
                    continue
                if not isinstance(val, torch.Tensor):
                    repeated_data[key] = val
                    continue
                repeated_data[key] = val.repeat_interleave(self.args.repeat_num, dim=0)

            res = self.sample(repeated_data, num_samples=1, device=device, init_pose=None)

        # ------------------------------------------------
        # 4. Reshape and convert to quaternion
        # ------------------------------------------------
        pred_pose = res.reshape(bs, self.args.repeat_num, -1)

        rot_matrix = get_rot_matrix(res[:, :-3], self.args.pose_mode)
        quat_wxyz = matrix_to_quaternion(rot_matrix)
        res_q_wxyz = torch.cat((quat_wxyz, res[:, -3:]), dim=-1)
        pred_pose_q_wxyz = res_q_wxyz.reshape(bs, self.args.repeat_num, -1)
        
        return pred_pose, pred_pose_q_wxyz

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
        self.load_state_dict(ckpt['model_state_dict'], strict=False)
            
        # Restore EMA state if available
        if 'ema_state_dict' in ckpt:
            self.ema.load_state_dict(ckpt['ema_state_dict'])
            # 切到 EMA 权重
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

