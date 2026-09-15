import torch
import torch.nn as nn
import sys
import os
import torch.optim as optim
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tensorboardX import SummaryWriter
from utils.networks_utils import zero_module, get_ckpt_and_writer_path

from utils.misc import encode_axes
from utils.clock import TrainClock
from networks.scale.scale_utils import ExponentialMovingAverage
    
class ScaleNet(nn.Module):
    def __init__(self, args, pts_dim, dino_dim=0, embedding_dim=180):
        super(ScaleNet, self).__init__()
        self.args = args
        self.clock = TrainClock()
        self.is_testing = False

        self.pts_dim = pts_dim
        self.dino_dim = dino_dim
        self.embedding_dim = embedding_dim
        assert embedding_dim % 18 == 0

        self.act = nn.ReLU(True)
        self.axes_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        self.fusion_tail_length = nn.Sequential(
            nn.Linear(pts_dim + dino_dim + 256, 256),
            self.act,
            zero_module(nn.Linear(256, 3))
        )

        # get checkpoint and writer path
        self.model_dir, writer_path = get_ckpt_and_writer_path(self.args)

        if self.args.is_train:
            self.writer = SummaryWriter(writer_path)
        
        self.to(args.device)
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.args.ema_rate)

    # forward function
    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, pts_dim]
                'rgb_feat': [bs, dino_dim] (optional)
                'axes': [bs, 3, 3]
            }
        
        Return: 
            Length: [bs, 3]
        '''
        axes_feat = self.axes_encoder(encode_axes(data['axes'], self.embedding_dim // 18))
        total_feat = torch.cat([data['pts_feat'], axes_feat], dim=-1)
        if self.dino_dim:
            total_feat = torch.cat([total_feat, data['rgb_feat']], dim=-1)
        return self.fusion_tail_length(total_feat)

    # loss function
    def loss_fn(self, pred_len, gt_len):
        '''
        pred_len: [bs, 3]
        gt_len: [bs, 3]
        '''
        # return torch.mean((pred_len - gt_len) ** 2) * 10000
        return torch.mean(nn.SmoothL1Loss(reduction='none')(pred_len, gt_len)) * 10000
    
    # set optimizer
    def set_optimizer(self):
        """set optimizer used in training"""
        params = self.parameters()            
        self.base_lr = self.args.lr
        if self.args.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif self.args.optimizer == 'Adam':
            optimizer = optim.Adam(params, betas=(0.9, 0.999), eps=1e-8, lr=self.args.lr)     
        else:
            raise NotImplementedError
        return optimizer

    # set scheduler
    def set_scheduler(self):
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.lr_decay)
        return scheduler
    
    # save checkpoint
    def save_ckpt(self, name=None):
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        self.ema.store(self.parameters())
        self.ema.copy_to(self.parameters())
        model_state_dict = self.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.to(self.args.device)
        self.ema.restore(self.parameters())

    # load checkpoint
    def load_ckpt(self, model_dir, load_model_only=False):
        if not os.path.exists(model_dir):
            raise ValueError("Checkpoint {} not exists.".format(model_dir))

        ckpt = torch.load(model_dir)
        print("Loading checkpoint from {} ...".format(model_dir))
        
        # we only use single gpu
        self.load_state_dict(ckpt['model_state_dict'])
        
        if not load_model_only:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.clock.restore_checkpoint(ckpt['clock'])

    # collect loss
    def collect_scale_loss(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'rgb_feat': [bs, dino_dim]
                'axes_training': [bs, cbs, 3, 3]
                'length_training': [bs, cbs, 3]
            }
        '''
        _data = {
            'axes': data['axes_training'].reshape(-1, 3, 3),
            'pts_feat': data['pts_feat'].repeat_interleave(self.args.scale_batch_size, dim=0),
        }
        if 'rgb_feat' in data and data['rgb_feat'] is not None:
            _data['rgb_feat'] = data['rgb_feat'].repeat_interleave(self.args.scale_batch_size, dim=0)
        pred_len = self(_data)
        gt_len = data['length_training'].reshape(-1, 3)
        len_loss = self.loss_fn(pred_len, gt_len)
        losses = {'length': len_loss}
        return losses

    # train one epoch
    def train_scale_one_step(self, data):
        self.train()

        cls_losses = self.collect_scale_loss(data)
        
        self.update_network(cls_losses)
        self.record_losses(cls_losses, 'train')
        self.record_lr()
        
        self.ema.update(self.parameters())
        self.pts_feature = False
        return cls_losses

    # eval function
    def eval_scale_func(self, data, data_mode):
        self.is_testing = True
        self.eval()

        with torch.no_grad(): 

            _data = data.copy()
            _data['axes'] = _data['axes_training'][:, 0, :, :] # no need for additional batching in evaluation
            pred_len = self(_data)
            
            metric = self.collect_length_metric(pred_len, data['length_training'][:, 0, :])
            
            self.record_metrics(metric, 'scale', data_mode)
            
        return metric
    
    # predict function
    def pred_scale_func(self, data):
        """
        predict length
        also return axes because of historical reasons
        data: {
            'axes': [bs, 3, 3]
            'pts_feat'
            'rgb_feat'
        }
        Return: {
            axes: [bs, 3, 3]
            length: [bs, 3]
        }
        """
        self.is_testing = True
        self.eval()

        with torch.no_grad(): 
            pred_len = self(data)
            
        return data['axes'], pred_len
    
    # back propagate
    def update_network(self, loss_dict):
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        if torch.isnan(loss).item():
            print("nan encountered")
            return
        loss.backward()
        if self.args.grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=self.args.grad_clip
            )
        self.optimizer.step()

    # update learning rate
    def update_learning_rate(self):
        if self.clock.step <= self.args.warmup:
            self.optimizer.param_groups[-1]['lr'] = self.base_lr / self.args.warmup * self.clock.step
        elif not self.optimizer.param_groups[-1]['lr'] < 1e-4:
            self.scheduler.step()

    # record losses
    def record_losses(self, loss_dict, mode='train'):
        losses_values = {k: v.item() for k, v in loss_dict.items()}
        for k, v in losses_values.items():
            self.writer.add_scalar(f'{mode}/{k}', v, self.clock.step)

    # record metrics
    def record_metrics(self, metric, sampler_mode, mode='val'):

        if 'rot_error' in metric:
            rot_error = metric['rot_error']
            for k, v in rot_error.items():
                if not k == 'item':
                    self.writer.add_scalar(f'{mode}/{sampler_mode}_{k}_rot_error', v, self.clock.epoch)
        
        if 'trans_error' in metric:
            trans_error = metric['trans_error']
            for k, v in trans_error.items():
                if not k == 'item':
                    self.writer.add_scalar(f'{mode}/{sampler_mode}_{k}_trans_error', v, self.clock.epoch)
        
        if 'length_error' in metric:
            length_error = metric['length_error']
            for k, v in length_error.items():
                if not k == 'item':
                    self.writer.add_scalar(f'{mode}/{sampler_mode}_{k}_length_error', v, self.clock.epoch)

    # record learning rate
    def record_lr(self):
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.clock.step)

    # collect length metrics
    def collect_length_metric(self, pred_length, gt_length):
        length_error = torch.norm(pred_length - gt_length, dim=1).cpu().numpy()
        length_error = {
            'mean': np.mean(length_error),
            'median': np.median(length_error),
            'item': length_error,
        }
        return {'length_error': length_error}