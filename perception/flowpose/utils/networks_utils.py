import os
import torch
import torch.nn as nn
import numpy as np
import sys

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_ckpt_and_writer_path(args):
    ''' init exp folder and writer '''
    ckpt_path = f'./results/ckpts/{args.log_dir}'
    writer_suffix = '_continue' if args.use_pretrain else ''
    writer_path = f'./results/logs/{args.log_dir}{writer_suffix}'
    
    if args.is_train:
        os.makedirs('./results', exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(writer_path, exist_ok=True)
    return ckpt_path, writer_path

def clear_cache(self):
    if hasattr(self, '_cached_pts_feat'):
        delattr(self, '_cached_pts_feat')

class  GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        # Ensure x is always 2D [batch_size, 1]
        if x.dim() == 1:
            x = x.view(-1, 1)
        
        # x: [bs, 1], W: [embed_dim//2]
        # Proper broadcasting: [bs, 1] * [1, embed_dim//2] = [bs, embed_dim//2]
        x_proj = x * self.W.unsqueeze(0) * 2 * np.pi
        
        # Concatenate sin/cos: [bs, embed_dim]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)