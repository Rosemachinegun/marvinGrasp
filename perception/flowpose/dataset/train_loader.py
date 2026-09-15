import webdataset as wds
import torch
from torch.utils.data import DataLoader
from dataset.dataset import OmniXTrainDataset, rawDataset
from dataset.augmentation import *

def get_train_dataloader(args, shard_path, batch_size=32, num_workers=4):
    
    transform = Compose([
        ParseMetaData(),
        CropAndResize(
            img_size=args.img_size,
            dynamic_zoom_params=args.DYNAMIC_ZOOM_IN_PARAMS
        ),
        GeneratePointCloud(
            n_pts=1024,
            deform_2d_params=args.DEFORM_2D_PARAMS
        ),
        # DinoAugmentation(),
        ToTensor(args=args)
    ])

    dataset = (
        wds.WebDataset(shard_path, shardshuffle=1000)
        .shuffle(1000)
        .map(OmniXTrainDataset._pose_data_decoder, handler=wds.warn_and_continue)
        .compose(lambda src: (x for x in src for _ in range(8)))  # 8x repetition
        .map(transform)
        .select(lambda x: x is not None)  # Filter failed transforms
        .batched(batch_size, partial=False)  # batching
    )
    
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,  # batching already done by WebDataset
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader

def get_train_dataloader_raw(args, data_path, batch_size=32, num_workers=4,
                             split: str = 'train'):
    """Training dataloader backed by rawDataset (directory of raw files).

    Uses the same transform pipeline as get_train_dataloader so the two
    loaders are interchangeable from the training loop's perspective.

    Args:
        args:        Parsed args (must have img_size, DYNAMIC_ZOOM_IN_PARAMS,
                     DEFORM_2D_PARAMS).
        data_path:   Root directory passed to rawDataset.  May be the dataset
                     root for a hierarchical layout
                     ({root}/{subset}/{train|val}/{scene}/{index}/) or a flat
                     directory of raw files.
        batch_size:  Mini-batch size.
        num_workers: DataLoader workers.
        split:       If given (default ``'train'``), only directories whose
                     path contains this segment are loaded.  Pass ``None``
                     to load all sub-directories regardless of split.
    """
    
    transform = Compose([
        ParseMetaData(),
        CropAndResize(
            img_size=args.img_size,
            dynamic_zoom_params=args.DYNAMIC_ZOOM_IN_PARAMS
        ),
        GeneratePointCloud(
            n_pts=1024,
            deform_2d_params=args.DEFORM_2D_PARAMS
        ),
        ToTensor(args=args)
    ])

    dataset = rawDataset(path=data_path, transform=transform, keypoint=False,
                         split=split)

    def collate_skip_none(batch):
        batch = [x for x in batch if x is not None]
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_skip_none,
        drop_last=True,
    )

    return dataloader