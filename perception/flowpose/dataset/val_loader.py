import webdataset as wds
from torch.utils.data import DataLoader
from dataset.dataset import OmniXValDataset, rawDataset, rawValDataset
from dataset.augmentation import *

def flatten_per_object(src):
    """Flatten samples when per_object=True returns a list of samples per image."""
    for sample in src:
        if isinstance(sample, list):
            for s in sample:
                yield s
        elif sample is not None:
            yield sample

def _drop_step_filter(src, step):
    """Keep every *step*-th sample.  Implemented as a composable pipeline
    stage so it works correctly with multi-worker DataLoader (each worker
    processes its own shard subset and the counter lives inside the
    generator, so it resets naturally on every new iteration)."""
    for i, sample in enumerate(src):
        if i % step == 0:
            yield sample

def get_validation_dataloader(args, shard_path, batch_size=32, num_workers=4):
    
    args.DYNAMIC_ZOOM_IN_PARAMS['DZI_TYPE'] = 'none' 
    args.DEFORM_2D_PARAMS['roi_mask_pro'] = 0

    transform = Compose([
        ParseMetaData(
            per_object=True
        ),
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

    drop_step = getattr(args, 'drop_step', 1)

    dataset = (
        wds.WebDataset(shard_path, shardshuffle=False)
        .map(OmniXValDataset._decode_sample, handler=wds.warn_and_continue)
    )
    
    # Apply drop_step as a composable pipeline stage so the counter
    # is created fresh each time the pipeline is iterated.
    if drop_step > 1:
        dataset = dataset.compose(lambda src: _drop_step_filter(src, drop_step))
    
    dataset = (
        dataset
        .map(transform)
        .compose(flatten_per_object)  # Flatten per-object samples before batching
        .select(lambda x: x is not None)  # Filter failed transforms
        .batched(batch_size, partial=False)  # Add batching
    )
    
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,  # batching already done by WebDataset
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def get_validation_dataloader_raw(args, shard_path, batch_size=32, num_workers=4):
    args.DYNAMIC_ZOOM_IN_PARAMS['DZI_TYPE'] = 'none' 
    args.DEFORM_2D_PARAMS['roi_mask_pro'] = 0

    transform = Compose([
        ParseMetaData(
            per_object=True
        ),
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

    dataset = rawDataset(path=shard_path, transform=transform, keypoint=False,
                         split='', per_object=True)
    
    def collate_fn_raw(batch):
        """Custom collate function that flattens per-object samples before batching."""
        # Flatten lists of samples (per_object mode)
        flattened = []
        for item in batch:
            if isinstance(item, list):
                flattened.extend([s for s in item if s is not None])
            elif item is not None:
                flattened.append(item)
        
        # Use default collate on flattened samples
        if not flattened:
            return {}
        
        from torch.utils.data._utils.collate import default_collate
        return default_collate(flattened)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_raw,
        drop_last=False,
    )
    
    return dataloader


def get_validation_dataloader_raw_iterable(args, data_path, batch_size=32, num_workers=4):
    """Create validation dataloader using rawValDataset (IterableDataset style).
    
    Similar to get_validation_dataloader but for raw file format instead of webdatasets.
    Uses rawValDataset which iterates through directories and applies per_object handling.
    """
    args.DYNAMIC_ZOOM_IN_PARAMS['DZI_TYPE'] = 'none' 
    args.DEFORM_2D_PARAMS['roi_mask_pro'] = 0

    transform = Compose([
        ParseMetaData(
            per_object=True
        ),
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

    drop_step = getattr(args, 'drop_step', 1)

    dataset = rawValDataset(
        data_path=data_path, 
        transform=transform, 
        drop_step=drop_step,
        per_object=True
    )
    
    # For IterableDataset, use DataLoader with batch_size directly
    # The dataset yields individual samples, and DataLoader batches them
    def batch_collate(samples):
        """Batch samples and apply default collate."""
        from torch.utils.data._utils.collate import default_collate
        return default_collate(samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # IterableDataset handles iteration, don't use workers
        pin_memory=True,
        collate_fn=batch_collate,
    )
    
    return dataloader