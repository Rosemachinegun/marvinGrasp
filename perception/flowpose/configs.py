import torch.nn as nn
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSGFus
from networks.scale.scalenet import ScaleNet

MODEL_ARCHS = {
    "pointnet": Pointnet2ClsMSGFus,
    "scalenet": ScaleNet,
    "keypoint": None
}

MODEL_CONFIGS = {
    "pointnet": {
        # TODO
    },
    "scalenet": {
        # TODO
    },
    "keypoint": {
        # TODO
    }
}

def instantiate_model(args) -> nn.Module:
    architechture = args.arch
    assert (
        architechture in MODEL_CONFIGS
    ), f"你在 干什么？"

    configs = MODEL_CONFIGS[architechture]
    arch = MODEL_ARCHS[architechture]

    if architechture == "scalenet":
        model = ScaleNet(args,
                         args.num_points,
                         dino_dim=0, # use pointwise dino
                         embedding_dim=args.scale_embedding)
    
    elif architechture == "keypoint":
        if getattr(args, "is_train", False):
            from networks.keypoint.keypoint import KeypointModel
            model = KeypointModel(
                args,
                num_classes=args.num_classes,
                dino=args.dino,
                img_size=args.img_size,
            )
        else:
            from networks.keypoint.decoder import DenseKeypointHead
            model = DenseKeypointHead(
                d_model=384,
                num_classes=args.num_classes
            )

    elif architechture == "pointnet":
        configs['dropout'] = args.dropout

        if args.use_edm_aug:
            configs['augment_dim'] = 6

        if getattr(args, "is_train", False):
            from networks.flow.meanflow_train import MeanFlow
        else:
            from networks.flow.meanflow_inference import MeanFlow

        model = MeanFlow(arch=arch, net_configs=configs, args=args)

    return model