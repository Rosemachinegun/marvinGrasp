import os
import torch
import cv2
import numpy as np

class DinoLoader:
    feat = None # static
    DEFAULT_LOCAL_REPO_PATH = '/workspace/model/facebookresearch_dinov2_main'

    def __init__(self, model_name: str = 'dinov2_vits14', device=None, local_repo_path=None, ckpt_path=None):
        self.model_name = model_name
        self.local_repo_path = local_repo_path
        self.ckpt_path = ckpt_path
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._load_model()
        

    def _load_model(self):
        try:
            repo_path = self.local_repo_path or self.DEFAULT_LOCAL_REPO_PATH
            # print(self.model_name)
            if os.path.isdir(repo_path):
                # self.model = torch.hub.load(repo_path, self.model_name, source='local')
                self.model = torch.hub.load(repo_path, self.model_name, source='local', pretrained=False).to(device=self.device)
            else:
                print(f"Warning: Local repository path '{repo_path}' does not exist. Attempting to load model from torch.hub.")
                self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            
            # ===== load local weights =====
            ckpt_path = self.ckpt_path
            if not ckpt_path or not os.path.exists(ckpt_path):
                print(f"Warning: DINO ckpt path '{ckpt_path}' does not exist, using model without pretrained weights.")
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if "model" in ckpt:
                    state_dict = ckpt["model"]
                else:
                    state_dict = ckpt
                self.model.load_state_dict(state_dict, strict=True)

            self.model.to(self.device)
            self.model.requires_grad_(False)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DINO model '{self.model_name}': {e}")

    def _tensor_to_bgr_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().cpu().numpy()
        # channel-first to HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        # handle float types: try to detect normalization and undo it
        if np.issubdtype(arr.dtype, np.floating):
            mn = float(arr.min())
            mx = float(arr.max())
            # common case: ImageNet-normalized floats (~-2..+2)
            if mn < -0.5 and mx <= 3.0 and arr.shape[2] == 3:
                mean = np.array([0.485, 0.456, 0.406], dtype=arr.dtype)
                std = np.array([0.229, 0.224, 0.225], dtype=arr.dtype)
                arr = arr * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        # convert to BGR for OpenCV display
        if arr.ndim == 3 and arr.shape[2] == 3:
            try:
                # arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGR)
            except Exception:
                pass

        return arr
    # extract features from roi_rgb and store in static variable feat
    def extract_features(self, roi_rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.model.get_intermediate_layers(roi_rgb)[0]
            # for i in range(len(roi_rgb)):
            #     cv2.imshow(f"Feature {i}", self._tensor_to_bgr_numpy(roi_rgb[i]))
            #     cv2.waitKey(0)
            self.__class__.feat = feat
    
    # return the extracted features from static variable feat
    def get_feature(self) -> torch.Tensor:
        if self.__class__.feat is None:
            raise ValueError("No features extracted yet. Call extract_features() first.")
        return self.__class__.feat
