from dataclasses import dataclass
import json
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

@dataclass
class Pose(object):
    """Object/Camera pose representation"""

    quaternion: "tuple[float, float, float, float]"
    """quaternion in scale-first format (wxyz)"""

    translation: "tuple[float, float, float]"
    """translation from object (centered) space to camera space"""

    def __post_init__(self):
        assert len(self.quaternion) == 4
        assert len(self.translation) == 3

    def to_affine(self, scale=None):
        """transform to affine transformation (with no additional scaling)

        :return: 4x4 numpy array.

        Here's an example of getting :class:`Pose` from rotation matrix:

        .. doctest::

            >>> from cutoop.data_types import Pose
            >>> from scipy.spatial.transform import Rotation
            >>> x, y, z, w = Rotation.from_matrix([
            ...     [ 0.,  0.,  1.],
            ...     [ 0.,  1.,  0.],
            ...     [-1.,  0.,  0.]
            ... ]).as_quat()
            >>> pose = Pose(quaternion=[w, x, y, z], translation=[1, 1, 1])
            >>> pose.to_affine()
            array([[ 0.,  0.,  1.,  1.],
                   [ 0.,  1.,  0.,  1.],
                   [-1.,  0.,  0.,  1.],
                   [ 0.,  0.,  0.,  1.]], dtype=float32)
        """
        q = self.quaternion
        rot = _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        if scale is not None:
            rot = rot * scale
        trans = np.array(self.translation)
        mtx = np.eye(4).astype(np.float32)
        mtx[:3, :3] = rot
        mtx[:3, 3] = trans
        return mtx


@dataclass
class CameraIntrinsicsBase:
    """Camera intrinsics data.
    The unit of ``fx, fy, cx, cy, width, height`` are all pixels.
    """

    fx: float  # unit: pixel
    fy: float  # unit: pixel
    cx: float  # unit: pixel
    cy: float  # unit: pixel
    width: float  # unit: pixel
    height: float  # unit: pixel

    def to_matrix(self):
        """Transform to 3x3 K matrix. i. e.::

            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1 ]]

        :return: 3x3 numpy array.
        """
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def fov_x(self):
        return np.rad2deg(2 * np.arctan2(self.width, 2 * self.fx))

@dataclass
class ViewInfo(Pose):
    intrinsics: CameraIntrinsicsBase
    """Camera intrinsics"""
    scene_obj_path: str
    """Scene object mesh path"""

    background_image_path: str
    background_depth_path: str
    distances: "list[float]"
    kind: str

    def __post_init__(self):
        if not isinstance(self.intrinsics, CameraIntrinsicsBase):
            self.intrinsics = CameraIntrinsicsBase(**self.intrinsics)

@dataclass
class ObjectMetaInfo:
    # --- fields present in both formats ---
    oid: str
    """Object ID (new format: 'object_id', old format: 'oid')"""
    class_name: str
    """Class/category name (new format: 'category', old format: 'class_name')"""
    bbox_side_len: "list[float]"

    # --- old-format-only fields ---
    class_label: int = None
    """1-indexed class label (old format only)"""
    instance_path: str = None
    """Path to the model mesh file (old format only)"""
    scale: "list[float]" = None
    """Scale from object space to camera space (old format only)"""
    is_background: bool = None
    """Whether it is a background object (old format only)"""

    # --- new-format-only fields ---
    canonical_name: str = None
    """Full canonical instance name, e.g. 'mug_blue_001' (new format only)"""

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectMetaInfo":
        """Parse from either old-format (oid/class_name) or new-format (object_id/category)."""
        if "object_id" in d:            # new format
            return cls(
                oid=d["object_id"],
                class_name=d["category"],
                bbox_side_len=d["bbox_side_len"],
                canonical_name=d.get("canonical_name"),
            )
        else:                           # old format
            return cls(
                oid=d["oid"],
                class_name=d["class_name"],
                bbox_side_len=d["bbox_side_len"],
                class_label=d.get("class_label"),
                instance_path=d.get("instance_path"),
                scale=d.get("scale"),
                is_background=d.get("is_background"),
            )


@dataclass
class ObjectPoseInfo:
    # --- fields present in both formats ---
    mask_id: int
    """the value identifying this object in mask image"""
    meta: ObjectMetaInfo
    """object meta information."""
    quaternion_wxyz: "list[float]"
    """object rotation in camera space"""
    translation: "list[float]"
    """object translation in camera space"""

    # --- new-format-only fields ---
    rotation: "list[float]" = None
    """Euler angle rotation vector (new format only)"""

    # --- old-format-only fields ---
    is_valid: bool = None
    """Whether the object meets requirements (old format only)"""
    id: int = None
    """Object id in image, deprecated (old format only)"""
    material: "list[str]" = None
    world_quaternion_wxyz: "list[float]" = None
    world_translation: "list[float]" = None

    def __post_init__(self):
        """Backward-compatible: auto-construct ObjectMetaInfo from a raw old-format dict."""
        if not isinstance(self.meta, ObjectMetaInfo):
            self.meta = ObjectMetaInfo(**self.meta)

    def pose(self) -> Pose:
        """Construct :class:`Pose` relative to the camera coordinate."""
        return Pose(quaternion=self.quaternion_wxyz, translation=self.translation)

    @classmethod
    def from_dict(cls, d: dict, mask_id: int = None) -> "ObjectPoseInfo":
        """Parse from either format.

        ``mask_id`` is the fallback used for old-format data where the id is
        encoded in the parent dict key.  New-format data carries ``mask_id``
        inside ``meta`` and takes priority.
        """
        meta_data = d["meta"]
        meta = ObjectMetaInfo.from_dict(meta_data)
        # new format stores mask_id inside meta; old format derives it from the key prefix
        resolved_mask_id = meta_data.get("mask_id", mask_id)
        return cls(
            mask_id=resolved_mask_id,
            meta=meta,
            quaternion_wxyz=d["quaternion_wxyz"],
            translation=d["translation"],
            rotation=d.get("rotation"),
            is_valid=d.get("is_valid"),
            id=d.get("id"),
            material=d.get("material"),
            world_quaternion_wxyz=d.get("world_quaternion_wxyz"),
            world_translation=d.get("world_translation"),
        )


@dataclass
class ImageMetaData:
    objects: "list[ObjectPoseInfo]"
    """A list of visible objects"""
    camera: ViewInfo
    """Camera information"""

    # Normalised to scene_dataset internally; from_dict accepts both 'dataset' and 'scene_dataset'
    scene_dataset: str = None

    # old-format simulation fields (all optional)
    env_param: dict = None
    face_up: bool = None
    concentrated: bool = None
    comments: str = None
    runtime_seed: int = None
    baseline_dis: int = None
    emitter_dist_l: int = None

    def __post_init__(self):
        """Backward-compatible: construct from raw old-format dicts passed via ``**``."""
        if isinstance(self.objects, dict):
            self.objects = [
                ObjectPoseInfo(**x, mask_id=int(k.split("_")[0]))
                for k, x in self.objects.items()
            ]
        if self.camera is not None and not isinstance(self.camera, (ViewInfo, CameraIntrinsicsBase)):
            self.camera = ViewInfo(**self.camera)

    @classmethod
    def from_dict(cls, d: dict) -> "ImageMetaData":
        """Parse from either old-format or new-format JSON dict.

        Format is detected automatically:

        * **New format** – top-level ``"dataset"`` key; camera has only
          ``"intrinsics"``; per-object meta uses ``"object_id"`` / ``"category"``.
        * **Old format** – top-level ``"scene_dataset"`` key; camera is a full
          ``ViewInfo`` dict; per-object meta uses ``"oid"`` / ``"class_name"``.
        """
        # --- camera ---
        cam_data = d["camera"]
        if set(cam_data.keys()) == {"intrinsics"}:   # new format: only intrinsics present
            intrinsics = CameraIntrinsicsBase(**cam_data["intrinsics"])
            camera = ViewInfo(None, None, intrinsics, None, None, None, None, None)
        else:                                         # old format: full ViewInfo data
            cam_copy = dict(cam_data)
            if not isinstance(cam_copy.get("intrinsics"), CameraIntrinsicsBase):
                cam_copy["intrinsics"] = CameraIntrinsicsBase(**cam_copy["intrinsics"])
            camera = ViewInfo(**cam_copy)

        # --- objects ---
        objects = []
        for key, obj_data in d["objects"].items():
            # New format key: {object_id}_{canonical_name}_{mask_id} — last token is mask_id.
            # Old format key: {mask_id}_{class_name}                 — first token is mask_id.
            # meta.mask_id (new format) takes priority inside ObjectPoseInfo.from_dict.
            meta_data = obj_data.get("meta", {})
            if "object_id" in meta_data:
                key_mask_id = int(key.split("_")[-1])
            else:
                key_mask_id = int(key.split("_")[0])
            objects.append(ObjectPoseInfo.from_dict(obj_data, mask_id=key_mask_id))

        # --- top-level ---
        # Accept both 'dataset' (new) and 'scene_dataset' (old), normalise to scene_dataset
        dataset = d.get("dataset") or d.get("scene_dataset")

        return cls(
            objects=objects,
            camera=camera,
            scene_dataset=dataset,
            env_param=d.get("env_param"),
            face_up=d.get("face_up"),
            concentrated=d.get("concentrated"),
            comments=d.get("comments"),
            runtime_seed=d.get("runtime_seed"),
            baseline_dis=d.get("baseline_dis"),
            emitter_dist_l=d.get("emitter_dist_l"),
        )

    @staticmethod
    def load_json(path: str) -> "ImageMetaData":
        """Load and parse a metadata JSON file (both old and new format supported)."""
        with open(path, "r") as f:
            return ImageMetaData.from_dict(json.load(f))