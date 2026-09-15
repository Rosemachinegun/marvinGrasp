import os
import sys
import numpy as np
import webdataset as wds
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

# Adjust path to find the dataset module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import OmniXTrainDataset

def get_grouped_class_name(raw_class_name):
    # group car1, car2, car3 -> car
    if raw_class_name.startswith('car'):
        return 'car'
    if raw_class_name.startswith('box'):
        return 'box'
    if raw_class_name.startswith('bag'):
        return 'bag'
    # Add other groupings if needed
    return raw_class_name

def main():
    # Provide the path to the training dataset shards
    shard_path = "/media/kewei/Training/dataSOPE_webdataset/train-{000000..000010}.tar"
    print(f"Loading dataset from: {shard_path}")
    
    # Use WebDataset map to decode the data
    dataset = wds.WebDataset(shard_path).map(OmniXTrainDataset._pose_data_decoder)
    
    # Dictionary to hold lists of quaternions per class
    class_quaternions = defaultdict(list)
    
    # Iterate through all samples in the dataset
    for i, sample in enumerate(tqdm(dataset, desc="Processing Shards")):
        meta = sample.get("meta", {})
        objects = meta.get("objects", {})
        
        # Meta dictionaries are often keyed by string identifiers
        for obj_id, obj_data in objects.items():
            if 'quaternion_wxyz' in obj_data:
                q_wxyz = obj_data['quaternion_wxyz']
                
                # Extract class name and group
                raw_cls_name = obj_data.get('meta', {}).get('class_name', 'unknown')
                cls_name = get_grouped_class_name(raw_cls_name)
                
                class_quaternions[cls_name].append(q_wxyz)

    if len(class_quaternions) == 0:
        print("No quaternions found in the dataset.")
        return

    print("\n=== Analysis by Class ===")
    
    save_dir = "results/pose_by_class"
    os.makedirs(save_dir, exist_ok=True)
    
    for cls_name, quats in class_quaternions.items():
        # Convert list of quaternions to a numpy array
        q_wxyz_array = np.array(quats)
        print(f"\n--- Class: {cls_name} ---")
        print(f"Extracted {len(q_wxyz_array)} quaternions. Shape: {q_wxyz_array.shape}")
        
        # Convert from w, x, y, z to x, y, z, w for SciPy Rotation
        w_vals = q_wxyz_array[:, 0:1]
        xyz_vals = q_wxyz_array[:, 1:4]
        q_xyzw_array = np.hstack((xyz_vals, w_vals))
        
        try:
            # Calculate rotations and their euler angle representations
            rot = R.from_quat(q_xyzw_array)
            euler_deg = rot.as_euler('xyz', degrees=True)  # Roll, Pitch, Yaw in degrees
            euler_rad = rot.as_euler('xyz', degrees=False) # Roll, Pitch, Yaw in radians
        except Exception as e:
            print(f"Error converting quaternions for class {cls_name}: {e}")
            continue

        # Save results directly to numpy file per class
        save_path = os.path.join(save_dir, f"pose_euler_analysis_{cls_name}.npz")
        
        np.savez(
            save_path,
            quaternion_wxyz=q_wxyz_array,
            euler_angles_deg=euler_deg,
            euler_angles_rad=euler_rad
        )
        print(f"Saved pose arrays to [{save_path}]")
        
        # Output some basic analysis statistics
        print(f"Mean (X, Y, Z): {euler_deg.mean(axis=0)}")
        print(f"Std  (X, Y, Z): {euler_deg.std(axis=0)}")
        print(f"Min  (X, Y, Z): {euler_deg.min(axis=0)}")
        print(f"Max  (X, Y, Z): {euler_deg.max(axis=0)}")

if __name__ == "__main__":
    main()
