import os
import sys
import numpy as np
import webdataset as wds
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Adjust path to find the dataset module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import OmniXTrainDataset

def main():
    # Provide the path to the training dataset shards
    shard_path = "/media/kewei/Training/dataSOPE_webdataset/train-{000000..000184}.tar"
    print(f"Loading dataset from: {shard_path}")
    
    # Use WebDataset map to decode the data
    dataset = wds.WebDataset(shard_path).map(OmniXTrainDataset._pose_data_decoder)
    
    all_quaternions = []
    
    # Iterate through all samples in the dataset
    for i, sample in enumerate(tqdm(dataset, desc="Processing Shards")):
        meta = sample.get("meta", {})
        objects = meta.get("objects", {})
        
        # Meta dictionaries are often keyed by string identifiers
        for obj_id, obj_data in objects.items():
            if 'quaternion_wxyz' in obj_data:
                q_wxyz = obj_data['quaternion_wxyz']
                all_quaternions.append(q_wxyz)

    if len(all_quaternions) == 0:
        print("No quaternions found in the dataset.")
        return

    # Convert list of quaternions to a numpy array
    q_wxyz_array = np.array(all_quaternions)
    print(f"\nExtracted {len(q_wxyz_array)} quaternions. Shape: {q_wxyz_array.shape}")
    
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
        print(f"Error converting quaternions to Euler angles: {e}")
        return

    # Save results directly to numpy file
    save_path = "results/pose_euler_analysis.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.savez(
        save_path,
        quaternion_wxyz=q_wxyz_array,
        euler_angles_deg=euler_deg,
        euler_angles_rad=euler_rad
    )
    print(f"\nSuccessfully saved pose arrays to [{save_path}]. ")
    print(f" - Contains keys: ['quaternion_wxyz', 'euler_angles_deg', 'euler_angles_rad']")
    
    # Output some basic analysis statistics
    print("\n--- Euler Angles (Degrees) Statistics ---")
    print(f"Mean (X, Y, Z): {euler_deg.mean(axis=0)}")
    print(f"Std  (X, Y, Z): {euler_deg.std(axis=0)}")
    print(f"Min  (X, Y, Z): {euler_deg.min(axis=0)}")
    print(f"Max  (X, Y, Z): {euler_deg.max(axis=0)}")

if __name__ == "__main__":
    main()
