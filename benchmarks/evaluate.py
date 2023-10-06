import numpy as np
from scipy.spatial.transform import Rotation

def compute_angle(transform):
    # an invitation to 3-d vision, p 27
    return np.arccos(((np.trace(transform[:, :3, :3], axis1=1, axis2=2) - 1)/2).clip(-1, 1))

def evaluate_registration(src_poses, tgt_poses, ang_th=5, pos_th=0.05):
    assert len(src_poses) == len(tgt_poses)

    relative_pose = src_poses @ np.linalg.inv(tgt_poses)
    # relative_pose = src_poses @ tgt_poses
    rotation = Rotation.from_matrix(relative_pose[:, :3, :3]).as_euler('xyz', True)
    translation = relative_pose[:, :3, 3]

    angle_diff = compute_angle(relative_pose[:, :3, :3]) * 180 / np.pi
    trans_diff = np.linalg.norm(translation, 2, axis=-1)
    success = (angle_diff < ang_th) & (trans_diff < pos_th)
    return success
