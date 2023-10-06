from registration import registration_features, registration_points
import argparse
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
from glob import glob
from evaluate import evaluate_registration

import random
random.seed(10)

glob_file = lambda x: sorted(glob(str(x)))
data_path = Path('.test-data/7scenes')
split_file = Path(__file__).parent / 'splits/7scenes.json'
splits = json.load(open(split_file))
scenes = list(splits.keys())
downsample = None
radius_normal = 0.3

def FPFH(point_cloud, downsample=None, radius_normal=None, radius_feature=0.3, max_nn=100):
    if downsample is not None:
        point_cloud = point_cloud.voxel_down_sample(downsample)

    if radius_normal:
        point_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    descriptors = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))
    return point_cloud, descriptors

for scene in scenes:
    train_points = o3d.io.read_point_cloud(str(data_path / scene / 'points.ply'))
    train_points, train_desc = FPFH(train_points, radius_normal=radius_normal, downsample=downsample)
    list_test_points = glob_file(data_path / scene / 'points_*.ply')
    list_gt_poses = glob_file(data_path / scene / 'pose_*.txt')

    estimate_poses = []
    gt_poses = np.stack([np.loadtxt(gt_pose) for gt_pose in list_gt_poses], 0)

    for test_points, gt_pose in zip(list_test_points, list_gt_poses):
        test_points = o3d.io.read_point_cloud(test_points)
        test_points, test_desc = FPFH(test_points, radius_normal=radius_normal, downsample=downsample)
        result = registration_features(
            test_points, test_desc, train_points, train_desc, distance_threshold=0.2)
        result = registration_points(test_points, train_points, result, distance_threshold=0.2)
        estimate_poses += [result]
    estimate_poses = np.stack(estimate_poses, 0)
    success1 = evaluate_registration(gt_poses, estimate_poses, ang_th=2, pos_th=0.02)
    success2 = evaluate_registration(gt_poses, estimate_poses, ang_th=5, pos_th=0.05)
    success3 = evaluate_registration(gt_poses, estimate_poses, ang_th=10, pos_th=0.1)
    print(f"success rate for {scene} is {np.count_nonzero(success1)/len(success1)}")
    print(f"success rate for {scene} is {np.count_nonzero(success2)/len(success2)}")
    print(f"success rate for {scene} is {np.count_nonzero(success3)/len(success3)}")