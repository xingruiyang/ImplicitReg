import os
import sys
import time
import numpy as np
import torch
import json
import os
from pathlib import Path
import importlib
import open3d as o3d
from glob import glob
from tqdm import tqdm
from evaluate import evaluate_registration
from registration import registration_features, registration_points
glob_file = lambda x: sorted(glob(str(x)))

def _ppf(point1, normal1, point2, normal2):
    d = point1 - point2  # [1024, 3]
    len_d = np.sqrt(np.diag(np.dot(d, d.transpose()))) / 0.3  # [1024, 1]
    # element wise multiply https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html
    y = np.sum(np.multiply(normal1, d), axis=1)
    x = np.linalg.norm(np.cross(normal1, d), axis=1)
    dim1_ = np.arctan2(x, y) / np.pi
    # dim1 = np.arccos(np.sum(np.multiply(normal1, d), axis=1) / len_d) / np.pi  # [1024, 1]
    y = np.sum(np.multiply(normal2, d), axis=1)
    x = np.linalg.norm(np.cross(normal2, d), axis=1)
    dim2_ = np.arctan2(x, y) / np.pi
    # dim2 = np.arccos(np.sum(np.multiply(normal2, d), axis=1) / len_d) / np.pi  # [1024, 1]
    y = np.sum(np.multiply(normal1, normal2), axis=1)
    x = np.linalg.norm(np.cross(normal1, normal2), axis=1)
    dim3_ = np.arctan2(x, y) / np.pi
    # dim3 = np.arccos(np.clip(np.sum(np.multiply(normal1, normal2), axis=1), a_min=-1, a_max=1)) / np.pi
    return np.array([dim1_, dim2_, dim3_, len_d]).transpose()

def build_ppf_input(pcd, keypts):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    keypts_id = []
    for i in range(keypts.shape[0]):
        _, id, _ = kdtree.search_knn_vector_3d(keypts[i], 1)
        keypts_id.append(id[0])
    neighbor, keypts_id = collect_local_neighbor(keypts_id, pcd, vicinity=0.3, num_points=1024)
    local_pachtes = build_local_patch(keypts_id, pcd, neighbor).astype(np.float32)
    return keypts[keypts_id], local_pachtes


def collect_local_neighbor(ids, pcd, vicinity=0.3, num_points=1024):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    res = []
    cleaned_id = []
    for id in ids:
        [k, idx, variant] = kdtree.search_radius_vector_3d(pcd.points[id], vicinity)
        if k == 1:
            continue
        if k > num_points:
            idx = np.random.choice(idx[1:], num_points, replace=False)
        else:
            idx = np.random.choice(idx[1:], num_points)
        res.append(idx)
        cleaned_id.append(id)
    return np.array(res), np.array(cleaned_id)


def build_local_patch(ref_inds, pcd, neighbor):
    pcd.estimate_normals()
    num_patches = len(ref_inds)
    num_points_per_patch = len(neighbor[0])
    # shape: num_ref_point, num_point_per_patch, 4
    local_patch = np.zeros([num_patches, num_points_per_patch, 4], dtype=float)
    for i, ref_ind, inds in zip(range(num_patches), ref_inds, neighbor):
        ppfs = _ppf(pcd.points[ref_ind], pcd.normals[ref_ind], np.asarray(pcd.points)[inds],
                    np.asarray(pcd.normals)[inds])
        local_patch[i] = ppfs
    return local_patch


def generate_descriptor(model, local_patches):
    batch = len(local_patches)
    input_ = torch.tensor(local_patches)
    input_ = input_.cuda()
    model = model.cuda()
        # cuda out of memry
    desc_list = []
    for batch_input in torch.split(input_, 50):
        # step_size = int(batch / 50)
        desc = model.encoder(batch_input)
        desc_list.append(desc.detach().cpu().numpy())
        del desc
    desc = np.concatenate(desc_list, 0).reshape([batch, 512])
    return desc

def to_o3d_points(inputs):
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(inputs)
    # features.data = inputs.transpose(0, 1)
    return points

def to_o3d_feature(inputs):
    features = o3d.pipelines.registration.Feature()
    features.data = inputs.transpose(1, 0)
    return features


if __name__ == '__main__':
    pretrained_model = '/home/xingrui/Workspace/third_party/3d/PPF-FoldNet/pretrained/sun3d_best.pkl'
    ppf_path = Path('/home/xingrui/Workspace/third_party/3d/PPF-FoldNet')

    data_path = Path(__file__).parent.parent / '.test-data/7scenes'
    split_file = Path(__file__).parent / 'splits/7scenes.json'
    # desc_out_path = Path(__file__).parent.parent/ '.output/desc/ppf'
    # desc_out_path.mkdir(exist_ok=True)
    split = json.load(open(split_file))
    scenes = list(split.keys())

    module_file_path = ppf_path / 'models/model_conv1d.py'
    module_name = 'models'
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    model = module.PPFFoldNet(50, 1024)
    model.load_state_dict(torch.load(pretrained_model))

    for scene in scenes:
        train_points = o3d.io.read_point_cloud(str(data_path / scene / 'points.ply'))
        key_points_train = np.asarray(train_points.points)
        key_points_train, local_patches = build_ppf_input(train_points, key_points_train)
      
        train_desc = generate_descriptor(model, local_patches)

        list_test_points = glob_file(data_path / scene / 'points_*.ply')
        list_gt_poses = glob_file(data_path / scene / 'pose_*.txt')
        estimate_poses = []
        gt_poses = np.stack([np.loadtxt(gt_pose) for gt_pose in list_gt_poses], 0)

        for test_points, gt_pose in tqdm(zip(list_test_points, list_gt_poses)):
            test_points = o3d.io.read_point_cloud(test_points)
            key_points_test = np.asarray(test_points.points)
            key_points_test, local_patches = build_ppf_input(test_points, key_points_test)
            test_desc = generate_descriptor(model, local_patches)

            result = registration_features(
                to_o3d_points(key_points_test), to_o3d_feature(test_desc), 
                to_o3d_points(key_points_train), to_o3d_feature(train_desc), 
                distance_threshold=0.2)
            result = registration_points(
                test_points, train_points, 
                result, distance_threshold=0.2)
            estimate_poses += [result]

        estimate_poses = np.stack(estimate_poses, 0)
        success1 = evaluate_registration(gt_poses, estimate_poses, ang_th=2, pos_th=0.02)
        success2 = evaluate_registration(gt_poses, estimate_poses, ang_th=5, pos_th=0.05)
        success3 = evaluate_registration(gt_poses, estimate_poses, ang_th=10, pos_th=0.1)
        print(f"success rate for {scene} is {np.count_nonzero(success1)/len(success1)}")
        print(f"success rate for {scene} is {np.count_nonzero(success2)/len(success2)}")
        print(f"success rate for {scene} is {np.count_nonzero(success3)/len(success3)}")