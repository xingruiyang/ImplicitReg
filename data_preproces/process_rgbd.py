import argparse
import glob
import json
import os
import pickle

import cv2
import natsort
import numpy as np
import torch

from data_utils import (display_pnts, display_sdf, display_voxels,
                        get_depth_files, get_intrinsics, get_pose_files,
                        get_voxels, read_poses, save_point_cloud_ply,
                        subsample_pnts)
from sampling import sample_rgbd, sample_voxels, save_samples

if __name__ == '__main__':
    '''Sample training/evaluation rgbd data from sequences
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--scene_type', type=str, default='general')
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--num_skip', type=int, default=10)
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=-1)
    parser.add_argument('--subsample', type=int, default=-1)
    parser.add_argument('--save_pnts', action='store_true')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    index_map = dict()

    intrinsics, scale = get_intrinsics(
        args.data_path, args.subsample, args.scene_type)
    depth_files = get_depth_files(args.data_path, args.scene_type)
    pose_filenames = get_pose_files(args.data_path, args.scene_type)
    init_pose, frame_poses = read_poses(pose_filenames, args.scene_type)

    print("total number of {} frames to be sampled".format(len(depth_files)))
    zero_pnts, pnt_normals, sdf_samples = sample_rgbd(
        depth_files,
        frame_poses,
        intrinsics=intrinsics,
        num_skip=args.num_skip,
        frame_range=(args.frame_start, args.frame_end),
        displacements=[0.01, 0.015],
        include_surface_pnts=True,
        use_kd_tree=True,
        depth_scale=scale,
        add_noise=args.add_noise,
        subsample=args.subsample
    )
    display_pnts(zero_pnts)
    # display_sdf(sdf_samples[:, :3], sdf_samples[:, 3], True)

    zero_pnts = torch.FloatTensor(zero_pnts).float()
    sdf_samples = torch.FloatTensor(sdf_samples).float()

    # get voxels
    voxel_size = args.voxel_size
    voxels = get_voxels(zero_pnts, voxel_size, method='uniform')
    # display_voxels(zero_pnts, voxels, voxel_size)

    # voxelize and save
    nsamples = save_samples(
        zero_pnts.cuda(),
        sdf_samples.cuda(),
        voxels.cuda(),
        voxel_size,
        args.out_path,
        norm_type=np.inf,
        subsample=-1,
        sample_free_space=True)

    for i in range(nsamples):
        local_filename = os.path.join('{}.npz'.format(i))
        index_map[i] = local_filename

    pickle.dump(index_map, open(
        os.path.join(args.out_path, 'index.pkl'), 'wb'))
    if args.save_pnts:
        zero_pnts, pnt_normals = subsample_pnts(zero_pnts, pnt_normals, 100000)
        save_point_cloud_ply(os.path.join(
            args.out_path, 'pnts.ply'), zero_pnts, pnt_normals)
        np.save(os.path.join(args.out_path, 'pnts.npy'), zero_pnts)
