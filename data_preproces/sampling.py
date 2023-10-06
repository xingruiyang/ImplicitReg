import os
import random

import numpy as np
import torch
from scipy.spatial import KDTree
from tqdm import tqdm

try:
    import data_utils as utils
except:
    import data.data_utils as utils


def sample_uniform_sphere(num_samples: int):
    '''Sample uniformly in [-range,range] bounding volume within voxels
    Args:
        centres (tensor)  : set of centres to sample from
        num_samples (int) : number of points to sample
        range (float): range to sample points
    '''
    samples = torch.rand(num_samples, 3) * 2.0 - 1
    samples = samples[torch.norm(samples, p=2, dim=-1) < 1, :]
    samples = samples * 1.1
    return samples.reshape(-1, 3)


def sample_voxels(centres: torch.Tensor, num_samples: int, range: float):
    '''Sample uniformly in [-range,range] bounding volume within voxels
    Args:
        centres (tensor)  : set of centres to sample from
        num_samples (int) : number of points to sample
        range (float): range to sample points
    '''
    samples = torch.rand(
        centres.shape[0], num_samples, 3, device=centres.device)
    samples = samples * 2.0 * range - range
    samples = centres[..., None, :3] + samples
    return samples.reshape(-1, 3)


def get_sample(
        surface: torch.Tensor,
        samples: torch.Tensor,
        voxel: torch.Tensor,
        voxel_size: float):
    '''Sample pnts/sdf within a given voxel
    Args:
        surface (tensor)  : on surface points
        samples (tensor) : near surface samples
        voxel (tensor): voxel centre in meters
        voxel_size (float): voxel size
    '''
    rotation = torch.eye(3)
    centroid = torch.zeros((3,))

    pnts = samples[:, :3] - voxel
    radius = 1.5 * voxel_size
    selector = torch.norm(pnts, p=np.inf, dim=-1)
    selector = selector < radius
    pnts = pnts[selector, :] / radius
    sdf = samples[selector, 3] / radius

    pos_sample_size = torch.count_nonzero(sdf > 0)
    neg_sample_size = torch.count_nonzero(sdf < 0)
    if pos_sample_size == 0 or neg_sample_size == 0:
        return None

    sample = torch.cat([pnts, sdf[:, None]], dim=-1)
    return sample, radius, rotation, centroid


def subsample_equal(samples: torch.Tensor, upper_limit: int):
    '''Subsample the training data. Inspired by DeepSDF
    Args:
        samples (tensor) : sdf samples to subsample
        upper_limit (int) : number of points upper limit
    '''
    half = upper_limit // 2
    pos_tensor = samples[samples[:, 3] > 0, :]
    neg_tensor = samples[samples[:, 3] < 0, :]

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    if pos_size >= half:
        pos_start_ind = random.randint(0, pos_size - half)
        pos_tensor = pos_tensor[pos_start_ind: (pos_start_ind + half)]
    if neg_size >= half:
        neg_start_ind = random.randint(0, neg_size - half)
        neg_tensor = neg_tensor[neg_start_ind: (neg_start_ind + half)]

    return torch.cat([pos_tensor, neg_tensor], dim=0)


def get_sample_oriented(
        surface: torch.Tensor,
        sdf_samples: torch.Tensor,
        voxel: torch.Tensor,
        voxel_size: float,
        norm_type: int = np.inf):
    '''Sample oriented pnts/sdf within a given voxel 
    Args:
        surface (tensor)  : on surface points
        samples (tensor) : near surface samples
        voxel (tensor): voxel centre in meters
    '''
    scale = 1.5
    radius = 1.5*voxel_size
    rotation = torch.eye(3)
    centroid = torch.zeros((3,))

    surface_pnts = surface - voxel
    selector = torch.norm(surface_pnts, p=2, dim=-1)
    selector = selector < scale*voxel_size

    num_selected = torch.count_nonzero(selector)
    if num_selected < 30:
        return None

    surface_pnts = surface_pnts[selector, :] / radius
    centroid = torch.mean(surface_pnts, dim=0)
    ref_pnts = surface_pnts - centroid

    if surface_pnts.shape[0] > 30:
        rotation = utils.cal_xyz_axis(ref_pnts)

    pnts = sdf_samples[:, :3] - voxel
    selector = torch.norm(pnts, p=norm_type, dim=-1)
    selector = selector < radius
    pnts = pnts[selector, :] / radius
    sdf = sdf_samples[selector, 3:4] / radius
    weight = sdf_samples[selector, 4:]

    pos_sample_size = torch.count_nonzero(sdf > 0)
    neg_sample_size = torch.count_nonzero(sdf < 0)
    if pos_sample_size == 0 or neg_sample_size == 0:
        return None

    # utils.display_sdf(pnts, sdf)
    sample = torch.cat([pnts, sdf, weight], dim=-1)
    return surface_pnts, sample, radius, rotation, centroid


def get_sample_adaptive(
        surface: torch.Tensor,
        samples: torch.Tensor,
        voxel: torch.Tensor,
        voxel_size: float):
    '''Sample pnts/sdf within a given voxel using adaptive radius
    Args:
        surface (tensor) : on surface points
        samples (tensor) : near surface samples
        voxel (tensor) : voxel centre in meters
    '''
    scale = 1.5
    rotation = torch.eye(3)
    centroid = torch.zeros((3,))

    surface_pnts = surface - voxel
    selector = torch.norm(surface_pnts, p=2, dim=-1)
    selector = selector < scale*voxel_size
    surface_pnts = surface_pnts[selector, :]

    centroid = torch.mean(surface_pnts, dim=0)
    ref_pnts = surface_pnts - centroid

    if surface_pnts.shape[0] > 30:
        rotation = utils.cal_xyz_axis(ref_pnts)

    ref_pnts = torch.matmul(ref_pnts, rotation.transpose(-1, -2))
    radius = torch.max(torch.norm(ref_pnts, p=2, dim=-1)).item()
    radius = max(radius, scale*voxel_size)
    new_centre = voxel + centroid

    pnts = samples[:, :3] - new_centre
    selector = torch.norm(pnts, p=2, dim=-1)
    selector = selector < radius
    pnts = pnts[selector, :] / radius + centroid
    sdf = samples[selector, 3] / radius

    pos_sample_size = torch.count_nonzero(sdf > 0)
    neg_sample_size = torch.count_nonzero(sdf < 0)
    if pos_sample_size == 0 or neg_sample_size == 0:
        return None

    sample = torch.cat([pnts, sdf[:, None]], dim=-1)
    return surface_pnts, sample, radius, rotation, centroid


def save_samples(surface: torch.Tensor,
                 sdf_samples: torch.Tensor,
                 total_voxels: torch.Tensor,
                 voxel_size: float,
                 out_path: str,
                 num_sampled: int = 0,
                 subsample: int = -1,
                 save_pnts: bool = False,
                 norm_type=np.inf,
                 sample_free_space: bool = False,
                 **kwargs):
    '''Save voxel samples from pnts-sdf pairs
    Args:
        surface (tensor) : on surface points
        samples (tensor) : near surface samples
        total_voxels (tensor) : voxel centres
        voxel_size (float) : voxel width
        out_path (str) : output path
        num_sampled (int) : number already sampled voxels
        upper_limit (int) : max number points one voxel can have
    '''
    centroids = []
    rotations = []
    voxel_radius = []

    num_voxels = total_voxels.shape[0]
    num_samples = 0
    voxels = []
    pbar = tqdm(range(num_voxels), leave=False)
    pbar.set_description("Saving voxels...")
    for i in pbar:
        voxel = total_voxels[i, :]
        voxel_samples = get_sample_oriented(
            surface, sdf_samples, voxel, voxel_size, norm_type)
        if voxel_samples is None:
            continue

        pnts, sample, radius, rotation, centroid = voxel_samples
        num_pnts = sample.shape[0]
        if subsample > 0 and num_pnts > subsample:
            sample = subsample_equal(sample, subsample)

        if sample_free_space:
            random_pnts = torch.rand((32, 3), device=pnts.device)*2-1
            sdf = torch.zeros((32, 1), device=pnts.device)
            weight = torch.zeros((32, 1), device=pnts.device)
            random_samples = torch.cat([random_pnts, sdf, weight], dim=-1)
            sample = torch.cat([sample, random_samples], dim=0)

        filename = '{}.npz'.format(num_samples)
        voxel = voxel.detach().cpu().numpy()
        sample = sample.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        centroid = centroid.detach().cpu().numpy()
        pnts = pnts.detach().cpu().numpy()

        np.savez(
            os.path.join(out_path, filename),
            pnts=pnts if save_pnts else None,
            samples=sample,
            rotation=rotation,
            centroid=centroid
        )

        voxels.append(voxel)
        voxel_radius.append(radius)
        centroids.append(centroid)
        rotations.append(rotation)
        num_samples += 1

    """
    Meta data:
        --voxels
        --voxel size
        --rotations
        --centroids
        --adaptive radius (if any)
        --latent_vec range
        --save surface points?
    """
    voxels = np.stack(voxels, axis=0)
    rotations = np.stack(rotations, axis=0)
    centroids = np.stack(centroids, axis=0)
    voxel_radius = np.asarray(voxel_radius)[:, None]
    voxels = np.concatenate([voxels, voxel_radius], axis=-1)
    # np.savez(
    #     os.path.join(out_path, 'meta.npz'),
    #     rotations=rotations,
    #     centroids=centroids,
    #     voxels=voxels,
    #     voxel_size=voxel_size,
    #     latent_vecs=(num_sampled, num_sampled+num_samples),
    #     **kwargs
    # )

    return num_samples


def voxelize_samples(surface: torch.Tensor,
                     samples: torch.Tensor,
                     total_voxels: torch.Tensor,
                     voxel_size: float,
                     out_path: str,
                     num_sampled: int = 0,
                     upper_limit: int = -1):
    '''Save voxel samples from pnts-sdf pairs
    Args:
        surface (tensor) : on surface points
        samples (tensor) : near surface samples
        total_voxels (tensor) : voxel centres
        voxel_size (float) : voxel width
        out_path (str) : output path
        num_sampled (int) : number already sampled voxels
        upper_limit (int) : max number points one voxel can have
    '''
    centroids = []
    rotations = []
    voxel_radius = []

    num_voxels = total_voxels.shape[0]
    num_samples = 0
    voxels = []
    all_samples = []
    for i in range(num_voxels):
        voxel = total_voxels[i, :]
        voxel_samples = get_sample_oriented(
            surface, samples, voxel, voxel_size)
        if voxel_samples is None:
            continue

        sample, radius, rotation, centroid = voxel_samples
        num_pnts = sample.shape[0]
        if upper_limit > 0 and num_pnts > upper_limit:
            sample = subsample_equal(sample, upper_limit)

        indices = torch.zeros(
            (sample.shape[0], 1), device=sample.device)+num_samples+i
        sample = torch.cat([indices, sample], dim=-1)

        voxel = voxel.detach().cpu().numpy()
        sample = sample.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        centroid = centroid.detach().cpu().numpy()

        voxels.append(voxel)
        all_samples.append(sample)
        voxel_radius.append(radius)
        centroids.append(centroid)
        rotations.append(rotation)
        num_samples += 1

    """
    Meta data:
        --voxels
        --voxel size
        --rotations
        --centroids
        --adaptive radius (if any)
        --latent_vec range
        --save surface points?
    """
    voxels = np.stack(voxels, axis=0)
    all_samples = np.concatenate(all_samples, axis=0)
    rotations = np.stack(rotations, axis=0)
    centroids = np.stack(centroids, axis=0)
    voxel_radius = np.asarray(voxel_radius)[:, None]
    voxels = np.concatenate([voxels, voxel_radius], axis=-1)

    return {
        'voxels': voxels,
        'samples': all_samples,
        'rotations': rotations,
        'centroids': centroids,
        'voxel_size': voxel_size,
    }


def sample_pnts_with_normal(
        depth_files: list,
        frame_pose: list,
        intrinsics: np.ndarray,
        num_skip: int = 10,
        depth_scale: float = 1000,
):
    '''sample pnts from input RGB-D frames
    Args:
        depth_files: (list) list of depth files
        frame_pose: (list) list of pose of the shape $4\times 4$
        intrinsics: (np.ndarray) camera intrinsics of the shape $3\times 3$
        num_skip: (int) skip after every sampled frame
    '''
    zero_pnts = []
    pnt_normals = []
    pbar = tqdm(range(0, len(depth_files), num_skip))
    pbar.set_description("Extracting RGB-D frames...")
    for i in pbar:
        filepath = depth_files[i]
        if isinstance(frame_pose, dict):
            pose = frame_pose[list(frame_pose.keys())[i]]
        else:
            pose = frame_pose[i]

        # load data
        depth = utils.load_depth_map(filepath, depth_scale)
        pcd = utils.depth_to_point_cloud(depth, intrinsics)
        normal = utils.compute_normal_map(
            pcd, inv_y_axis=True if intrinsics[1, 1] < 0 else False)

        normal = normal.reshape(-1, 3)
        depth = depth.reshape(-1, 1)
        pcd = pcd.reshape(-1, 3)

        # only retain data that contains valid normal & depth
        nonzeros = np.logical_and(depth[:, 0] != 0, normal[:, 2] != 0)
        depth = depth[nonzeros, :]
        pcd = pcd[nonzeros, :]
        normal = normal[nonzeros, :]

        normal = np.matmul(
            normal, pose[:3, :3].transpose())
        pcd = np.matmul(
            pcd, pose[:3, :3].transpose()) + pose[:3, 3]

        zero_pnts.append(pcd)
        pnt_normals.append(normal)

    zero_pnts = np.concatenate(zero_pnts, axis=0)
    pnt_normals = np.concatenate(pnt_normals, axis=0)
    return zero_pnts, pnt_normals


def get_empty_space_samples(
    pnts, normals, use_kd_tree
):
    N,  _ = pnts.shape
    dist = (1 - np.random.rand(N, 1) * 0.4)
    pnt_sample = pnts * dist
    if use_kd_tree:
        ref_pnts = utils.subsample_pnts(pnts, subsample=2**15)
        kd_tree = KDTree(ref_pnts)
        dist, _ = kd_tree.query(pnt_sample)
        sdf = dist
    else:
        sdf = np.sum((pnts * (dist-1))*normals, axis=-1)
    return pnt_sample, sdf


def sample_rgbd(
        depth_files: list,
        frame_pose: list,
        intrinsics: np.ndarray,
        num_skip: int = 10,
        frame_range: tuple = (0, -1),
        displacements: list = [0.01, 0.015, 0.02],
        include_surface_pnts: bool = False,
        use_kd_tree: bool = True,
        depth_scale: float = 1000,
        add_noise: bool = True,
        subsample: int = -1
):
    '''sample sdf-pnts from input RGB-D frames
    Args:
        depth_files: (list) list of depth files
        frame_pose: (list) list of pose of the shape $4\times 4$
        intrinsics: (np.ndarray) camera intrinsics of the shape $3\times 3$
        num_skip: (int) skip after every sampled frame
        displacements: (list) list of sdf displacements
        sample_empty_space: (bool) also sample camera frustum
        use_kd_tree: (bool) use kd tree to calculate sdf for free space samples
    '''
    zero_pnts = []
    pnt_normals = []
    sdf_samples = []
    frame_start = frame_range[0]
    frame_end = len(depth_files) if frame_range[1] < 0 else frame_range[1]
    pbar = tqdm(range(frame_start, frame_end, num_skip))
    # pbar = tqdm(range(0, 100, num_skip))
    pbar.set_description("Extracting RGB-D frames...")
    for i in pbar:
        filepath = depth_files[i]
        pose = frame_pose[i]
        # if isinstance(frame_pose, dict):
        #     pose = frame_pose[list(frame_pose.keys())[i]]
        # else:
        #     pose = frame_pose[i]
        # continue

        # load data
        depth = utils.load_depth_map(filepath, subsample, depth_scale)
        pcd = utils.depth_to_point_cloud(depth, intrinsics)
        normal = utils.compute_normal_map(
            pcd, inv_y_axis=True if intrinsics[1, 1] < 0 else False)

        normal = normal.reshape(-1, 3)
        depth = depth.reshape(-1, 1)
        pcd = pcd.reshape(-1, 3)

        # only retain data that contains valid normal & depth
        nonzeros = np.logical_and(depth[:, 0] != 0, normal[:, 2] != 0)
        depth = depth[nonzeros, :]
        pcd = pcd[nonzeros, :]
        normal = normal[nonzeros, :]

        # take random samples along the visual ray
        pnt_sample, sdf_sample = get_empty_space_samples(
            pcd, normal, use_kd_tree)

        normal = np.matmul(
            normal, pose[:3, :3].transpose())
        pcd = np.matmul(
            pcd, pose[:3, :3].transpose()) + pose[:3, 3]
        pnt_sample = np.matmul(
            pnt_sample, pose[:3, :3].transpose()) + pose[:3, 3]

        # nonzeros = sdf_sample > 0
        # pcd = pcd[nonzeros, :]
        # normal = normal[nonzeros, :]
        # sdf_sample = sdf_sample[nonzeros, None]
        # pnt_sample = pnt_sample[nonzeros, :]

        samples = [pnt_sample]
        sdf = [sdf_sample[:, None]]
        weights = [1.0 / depth]

        for disp in displacements:
            disp = np.random.randn(pcd.shape[0], 1) * disp
            samples.append(pcd + normal * disp)
            # samples.append(pcd - normal * disp)
            sdf.append(np.zeros((pcd.shape[0], 1)) + disp)
            # sdf.append(np.zeros((pcd.shape[0], 1)) - disp)
            weights.append(1.0 / depth)
            # weights.append(1.0 / depth)

        if include_surface_pnts:
            samples.append(pcd)
            sdf.append(np.zeros((pcd.shape[0], 1)))
            weights.append(1.0 / depth)

        zero_pnts.append(pcd)
        pnt_normals.append(normal)
        samples = np.concatenate(samples, axis=0)
        sdf = np.concatenate(sdf, axis=0)
        weights = np.concatenate(weights, axis=0)
        sdf_samples.append(np.concatenate([samples, sdf, weights], axis=-1))

    zero_pnts = np.concatenate(zero_pnts, axis=0)
    pnt_normals = np.concatenate(pnt_normals, axis=0)
    sdf_samples = np.concatenate(sdf_samples, axis=0)
    return zero_pnts, pnt_normals, sdf_samples
