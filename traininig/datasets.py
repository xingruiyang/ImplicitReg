import os
import pickle

import numpy as np
from torch.utils.data import Dataset

import open3d as o3d
import torch
from glob import glob

def collate_fn(batch):
    data = np.concatenate(batch, axis=0)
    return torch.FloatTensor(data)


def display_sdf(pts, sdf, only_negative=False, sdf_limit=-1):
    """Visualize pnts with corresponding sdf values
    Args:
        pts (ndarray|tensor): pnts of shape Nx3
        sdf (ndarray|tensor): sdf values of shape Nx3
        only_negative (bool, optional): only show negative samples
    """

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()

    if only_negative:
        pts = pts[sdf < 0, :]
        sdf = sdf[sdf < 0]

    if sdf_limit > 0:
        selector = np.abs(sdf) < sdf_limit
        pts = pts[selector, :]
        sdf = sdf[selector]

    color = np.zeros_like(pts)
    color[sdf > 0, 0] = 1
    color[sdf < 0, 2] = 1
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([point_cloud])


class SDFSamples(Dataset):
    def __init__(
        self,
        data_path: str,
        orient: bool = False,
        subsample: int = -1,
        **kwargs
    ) -> None:
        super(SDFSamples, self).__init__(**kwargs)
        self.data_path = data_path
        self.shapes = glob(os.path.join(self.data_path, 'samples/*.npz'))
        self.num_latents = len(self.shapes)
        # index_filename = os.path.join(data_path, 'index.pkl')
        # self.index_map = pickle.load(open(index_filename, 'rb'))
        # self.num_latents = len(self.index_map.keys())
        self.subsample = subsample
        self.orient = orient

    def __len__(self):
        return self.num_latents

    def load_pcd(self, filename):
        data = np.load(filename)
        samples = data['samples']

        if self.subsample > 0:
            sdf = samples[:, 3]
            pos_tensor = samples[sdf > 0, :]
            neg_tensor = samples[sdf < 0, :]
            half = self.subsample // 2

            pos_size = pos_tensor.shape[0]
            neg_size = neg_tensor.shape[0]

            if pos_size <= half:
                random_pos = (np.random.rand(half) *
                              pos_tensor.shape[0]).astype(int)
                sample_pos = pos_tensor[random_pos, :]
            else:
                pos_ind = np.random.permutation(pos_size)[:half]
                sample_pos = pos_tensor[pos_ind, :]

            if neg_size <= half:
                random_neg = (np.random.rand(half) *
                              neg_tensor.shape[0]).astype(int)
                sample_neg = neg_tensor[random_neg, :]
            else:
                neg_ind = np.random.permutation(neg_size)[:half]
                sample_neg = neg_tensor[neg_ind, :]

            samples = np.concatenate([sample_pos, sample_neg], axis=0)

        if self.orient:
            centroid = data['centroid']
            rotation = data['rotation']
            samples[:, :3] -= centroid
            samples[:, :3] = np.matmul(
                samples[:, :3], rotation.transpose(-1, -2))

        return samples

    def __getitem__(self, index):
        samples = self.load_pcd(self.shapes[index])
        indices = np.zeros((samples.shape[0], 1)) + index
        samples[:, :3] += np.random.randn(samples.shape[0], 3) * 0.005
        # display_sdf(samples[:, :3], samples[:, 3])
        return np.concatenate([indices, samples], axis=-1).astype(np.float32)


class SDFSamplesRGBD(Dataset):
    def __init__(
        self,
        data_path: str,
        orient: bool = False,
        subsample: int = -1,
        **kwargs
    ) -> None:
        super(SDFSamplesRGBD, self).__init__(**kwargs)
        self.data_path = data_path
        index_filename = os.path.join(data_path, 'index.pkl')
        self.index_map = pickle.load(open(index_filename, 'rb'))
        self.num_latents = len(self.index_map.keys())
        self.subsample = subsample
        self.orient = orient

    def __len__(self):
        return self.num_latents

    def load_pcd(self, filename):
        data = np.load(os.path.join(self.data_path, filename))
        samples = data['samples']

        if self.subsample > 0:
            sdf = samples[:, 3]
            pos_tensor = samples[sdf > 0, :]
            neg_tensor = samples[sdf < 0, :]
            zero_tensor = samples[sdf == 0, :]
            half = self.subsample // 3

            pos_size = pos_tensor.shape[0]
            neg_size = neg_tensor.shape[0]
            zero_size = zero_tensor.shape[0]

            if zero_size <= half:
                random_pos = (np.random.rand(half) *
                              zero_tensor.shape[0]).astype(int)
                sample_zero = zero_tensor[random_pos, :]
            else:
                zero_ind = np.random.permutation(zero_size)[:half]
                sample_zero = zero_tensor[zero_ind, :]

            if pos_size <= half:
                random_pos = (np.random.rand(half) *
                              pos_tensor.shape[0]).astype(int)
                sample_pos = pos_tensor[random_pos, :]
            else:
                pos_ind = np.random.permutation(pos_size)[:half]
                sample_pos = pos_tensor[pos_ind, :]

            if neg_size <= half:
                random_neg = (np.random.rand(half) *
                              neg_tensor.shape[0]).astype(int)
                sample_neg = neg_tensor[random_neg, :]
            else:
                neg_ind = np.random.permutation(neg_size)[:half]
                sample_neg = neg_tensor[neg_ind, :]

            samples = np.concatenate(
                [sample_zero, sample_pos, sample_neg], axis=0)

        if self.orient:
            centroid = data['centroid']
            rotation = data['rotation']
            samples[:, :3] -= centroid
            samples[:, :3] = np.matmul(
                samples[:, :3], rotation.transpose(-1, -2))

        return samples

    def __getitem__(self, index):
        samples = self.load_pcd(self.index_map[index])
        indices = np.zeros((samples.shape[0], 1)) + index
        # display_sdf(samples[:, :3], samples[:, 3])
        return np.concatenate([indices, samples], axis=-1).astype(np.float32)


class SDFSamplesEpochLoad(Dataset):
    def __init__(
        self,
        data_path: str,
        orient: bool = False,
        subsample: int = -1,
        ** kwargs
    ) -> None:
        super(SDFSamplesEpochLoad, self).__init__(**kwargs)
        self.data_path = data_path
        # index_filename = os.path.join(data_path, 'index.pkl')
        # self.index_map = pickle.load(open(index_filename, 'rb'))
        self.num_latents = len(glob(os.path.join(self.data_path, 'samples/*.npz')))
        # self.num_latents = len(self.index_map.keys())
        self.orient = orient
        self.subsample = subsample
        self.samples = self.load_all()

    def __len__(self):
        return self.samples.shape[0]

    def load_pcd(self, filename):
        data = np.load(os.path.join(self.data_path, filename))
        samples = data['samples']

        if self.subsample > 0:
            sdf = samples[:, 3]
            pos_tensor = samples[sdf > 0, :]
            neg_tensor = samples[sdf < 0, :]
            half = self.subsample // 2

            pos_size = pos_tensor.shape[0]
            neg_size = neg_tensor.shape[0]

            if pos_size <= half:
                random_pos = (np.random.rand(half) *
                              pos_tensor.shape[0]).astype(int)
                sample_pos = pos_tensor[random_pos, :]
            else:
                pos_ind = np.random.permutation(pos_size)[:half]
                sample_pos = pos_tensor[pos_ind, :]

            if neg_size <= half:
                random_neg = (np.random.rand(half) *
                              neg_tensor.shape[0]).astype(int)
                sample_neg = neg_tensor[random_neg, :]
            else:
                neg_ind = np.random.permutation(neg_size)[:half]
                sample_neg = neg_tensor[neg_ind, :]

            samples = np.concatenate([sample_pos, sample_neg], axis=0)

        if self.orient:
            centroid = data['centroid']
            rotation = data['rotation']
            samples[:, :3] -= centroid
            samples[:, :3] = np.matmul(
                samples[:, :3], rotation.transpose(-1, -2))

        return samples

    def load_all(self):
        samples = []
        for i in range(self.num_latents):
            data = self.load_pcd(self.index_map[i])
            indices = np.zeros((data.shape[0], 1)) + i
            samples += [np.concatenate([indices, data], axis=-1)]
        return np.concatenate(samples, axis=0).astype(np.float32)

    def __getitem__(self, index):
        return self.samples[index, ...]
