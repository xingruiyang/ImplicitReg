import json
import os
import shutil
from copy import deepcopy

import cv2
import numpy as np
import open3d as o3d
import torch


def to_o3d(arr: np.ndarray, color: np.ndarray = None):
    """Converts NumPy pnts to open3d format
    Args:
        arr (ndarray|tensor): pnts of shape Nx3
        color (ndarray|tensor, optional): colors of shape Nx3
    """
    if isinstance(arr, torch.Tensor):
        arr = deepcopy(arr).detach().cpu().numpy()
    if isinstance(color, torch.Tensor):
        color = deepcopy(color).detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def draw_voxels(pnts: np.ndarray,
                voxels: np.ndarray,
                voxel_size: float = 0.1):
    """Visualize pnts with a set of voxels
    Args:
        pnts (ndarray|tensor): pnts of shape Nx3
        voxels (ndarray|tensor): voxels of shape Mx3
        voxel_size (float, optional): the length of voxels
    """
    if isinstance(pnts, torch.Tensor):
        pts = pnts.detach().cpu().numpy()
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()
    num_voxels = voxels.shape[0]
    geometries = []
    point_cloud = to_o3d(pts)
    for i in range(num_voxels):
        voxel = voxels[i, :]
        bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * .5 * voxel_size + voxel,
            np.ones((3, )) * .5 * voxel_size + voxel
        )
        geometries.append(bbox_inner)
    o3d.visualization.draw_geometries(geometries+[point_cloud])


def display_sdf(pts, sdf, only_negative=False):
    """Visualize pnts with corresponding sdf values
    Args:
        pts (ndarray|tensor): pnts of shape Nx3
        sdf (ndarray|tensor): sdf values of shape Nx3
        only_negative (bool, optional): only show negative samples
    """

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    if isinstance(pts, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()

    if only_negative:
        pts = pts[sdf < 0, :]
        sdf = sdf[sdf < 0]

    color = np.zeros_like(pts)
    color[sdf > 0, 0] = 1
    color[sdf < 0, 2] = 1
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([point_cloud])


def load_config(config_file):
    return json.load(open(config_file, 'r'))


def make_backups(out_path, in_path=None, cfg_name='main.cfg', flag='train'):
    """Make backups before train/inference
    Args:
        out_path (str): target path
    """
    os.makedirs(out_path, exist_ok=True)

    if flag == 'train':
        shutil.copy('network.py', os.path.join(out_path, 'arch.py'))
        shutil.copy(cfg_name, os.path.join(out_path, 'cfg.json'))
    elif in_path is not None:
        try:
            shutil.copy(os.path.join(in_path, 'arch.py'),
                        os.path.join(out_path, 'arch.py'))
            shutil.copy(os.path.join(in_path, 'cfg.json'),
                        os.path.join(out_path, 'cfg.json'))
        except:
            pass


def compute_gradient(
        y: torch.Tensor,
        x: torch.Tensor,
        grad_outputs: bool = None
):
    """Computing gradients dx of dy
    Args:
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def import_class(name: str):
    '''Recursively imporing a class, using dots as delimiters
    Args:
        name (str): full path (relative) to the class
    '''
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


