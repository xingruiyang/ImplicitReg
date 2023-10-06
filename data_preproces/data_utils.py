import copy
import glob
import os

import cv2
import natsort
import numpy as np
import open3d as o3d
import torch


def normalize(vec: torch.Tensor):
    if len(vec.shape) < 3:
        vec_norm = torch.norm(vec, p=2)
        return vec / vec_norm if vec_norm != 0 else vec


def cal_z_axis(ref_points: torch.Tensor):
    '''Calculate the z-axis from point clouds
    Args:
        ref_points(Tensor): needs to be centred
    '''
    cov_mat = torch.matmul(ref_points.transpose(-1, -2), ref_points)
    _, eig_vecs = torch.linalg.eigh(cov_mat)
    z_axis = eig_vecs[:, 0]
    mask = (torch.sum(-ref_points * z_axis) < 0).float()
    return z_axis if mask > 0 else -z_axis


def cal_x_axis(ref_pnts, z_axis):
    '''Calculate the x-axis from point clouds
    Args:
        ref_points (tensor): needs to be centred
        z_axis (tensor): the estimated z_axis
    '''
    z_proj = torch.sum(ref_pnts*z_axis, dim=-1, keepdim=True)
    sign_weight = z_proj**2
    sign_weight[z_proj < 0] *= -1

    vec_proj = ref_pnts - z_proj * z_axis
    dist = torch.norm(ref_pnts, p=2, dim=-1, keepdim=True)
    supp = torch.max(dist)
    dist_weight = (supp - dist)**2

    x_axis = dist_weight * sign_weight * vec_proj
    x_axis = torch.sum(x_axis, dim=0)

    return normalize(x_axis)


def cal_xyz_axis(ref_pnts):
    '''Calculate the local reference frame of a point patch
    Args:
        ref_points(Tensor): needs to be centred
    '''
    z_axis = cal_z_axis(ref_pnts)
    x_axis = cal_x_axis(ref_pnts, z_axis)
    y_axis = torch.cross(x_axis, z_axis)
    return torch.stack([x_axis, y_axis, z_axis], dim=0)


def get_voxels(pnts: torch.Tensor,
               voxel_size: float = 0.1,
               method: str = 'uniform',
               voxel_num: int = 1500):
    """Compute voxels from the given point cloud
    Args:
        pnts (tensor): pnts of shape Nx3
        voxels (float, optional): the length of voxels
        method (str, optional): sampling strategy
        voxel_num (int, optional): only used when `method` is set to `random`
    """
    return_nparr = False
    if isinstance(pnts, np.ndarray):
        return_nparr = True
        pnts = torch.FloatTensor(pnts)

    if method == 'uniform':
        voxels = torch.divide(pnts, voxel_size, rounding_mode='floor')
        voxels = torch.unique(voxels, dim=0)
        voxels += .5
        voxels *= voxel_size
        if return_nparr:
            return voxels.numpy()
        else:
            return voxels
    elif method == 'random':
        voxels = torch.randperm(pnts.shape[0])[:voxel_num]
        voxels = pnts[voxels, :]
        return voxels


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


def display_pnts(pts):
    """Visualize pnts
    Args:
        pts (ndarray|tensor): pnts of shape Nx3
    """

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([point_cloud])


def display_pnt_pair(pair):
    """Visualize pnts
    Args:
        pts (ndarray|tensor): pnts of shape Nx3
    """
    pnts1, pnts2 = pair
    if isinstance(pnts1, torch.Tensor):
        pnts1 = pnts1.detach().cpu().numpy()
    if isinstance(pnts2, torch.Tensor):
        pnts2 = pnts2.detach().cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pnts1)
    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(pnts2)
    o3d.visualization.draw_geometries([point_cloud, point_cloud2])


def display_voxels(pts, voxels, voxel_size=0.1):
    """Visualize pnts with corresponding voxels
    Args:
        pts (ndarray|tensor): pnts of shape Nx3
        voxels (ndarray|tensor): voxels of shape Mx3
        voxel_size (float): voxel width in meters
    """
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    geometries = []
    num_voxels = voxels.shape[0]
    for i in range(num_voxels):
        voxel = voxels[i, :]
        bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * .5 * voxel_size + voxel,
            np.ones((3, )) * .5 * voxel_size + voxel
        )
        geometries.append(bbox_inner)
    o3d.visualization.draw_geometries(geometries+[point_cloud])


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def normalized(a, axis=-1, order=2):
    l2 = np.linalg.norm(a, order, axis)
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def depth_to_point_cloud(
        depth: torch.Tensor,
        intr: torch.Tensor,
        return_template: bool = False
):
    '''Compute vertex map from depth images
    Args:
        depth (tensor): depth map of the shape $I\times J\times 1$
        intr (tensor): intrinsic matrix of shape $3\times 3$
        return_template (bool): if a template is returned
    '''
    pcd_template = np.ones((depth.shape[0], depth.shape[1], 3))
    pcd_template[:, :, 0], pcd_template[:, :, 1] = np.meshgrid(
        np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    pcd_template = np.matmul(
        pcd_template, np.transpose(np.linalg.inv(intr)))
    pcd = depth[..., None]*pcd_template
    if return_template:
        return pcd, pcd_template
    else:
        return pcd


def compute_normal_map(pcd, inv_y_axis=False):
    dx = np.zeros_like(pcd)
    dy = np.zeros_like(pcd)
    dx[1:-1, ...] = pcd[2:, :, :] - pcd[:-2, :, :]
    dy[:, 1:-1, :] = pcd[:, 2:, :] - pcd[:, :-2, :]
    ldx = np.linalg.norm(dx, axis=-1)
    ldy = np.linalg.norm(dx, axis=-1)
    if inv_y_axis:
        normal = np.cross(dy, dx, axis=-1)
    else:
        normal = np.cross(dx, dy, axis=-1)
    normal = normalized(normal)
    # visual_ray = normalized(-copy.deepcopy(pcd))
    # ray_normal_angle = np.einsum('ijk, ijk->ij', normal, visual_ray)
    # ray_normal_angle = np.arccos(ray_normal_angle)
    # normal[ray_normal_angle > 90, :] = 0

    normal[-1, :, 2] = 0
    normal[0, :, 2] = 0
    normal[:, -1, 2] = 0
    normal[:, 0, 2] = 0
    normal[np.logical_or(ldx > 0.05, ldy > 0.05)] = 0
    return normal


def load_depth_map(depth_path, subsample=-1, depth_scale=1000.0):
    depth = cv2.imread(depth_path, -1)

    depth = depth / depth_scale
    depth[depth > 5] = 0
    if subsample > 0:
        shape = [s // 2**subsample for s in depth.shape]
        depth = cv2.resize(
            depth, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32)


def get_intrinsics(scene_path, subsample=-1, type='general'):
    '''load intrinsic matrix
    '''
    intrinsics = np.eye(3)
    depth_scale = 1000
    if type == 'general':
        intr = np.loadtxt(os.path.join(scene_path, 'calib/calib.txt'))
        intrinsics = np.eye(3)
        intrinsics[0, 0] = intr[0]
        intrinsics[1, 1] = intr[1]
        intrinsics[0, 2] = intr[2]
        intrinsics[1, 2] = intr[3]
        depth_scale = intr[4]
    elif type == '3dmatch':
        intrinsics = np.loadtxt(os.path.join(
            scene_path, 'camera-intrinsics.txt'))
        depth_scale = 1000
    if subsample > 0:
        intrinsics //= (2**subsample)
        intrinsics[2, 2] = 1
    return intrinsics, depth_scale


def get_depth_files(scene_path, type='general'):
    '''glob depth files
    '''
    if type == 'general':
        depth_files = glob.glob(os.path.join(scene_path, "depth/*.png"))
        depth_files = natsort.natsorted(depth_files)
        return depth_files#[1:]
    elif type == '3dmatch':
        depth_files = glob.glob(os.path.join(scene_path, "seq-01/*.depth.png"))
        depth_files = natsort.natsorted(depth_files)
        return depth_files


def get_pose_files(scene_path, type='general'):
    '''glob pose files
    '''
    if type == '3dmatch':
        pose_files = glob.glob(os.path.join(scene_path, "seq-01/*.pose.txt"))
        pose_files = natsort.natsorted(pose_files)
        return pose_files
    elif type == 'general':
        return os.path.join(scene_path, 'groundtruth.txt')


def transform44(l):
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < np.finfo(float).eps * 4.0:
        return np.array([
            [1.0, 0.0, 0.0, t[0]],
            [0.0, 1.0, 0.0, t[1]],
            [0.0, 0.0, 1.0, t[2]],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2], q[0, 1]-q[2, 3], q[0, 2]+q[1, 3], t[0]),
        (q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2], q[1, 2]-q[0, 3], t[1]),
        (q[0, 2]-q[1, 3], q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)


def read_trajectory(filename, matrix=True, offset=0):
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[float(v.strip()) for v in line.split(" ") if v.strip() != ""]
            for line in lines if len(line) > 0 and line[0] != "#"]
    list_ok = []
    for i, l in enumerate(list):
        if l[4:8] == [0, 0, 0, 0]:
            continue
        isnan = False
        for v in l:
            if np.isnan(v):
                isnan = True
                break
        if isnan:
            print("Warning: line {} of file '{}' has NaNs, skipping line".format(
                i, filename))
            continue
        list_ok.append(l)
    if matrix:
        traj = dict([(int(l[0]+offset), transform44(l[0:]))
                    for l in list_ok])
    else:
        traj = dict([(int(l[0]+offset), l[1:8]) for l in list_ok])
    return traj


def read_poses(pose_files, type='general', transform_poses=False):
    '''Read poses from the file list, 
    return the first frame pose and a frame pose list
    '''
    if type == 'general':
        poses = read_trajectory(pose_files)
        poses = list(poses.values())
        return poses[0], poses
    else:
        num_pose = len(pose_files)
        init_pose = None
        frame_poses = []
        for i in range(num_pose):
            i_pose = np.loadtxt(pose_files[i]).astype(float)
            if init_pose is None:
                init_pose = i_pose
            if transform_poses:
                i_pose = np.matmul(np.linalg.inv(init_pose), i_pose)
            frame_poses.append(i_pose)
        assert(len(frame_poses) == len(pose_files))
        return init_pose, frame_poses


def subsample_pnts(pnts, normals=None, subsample=-1):
    if isinstance(pnts, torch.Tensor):
        pnts = pnts.detach().cpu().numpy()

    if pnts.shape[0] > subsample:
        choices = np.random.permutation(pnts.shape[0])[:subsample]
        pnts = pnts[choices, :]
        if normals is not None:
            normals = normals[choices, :]
    if normals is not None:
        return pnts, normals
    else:
        return pnts


def save_point_cloud_ply(
        filename: str,
        points: np.ndarray,
        normals: np.ndarray = None
) -> None:
    '''Write numpy array to a PLY file
    '''
    with open(filename, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write("element vertex {}\n".format(points.shape[0]).encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if normals is not None:
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")
        f.write(b"end_header\n")

        if normals is not None:
            points = np.concatenate([points, normals], axis=-1)
            f.write(np.ascontiguousarray(points.astype(
                np.float32), dtype='<f4').tobytes())
        else:
            f.write(np.ascontiguousarray(points.astype(
                np.float32), dtype='<f4').tobytes())
