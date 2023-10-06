# adapted from https://github.com/zju3dv/manhattan_sdf
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import torch
import os
import pyrender
import cv2
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from glob import glob
import argparse
from PIL import Image

os.environ['PYOPENGL_PLATFORM'] = 'egl'
def globf(x): return natsorted(glob(str(x)))

class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
        

def refuse(mesh, poses, K):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.005,
        sdf_trunc=3 * 0.005,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for pose in tqdm(poses):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

ply_file = "~/Workspace/active/ImplicitReg/scene0050_00/2023-10-01_20-13-17/mesh/00145000.ply"
# ply_file = "~/Workspace/active/experiments/mesh_outputs/monosdf/scannet/scene0050_00/mesh.ply"
mesh = trimesh.load(ply_file)
data_input = Path("~/Workspace/exp_data/recon/scannet/scene0050_00")
pose_paths = globf(data_input / 'pose/*_gt.txt')

poses = np.stack([np.loadtxt(path) for path in pose_paths]).astype(np.float32)
K = np.loadtxt(data_input / 'intrinsics.txt').astype(np.float32)[:3, :3]

img = data_input / 'image/000000.jpg'
H, W = Image.open(img).size[:2]
# H, W = 480, 640

mesh = refuse(mesh, poses, K)

# save mesh
out_mesh_path = "./fuse.ply"
o3d.io.write_triangle_mesh(out_mesh_path, mesh)
mesh = trimesh.load(out_mesh_path)