import open3d as o3d
from pathlib import Path
import numpy as np
import json
from glob import glob
from tqdm import tqdm

glob_file = lambda x: sorted(glob(str(x)))

# K = np.array([528.0, 0, 319.5, 0, 528.0, 239.5, 0, 0, 1]).reshape(3, 3)
root_path = Path('/mnt/dataset1/7-Scenes')
output_path = Path('.test-data/7scenes')
output_path.mkdir(exist_ok=True)
split_file = Path(__file__).parent / 'splits/7scenes.json'
splits = json.load(open(split_file))

num_points_train = 10000
num_points_test = 2000
nview_per_fragment = 50

for key, val in tqdm(splits.items()):
    scene_name = key
    scene_out = output_path / scene_name
    train_splits = val['train']
    test_splits = val['test']

    scene_out.mkdir(exist_ok=True)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for seq in train_splits:
        seq_path = root_path / scene_name / f'seq-{seq:02d}'
        imgs = glob_file(seq_path / 'frame-*.color.png')
        depths = glob_file(seq_path / 'frame-*.depth.png')
        poses = glob_file(seq_path / 'frame-*.pose.txt')
        n_imgs = len(imgs)
        for i in tqdm(range(n_imgs), leave=False):
            img = o3d.io.read_image(imgs[i])
            depth = o3d.io.read_image(depths[i])
            pose = np.loadtxt(poses[i])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                img, depth, depth_trunc=3.0, depth_scale=1000, convert_rgb_to_intensity=False)
            volume.integrate(rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                np.linalg.inv(pose))
    
        print("extracting train mesh...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(scene_out / "mesh.ply"), mesh)
        train_points = mesh.sample_points_uniformly(num_points_train)
        o3d.io.write_point_cloud(str(scene_out / 'points.ply'), train_points)

    num_test_views = 0
    test_imgs = []
    test_depths = []
    test_poses = []
    for seq in test_splits:
        seq_path = root_path / scene_name / f'seq-{seq:02d}'
        imgs = glob_file(seq_path / 'frame-*.color.png')
        depths = glob_file(seq_path / 'frame-*.depth.png')
        poses = glob_file(seq_path / 'frame-*.pose.txt')
        n_imgs = len(poses)
        num_test_views += n_imgs
        test_imgs += imgs
        test_depths += depths
        test_poses += poses

    n_frag = num_test_views // nview_per_fragment
    print(f"generating {n_frag} test fragments for scene {scene_name}")

    for i in tqdm(range(n_frag)):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for j in range(nview_per_fragment):
            img = o3d.io.read_image(test_imgs[i * nview_per_fragment + j])
            depth = o3d.io.read_image(test_depths[i * nview_per_fragment + j])
            pose = np.loadtxt(test_poses[i * nview_per_fragment + j])

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                img, depth, depth_trunc=3.0, depth_scale=1000, convert_rgb_to_intensity=False)
            volume.integrate(rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                np.linalg.inv(pose))

        np.savetxt(str(scene_out / f'pose_{i:04d}.txt'), test_poses[i * nview_per_fragment])
        mesh = volume.extract_triangle_mesh()
        mesh.transform(np.linalg.inv(test_poses[i * nview_per_fragment]))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(scene_out / f"mesh_{i:04d}.ply"), mesh)
        test_points = mesh.sample_points_uniformly(num_points_test)
        o3d.io.write_point_cloud(str(scene_out / f'points_{i:04d}.ply'), test_points)
