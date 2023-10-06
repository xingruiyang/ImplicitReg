import argparse
import torch
import open3d as o3d
import os

# os.environ["DISPLAY"] = ":0"


def load_scene_data(scene_file):
    data = torch.load(scene_file)
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(data['voxels'].numpy())

    features = o3d.pipelines.registration.Feature()
    features.data = data['latents'].numpy().transpose()
    return features, points


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(300000, 0.999))
    return result


def draw_registration_result(source, target, transformation):
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    source.transform(transformation)
    o3d.visualization.draw_geometries([
        source, target],
        zoom=0.4559,
        front=[0.6452, -0.3036, -0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.9482, 0.1556])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene1")
    parser.add_argument("scene2")
    args = parser.parse_args()

    desc1, kpts1 = load_scene_data(args.scene1)
    desc2, kpts2 = load_scene_data(args.scene2)
    result_ransac = execute_global_registration(
        kpts1, kpts2, desc1, desc2, voxel_size=0.05)
    print(result_ransac)
    draw_registration_result(kpts1, kpts2, result_ransac.transformation)
