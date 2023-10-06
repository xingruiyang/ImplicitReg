import open3d as o3d
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def registration_features(
        src_points, src_descriptors, 
        tgt_points, tgt_descriptors, 
        distance_threshold):
    # distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_points, tgt_points, src_descriptors, tgt_descriptors, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, 
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result.transformation


def registration_points(
        src_points, tgt_points,
        initial_alignment, distance_threshold):
    result = o3d.pipelines.registration.registration_icp(
        src_points, 
        tgt_points, 
        distance_threshold, 
        initial_alignment,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result.transformation