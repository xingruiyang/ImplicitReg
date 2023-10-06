import numpy as np
import open3d as o3d
import os
os.environ["DISPLAY"] = ":0"

CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
)

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [
        0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)


def create_camera_actor(g, scale=0.05):
    """build open3d camera polydata"""
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES),
    )

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


# mesh = o3d.io.read_triangle_mesh("test.ply")
# mesh.compute_vertex_normals()
train_cam = []
test_cam = []
scene = 'office'
num = 100
for i in range(num):
    pose = np.loadtxt(f"data/{scene}/{i}.txt")
    cam = create_camera_actor(0.9, 0.05)
    cam.transform(pose)
    train_cam += [cam]

for i in range(num):
    pose = np.loadtxt(f"data/{scene}/reg_{i}.txt")
    cam = create_camera_actor(0.1, 0.05)
    cam.transform(pose)
    test_cam += [cam]

o3d.visualization.draw_geometries(train_cam+test_cam)
