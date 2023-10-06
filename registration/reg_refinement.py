import argparse
import copy
import os

import numpy as np
import torch
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
from skimage.measure import marching_cubes
from sklearn.neighbors import NearestNeighbors
from network import ImplicitNet
from reg_two_segments import (draw_registration_result,
                              execute_global_registration, load_scene_data)

# os.environ["DISPLAY"] = ":0"


def gram_schmidt(rots):
    v1 = rots[..., :3]
    v1 = v1 / torch.max(torch.sqrt(torch.sum(v1**2, dim=-1, keepdim=True)),
                        torch.tensor(1e-6, dtype=torch.float32, device=v1.device))
    v2 = rots[..., 3:] - \
        torch.sum(v1 * rots[..., 3:], dim=-1, keepdim=True) * v1
    v2 = v2 / torch.max(torch.sqrt(torch.sum(v2**2, dim=-1, keepdim=True)),
                        torch.tensor(1e-6, dtype=torch.float32, device=v1.device))
    v3 = v1.cross(v2)

    rots = torch.stack([v1, v2, v3], dim=2)

    return rots[0, ...]


def run_icp_refine(
    network: torch.nn.Module,
    dst_voxels: np.ndarray,
    dst_latents: np.ndarray,
    query_pts: np.ndarray,
    voxel_size: float,
    transform: np.ndarray,
    num_iterations: int = 10,
    min_inlier_points: int = 1,
    centroids: np.ndarray = None,
    rotations: np.ndarray = None
):
    device = torch.device('cuda:0')
    network.to(device).eval()

    query_pcd = torch.from_numpy(query_pts).float().to(device)
    train_latents = torch.from_numpy(dst_latents).float().to(device)

    if centroids is not None:
        centroids = torch.from_numpy(centroids).float().to(device)
    if rotations is not None:
        rotations = torch.from_numpy(rotations).float().to(device)

    transform = copy.deepcopy(transform)
    init_r = transform[:3, :3]
    init_t = transform[:3, 3]
    print("before: ", transform)

    best_r = torch.from_numpy(init_r).float().to(device)
    # column-wise flatten
    best_r = best_r[:3, :2].swapaxes(0, 1).reshape(-1, 6)
    best_t = torch.from_numpy(init_t).float().to(device)

    best_r.requires_grad_()
    best_t.requires_grad_()

    train_voxels = dst_voxels/voxel_size - .5
    train_voxels = np.round(train_voxels).astype(int)
    oct_tree = cKDTree(train_voxels)

    optimizer = torch.optim.Adam([best_r, best_t], lr=1e-2)
    pbar = tqdm(range(num_iterations))
    for n_iter in pbar:
        best_r_mat = gram_schmidt(best_r)
        # query_pts = torch.matmul(query_pcd - best_t, best_r_mat)
        query_pts = torch.matmul(
            query_pcd, best_r_mat.transpose(0, 1)) + best_t

        query_np = query_pts.detach().cpu().numpy()
        query_np = query_np // voxel_size
        dist, indices = oct_tree.query(query_np, k=1)
        dist = dist.astype(int)

        indices = indices[dist == 0]
        point_indices = np.where(dist == 0)[0]
        if point_indices.shape[0] < min_inlier_points:
            print("icp failed: no sufficient inlier points {}".format(
                point_indices.shape[0]))
            break

        point_indices = torch.from_numpy(point_indices).int().to(device)
        indices = torch.from_numpy(indices).int().to(device)
        query_np = torch.from_numpy(query_np).float().to(device)
        query_np += 0.5

        points = torch.index_select(
            query_pts/voxel_size-query_np, 0, point_indices)
        latents = torch.index_select(train_latents, 0, indices)
        if centroids is not None and rotations is not None:
            centroid = torch.index_select(centroids, 0, indices)
            rotation = torch.index_select(rotations, 0, indices)
            points = torch.matmul(
                (points-centroid).unsqueeze(1),
                rotation.transpose(1, 2)).squeeze()

        inputs = torch.cat([points, latents], dim=-1)
        sdf_pred = network(inputs).squeeze()

        loss = (sdf_pred.abs()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("inliers: {}/{} loss: {:.2f}".format(
            point_indices.shape[0], query_pts.shape[0], loss.item()))

    best_r = gram_schmidt(best_r)
    best_transform = np.eye(4)
    best_transform[:3, :3] = best_r.detach().cpu().numpy()
    best_transform[:3, 3] = best_t.detach().cpu().numpy()
    return best_transform
# def run_icp_refine(
#     network,
#     train_voxels,
#     train_latents,
#     query_pts,
#     voxel_size,
#     transform,
#     num_iterations=10,
#     min_inlier_points=1,
#     centroids=None,
#     rotations=None
# ):
#     network.cuda().eval()

#     # query_pcd = torch.from_numpy(query_pts).float().cuda()
#     # train_latents = torch.from_numpy(dst_latents).float().cuda()

#     # if centroids is not None:
#     #     centroids = torch.from_numpy(centroids).float().cuda()
#     # if rotations is not None:
#     #     rotations = torch.from_numpy(rotations).float().cuda()
#     transform = copy.deepcopy(transform)
#     init_r = transform[:3, :3]
#     init_t = transform[:3, 3]
#     print("before: ", transform)

#     best_r = torch.from_numpy(init_r).float().cuda()
#     # column-wise flatten
#     best_r = best_r[:3, :2].swapaxes(0, 1).reshape(-1, 6)
#     best_t = torch.from_numpy(init_t).float().cuda()

#     best_r.requires_grad_()
#     best_t.requires_grad_()

#     train_voxels = train_voxels / voxel_size  # + .5
#     train_voxels = torch.round(train_voxels).int()
#     # trimesh.PointCloud(train_voxels.detach().cpu().numpy()).show()
#     # oct_tree = cKDTree(train_voxels.detach().cpu().numpy())
#     oct_tree = cKDTree(train_voxels.detach().cpu().numpy())

#     optim = torch.optim.Adam([best_r, best_t], lr=1e-2)
#     pbar = tqdm(range(num_iterations))
#     for n_iter in pbar:
#         optim.zero_grad()
#         best_r_mat = gram_schmidt(best_r)
#         # query_pts = torch.matmul(query_pcd - best_t, best_r_mat)
#         query_pts = torch.matmul(
#             query_pts, best_r_mat.transpose(0, 1)) + best_t

#         query_np = query_pts.detach().cpu().numpy()
#         query_np = query_np // voxel_size
#         dist, indices = oct_tree.query(query_np, k=1)
#         dist = dist.astype(int)

#         indices = indices[dist == 0]
#         point_indices = np.where(dist == 0)[0]
#         if point_indices.shape[0] < min_inlier_points:
#             print("icp failed: no sufficient inlier points {}".format(
#                 point_indices.shape[0]))
#             break

#         point_indices = torch.from_numpy(point_indices).int().cuda()
#         indices = torch.from_numpy(indices).int().cuda()
#         query_np = torch.from_numpy(query_np).float().cuda()
#         query_np += 0.5

#         points = torch.index_select(
#             query_pts/voxel_size-query_np, 0, point_indices)
#         latents = torch.index_select(train_latents, 0, indices)
#         if centroids is not None and rotations is not None:
#             centroid = torch.index_select(centroids, 0, indices)
#             rotation = torch.index_select(rotations, 0, indices)
#             points = torch.matmul(
#                 (points-centroid).unsqueeze(1),
#                 rotation.transpose(1, 2)).squeeze()

#         inputs = torch.cat([points, latents], dim=-1)
#         sdf_pred = network(inputs).squeeze()

#         loss = (sdf_pred**2).mean()

#         loss.backward(retain_graph=True)
#         # print(best_t.grad)
#         optim.step()

#         pbar.set_description("inliers: {}/{} loss: {:.2f}".format(
#             point_indices.shape[0], query_pts.shape[0], loss.item()))

#     best_r = gram_schmidt(best_r)
#     best_transform = np.eye(4)
#     best_transform[:3, :3] = best_r.detach().cpu().numpy()
#     best_transform[:3, 3] = best_t.detach().cpu().numpy()
#     return best_transform


def load_src_data(data, max_num_pts=8192):
    data = torch.load(data)
    voxels = data['voxels']
    latents = data['latents']
    rotations = data['rotations']
    centroids = data['centroids']
    voxel_size = float(data['voxel_size'])
    # src_pts = data['pts'].squeeze(1)
    # src_pts = src_pts[torch.randperm(src_pts.shape[0])[:max_num_pts], :]
    voxels_o3d = o3d.geometry.PointCloud()
    voxels_o3d.points = o3d.utility.Vector3dVector(voxels.numpy())

    features = o3d.pipelines.registration.Feature()
    features.data = data['latents'].numpy().transpose()

    network = ImplicitNet(**data['config']).cuda()
    network.load_state_dict(data['model'])

    # trimesh.PointCloud(src_pts.detach().cpu().numpy()).show()

    src_pts = create_mesh(network, voxels, latents.cuda(), None,
                          voxel_size, rotations, centroids)

    return features, voxels_o3d, src_pts, voxels, network


def load_dst_data(data):
    data = torch.load(data)
    voxels = data['voxels']
    voxels_o3d = o3d.geometry.PointCloud()
    voxels_o3d.points = o3d.utility.Vector3dVector(voxels.numpy())

    latents = data['latents']
    voxel_size = float(data['voxel_size'])
    features = o3d.pipelines.registration.Feature()
    features.data = latents.numpy().transpose()

    rotations = data['rotations']
    centroids = data['centroids']

    return features, voxels_o3d, voxels, latents, voxel_size, rotations, centroids


@ torch.no_grad()
def create_mesh(model, voxels, latents, pts=None, scale=1,
                rotations=None, centroids=None, bit=8):
    x = y = z = torch.linspace(-.5, .5, bit)
    xx, yy, zz = torch.meshgrid(x, y, z)
    grid = torch.stack([xx, yy, zz], dim=-1).float().cuda()

    sample_pts = grid.reshape(1, -1, 3).expand(voxels.shape[0], -1, -1)
    sample_pts = sample_pts - centroids.unsqueeze(1).cuda().float()
    sample_pts = sample_pts @ rotations.transpose(-1, -2).cuda().float()

    latent = latents.unsqueeze(1).expand(sample_pts.shape[:-1]+(-1,))
    # inputs = torch.cat([latent, sample_pts], dim=-1)
    inputs = torch.cat([sample_pts, latent], dim=-1)
    inputs = inputs.reshape(inputs.shape[0]*inputs.shape[1], -1)

    chunk_size = 10000
    sdf = torch.cat([model(inputs[i: i + chunk_size, :]).squeeze().detach().cpu()
                     for i in range(0, inputs.size(0), chunk_size)], 0)
    sdf_grid = sdf.reshape(-1, bit, bit, bit).numpy()

    vertices = []
    faces = []
    num_verts = 0

    for ind in tqdm(range(voxels.shape[0])):
        sdf = sdf_grid[ind]
        voxel = voxels[ind, :].cpu().numpy()
        if np.min(sdf) > 0 or np.max(sdf) < 0:
            continue
        a = scale / (bit-1)
        spacing = [a, a, a]
        try:
            vert, face, _, _ = marching_cubes(
                sdf, level=0, spacing=spacing, mask=None)
            vert -= 0.5 * scale
            vertices += [vert + voxel]
            faces += [face + num_verts]
            num_verts += vert.shape[0]
        except:
            pass

    faces = np.concatenate(faces, 0)
    vertices = np.concatenate(vertices, 0)

    return trimesh.Trimesh(vertices, faces).sample(8192)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene1")
    parser.add_argument("scene2")
    parser.add_argument("--out", default=None)
    parser.add_argument("--show", action='store_true')
    args = parser.parse_args()
    # torch.autograd.set_detect_anomaly(True)
    desc1, kpts1, src_pts, voxels, network = load_src_data(args.scene1)
    desc2, kpts2, voxels, latents, voxel_size, rotations, centroids = load_dst_data(
        args.scene2)

    result_ransac = execute_global_registration(
        kpts1, kpts2, desc1, desc2, voxel_size=0.05)
    print("coarse registration:\n", result_ransac)
    transform = result_ransac.transformation

    # transform = run_icp_refine(
    #     network,
    #     voxels.cuda(),
    #     latents.cuda(),
    #     src_pts,
    #     voxel_size,
    #     coarse_transform,
    #     num_iterations=10,
    #     min_inlier_points=1,
    #     centroids=None,
    #     rotations=None
    # )

    transform = run_icp_refine(
        network,
        voxels.detach().cpu().numpy(),
        latents.detach().cpu().numpy(),
        src_pts,
        voxel_size,
        transform,
        num_iterations=200,
        min_inlier_points=1,
        centroids=centroids.detach().cpu().numpy(),
        rotations=rotations.detach().cpu().numpy()
    )

    if args.out is not None:
        np.savetxt(args.out, transform)

    if args.show:
        os.environ["DISPLAY"] = ":0"
        draw_registration_result(kpts1, kpts2, transform)
