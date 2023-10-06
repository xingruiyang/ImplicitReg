import argparse
import math
import torch

import marching_cubes
import numpy as np
import open3d as o3d
import trimesh

from network import ImplicitNet

# os.environ["DISPLAY"] = ":0"
# from postprocess import remove_backface

class MeshExtractor(object):
    def __init__(self,  model, voxels, latents, voxel_size,
                 rotations=None, centroids=None, bit=8, interp=False):
        self.model = model
        self.voxels = voxels
        self.latents = latents
        self.voxel_size = voxel_size
        self.rotations = None
        self.centroids = None

        self.resolution = bit
        self.interp = interp
        if interp:
            self.start = -1.
            self.end = 1. - 2./bit
            self.resolution *= 2
        else:
            self.start = -.5
            self.end = .5 - 1./bit
            self.resolution -= 1

        if rotations is not None:
            self.rotations = rotations.to(latents.device).float()
            self.centroids = centroids.to(latents.device).float()

    @ torch.no_grad()
    def get_sdf(self):
        x = y = z = torch.linspace(self.start, self.end, self.resolution)
        xx, yy, zz = torch.meshgrid(x, y, z)
        samples = torch.stack([xx, yy, zz], dim=-1).float()
        samples = samples.reshape(
            1, -1, 3).expand(self.voxels.shape[0], -1, -1)

        if self.centroids is not None:
            samples = samples - self.centroids.unsqueeze(1)
        if self.rotations is not None:
            samples = samples @ self.rotations.transpose(-1, -2)

        latent = self.latents.unsqueeze(1).expand(samples.shape[:-1]+(-1,))
        inputs = torch.cat([samples, latent], dim=-1)
        inputs = inputs.reshape(inputs.shape[0]*inputs.shape[1], -1)

        chunk_size = 10000
        sdf = torch.cat([self.model(inputs[i: i + chunk_size, :].cuda()).squeeze().detach().cpu()
                        for i in range(0, inputs.size(0), chunk_size)], 0)
        sdf_grid = sdf.reshape(-1, self.resolution,
                               self.resolution, self.resolution)
        return sdf_grid

    @ torch.no_grad()
    def linearize_id(self, xyz, n_xyz):
        return xyz[:, 2] + n_xyz[-1] * xyz[:, 1] + (n_xyz[-1] * n_xyz[-2]) * xyz[:, 0]

    @torch.no_grad()
    def create_mesh_overlay(self):
        sdf = self.get_sdf()

        from skimage.measure import marching_cubes
        verts_all = []
        faces_all = []
        num_verts = 0
        for i in range(len(voxels)):
            isdf = sdf[i].detach().cpu().numpy()
            if np.max(isdf) < 0 or np.min(isdf) > 0:
                continue
            results = marching_cubes(isdf, 0)
            if results is not None:
                verts, faces, _, _ = results
                verts /= (self.resolution-1)
                verts -= 0.5
                verts *= self.voxel_size
                verts += voxels[i].numpy()
                faces_all += [faces+num_verts]
                verts_all += [verts]
                num_verts += len(verts)
        verts = np.concatenate(verts_all)
        faces = np.concatenate(faces_all)

        if len(verts) != 0:
            mesh = trimesh.Trimesh(verts, faces).as_open3d
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])
        else:
            print("failed extracting surface!")

    @ torch.no_grad()
    def create_mesh(self, output=None):
        sdf = self.get_sdf()
        voxels = self.voxels / self.voxel_size
        min_bound = torch.floor(torch.amin(voxels, dim=0)).int()
        max_bound = torch.ceil(torch.amax(voxels, dim=0)).int()
        n_xyz = (max_bound - min_bound + 1).numpy().tolist()
        indexer = torch.ones((math.prod(n_xyz),),
                             device=sdf.device, dtype=torch.long) * -1
        vec_id_batch_mapping = torch.arange(voxels.shape[0]).int()
        voxels = torch.floor(voxels) - min_bound

        latent_vecs_pos = self.linearize_id(voxels, n_xyz).long()
        indexer[latent_vecs_pos] = vec_id_batch_mapping.long()
        std = torch.ones_like(sdf)
        indexer = indexer.view(n_xyz)
        sdf = -sdf

        if self.interp:
            verts, _, _ = marching_cubes.marching_cubes_sparse_interp(
                indexer.cuda(), latent_vecs_pos.cuda(), vec_id_batch_mapping.cuda(),
                sdf.cuda(), std.cuda(), n_xyz, 2000.0, int(2e7)
            )
        else:
            verts, _, _ = marching_cubes.marching_cubes(
                indexer.cuda(), latent_vecs_pos.cuda(), vec_id_batch_mapping.cuda(),
                sdf.cuda(), std.cuda(), n_xyz, 2000.0, int(2e7)
            )

        if len(verts) != 0:
            print(verts.shape)
            verts = verts.cpu().numpy()
            bound_min = min_bound.numpy()
            verts = verts + bound_min
            verts *= voxel_size
            verts = verts.reshape(-1, 3)
            faces = np.arange(verts.shape[0]).reshape(-1, 3)
            mesh = trimesh.Trimesh(verts, faces).as_open3d
            mesh.compute_vertex_normals()

            if output is not None:
                o3d.io.write_triangle_mesh(output, mesh)
            else:
                o3d.visualization.draw_geometries([mesh])
        else:
            print("failed extracting surface!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--interp', action='store_true')
    parser.add_argument('--res', type=int, default=8)
    args = parser.parse_args()

    data = torch.load(args.data)
    latents = data['latents']
    voxels = data['voxels']
    ckpt = data['model']
    cfg = data["config"]
    rotations = data['rotations']
    centroids = data['centroids']
    voxel_size = data['voxel_size']

    model = ImplicitNet(**cfg)
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    extractor = MeshExtractor(
        model, voxels, latents,
        voxel_size, rotations,
        centroids, bit=args.res,
        interp=args.interp)
    if args.naive:
        extractor.create_mesh_overlay()
    else:
        extractor.create_mesh(args.output)
    