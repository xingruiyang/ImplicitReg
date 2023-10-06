import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import trimesh
from data_utils import display_sdf, get_voxels
from mesh_to_sdf import sample_sdf_near_surface
from scipy.spatial.transform import Rotation
from torchgp import compute_sdf, load_obj, sample_near_surface, sample_surface
from tqdm import tqdm


def sample_voxels(centres: torch.Tensor, num_samples: int, range: float):
    samples = torch.rand(
        centres.shape[0], num_samples, 3, device=centres.device)
    samples = samples * 2.0 * range - range
    samples = centres[..., None, :3] + samples
    return samples.reshape(-1, 3)


def get_voxel_samples(
        surface, samples, voxels, voxel_size,
        min_surface_points=256, min_sample_points=2048):
    all_samples = []
    all_surface = []
    for voxel in voxels:
        surface_pnts = surface - voxel
        sample_pnts = samples[:, :3] - voxel
        voxel_surface = surface_pnts[torch.norm(
            surface_pnts, p=np.inf, dim=-1) <= voxel_size]
        voxel_samples = samples[torch.norm(
            sample_pnts, p=np.inf, dim=-1) <= voxel_size]
        num_pos = len(voxel_samples[voxel_samples[:, 3] >= 0])
        num_neg = len(voxel_samples[voxel_samples[:, 3] < 0])
        if len(voxel_surface) >= min_surface_points and \
            len(voxel_samples) >= min_sample_points and \
                num_pos >= 128 and num_neg >= 128:
            voxel_surface = voxel_surface / voxel_size
            voxel_samples = voxel_samples / voxel_size
            all_samples += [voxel_samples]
            all_surface += [voxel_surface]
    return all_surface, all_samples


@torch.no_grad()
def main(args):
    # total_object_samples = 0
    total_shape_generated = 0

    IN_PATH = Path(args.data_path)
    OUT_PATH = Path(args.out_path)
    OUT_PATH.mkdir(exist_ok=True)

    SAMPLE_OUT_PATH = OUT_PATH / 'samples'
    SAMPLE_OUT_PATH.mkdir(exist_ok=True)

    splits = json.load(open(args.split_file, 'r'))
    class_progbar = tqdm(splits.items())
    object_index_file = []
    for object_class, object_list in class_progbar:
        class_progbar.set_description(
            f"processing object class {object_class}")
        object_list = list(object_list)
        if args.shuffle:
            random.shuffle(object_list)

        num_models_sampled = 0
        object_progbar = tqdm(total=args.objects_per_class, leave=False)
        for object_name in object_list:
            object_progbar.set_description(f"processing object {object_name}")
            object_path = IN_PATH / object_class / \
                object_name / 'models/model_normalized.obj'

            verts, faces = load_obj(str(object_path))
            rotation = torch.eye(3)
            if args.rotate:
                rotation = torch.FloatTensor(Rotation.random().as_matrix())
                verts = torch.matmul(verts, rotation.transpose(-1, -2))

            # get voxels
            voxel_size = (torch.max(torch.norm(
                verts, p=2, dim=-1)) / 16).item() if args.voxel_size < 0 else args.voxel_size
            surface_pnts, _ = sample_surface(verts, faces, 200000)
            voxels = get_voxels(surface_pnts, voxel_size, method='uniform')

            if args.method == 0:
                # use nglod
                surface_samples = torch.cat([
                    sample_near_surface(verts, faces, 250000, 0.005),
                    sample_near_surface(verts, faces, 250000, 0.0005),
                    sample_voxels(voxels, 32, 2 * voxel_size)
                ], dim=0)
                # calculate sdf
                sdf = compute_sdf(
                    verts.cuda(), faces.cuda(), surface_samples.cuda())
            elif args.method == 1:
                # use mesh_to_sdf
                mesh = trimesh.Trimesh(verts, faces)
                surface_samples, sdf = sample_sdf_near_surface(
                    mesh, voxels, voxel_size)
                surface_samples = torch.FloatTensor(surface_samples)
                sdf = torch.FloatTensor(sdf)
            else:
                # combine nglod and mesh_to_sdf
                mesh = trimesh.Trimesh(verts, faces)
                surface_samples, sdf = sample_sdf_near_surface(
                    mesh, voxels, voxel_size)
                surface_samples = torch.FloatTensor(surface_samples)
                sdf = compute_sdf(verts.cuda(), faces.cuda(),
                                  surface_samples.cuda())

            if args.inspect:
                display_sdf(surface_samples, sdf.cpu(), only_negative=False)

            sdf = sdf.cpu()[:, None]
            weights = torch.ones_like(sdf)
            sdf_samples = torch.cat([surface_samples, sdf, weights], dim=-1)

            # voxelize samples
            # nsamples = save_samples(
            #     surface_pnts.cuda(),
            #     sdf_samples.cuda(),
            #     voxels.cuda(),
            #     voxel_size,
            #     samples_path,
            #     num_sampled,
            #     upper_limit=-1,
            #     save_pnts=args.save_pnts,
            #     sample_free_space=False,
            # )
            surface, samples = get_voxel_samples(
                surface_pnts.cuda(),
                sdf_samples.cuda(),
                voxels.cuda(),
                voxel_size,
                min_surface_points=256,
                min_sample_points=2048)
            num_shapes = len(surface)
            if num_shapes == 0:
                continue

            object_index_file.append({
                "class": object_class,
                "filename": object_name,
                "index_begin": total_shape_generated,
                "index_end": total_shape_generated+num_shapes,
                "rotation": rotation.cpu().numpy().tolist()
            })

            for idx in range(num_shapes):
                np.savez(
                    SAMPLE_OUT_PATH / f'{total_shape_generated+idx}.npz',
                    surface=surface[idx].cpu().numpy(),
                    samples=samples[idx].cpu().numpy())

            object_progbar.update(1)
            total_shape_generated += num_shapes
            num_models_sampled += 1
            if num_models_sampled >= args.objects_per_class:
                break
    json.dump(object_index_file, open(OUT_PATH / 'meta.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('split_file', type=str)
    parser.add_argument('--method', type=int, default=0)
    parser.add_argument('--objects_per_class', type=int, default=50)
    parser.add_argument('--voxel_size', type=float, default=-1)
    # parser.add_argument('--test', action='store_true')
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    # parser.add_argument('--save_pnts', action='store_true')
    args = parser.parse_args()

    main(args)
