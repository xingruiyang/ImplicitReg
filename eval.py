import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from network import ImplicitNet
from utils import display_sdf

if __name__ == '__main__':
    os.environ["DISPLAY"] = ":0"

    parser = argparse.ArgumentParser()
    parser.add_argument('pnts', type=str)
    parser.add_argument('--out_data', type=str, default="data/test.pth")
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=50000)
    parser.add_argument('--orient', action='store_true')
    args = parser.parse_args()

    net_args = {
        "latent_dim": 64,
        "hidden_dims": [
            128,
            128,
            128,
            128
        ],
        "use_tanh": True,
        "clamp_dist": 0.1
    }

    # pnts = 'data/sdf_samples.npz'
    if args.orient:
        ckpt = '/mnt/data/deep_shapes/models/test/64/latest_model.pth'
    else:
        ckpt = '/mnt/data/deep_shapes/models/recon/un64d/latest_model.pth'

    device = torch.device('cuda')

    model = ImplicitNet(**net_args).cuda()
    data = torch.load(ckpt, map_location=device)
    model_state_dict = {
        k.replace('module.', ''): v for k, v in data["model_state_dict"].items() if 'generator' not in k}
    model.load_state_dict(model_state_dict)
    model.train()

    samples = np.load(args.pnts)
    voxel_size = samples["voxel_size"]
    voxels = samples["voxels"]
    sdf_samples = samples["sdf_samples"]
    rotations = samples["rotations"]
    centroids = samples["centroids"]
    # xyz = samples["xyz"]

    voxels = torch.from_numpy(voxels)
    sdf_samples = torch.from_numpy(sdf_samples)
    rotations = torch.from_numpy(rotations)
    centroids = torch.from_numpy(centroids)

    num_voxels = voxels.shape[0]
    latent_vecs = torch.zeros((num_voxels, 64)).cuda()
    latent_vecs.requires_grad_(True)
    torch.nn.init.normal_(latent_vecs, std=0.01)

    optim = torch.optim.Adam([latent_vecs], lr=1e-3)
    scale = float(voxel_size)
    global_steps = 0
    avg_loss = 0

    for i in tqdm(range(args.max_steps)):
        shuffle = torch.randperm(sdf_samples.shape[0])
        sdf_samples = sdf_samples[shuffle, :]
        batch_samples = torch.split(sdf_samples, args.batch_size)

        # pt = sdf_samples[sdf_samples[:, 0] == sdf_samples[0, 0], 1:4]
        # sdf = sdf_samples[sdf_samples[:, 0] == sdf_samples[0, 0], 4]
        # display_sdf(pt, sdf)

        for g in optim.param_groups:
            if i < 2:
                g['lr'] = 1e-2
            elif i < 5:
                g['lr'] = 1e-3
            else:
                g['lr'] = 1e-4

        pbar = tqdm(batch_samples, leave=False)
        for samples in pbar:
            samples = samples.cuda()
            index = samples[:, 0].long()
            points = samples[:, 1:4].float() / scale
            gt_sdf = samples[:, 4].float() / scale
            weight = samples[:, 5].float()

            if model.use_tanh:
                gt_sdf = torch.tanh(gt_sdf)
            if model.clamp:
                gt_sdf = gt_sdf * model.clamp_dist
            # if model.clamp:
            #     gt_sdf = torch.clamp(
            #         gt_sdf, -model.clamp_dist, model.clamp_dist)

            if args.orient:
                rotation = rotations[index, :, :].cuda().float()
                centroid = centroids[index, :].cuda().float()
                points -= centroid
                points = points.unsqueeze(1) @ rotation.transpose(-1, -2)

            latents = latent_vecs[index, :]
            inputs = torch.cat([points.squeeze(1), latents], dim=-1)
            # inputs = torch.cat([latents, points.squeeze(1)], dim=-1)
            pred_sdf = model(inputs).squeeze()

            weight = 1/(weight**2)
            loss = (weight*(pred_sdf - gt_sdf).abs()).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            avg_loss += loss.item()
            global_steps += 1
            pbar.set_description(
                "loss {:.4f}".format(avg_loss/global_steps))

print(voxels)
torch.save({
    'model': model.cpu().state_dict(),
    'voxels': voxels.detach().cpu(),
    'latents': latent_vecs.detach().cpu(),
    'voxel_size': voxel_size,
    # 'pts': xyz,
    'config': net_args,
    'rotations': rotations,
    'centroids': centroids
}, args.out_data)

# model.cuda()
# create_mesh(model, voxels, latent_vecs, points, voxel_size)
# create_mesh(model, voxels, latent_vecs, None, voxel_size, rotations, centroids)
