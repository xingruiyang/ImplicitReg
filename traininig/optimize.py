import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets import collate_fn
from lrschedualer import load_schedualers
from network import ImplicitNet
from trainer import Trainer
from utils import import_class, load_config, make_backups

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('ckpt')
    parser.add_argument('out_path')
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--latents', type=str, default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--load_optim', action='store_true')
    parser.add_argument('--load_epoch', action='store_true')
    args = parser.parse_args()

    # make backups
    make_backups(args.out_path, args.ckpt, flag='test')
    config_file = args.cfg if args.cfg is not None else os.path.join(
        args.ckpt, 'cfg.json')
    test_cfg = load_config(config_file)["test"]

    # loading test data
    device = torch.device('cpu' if args.cpu else 'cuda')
    DatasetName = import_class('datasets.'+test_cfg["dataset"]["name"])
    test_data = DatasetName(args.data, args.orient, **
                            test_cfg["dataset"]["args"])
    test_loader = DataLoader(test_data,
                             batch_size=test_cfg["batch_size"],
                             num_workers=args.threads,
                             shuffle=True,
                             collate_fn=collate_fn)

    model_ckpt = os.path.join(args.ckpt, 'latest_model.pth')
    model_cfg = os.path.join(
        args.ckpt, 'cfg.json') if args.cfg is None else args.cfg
    # model_ckpt = test_cfg['model_ckpt']
    model, epoch = ImplicitNet.create_from_cfg(
        cfg=model_cfg, ckpt=model_ckpt, device=device)

    epoch = epoch if args.load_epoch else 0
    print("starting from epoch {}".format(epoch))

    model.initialize_latents(
        test_data.num_latents,
        ckpt=args.latents,
        device=device)
    model.freeze_decoder = True
    model.eval()

    trainer = Trainer(
        device=device,
        out_path=args.out_path,
        ckpt_freq=test_cfg["ckpt_frequency"],
        num_epochs=test_cfg["num_epoch"],
        epoch_begin=epoch,
        lrschedual=load_schedualers(test_cfg["schedualers"]),
        mini_batch=test_cfg["mini_batch"]
    )

    trainer.fit(
        model,
        test_loader,
        optim_ckpt=model_ckpt if args.load_optim else None)
    model.save_latents(os.path.join(args.out_path, 'last_latents.npy'))
