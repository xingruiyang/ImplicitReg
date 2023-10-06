import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lrschedualer import load_schedualers
from torch.optim.lr_scheduler import ReduceLROnPlateau
from network import ImplicitNet
from datasets import collate_fn
from utils import import_class, load_config, make_backups
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(
            self,
            out_path,
            device: torch.device = torch.device('cpu'),
            lrschedual: list = None,
            ckpt_freq: int = -1,
            epoch_begin: int = 0,
            num_epochs: int = 1,
            mini_batch: int = 1
    ) -> None:
        super().__init__()
        self.out_path = out_path
        self.num_epochs = num_epochs
        self.ckpt_freq = ckpt_freq
        self.device = device
        self.epoch_begin = epoch_begin
        self.lrschedual = lrschedual
        self.mini_batch = mini_batch
        self.logger = SummaryWriter(os.path.join(out_path, 'logs'))
        self.scaler = torch.cuda.amp.GradScaler()

    def update_lr(self, optim, n_epoch, pbar):
        '''update learning rate from the provided schedualers
        '''
        if self.lrschedual is None:
            return

        model_schedualer = self.lrschedual.get("model")
        latent_schedualer = self.lrschedual.get("latent")

        if model_schedualer is not None:
            last_lr_model = optim.param_groups[0]['lr']
            new_lr_model = model_schedualer.get_lr(n_epoch)
            if last_lr_model != new_lr_model:
                optim.param_groups[0]['lr'] = new_lr_model
                pbar.write("model lr is set to {}".format(new_lr_model))

        if latent_schedualer is not None:
            new_lr_latent = latent_schedualer.get_lr(n_epoch)
            last_lr_latent = optim.param_groups[1]['lr']
            if last_lr_latent != new_lr_latent:
                optim.param_groups[1]['lr'] = new_lr_latent
                pbar.write("latent lr is set to {}".format(new_lr_latent))

    def fit(
        self,
            model: torch.nn.Module,
            data_loader: DataLoader,
            optim_ckpt: str = None
    ) -> None:
        '''train/optimise a network/latents
        Args:
            model (moduel): initialized network,
            data_loader (DataLoader): train/eval data loader
            optim_ckpt (str): ckpt path to the optimizer (if any)
        '''
        # model.train()
        step_global = 0
        optimizer = model.configure_optimizers(ckpt=optim_ckpt)
        for n_epoch in range(self.epoch_begin, self.num_epochs):
            batch_loss = 0
            sdf_loss = 0
            latent_loss = 0
            # self.pbar = tqdm(data_loader)
            pbar = tqdm(total=len(data_loader)*self.mini_batch)
            self.update_lr(optimizer, n_epoch, pbar)

            num_batch_step = 0
            for n_batch, batch_data in enumerate(data_loader):
                batch_data = batch_data.view(-1, 6)
                batch_data = batch_data[torch.randperm(batch_data.shape[0]), :]
                chunk_data = torch.chunk(batch_data, self.mini_batch)

                for n_chunk in range(len(chunk_data)):
                    mini_batch_data = chunk_data[n_chunk].to(self.device)

                    with torch.cuda.amp.autocast():
                        losses = model.training_step(
                            mini_batch_data, n_epoch)

                    loss = losses['loss']

                    optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    batch_loss += loss.item()
                    sdf_loss += losses['sdf_loss']
                    latent_loss += losses['latent_loss']
                    step_global += 1
                    num_batch_step += 1

                    pbar.set_description(
                        "Epoch {} loss: {:.4f} sdf: {:.4f} latent: {:.4f}".format(
                            n_epoch,
                            batch_loss/num_batch_step,
                            sdf_loss/num_batch_step,
                            latent_loss/num_batch_step
                        )
                    )

                    pbar.update(1)
                    self.logger.add_scalar(
                        'train/loss', loss.item(), step_global)
                    self.logger.add_scalar(
                        'train/sdf_loss', losses['sdf_loss'], step_global)
                    self.logger.add_scalar(
                        'train/latent_loss', losses['latent_loss'], step_global)

            if (n_epoch > 0 and self.ckpt_freq > 0):
                if n_epoch % self.ckpt_freq == 0:
                    self.save_ckpt(model, optim=optimizer, epoch=n_epoch)
            self.save_latest(model, optim=optimizer, epoch=n_epoch)

    def save_ckpt(self, model, optim=None, epoch=0):
        model.save_ckpt(os.path.join(
            self.out_path, 'ckpt_e{}_model.pth'.format(epoch)), optim, epoch)
        model.save_latents(os.path.join(
            self.out_path, 'ckpt_e{}_latents.npy'.format(epoch)), epoch)

    def save_latest(self, model, optim=None, epoch=0):
        model.save_ckpt(os.path.join(
            self.out_path, 'latest_model.pth'), optim, epoch)
        model.save_latents(os.path.join(
            self.out_path, 'latest_latents.npy'), epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('out_path')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--latents', type=str, default=None)
    parser.add_argument('--cfg', type=str, default='configs/basic.json')
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--mini_batch', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--load_optim', action='store_true')
    parser.add_argument('--reset_epoch', action='store_true')
    args = parser.parse_args()

    # backup arch and config
    # make_backups(args.out_path, cfg_name=args.cfg)

    device = torch.device('cpu' if args.cpu else 'cuda')
    train_cfg = load_config(args.cfg)["train"]

    # load training data
    DatasetName = import_class('datasets.'+train_cfg["dataset"]["name"])
    train_data = DatasetName(
        args.data, args.orient,
        **train_cfg["dataset"]["args"])
    train_loader = DataLoader(
        train_data,
        batch_size=train_cfg["batch_size"],
        num_workers=args.threads,
        shuffle=True,
        collate_fn=collate_fn)

    # load model arch and (optionally) ckpts
    model, epoch = ImplicitNet.create_from_cfg(
        args.cfg, ckpt=args.ckpt, device=device)

    # initialize latent vecs and (optionally) load from ckpts
    model.initialize_latents(
        train_data.num_latents,
        ckpt=args.latents, device=device)
    model.train()

    trainer = Trainer(
        device=device,
        out_path=args.out_path,
        epoch_begin=0 if args.reset_epoch else epoch,
        lrschedual=load_schedualers(train_cfg["schedualers"]),
        ckpt_freq=train_cfg["ckpt_frequency"],
        num_epochs=train_cfg["num_epoch"],
        mini_batch=train_cfg["mini_batch"]
    )

    trainer.fit(
        model, train_loader,
        optim_ckpt=args.ckpt if args.load_optim else None)

    # final saves
    model.save_ckpt(
        os.path.join(args.out_path, 'last_ckpt.pth'))
    model.save_latents(
        os.path.join(args.out_path, 'last_latents.npy'))
