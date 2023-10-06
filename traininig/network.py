import numpy as np
import torch
import torch.nn as nn

try:
    import traininig.utils as utils
except:
    import utils

activations = {
    "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
    "soft_plus": nn.Softplus(beta=100)
}

optimizers = {
    "Adam": torch.optim.Adam
}

loss_functions = {
    "l1": torch.nn.L1Loss,
    "l2": torch.nn.MSELoss
}


class ImplicitNet(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dims,
                 use_tanh=True,
                 clamp_dist=-1,
                 loss_fn="l1",
                 act_fn="leaky_relu",
                 optimizer="Adam",
                 use_gradients=False,
                 gradient_weight=0,
                 latent_reg_weight=0,
                 geometric_init=True,
                 radius_init=1,
                 freeze_decoder=False):
        super(ImplicitNet, self).__init__()

        dims = [latent_dim+3] + hidden_dims + [1]
        self.num_layers = len(dims)

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(
                        np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            lin = nn.DataParallel(lin)
            setattr(self, "lin_"+str(layer), lin)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

        self.act_fn = activations.get(
            act_fn, nn.ReLU())
        self.optimizer = optimizers.get(
            optimizer, torch.optim.Adam)
        self.loss_fn = loss_functions.get(
            loss_fn, torch.nn.L1Loss
        )(reduction='none')

        self.clamp_dist = clamp_dist
        self.clamp = clamp_dist > 0
        self.latent_vecs = None
        self.latent_dim = latent_dim
        self.use_gradients = use_gradients
        self.freeze_decoder = freeze_decoder
        self.latent_reg_weight = latent_reg_weight
        self.gradient_weight = gradient_weight

    def forward(self, inputs):
        x = inputs
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin_" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act_fn(x)
        x = self.tanh(x) if self.use_tanh else x
        return torch.clamp(x, -self.clamp_dist, self.clamp_dist) if self.clamp else x

    def configure_optimizers(self, ckpt=None):
        if self.freeze_decoder:
            trainable_params = [{
                'params': [], 'lr': 1},
                {'params': self.latent_vecs, 'lr': 1}]
        else:
            trainable_params = [{
                'params': self.parameters(), 'lr': 1},
                {'params': self.latent_vecs, 'lr': 1}]
        optimizer = self.optimizer(trainable_params)

        if ckpt is not None:
            print("loading optim from ckpt {}".format(ckpt))
            checkpoint = torch.load(ckpt)
            state_dict = checkpoint['optim_state_dict']
            if state_dict is not None:
                optimizer.load_state_dict(state_dict)

        return optimizer

    def training_step(self, train_batch, n_epoch=0):
        pnts = train_batch[..., 1:4].view(-1, 3)
        gt_sdf = train_batch[..., 4].view(-1, 1)
        indices = train_batch[..., 0].view(-1).long()
        # weights = train_batch[..., 5].view(-1, 1)
        weights = torch.ones_like(gt_sdf).float()

        # latents = torch.index_select(self.latent_vecs, 0, indices)
        latents = self.latent_vecs[indices, :]
        inputs = torch.cat([latents, pnts], dim=-1)

        pred_sdf = self.forward(inputs)

        if self.use_tanh:
            gt_sdf = torch.tanh(gt_sdf)
        if self.clamp:
            gt_sdf = torch.clamp(gt_sdf, -self.clamp_dist, self.clamp_dist)

        sdf_loss = self.loss_fn(gt_sdf, pred_sdf)
        sdf_loss = (sdf_loss * weights).mean()
        latent_loss = torch.norm(latents, p=2, dim=-1).mean()
        loss = sdf_loss + self.latent_reg_weight * latent_loss

        if self.use_gradients:
            gradient = utils.compute_gradient(pred_sdf, inputs)[:, -3:]
            grad_loss = torch.abs(gradient.norm(dim=-1) - 1).mean()
            loss += self.gradient_weight * grad_loss

        return {
            'loss': loss,
            'sdf_loss': sdf_loss.item(),
            'latent_loss': latent_loss.item()
        }

    def save_ckpt(self, filename: str, optim=None, epoch: int = 0):
        model_state_dict = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optim_state_dict": optim.state_dict() if optim is not None else None
        }
        torch.save(model_state_dict, filename)

    def save_latents(self, filename: str, epoch: int = 0):
        latent_vecs = self.latent_vecs.detach().cpu().numpy()
        np.savez(filename, latent_vecs=latent_vecs, epoch=epoch)

    def initialize_latents(self, num_latents=0, ckpt=None, device=torch.device('cpu')):
        print("trying to allocate {} latents".format(num_latents))
        if ckpt is None:
            self.latent_vecs = torch.zeros(
                (int(num_latents), self.latent_dim)).to(device)
            torch.nn.init.normal_(self.latent_vecs, 0, 0.01)
            self.latent_vecs.requires_grad_()
        else:
            print("loading latents from check point {}".format(ckpt))
            latent_vecs = np.load(ckpt)['latent_vecs']
            self.latent_vecs = torch.from_numpy(latent_vecs).to(device)
            self.latent_vecs.requires_grad_()

    @staticmethod
    def create_from_cfg(cfg, ckpt=None, device=torch.device('cpu')):
        net_args = utils.load_config(cfg)
        print("creating model from cfg {}".format(cfg))
        network = ImplicitNet(**net_args['params']).to(device)
        epoch = 0
        if ckpt is not None:
            print("loading decoder from check point {}".format(ckpt))
            data = torch.load(ckpt, map_location=device)
            epoch = data["epoch"]+1
            network.load_state_dict(data["model_state_dict"])
        return network, epoch
