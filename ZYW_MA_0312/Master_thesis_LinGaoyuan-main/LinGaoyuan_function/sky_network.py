import functools
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class StyleMLP(nn.Module):
    r"""MLP converting style code to intermediate style representation."""

    def __init__(self, style_dim, out_dim, hidden_channels=256, leaky_relu=True, num_layers=5, normalize_input=True,
                 output_act=True):
        super(StyleMLP, self).__init__()

        self.normalize_input = normalize_input
        self.output_act = output_act
        fc_layers = []
        fc_layers.append(nn.Linear(style_dim, hidden_channels, bias=True))
        for i in range(num_layers-1):
            fc_layers.append(nn.Linear(hidden_channels, hidden_channels, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)

        self.fc_out = nn.Linear(hidden_channels, out_dim, bias=True)

        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = functools.partial(F.relu, inplace=True)

    def forward(self, z):
        r""" Forward network

        Args:
            z (N x style_dim tensor): Style codes.
        """
        if self.normalize_input:
            z = F.normalize(z, p=2, dim=-1)
        for fc_layer in self.fc_layers:
            z = self.act(fc_layer(z))
        z = self.fc_out(z)
        if self.output_act:
            z = self.act(z)
        return z

class SKYMLP(nn.Module):
    r"""MLP converting ray directions to sky features."""

    def __init__(self, in_channels, style_dim, out_channels_c=3,
                 hidden_channels=256, leaky_relu=True):
        super(SKYMLP, self).__init__()
        self.fc_z_a = nn.Linear(style_dim, hidden_channels, bias=False)

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, hidden_channels)
        self.fc5 = nn.Linear(hidden_channels, hidden_channels)

        self.fc_out_c = nn.Linear(hidden_channels, out_channels_c)

        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = functools.partial(F.relu, inplace=True)

        self.act_out = nn.Softmax()

    def forward(self, x, z):
        r"""Forward network

        Args:
            x (... x in_channels tensor): Ray direction embeddings.
            z (... x style_dim tensor): Style codes.
        """

        z = self.fc_z_a(z)
        while z.dim() < x.dim():
            z = z.unsqueeze(1)

        y = self.act(self.fc1(x) + z)
        # y = self.act(self.fc1(x))

        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))
        y = self.act(self.fc4(y))
        y = self.act(self.fc5(y))
        c = self.fc_out_c(y)

        # c = self.act_out(c)

        return c

class SkyModel(nn.Module):
    def __init__(self, args, style_dims=128, load_opt=True, load_scheduler=True):
        super().__init__()
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))

        self.sky_model = SKYMLP(in_channels = 3, style_dim=style_dims).to(device)
        self.sky_style_model = StyleMLP(style_dim = style_dims, out_dim = style_dims).to(device)

        self.sky_optimizer = torch.optim.Adam(self.sky_model.parameters(), lr=args.lrate_sky_model)
        self.sky_style_optimizer = torch.optim.Adam(self.sky_style_model.parameters(), lr=args.lrate_sky_style_model)

        self.sky_scheduler = torch.optim.lr_scheduler.StepLR(
            self.sky_optimizer, step_size=args.lrate_decay_steps_sky_model, gamma=args.lrate_decay_factor_sky_model
        )
        self.sky_style_scheduler = torch.optim.lr_scheduler.StepLR(
            self.sky_style_optimizer, step_size=args.lrate_decay_steps_sky_style_model,
            gamma=args.lrate_decay_factor_sky_style_model
        )

        if args.distributed:
            self.sky_model = torch.nn.parallel.DistributedDataParallel(
                self.sky_model, device_ids=[args.local_rank], output_device=args.local_rank
            )
            self.sky_style_model = torch.nn.parallel.DistributedDataParallel(
                self.sky_style_model, device_ids=[args.local_rank], output_device=args.local_rank
            )

        self.black_background = torch.zeros((3,)).cuda()

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )



    def forward(self, ray_d, sky_style_code, sky_mask):
        ray_d = ray_d.cuda()

        sky_style_code = self.sky_style_model(sky_style_code)

        self.sky_style_code = sky_style_code

        skynet_out_color = self.sky_model(ray_d, sky_style_code)

        skynet_out_c = skynet_out_color * (1.0 - sky_mask) + self.black_background * (sky_mask)

        rgb_sky = skynet_out_c

        return rgb_sky, sky_style_code

    def save_model(self, filename):
        to_save = {
            "sky_optimizer": self.sky_optimizer.state_dict(),
            "sky_scheduler": self.sky_scheduler.state_dict(),
            "sky_style_optimizer": self.sky_style_optimizer.state_dict(),
            "sky_style_scheduler": self.sky_style_scheduler.state_dict(),
            "sky_model": self.sky_model.state_dict(),
            "sky_style_model": self.sky_style_model.state_dict(),
            "sky_style_code": self.sky_style_code,
        }

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            to_load = torch.load(filename)
        # print(to_load["net_coarse"].keys())
        # exit()
        if load_opt:
            self.sky_optimizer.load_state_dict(to_load["sky_optimizer"])
            self.sky_style_optimizer.load_state_dict(to_load["sky_style_optimizer"])
        if load_scheduler:
            self.sky_scheduler.load_state_dict(to_load["sky_scheduler"])
            self.sky_style_scheduler.load_state_dict(to_load["sky_style_scheduler"])

        self.sky_model.load_state_dict(to_load["sky_model"])
        self.sky_style_model.load_state_dict(to_load["sky_style_model"])
        self.sky_style_code = to_load["sky_style_code"]


    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        sky_ckpts = []
        if os.path.exists(out_folder):
            sky_ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        'LinGaoyuan code: remove all the elements in ckpts which name contains sky_model'
        ckpts_copy = []
        for i in sky_ckpts:
            if i.split("/")[-1].find('sky_model') == 0:
                ckpts_copy.append(i)
        sky_ckpts = ckpts_copy

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                sky_ckpts = [self.args.ckpt_path]

        if len(sky_ckpts) > 0 and not self.args.no_reload:
            fpath = sky_ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0

        return step

    def switch_to_eval(self):
        self.sky_model.eval()
        self.sky_style_model.eval()

    def switch_to_train(self):
        self.sky_model.train()
        self.sky_style_model.train()

