import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
import os

'use the ray_d and the mask of sky area of image to predict sky rgb'
class SkyAttention(nn.Module):
    def __init__(self, dim_input = 256, dim_embed = 256, input_embedding = False, num_head = 1, dropout=0.1):
        super().__init__()
        self.dim_input = dim_input
        self.num_head = num_head
        if dim_embed == None:
            self.dim_embed = self.dim_input
        else:
            self.dim_embed = dim_embed
        self.dim_head = self.dim_embed // self.num_head

        if input_embedding is True:
            self.embed_fc = nn.Linear(self.dim_input, self.dim_embed)


        self.q_fc = nn.Linear(self.dim_input, self.dim_embed, bias=False)
        self.k_fc = nn.Linear(self.dim_input, self.dim_embed, bias=False)
        self.v_fc = nn.Linear(self.dim_input, self.dim_embed, bias=False)

        self.layer_norm = nn.LayerNorm(self.dim_input)

        self.out_fc = nn.Linear(self.dim_embed, self.dim_input, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, mask = None):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        residual = input
        input = self.layer_norm(input)

        q = self.q_fc(input)
        q = q.view(input.shape[0], input.shape[1], self.num_head, self.dim_head)
        # q = q.permute(2, 0, 1, 3)  # (self.num_head * batch_input) x len_input x self.dim_head
        q = q.permute(0, 2, 1, 3)  # (batch_input * self.num_head) x len_input x self.dim_head

        k = self.k_fc(input)
        k = k.view(input.shape[0], input.shape[1], self.num_head, self.dim_head)
        # k = k.permute(2, 0, 1, 3)  # (self.num_head * batch_input) x len_input x self.dim_head
        k = k.permute(0, 2, 1, 3)  # (batch_input * self.num_head) x len_input x self.dim_head

        v = self.v_fc(input)
        v = v.view(input.shape[0], input.shape[1], self.num_head, self.dim_head)
        # v = v.permute(2, 0, 1, 3)  # (self.num_head * batch_input) x len_input x self.dim_head
        v = v.permute(0, 2, 1, 3)  # (batch_input * self.num_head) x len_input x self.dim_head

        # attn = torch.bmm(q, k.transpose(1, 2)) / np.power(self.dim_head, 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.power(self.dim_head, 0.5)

        if mask is not None:
            mask = mask.repeat(self.num_head, 1, 1)  # (self.num_head * batch_input) x .. x ..
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # output = torch.bmm(attn, v)
        output = torch.matmul(attn, v)

        # output = output.view(self.num_head, input.shape[0], input.shape[1], self.dim_head)
        # output = output.permute(1, 2, 0, 3)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(input.shape[0], input.shape[1], -1)

        output = self.dropout(self.out_fc(output))
        output = self.layer_norm(output + residual)

        return output[0]


class SkyStyleAttention(nn.Module):
    def __init__(self, dim_input = 128, dim_embed = 256, input_embedding = False, num_head = 1, dropout=0.1):
        super().__init__()
        self.dim_input = dim_input
        self.num_head = num_head
        if dim_embed == None:
            self.dim_embed = self.dim_input
        else:
            self.dim_embed = dim_embed
        self.dim_head = self.dim_embed // self.num_head

        if input_embedding is True:
            self.embed_fc = nn.Linear(self.dim_input, self.dim_embed)

        self.q_fc = nn.Linear(self.dim_input, self.dim_embed, bias=False)
        self.k_fc = nn.Linear(self.dim_input, self.dim_embed, bias=False)
        self.v_fc = nn.Linear(self.dim_input, self.dim_embed, bias=False)

        self.layer_norm = nn.LayerNorm(self.dim_input)

        self.out_fc = nn.Linear(self.dim_embed, self.dim_input, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, mask = None):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        residual = input
        input = self.layer_norm(input)

        q = self.q_fc(input)
        q = q.view(input.shape[0], input.shape[1], self.num_head, self.dim_head)
        # q = q.permute(2, 0, 1, 3)  # (self.num_head * batch_input) x len_input x self.dim_head
        q = q.permute(0, 2, 1, 3)  # (batch_input * self.num_head) x len_input x self.dim_head

        k = self.k_fc(input)
        k = k.view(input.shape[0], input.shape[1], self.num_head, self.dim_head)
        # k = k.permute(2, 0, 1, 3)  # (self.num_head * batch_input) x len_input x self.dim_head
        k = k.permute(0, 2, 1, 3)  # (batch_input * self.num_head) x len_input x self.dim_head

        v = self.v_fc(input)
        v = v.view(input.shape[0], input.shape[1], self.num_head, self.dim_head)
        # v = v.permute(2, 0, 1, 3)  # (self.num_head * batch_input) x len_input x self.dim_head
        k = k.permute(0, 2, 1, 3)  # (batch_input * self.num_head) x len_input x self.dim_head

        # attn = torch.bmm(q, k.transpose(1, 2)) / np.power(self.dim_head, 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.power(self.dim_head, 0.5)

        if mask is not None:
            mask = mask.repeat(self.num_head, 1, 1)  # (self.num_head * batch_input) x .. x ..
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # output = torch.bmm(attn, v)
        output = torch.matmul(attn, v)

        # output = output.view(self.num_head, input.shape[0], input.shape[1], self.dim_head)
        # output = output.permute(1, 2, 0, 3)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(input.shape[0], input.shape[1], -1)

        output = self.dropout(self.out_fc(output))
        output = self.layer_norm(output + residual)

        return output[0]

class SkyTransformer(nn.Module):
    def __init__(self, args, ray_d_input=3, style_dim=128, dim_embed=256, num_head=2, input_embedding = False,
                 dropout=0.05, num_layers=5):
        super(SkyTransformer, self).__init__()
        self.SkyTrans = nn.ModuleList([])
        self.SkyStyleTrans = nn.ModuleList([])
        self.num_layers = num_layers

        self.fc_z_a = nn.Linear(style_dim, dim_embed, bias=False)
        self.fc1 = nn.Linear(ray_d_input, dim_embed)
        self.fc_out_c = nn.Linear(dim_embed, ray_d_input)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        for i in range(self.num_layers):
            transformer = SkyAttention(dim_input=dim_embed, dim_embed=dim_embed, input_embedding=input_embedding,
                                          num_head=num_head, dropout=dropout)
            self.SkyTrans.append(transformer)

            sky_style_transformer = SkyStyleAttention(dim_input=style_dim, dim_embed=dim_embed, input_embedding=input_embedding,
                                                      num_head=num_head, dropout=dropout)
            self.SkyStyleTrans.append(sky_style_transformer)


        # self.sky_attention = SkyAttention(dim_input=ray_d_input, dim_embed=dim_embed, input_embedding=input_embedding,
        #                                   num_head=num_head, dropout=dropout)
        # self.sky_style_attention = SkyStyleAttention(dim_input=style_dim, dim_embed=dim_embed, input_embedding=input_embedding,
        #                                              num_head=num_head, dropout=dropout)
        # self.fc_z_a = nn.Linear(style_dim, dim_embed, bias=False)
        # self.fc_x_a = nn.Linear(style_dim, dim_embed, bias=False)

    def forward(self, x, z=None, sky_mask=None):
        # for i in range(self.num_layers):
        #     z = self.sky_style_attention(z)
        # z = self.fc_z_a(z)
        #
        # while z.dim() < x.dim():
        #     z = z.unsqueeze(1)

        # for i in range(self.num_layers):
        #     x = self.sky_attention(x)
        for i, skystyletran in enumerate(self.SkyStyleTrans):
            z = skystyletran(z)

        sky_style_code = z

        z = self.fc_z_a(z)

        x = self.act(self.fc1(x) + z)

        for i, skytran in enumerate(self.SkyTrans):
            x = skytran(x)

        skynet_out_color = self.fc_out_c(x)

        # skynet_out_c = skynet_out_color * (1.0 - sky_mask) + self.black_background * (sky_mask)
        #
        # rgb_sky = skynet_out_c

        return skynet_out_color, sky_style_code

class SkyTransformerModel(nn.Module):
    def __init__(self, args, ray_d_input=3, style_dim=128, dim_embed=256, num_head=2, input_embedding = False,
                 dropout=0.05, num_layers=5, load_opt=True, load_scheduler=True):
        super().__init__()
        self.device = "cuda:{}".format(args.local_rank)
        self.args = args
        self.sky_transformer = SkyTransformer(args, ray_d_input=3, style_dim=128, dim_embed=256, num_head=2, input_embedding=False,
                                              dropout=0.05, num_layers=5)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.sky_transformer.SkyTrans.parameters(), "lr": args.lrate_sky_model},
                {"params": self.sky_transformer.SkyStyleTrans.parameters(), "lr": args.lrate_sky_style_model}
                ]
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps_sky_model, gamma=args.lrate_decay_factor_sky_model
        )

        self.black_background = torch.zeros((3,)).cuda()

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(out_folder, load_opt=load_opt, load_scheduler=load_scheduler)

    def forward(self, x, z=None, sky_mask=None):
        x = x.cuda()
        z = z.cuda()
        skynet_out_c, sky_style_code = self.sky_transformer(x,z)

        self.sky_style_code = sky_style_code

        skynet_out_c = skynet_out_c * (1.0 - sky_mask) + self.black_background * (sky_mask)

        rgb_sky = skynet_out_c

        return rgb_sky, sky_style_code

    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "sky_transformer": self.sky_transformer.state_dict(),
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
            self.optimizer.load_state_dict(to_load["optimizer"])
            self.optimizer_to(self.optimizer, self.device)
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        self.sky_transformer.load_state_dict(to_load["sky_transformer"])
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
        self.sky_transformer.eval()

    def switch_to_train(self):
        self.sky_transformer.train()

    def optimizer_to(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)