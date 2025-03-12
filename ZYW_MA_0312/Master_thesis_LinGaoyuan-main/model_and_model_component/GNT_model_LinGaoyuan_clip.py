import numpy as np
import torch
import torch.nn as nn

from LinGaoyuan_function.aliasing import exercute_aliasing_filter

# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None, aliasing_filter = False, aliasing_filter_type = 'filter bank'):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)

        'anti-aliasing filter should be used in here after the completion of ff layer'
        if aliasing_filter is True:
            # aliasing_filter_type = 'filter bank' # 'filter bank' or 'single filter'
            x = exercute_aliasing_filter(x,aliasing_filter_type)

        x = x + residue

        return x


# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None):
        super(Attention, self).__init__()
        if attn_mode in ["qk", "gate"]:
            self.q_fc = nn.Linear(dim, dim, bias=False)
            self.k_fc = nn.Linear(dim, dim, bias=False)
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode

    def forward(self, x, pos=None, ret_attn=False):
        if self.attn_mode in ["qk", "gate"]:
            q = self.q_fc(x)
            q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
            k = self.k_fc(x)
            k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.v_fc(x)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


# Ray Transformer
class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None
    ):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim)

    def forward(self, x, pos=None, ret_attn=False):
        residue = x
        x = self.attn_norm(x)
        x = self.attn(x, pos, ret_attn)
        if ret_attn:
            x, attn = x
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x


class GNT(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False, dim_clip_shape_code=0):
        super(GNT, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        '''
        LinGaoyuan_20240930: add dimension of shape code, if the model is used for building and street, the self.dim_clip_shape_code = 0, 
        and if model is used for car, self.dim_clip_shape_code = 128
        '''
        self.dim_clip_shape_code = dim_clip_shape_code
        self.dim_clip_appearance_code = 128

        # NOTE: Apologies for the confusing naming scheme, here view_crosstrans refers to the view transformer, while the view_selftrans refers to the ray transformer
        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])
        for i in range(args.trans_depth):  # trans_depth = 4 in config file
            # view transformer
            view_trans = Transformer2D(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_crosstrans.append(view_trans)

            '''
            LinGaoyuan_20240930: add dimension of shape code, if the model is used for building and street, the self.dim_clip_shape_code = 0, 
            and if model is used for car, self.dim_clip_shape_code = 128
            '''
            # ray transformer
            # ray_trans = Transformer(
            #     dim=args.netwidth,
            #     ff_hid_dim=int(args.netwidth * 4),
            #     n_heads=4,
            #     ff_dp_rate=0.1,
            #     attn_dp_rate=0.1,
            # )
            ray_trans = Transformer(
                dim=args.netwidth + self.dim_clip_shape_code,
                ff_hid_dim=int(args.netwidth * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_selftrans.append(ray_trans)
            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth + posenc_dim + viewenc_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)

        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(args.netwidth)

        'LinGaoyuan_20240930: add dimension of clip appearance code to radiance mlp. '
        '''
        LinGaoyuan_20240930: add dimension of shape code, if the model is used for building and street, the self.dim_clip_shape_code = 0, 
        and if model is used for car, self.dim_clip_shape_code = 128
        '''
        # self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.rgb_fc = nn.Linear(args.netwidth + self.dim_clip_appearance_code + self.dim_clip_shape_code, 3)

        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

        self.aliasing_filter = args.aliasing_filter
        self.aliasing_filter_type = args.aliasing_filter_type

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d):
        # compute positional embeddings
        viewdirs = ray_d  # (N_rand, 3)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # (N_rand, 3)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # (N_rand, 3)
        viewdirs = self.view_enc(viewdirs)  # (N_rand, 63)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()  # (N_rand*N_samples, 3)
        pts_ = self.pos_enc(pts_)  # (N_rand*N_samples, 63)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])  # (N_rand, N_samples, 63)
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)  # (N_rand, N_samples, 63)
        embed = torch.cat([pts_, viewdirs_], dim=-1)  # (N_rand, N_samples, 126)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)  # both (N_rand, N_samples, 63)

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)  # (N_rand, N_samples, num_source_views, 64)
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]  # (N_rand, N_samples, 64)

        # transformer modules
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            aliasing_filter = False
            if self.aliasing_filter is True and i < 2:
                aliasing_filter = True

            q = crosstrans(q, rgb_feat, ray_diff, mask, aliasing_filter, self.aliasing_filter_type)  # (N_rand, N_samples, 64)
            # embed positional information
            if i % 2 == 0:
                q = torch.cat((q, input_pts, input_views), dim=-1)  # (N_rand, N_samples, 190)  190 = 64+63+63
                q = q_fc(q)  # (N_rand, N_samples, 64)
            # ray transformer
            q = selftrans(q, ret_attn=self.ret_alpha)
            # 'learned' density
            if self.ret_alpha:
                q, attn = q
        # normalize & rgb
        h = self.norm(q)
        outputs = self.rgb_fc(h.mean(dim=1))  # (N_rand, 3)
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs

    'LinGaoyuan_operation_20240930: create the version of forward() function which use clip latent code'
    def forward_clip(self, rgb_feat, ray_diff, mask, pts, ray_d, latent_code=None):
        # compute positional embeddings
        viewdirs = ray_d  # (N_rand, 3)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # (N_rand, 3)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # (N_rand, 3)
        viewdirs = self.view_enc(viewdirs)  # (N_rand, 63)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()  # (N_rand*N_samples, 3)
        pts_ = self.pos_enc(pts_)  # (N_rand*N_samples, 63)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])  # (N_rand, N_samples, 63)
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)  # (N_rand, N_samples, 63)
        embed = torch.cat([pts_, viewdirs_], dim=-1)  # (N_rand, N_samples, 126)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)  # both (N_rand, N_samples, 63)

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)  # (N_rand, N_samples, num_source_views, 64)
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]  # (N_rand, N_samples, 64)

        'LinGaoyuan_20240930: create clip shape code and appearance code, the size of latent_code should be (128) or (2,128)'
        if (latent_code.shape)[0] == 2:
            clip_shape_code = latent_code[0,:]
            clip_appearance_code = latent_code[1,:]
        else:
            clip_shape_code = None
            clip_appearance_code = latent_code.unsqueeze(0)  # we want (128) -> (1,128)

        # transformer modules
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            aliasing_filter = False
            if self.aliasing_filter is True and i < 2:
                aliasing_filter = True

            q = crosstrans(q, rgb_feat, ray_diff, mask, aliasing_filter, self.aliasing_filter_type)  # (N_rand, N_samples, 64)
            # embed positional information
            if i % 2 == 0:
                q = torch.cat((q, input_pts, input_views), dim=-1)  # (N_rand, N_samples, 190)  190 = 64+63+63
                q = q_fc(q)  # (N_rand, N_samples, 64)

            'LinGaoyuan_20240930: LinGaoyuan_operation_20240930: determine to whether add the clip shape code to ray_transformer or not based on the size of latent_code'
            # # ray transformer
            # q = selftrans(q, ret_attn=self.ret_alpha)
            if clip_shape_code == None:
                q = selftrans(q, ret_attn=self.ret_alpha)
            else:
                clip_shape_code = clip_shape_code.repeat((q.shape)[0], (q.shape)[1], 1)
                q = torch.cat((q, clip_shape_code), dim=-1)
                q = selftrans(q, ret_attn=self.ret_alpha)

            # 'learned' density
            if self.ret_alpha:
                q, attn = q
        # normalize & rgb
        h = self.norm(q)

        'LinGaoyuan_20240930: cat clip radiance code as the input of self.rgb_fc()'
        clip_appearance_code = clip_appearance_code.repeat((h.shape)[0], (h.shape)[1], 1)
        h = torch.cat((h, clip_appearance_code), dim=-1)

        outputs = self.rgb_fc(h.mean(dim=1))  # (N_rand, 3)

        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs


class GNT_prop(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False):
        super(GNT_prop, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        # NOTE: Apologies for the confusing naming scheme, here view_crosstrans refers to the view transformer, while the view_selftrans refers to the ray transformer
        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])
        for i in range(args.trans_depth/2):  # trans_depth = 4 in config file
            # view transformer
            view_trans = Transformer2D(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_crosstrans.append(view_trans)
            # ray transformer
            ray_trans = Transformer(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_selftrans.append(ray_trans)
            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth + posenc_dim + viewenc_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)

        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(args.netwidth)
        self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

        self.aliasing_filter = args.aliasing_filter
        self.aliasing_filter_type = args.aliasing_filter_type

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d):
        # compute positional embeddings
        viewdirs = ray_d  # (N_rand, 63)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # (N_rand, 63)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # (N_rand, 63)
        viewdirs = self.view_enc(viewdirs)  # (N_rand, 63)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()  # (N_rand*N_samples, 3)
        pts_ = self.pos_enc(pts_)  # (N_rand*N_samples, 63)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])  # (N_rand, N_samples, 63)
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)  # (N_rand, N_samples, 63)
        embed = torch.cat([pts_, viewdirs_], dim=-1)  # (N_rand, N_samples, 126)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)  # both (N_rand, N_samples, 63)

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)  # (N_rand, N_samples, num_source_views, 64)
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]  # (N_rand, N_samples, 64)

        # transformer modules
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            aliasing_filter = False
            if self.aliasing_filter is True and i < 2:
                aliasing_filter = True

            q = crosstrans(q, rgb_feat, ray_diff, mask, aliasing_filter, self.aliasing_filter_type)  # (N_rand, N_samples, 64)
            # embed positional information
            if i % 2 == 0:
                q = torch.cat((q, input_pts, input_views), dim=-1)  # (N_rand, N_samples, 190)  190 = 64+63+63
                q = q_fc(q)  # (N_rand, N_samples, 64)
            # ray transformer
            q = selftrans(q, ret_attn=self.ret_alpha)
            # 'learned' density
            if self.ret_alpha:
                q, attn = q
        # normalize & rgb
        h = self.norm(q)
        outputs = self.rgb_fc(h.mean(dim=1))  # (N_rand, 3)
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs


