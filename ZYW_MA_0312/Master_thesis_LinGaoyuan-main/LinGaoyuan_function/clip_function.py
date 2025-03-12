import torch
import torch.nn as nn
import numpy as np

class mapper(nn.Module):
    def __init__(self,
                 input_dim = 512,
                 hid_layers = [128, 256, 128],
                 activation='Relu',
                 ):
        super().__init__()
        self.activation_dict = {
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'softplus': nn.Softplus,
            }

        self.activation = self.activation_dict[activation.lower()]

        mapper_net = []
        mapper_net.append(nn.Linear(input_dim, hid_layers[0]))
        for i in range(len(hid_layers)-1):
            mapper_net.append(nn.Linear(hid_layers[i], hid_layers[i + 1]))
            mapper_net.append(self.activation())
        self.mapper_net = nn.Sequential(*mapper_net)

    def forward(self, clip_feature):
        if len(clip_feature.shape) == 2:
            clip_feature = clip_feature.squeeze()
        latent_code = self.mapper_net(clip_feature)

        return latent_code


class deformation(nn.Module):
    def __init__(self,
                 input_dim = 128,
                 hid_layers = [256, 256, 256, 256],
                 output_dim = 128,
                 activation='Relu',
                 output_activation = 'Tanh',
                 ):
        super().__init__()
        self.activation_dict = {
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'softplus': nn.Softplus,
            'tanh' : nn.Tanh,
            }

        self.activation = self.activation_dict[activation.lower()]
        self.output_activation = self.activation_dict[output_activation.lower()]

        deformation_net = []
        deformation_net.append(nn.Linear(input_dim, hid_layers[0]))
        for i in range(len(hid_layers)-1):
            deformation_net.append(nn.Linear(hid_layers[i], hid_layers[i + 1]))
            deformation_net.append(self.activation())

        deformation_net.append(nn.Linear(hid_layers[-1], output_dim))
        deformation_net.append(self.output_activation())
        self.deformation_net = nn.Sequential(*deformation_net)

    def forward(self, latent_code):
        if len(latent_code.shape) == 2:
            latent_code = latent_code.squeeze()
        output = self.deformation_net(latent_code)

        return output


class mapper_attention(nn.Module):
    def __init__(self, input_dim=512, embed_dim=None, input_embedding=False, num_head=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_head = num_head
        if embed_dim == None:
            self.embed_dim = self.input_dim
        else:
            self.embed_dim = embed_dim
        self.dim_head = self.embed_dim // self.num_head

        if input_embedding is True:
            self.embed_fc = nn.Linear(self.input_dim, self.embed_dim)

        self.q_fc = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.k_fc = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.v_fc = nn.Linear(self.input_dim, self.embed_dim, bias=False)

        self.layer_norm = nn.LayerNorm(self.input_dim)

        self.out_fc = nn.Linear(self.embed_dim, self.input_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, mask=None):

        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        residual = input
        input = self.layer_norm(input)

        'clip image_encoder map each image to feature with shape (1,512)or(1,768), so the shape of input is (batch_size, 1, 512) or (batch_size, 1, 768)'
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

        output = output.view(self.num_head, input.shape[0], input.shape[1], self.dim_head)
        # output = output.permute(1, 2, 0, 3)
        output = output.permute(0, 2, 1, 3)

        output = self.dropout(self.out_fc(output))
        output = self.layer_norm(output + residual)

        return output


'LinGaoyuan_20240924: both clip-nerf and model_and_model_component use this Embedder for position embedding'
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


def Loss_clip(clip_model, clip_preprocess, input_img, rendered_img, device):
    input_img = clip_preprocess(input_img).unsqueeze(0).to(device)
    rendered_img = clip_preprocess(rendered_img).unsqueeze(0).to(device)
    with torch.no_grad():
        input_img_feature = clip_model.encode_image(input_img)
        rendered_img_feature = clip_model.encode_image(rendered_img)
    input_img_feature /= input_img_feature.norm(dim=-1, keepdim=True)
    rendered_img_feature /= rendered_img_feature.norm(dim=-1, keepdim=True)
    loss_clip = 1 - (input_img_feature @ rendered_img_feature.T)

    return loss_clip

def Loss_clip_version_2(clip_model, clip_preprocess, input_img_feature, rendered_img, device):
    rendered_img = clip_preprocess(rendered_img).unsqueeze(0).to(device)
    with torch.no_grad():
        rendered_img_feature = clip_model.encode_image(rendered_img)
    # input_img_feature /= input_img_feature.norm(dim=-1, keepdim=True)
    rendered_img_feature /= rendered_img_feature.norm(dim=-1, keepdim=True)
    loss_clip = 1 - (input_img_feature @ rendered_img_feature.T)

    return loss_clip