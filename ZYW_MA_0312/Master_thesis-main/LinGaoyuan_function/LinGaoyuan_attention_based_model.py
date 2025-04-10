import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate = 0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x

'cross-attention'
class View_attention(nn.Module):
    def __init__(self, input_dim = 512, embed_dim = None, dim_q = 512, dim_k = 512, num_head=1, dropout=0.1, arm = None, arm_type = None):
        super(self).__init__()

        self.input_dim = input_dim

        if embed_dim == None:
            self.embed_dim = self.input_dim
        else:
            self.embed_dim = embed_dim

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = self.embed_dim // self.num_head
        self.dim_k = dim_k
        self.dim_q = dim_q

        '''
        根据Transformers这个GitHub库(https://huggingface.co/docs/transformers/main/en/tasks/image_feature_extraction)里的关于q,k,v层的代码基本上都是输入输出同shape的连接层.
        但也有输入为self.input_dim, 输出为self.num_head*self.head_dim, 即不一定输入的维度就是embed_dim, 也可以是一个升维的连接层.
        即 self.q_fc = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=False)
        '''
        self.q_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.pos_fc = nn.Sequential(
            nn.Linear(4, embed_dim // 8),
            nn.ReLU(),
            nn.Linear(embed_dim // 8, embed_dim),
        )

        self.scaling = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.input_dim, eps=1e-6)

        self.out_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        'arm是论文: Blending Anti-Aliasing into Vision Transformer 中负责 anti aliasing 的 Aliasing-Reduction module'
        if arm_type is not None:
            self.arm_type = arm_type  # gaussian_filter or filter_bank or cnn
            self.arm = arm  # 在是gaussian_filter或者filter_bank的情况下, self.arm应该是一个矩阵, 若是cnn, 则应该是一个cnn网络

    '对应很多attention网络模型的代码中q,k,v经过全连接层后permute(0,2,1,3)的操作, 无论如何最终的shape都为(batch, num_head, seq_length, head_dim)'
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, q, k, v, pos,
                attention_mask: Optional[torch.Tensor] = None,
                layer_head_mask: Optional[torch.Tensor] = None
                ):

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        batch, length_q, _ = q.size()
        batch, length_k, _ = k.size()
        batch, length_v, _ = v.size()

        residual = q

        q = self.layer_norm(q)

        # q = self.q_fc(q).view(batch, length_q, self.num_head, self.dim_k)
        # k = self.k_fc(k).view(batch, length_k, self.num_head, self.dim_k)
        # v = self.v_fc(v).view(batch, length_v, self.num_head, self.dim_v)
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(v)

        pos = self.pos_fc(pos)

        q = self._shape(q, -1, batch)  # b x n x lq x dk
        k = self._shape(k, -1, batch)  # b x n x lq x dk
        v = self._shape(v, -1, batch)  # b x n x lq x dk

        proj_shape = (batch * self.num_heads, -1, self.head_dim)

        q = q.reshape(*proj_shape)  # (b*n) x lq x dk
        k = k.reshape(*proj_shape)  # (b*n) x lq x dk
        v = v.reshape(*proj_shape)  # (b*n) x lq x dk

        # q = q.permute(2, 0, 1, 3).contiguous().view(-1, length_q, self.dim_k) # (n*b) x lq x dk
        # k = k.permute(2, 0, 1, 3).contiguous().view(-1, length_k, self.dim_k) # (n*b) x lk x dk
        # v = v.permute(2, 0, 1, 3).contiguous().view(-1, length_v, self.dim_v) # (n*b) x lv x dv

        src_len = k.size(1)

        '根据原论文公式, 可以先q*k再除np.power(self.dim_head, 0.5), 也可以先用q除self.scaling再乘k, 分别对应一下第一和第二个代码'
        # attn = torch.matmul(q, k.transpose(1, 2)) / np.power(self.dim_head, 0.5)
        attn = torch.matmul(q, k.transpose(1, 2)) * self.scaling  # 这里 *self.scaling 对应 /np.power(self.dim_head, 0.5)

        if attention_mask is not None:
            mask = attention_mask.repeat(self.num_head, 1, 1)  # (self.num_head * batch_input) x .. x ..
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)

        '其他attention_based的模型中有关于attention_mask的代码, 暂时不知道和之前先的有什么区别, 先加在此处'
        # if attention_mask is not None:
        #     if attention_mask.size() != (batch, 1, length_q, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(batch, 1, length_q, src_len)}, but is {attention_mask.size()}"
        #         )
        #     attn = attn.view(batch, self.num_heads, length_q, src_len) + attention_mask
        #     attn = attn.view(batch * self.num_heads, length_q, src_len)


        '''
        其他attention_based的模型中有关于layer_head_mask的代码
        layer_head_mask是用来判断在multi-head-attention模型中指定哪个head-attention有效, 哪个head-attention无效的mask
        暂时应该用不上, 但代码先加在这里.
        '''
        # if layer_head_mask is not None:
        #     if layer_head_mask.size() != (self.num_heads,):
        #         raise ValueError(
        #             f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
        #             f" {layer_head_mask.size()}"
        #         )
        #     attn = layer_head_mask.view(1, -1, 1, 1) * attn.view(batch, self.num_heads, length_q, src_len)
        #     attn = attn.view(batch * self.num_heads, length_q, src_len)


        attn = self.dropout(attn)

        output = torch.matmul(attn, (v + pos))

        output = output.view(batch, self.num_head, length_q, self.head_dim)
        output = output.permute(1, 2)  # batch * length_q * (num_head * self.dim_v)
        output = output.reshape(batch, length_q, self.embed_dim)

        '原版的GNT模型在输出层这没有用激活函数, 未来我们自己的研究要考虑是否加激活函数'
        output = self.out_fc(output)

        '判断是否执行anti-aliasing模块'
        if self.arm_type is not None:
            if self.arm_type == 'gaussian_filter' or self.arm_type == 'filter_bank':
                output = output * self.arm
            elif self.arm_type == 'cnn':
                output = self.arm(output)

        output = output + residual

        return output

    class view_transformer(nn.Module):
        def __init__(
                self, input_dim = 512, embed_dim = None, input_embedding = True, num_head = 1, dropout=0.1, ff_hid_dim = None, ff_dp_rate = 0.1
        ):
            super(self).__init__()
            self.input_dim = input_dim

            if embed_dim == None:
                self.embed_dim = self.input_dim
            else:
                self.embed_dim = embed_dim

            self.ff_dp_rate = ff_dp_rate

            if ff_hid_dim is None:
                self.ff_hid_dim = self.embed_dim
            else:
                self.ff_hid_dim = ff_hid_dim

            self.ff = FeedForward(self.embed_dim, self.ff_hid_dim, ff_dp_rate)

            self.ff_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

            self.attn = View_attention(input_dim, embed_dim, input_embedding, num_head, dropout)

        def forward(self, q, k, v, pos,
                    attention_mask: Optional[torch.Tensor] = None,
                    layer_head_mask: Optional[torch.Tensor] = None,
                    ):
            x = self.attn(q, k, v, pos, attention_mask, layer_head_mask)

            ' 这个过程对应论文Fig 2中间的那个Feed-Forward Network的结构图 '
            residue = x
            x = self.ff_norm(x)
            x = self.ff(x)
            x = x + residue

            return x


'self-attention, 和原版gnt模型的不同是我这这个模型里就直接进行 +residul 的操作, 原版是在transformer中加'
class ray_attention(nn.Module):
    def __init__(self, input_dim = 512, embed_dim = None, input_embedding = True, num_head = 1, dropout=0.1, ff_hid_dim = None, ff_dp_rate = 0.1):
        super().__init__()

        self.input_dim = input_dim

        self.num_head = num_head

        if embed_dim == None:
            self.embed_dim = self.input_dim
        else:
            self.embed_dim = embed_dim

        self.head_dim = self.embed_dim // self.num_head

        self.input_embedding = input_embedding
        if self.input_embedding is True:
            self.embed_fc = nn.Linear(self.input_dim, self.embed_dim)

        '''
        根据Transformers这个GitHub库(https://huggingface.co/docs/transformers/main/en/tasks/image_feature_extraction)里的关于q,k,v层的代码基本上都是输入输出同shape的连接层.
        但也有输入为self.input_dim, 输出为self.num_head*self.head_dim, 即不一定输入的维度就是embed_dim, 也可以是一个升维的连接层.
        即 self.q_fc = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=False)
        '''
        self.q_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.scaling = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.input_dim, eps=1e-6)

        self.out_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    '对应很多attention网络模型的代码中q,k,v经过全连接层后permute(0,2,1,3)的操作, 无论如何最终的shape都为(batch, num_head, seq_length, head_dim)'
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, input,
                attention_mask: Optional[torch.Tensor] = None,
                layer_head_mask: Optional[torch.Tensor] = None,
                use_original_gnt_model: bool = True,
                input_pos_emd = None,
                input_views_emd = None
                ):
        ''
        '原GNT中位置和方向的embedding之后的在整体GNT模型中已经合并到input中了, 所以在attention模型中并没有包含合并的代码, 这里也只是预留一个接口, 可忽略这段代码'
        if input_pos_emd is not None:
            input = torch.cat((input, input_pos_emd), dim=-1)
        if input_views_emd is not None:
            input = torch.cat((input, input_views_emd), dim=-1)

        input = input.to(torch.float32)

        if len(input.shape) == 2:
            input = input.unsqueeze(0)

        if self.input_embedding is True:
            input = self.embed_fc(input)
        residual = input
        input = self.layer_norm(input)

        batch, seq_length, _ = input.size()

        if use_original_gnt_model is True:
            q = self.q_fc(input)
            q = q.view(batch, seq_length, self.n_heads, -1).permute(0, 2, 1, 3)
            k = self.k_fc(input)
            k = k.view(batch, seq_length, self.n_heads, -1).permute(0, 2, 1, 3)
            v = self.v_fc(input)
            v = v.view(batch, seq_length, self.n_heads, -1).permute(0, 2, 1, 3)

            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
            out = out.view(batch, seq_length, -1)
            out = self.dropout(self.out_fc(out))

            output = out + residual


        else:
            q = self.q_fc(input)
            k = self.k_fc(input)
            v = self.v_fc(input)

            q = self._shape(q, -1, batch)  # b x n x lq x dk
            k = self._shape(k, -1, batch)  # b x n x lq x dk
            v = self._shape(v, -1, batch)  # b x n x lq x dk

            proj_shape = (batch * self.num_heads, -1, self.head_dim)

            q = q.reshape(*proj_shape)  # (b*n) x lq x dk
            k = k.reshape(*proj_shape)  # (b*n) x lq x dk
            v = v.reshape(*proj_shape)  # (b*n) x lq x dk

            src_len = k.size(1)

            '根据原论文公式, 可以先q*k再除np.power(self.dim_head, 0.5), 也可以先用q除self.scaling再乘k, 分别对应一下第一和第二个代码'
            # (batch_size, num_heads, len_q, dim_head) @ (batch_size, num_heads, dim_head, len_k) -> (batch_size, num_heads, len_q, len_k)
            # attn = torch.matmul(q, k.transpose(1, 2)) / np.power(self.dim_head, 0.5)
            attn = torch.matmul(q, k.transpose(1, 2)) * self.scaling  # 这里算的 self.scaling 对应 np.power(self.dim_head, 0.5)

            if attn.size() != (batch * self.num_heads, seq_length, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(batch * self.num_heads, seq_length, src_len)}, but is"
                    f" {attn.size()}"
                )

            if attention_mask is not None:
                mask = attention_mask.repeat(self.num_head, 1, 1)  # (self.num_head * batch_input) x .. x ..
                attn = attn.masked_fill(mask, -np.inf)
            attn = self.softmax(attn)

            '其他attention_based的模型中有关于attention_mask的代码, 暂时不知道和之前先的有什么区别, 先加在此处'
            # if attention_mask is not None:
            #     if attention_mask.size() != (batch, 1, length_q, src_len):
            #         raise ValueError(
            #             f"Attention mask should be of size {(batch, 1, length_q, src_len)}, but is {attention_mask.size()}"
            #         )
            #     attn = attn.view(batch, self.num_heads, length_q, src_len) + attention_mask
            #     attn = attn.view(batch * self.num_heads, length_q, src_len)


            '''
            其他attention_based的模型中有关于layer_head_mask的代码
            layer_head_mask是用来判断在multi-head-attention模型中指定哪个head-attention有效, 哪个head-attention无效的mask
            暂时应该用不上, 但代码先加在这里.
            '''
            # if layer_head_mask is not None:
            #     if layer_head_mask.size() != (self.num_heads,):
            #         raise ValueError(
            #             f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
            #             f" {layer_head_mask.size()}"
            #         )
            #     attn = layer_head_mask.view(1, -1, 1, 1) * attn.view(batch, self.num_heads, length_q, src_len)
            #     attn = attn.view(batch * self.num_heads, length_q, src_len)


            attn = self.dropout(attn)

            # output = torch.bmm(attn, v)
            output = torch.matmul(attn, v)

            output = output.view(batch, self.num_head, seq_length, self.head_dim)
            output = output.permute(1, 2)  # batch * length_q * (num_head * self.dim_v)
            output = output.reshape(batch, seq_length, self.embed_dim)

            '原版的GNT模型在输出层这没有用激活函数, 未来我们自己的研究要考虑是否加激活函数'
            output = self.dropout(self.out_fc(output))

            output = output + residual


        return output


    class ray_transformer(nn.Module):
        def __init__(
                self, input_dim = 512, embed_dim = None, input_embedding = True, num_head = 1, dropout=0.1, ff_hid_dim = None, ff_dp_rate = 0.1
        ):
            super(self).__init__()
            self.input_dim = input_dim

            if embed_dim == None:
                self.embed_dim = self.input_dim
            else:
                self.embed_dim = embed_dim

            self.ff_dp_rate = ff_dp_rate

            if ff_hid_dim is None:
                self.ff_hid_dim = self.embed_dim
            else:
                self.ff_hid_dim = ff_hid_dim

            self.ff = FeedForward(self.embed_dim, self.ff_hid_dim, ff_dp_rate)

            self.ff_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

            self.attn = ray_attention(input_dim, embed_dim, input_embedding, num_head, dropout, ff_hid_dim, ff_dp_rate)

        def forward(self, input,
                    attention_mask: Optional[torch.Tensor] = None,
                    layer_head_mask: Optional[torch.Tensor] = None,
                    ):
            x = self.attn(input)

            ' 这个过程对应论文Fig 2中间的那个Feed-Forward Network的结构图 '

            residue = x
            x = self.ff_norm(x)
            x = self.ff(x)
            x = x + residue

            return x


'''
来自GNT模型中的对位置和方向信息进行embedding的代码, 在clip-nerf中也有一模一样的代码, 用于输入到ray-transformer中的pos_embed和view_direction_embed.
注意合并原始input,pos_embed和view_direction_embed需要用到torch.cat()函数而不是直接相加, 同时注意输入维度也需要额外调整, 这时候的输入维度就不一定等于attention模型中默认设置的512了
'''
# class Embedder(nn.Module):
#     def __init__(self, **kwargs):
#         super(Embedder, self).__init__()
#         self.kwargs = kwargs
#         self.create_embedding_fn()
#
#     def create_embedding_fn(self):
#         embed_fns = []
#         d = self.kwargs["input_dims"]
#         out_dim = 0
#         if self.kwargs["include_input"]:
#             #猜测: 这里调用lambda应该是定义一个函数，在下面forward()中调用fn(inputs)时的inputs就是现在定义lambda中的x，到时候就直接将inputs的值加入到embed_fns中
#             embed_fns.append(lambda x: x)
#             out_dim += d
#
#         max_freq = self.kwargs["max_freq_log2"]
#         N_freqs = self.kwargs["num_freqs"]
#
#         if self.kwargs["log_sampling"]:
#             freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
#         else:
#             freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
#
#         for freq in freq_bands:
#             for p_fn in self.kwargs["periodic_fns"]:
#                 embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
#                 out_dim += d
#
#         # self.embved_fns的维度应该 = 最上面的append(lambda x: x)中创建的一层 + 下面两轮for循环中创建的层数
#         self.embed_fns = embed_fns
#         self.out_dim = out_dim
#
#     def forward(self, inputs):
#         return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


'''
todo: 整体模型的代码还没有写, 因为整体模型要考虑view transformer的输入, 在整体模型中要调用两个transformer并对position和view direction进行embedding操作并加入到ray transformer的input中

'''





