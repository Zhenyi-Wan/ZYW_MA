import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import (rearrange, reduce, repeat)

from LinGaoyuan_function.ReTR_function.ReTR_grid_sample import grid_sample_2d, grid_sample_3d
from LinGaoyuan_function.ReTR_function.ReTR_transformer import LocalFeatureTransformer
from LinGaoyuan_function.ReTR_function.ReTR_cnn2d import ResidualBlock
import math


class ZYW_ReTR_model(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False, use_volume_feature = False):
        super().__init__()

        self.args = args
        self.in_feat_ch = in_feat_ch
        self.posenc_dim = posenc_dim
        self.ret_alpha = ret_alpha
        self.PE_d_hid = 8
        self.use_volume_feature = use_volume_feature

        'LinGaoyuan_operation_20241023: add anti-aliasing module to retr model'
        self.aliasing_filter = args.aliasing_filter
        self.aliasing_filter_type = args.aliasing_filter_type

        'LinGaoyuan_20240915: 3 Transformer network in ReTR'
        self.view_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch, nhead=8, layer_names=['self'],
                                                        attention='linear')

        # Zhenyi Wan [2025/3/12] add four new(view) transformers
        #self.metallic_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch, nhead=8, layer_names=['self'],
        #                                                    attention='linear')
        #self.roughnesss_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch, nhead=8, layer_names=['self'],
        #                                                      attention='linear')
        #self.albedo_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch, nhead=8, layer_names=['self'],
        #                                                  attention='linear')
        #self.normals_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch, nhead=8, layer_names=['self'],
        #                                                   attention='linear')
        self.brdf_attributes = ['metallic', 'roughness', 'albedo', 'normals']
        self.brdf_transformers = nn.ModuleDict({
            name: LocalFeatureTransformer(d_model=self.in_feat_ch, nhead=8,
                                          layer_names=['self'], attention='linear')
            for name in self.brdf_attributes
        })

        if self.use_volume_feature and self.args.use_retr_feature_extractor:
            self.occu_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch * 2 + self.PE_d_hid, nhead=8, layer_names=['self'],
                                                            attention='full')

            # Zhenyi Wan [2025/3/12] every new view transformer needs a specific occlusion transformer
            self.brdf_occ_transformers = nn.ModuleDict({
                name: LocalFeatureTransformer(
                    d_model=self.in_feat_ch * 2 + self.PE_d_hid,
                    nhead=8,
                    layer_names=['self'],
                    attention='full'
                )
                for name in self.brdf_attributes
            })

            self.ray_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch * 2 + self.PE_d_hid, nhead=1,
                                                           layer_names=['cross'],
                                                           attention='full')
            # Zhenyi Wan [2025/3/12] add four ray transformers to generate the material, diffuse color and normals features.
            # Inorder to better understand, rename the ray transformer as decoder(we do not output radiance)
            #self.metallic_decoder = LocalFeatureTransformer(
            #    d_model=self.in_feat_ch * 2 + self.PE_d_hid, nhead=1, layer_names=['cross'],
            #    attention='full')
            #self.roughness_decoder = LocalFeatureTransformer(
            #    d_model=self.in_feat_ch * 2 + self.PE_d_hid, nhead=1, layer_names=['cross'],
            #    attention='full')
            #self.albedo_decoder = LocalFeatureTransformer(
            #    d_model=self.in_feat_ch * 2 + self.PE_d_hid, nhead=1, layer_names=['cross'],
            #    attention='full')
            #self.normals_decoder = LocalFeatureTransformer(
            #    d_model=self.in_feat_ch * 2 + self.PE_d_hid, nhead=1, layer_names=['cross'],
            #    attention='full')

            self.brdf_decoders = nn.ModuleDict({
                name: LocalFeatureTransformer(
                    d_model=self.in_feat_ch * 2 + self.PE_d_hid,
                    nhead=1,
                    layer_names=['cross'],
                    attention='full'
                )
                for name in self.brdf_attributes
            })

            'LinGaoyuan_20240915: MLP network in ReTR'
            # Zhenyi Wan [2025/3/12] This MLP is used for radiance output, albedo and normals maps are both 3 channels
            # maps. So keep it and used both for albedo and normals decoder(3 channels decoder)
            self.RadianceMLP = nn.Sequential(
                nn.Linear(self.in_feat_ch * 2 + self.PE_d_hid, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 3)
            )
            # Zhenyi Wan [2025/3/12] metallic and roughness maps(material maps) have only 1 channel. Add a one channel
            # called MaterialMLP to generate 1 channel map(1 channel decoder)
            self.MaterialMLP = nn.Sequential(
                nn.Linear(
                    self.in_feat_ch * 2 + self.PE_d_hid, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1)
            )

            self.linear_radianceweight_1_softmax = nn.Sequential(
                nn.Linear(self.in_feat_ch+3, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 8), nn.ReLU(inplace=True),
                nn.Linear(8, 1),
            )
            self.RadianceToken = ViewTokenNetwork(dim=self.in_feat_ch * 2 + self.PE_d_hid)
        else:
            self.occu_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch + self.PE_d_hid, nhead=8, layer_names=['self'],
                                                            attention='full')

            # Zhenyi Wan [2025/3/12] every new view transformer needs a specific occlusion transformer
            self.brdf_occ_transformers = nn.ModuleDict({
                name: LocalFeatureTransformer(
                    d_model=self.in_feat_ch + self.PE_d_hid,
                    nhead=8,
                    layer_names=['self'],
                    attention='full'
                )
                for name in self.brdf_attributes
            })

            self.ray_transformer = LocalFeatureTransformer(d_model=self.in_feat_ch + self.PE_d_hid, nhead=1, layer_names=['cross'],
                                                           attention='full')

            # Zhenyi Wan [2025/3/12] add four ray transformers to generate the material, diffuse color and normals features.
            # Inorder to better understand, rename the ray transformer as decoder(we do not output radiance)
            self.brdf_decoders = nn.ModuleDict({
                name: LocalFeatureTransformer(
                    d_model=self.in_feat_ch + self.PE_d_hid,
                    nhead=1,
                    layer_names=['cross'],
                    attention='full'
                )
                for name in self.brdf_attributes
            })

            'LinGaoyuan_20240915: MLP network in ReTR'
            self.RadianceMLP = nn.Sequential(
                nn.Linear(self.in_feat_ch + self.PE_d_hid, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 3)
            )
            # Zhenyi Wan [2025/3/20] material map and roughness map are 1 channel map
            self.MaterialMLP = nn.Sequential(
                nn.Linear(
                    self.in_feat_ch + self.PE_d_hid, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1)
            )
            'LinGaoyuan_operation_20240917: change the input dim of self.linear_radianceweight_1_softmax to self.in_feat_ch+3 when use_volume_feature is False'
            self.linear_radianceweight_1_softmax = nn.Sequential(
                nn.Linear(self.in_feat_ch+3, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 8), nn.ReLU(inplace=True),
                nn.Linear(8, 1),
            )
            self.RadianceToken = ViewTokenNetwork(dim=self.in_feat_ch + self.PE_d_hid)

        self.fuse_layer = nn.Linear(self.in_feat_ch + 3, self.in_feat_ch)

        self.rgbfeat_fc = nn.Linear(self.in_feat_ch + 3, self.in_feat_ch)

        self.softmax = nn.Softmax(dim=-2)

        self.div_term = torch.exp((torch.arange(0, self.PE_d_hid, 2, dtype=torch.float) *
                            -(math.log(10000.0) / self.PE_d_hid)))

    def order_posenc(self, z_vals):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.PE_d_hid  % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(self.PE_d_hid))
        pe = torch.zeros(z_vals.shape[0],z_vals.shape[1], self.PE_d_hid).to(z_vals.device)
        position = z_vals.unsqueeze(-1)
        # div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
        #                     -(math.log(10000.0) / d_model))).to(z_vals.device)
        pe[:, :, 0::2] = torch.sin(position.float() * self.div_term.to(z_vals.device))
        pe[:, :, 1::2] = torch.cos(position.float() * self.div_term.to(z_vals.device))
        return pe

    # Zhenyi Wan [2025/3/14] create a upper triangle matrix of size(i, n_samples+1, n_samples+1)
    def get_attn_mask(self, num_points):
        mask = (torch.triu(torch.ones(1, num_points+1, num_points+1)) == 1).transpose(1, 2)
    #    mask[:,0, 1:] = 0
        return mask

    'LinGaoyuan_20240930: retr model + model_and_model_component feature extractor or retr model + retr feature extractor'
    def forward(self, point3D, ray_batch, source_imgs_feat, z_vals, mask, ray_d, ray_diff, fea_volume=None, ret_alpha=True):

        B, n_views, H, W, rgb_channel = ray_batch['src_rgbs'].shape  # LinGaoyuan_20240916: B = Batch NV = num_source_views
        N_rand, N_samples, _ = point3D.shape

        img_rgb_sampled = source_imgs_feat[:, :, :, :3]  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 3)
        img_feat_sampled = source_imgs_feat[:,:,:,3:]  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 32)
        dir_relative = ray_diff[:,:,:,:3]

        # input_view = self.rgbfeat_fc(source_imgs_feat)  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 35) -> (N_rand, N_samples, n_views, 32)
        input_view = img_feat_sampled  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 32)
        input_view = rearrange(input_view, 'N_rand N_samples n_views C -> (N_rand N_samples) n_views C')

        output_view = self.view_transformer(input_view, aliasing_filter=self.aliasing_filter, aliasing_filter_type=self.aliasing_filter_type)

        # output_view = output_view.permute(2,0,1,3)  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 32) -> (n_views, N_rand, N_samples, 32)

        # input_occ = output_view[:,:,0,:]  # LinGaoyuan_20240916: (N_rand, N_samples, 1, 32)
        # view_feature = output_view[:,:,1:,:]  # LinGaoyuan_20240916: (N_rand, N_samples, NV - 1, 32)
        view_feature = output_view  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 32)
        view_feature = rearrange(view_feature, '(N_rand N_samples) n_views C -> N_rand N_samples n_views C', N_rand = N_rand, N_samples = N_samples)

        x_weight = torch.cat([view_feature, dir_relative], dim=-1)  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 35)
        x_weight = self.linear_radianceweight_1_softmax(x_weight)  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 1)
        if x_weight.dtype == torch.float32:
            x_weight[mask==0] = -1e9
        else:
            x_weight[mask==0] = -1e4
        weight = self.softmax(x_weight)  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 1)

        source_imgs_feat = rearrange(source_imgs_feat, "N_rand N_samples n_views C -> n_views C N_rand N_samples")#(N_views, 35, N_rand，N_samples)
        radiance = (source_imgs_feat * rearrange(weight,"N_rand N_samples n_views 1 -> n_views 1 N_rand N_samples", N_rand=N_rand, N_samples = N_samples)).sum(axis=0)
        radiance = rearrange(radiance, "DimRGB N_rand N_samples -> N_rand N_samples DimRGB")#(N_rand N_samples 35)

        attn_mask = self.get_attn_mask(N_samples).type_as(radiance)
        input_occ = torch.cat((self.fuse_layer(radiance), self.order_posenc(100 * z_vals.reshape(-1,z_vals.shape[-1])).type_as(radiance)), dim=-1)#(N_rand N_samples 34)
        radiance_tokens = self.RadianceToken(input_occ).unsqueeze(1)
        input_occ = torch.cat((radiance_tokens, input_occ), dim=1)

        output_occ = self.occu_transformer(input_occ)

        output_ray = self.ray_transformer(output_occ[:,:1], output_occ[:,1:])
        weight = self.ray_transformer.atten_weight.squeeze()# Zhenyi Wan [2025/3/20] (N_rand, N_samples)

        rgb = torch.sigmoid(self.RadianceMLP(output_ray))

        if len(rgb.shape) == 3:
            rgb = rgb.squeeze()

        if ret_alpha  is True:
            rgb = torch.cat([rgb,weight], dim=1)

        return rgb

    def forward_BDRF(self, point3D, ray_batch, source_imgs_feat, z_vals, mask, ray_d, ray_diff, fea_volume=None, ret_alpha=True):
        B, n_views, H, W, rgb_channel = ray_batch['src_rgbs'].shape  # LinGaoyuan_20240916: B = Batch NV = num_source_views
        N_rand, N_samples, _ = point3D.shape

        img_rgb_sampled = source_imgs_feat[:, :, :, :3]  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 3)
        img_feat_sampled = source_imgs_feat[:, :, :, 3:]  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 32)
        dir_relative = ray_diff[:, :, :, :3]  # (N_rand, N_samples, n_views, 3)

        # input_view = self.rgbfeat_fc(source_imgs_feat)  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 35) -> (N_rand, N_samples, n_views, 32)
        input_view = img_feat_sampled  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 32)
        input_view = rearrange(input_view, 'N_rand N_samples n_views C -> (N_rand N_samples) n_views C')

        def process_brdf_attribute(transformer, input_view):
            # Transformer
            feature = transformer(input_view)
            # Rearrange
            feature = rearrange(feature, '(N_rand N_samples) n_views C -> N_rand N_samples n_views C',
                                N_rand=N_rand, N_samples=N_samples)
            return feature

        #BRDF_features = {
        #    name: process_brdf_attribute(getattr(self, f'{name}_transformer'), input_view)
        #    for name in ['metallic', 'roughness', 'albedo', 'normals']
        #}

        BRDF_features = {
             name: process_brdf_attribute(self.brdf_transformers[name], input_view)
             for name in self.brdf_attributes
       }

        # Zhenyi Wan [2025/3/14] BRDF Features Encoder:Use four transformers as encoder for metallic, roughness, albedo and normals. The inputs are all the same,
        # sampled scr image features.
        #roughness_feature = self.roughnesss_transformer(input_view)  # (N_rand, N_samples, n_views, 32)
        #albedo_feature = self.albedo_transformer(input_view)  # (N_rand, N_samples, n_views, 32)
        #normals_feature = self.normals_transformer(input_view)  # (N_rand, N_samples, n_views, 32)

        # Zhenyi Wan [2025/3/14] rearrange the BRDF features
        #metallic_feature = rearrange(metallic_feature, '(N_rand N_samples) n_views C -> N_rand N_samples n_views C', N_rand = N_rand, N_samples = N_samples)
        #roughness_feature = rearrange(roughness_feature, '(N_rand N_samples) n_views C -> N_rand N_samples n_views C',
        #                             N_rand=N_rand, N_samples=N_samples)
        #albedo_feature = rearrange(albedo_feature, '(N_rand N_samples) n_views C -> N_rand N_samples n_views C',
        #                             N_rand=N_rand, N_samples=N_samples)
        #normals_feature = rearrange(normals_feature, '(N_rand N_samples) n_views C -> N_rand N_samples n_views C',
        #                             N_rand=N_rand, N_samples=N_samples)
        #BRDF_features = [metallic_feature, roughness_feature, albedo_feature, normals_feature]

        # Zhenyi Wan [2025/3/14] BRDF occlusion weights:combine the BRDF features with relative direction (N_rand, N_samples, n_views, 35)
        #BRDF_weights = []
        #for i, BRDF_feature in enumerate(BRDF_features):
        #    BRDF_weight = torch.cat([BRDF_feature, dir_relative], dim=-1)
        #    BRDF_weight = self.linear_radianceweight_1_softmax(BRDF_weight)
        #    if BRDF_weight.dtype == torch.float32:
        #        BRDF_weight[mask == 0] = -1e9
        #    else:
        #        BRDF_weight[mask == 0] = -1e4
        #   BRDF_weight = self.softmax(BRDF_weight)
        #    BRDF_weights.append(BRDF_weight)
        #[metallic_weight, roughness_weight, albedo_weight, normals_weight] = BRDF_weights

        def calculate_brdf_weight(feature, dir_relative, mask):
            weight = torch.cat([feature, dir_relative], dim=-1)
            weight = self.linear_radianceweight_1_softmax(weight)
            weight[mask == 0] = -1e9 if weight.dtype == torch.float32 else -1e4
            return self.softmax(weight)

        BRDF_weights = {
            name: calculate_brdf_weight(feature, dir_relative, mask)
            for name, feature in BRDF_features.items()
        }

        # Zhenyi Wan [2025/3/14] BRDF radiances
        #BRDF_radiances = []
        #for i, BRDF_weight in enumerate(BRDF_weights):
        #    BRDF_radiance = (source_imgs_feat * rearrange(BRDF_weight,
        #                                                  "N_rand N_samples n_views 1 -> n_views 1 N_rand N_samples",
        #                                                  N_rand=N_rand, N_samples=N_samples)).sum(axis=0)
        #    BRDF_radiance = rearrange(BRDF_radiance, "DimRGB N_rand N_samples -> N_rand N_samples DimRGB")
        #    BRDF_radiances.append(BRDF_radiance)
        #[metallic_radiance, roughness_radiance, albedo_radiance, normals_radiance] = BRDF_radiances
        source_imgs_feat = rearrange(source_imgs_feat, "N_rand N_samples n_views C -> n_views C N_rand N_samples")
        
        
        def calculate_brdf_radiance(weight):
            radiance = (source_imgs_feat * rearrange(weight,
                                                     "N_rand N_samples n_views C -> n_views C N_rand N_samples",
                                                     N_rand=N_rand, N_samples=N_samples)).sum(axis=0)
            return rearrange(radiance, "DimRGB N_rand N_samples -> N_rand N_samples DimRGB")
            

        BRDF_radiances = {
            name: calculate_brdf_radiance(weight)
            for name, weight in BRDF_weights.items()
        }

        # Zhenyi Wan [2025/3/14] BRDF occlusion input:
        #BRDFocc_inputs = []
        #BDRFradiance_tokens = []
        #for i, BRDF_radiance in enumerate(BRDF_radiances):
        #    BRDFocc_input = torch.cat((self.fuse_layer(BRDF_radiance), self.order_posenc(100 * z_vals.reshape(-1,z_vals.shape[-1])).type_as(BRDF_radiance)), dim=-1)
        #    BDRFradiance_token = self.RadianceToken(BRDFocc_input).unsqueeze(1)
        #    BRDFocc_input = torch.cat((BDRFradiance_token, BRDFocc_input), dim=1)
        #    BRDFocc_inputs.append(BRDFocc_input)
        #    BDRFradiance_tokens.append(BDRFradiance_token)
        #[metallicocc_input, roughnessocc_input, albedoocc_input, normalsocc_input] = BRDFocc_inputs
        #[metallicradiance_token, roughnessradiance_token, albedoradiance_token, normalsradiance_token] = BDRFradiance_tokens

        def prepare_brdf_occ_input(radiance):
            occ_input = torch.cat((self.fuse_layer(radiance),
                                   self.order_posenc(100 * z_vals.reshape(-1, z_vals.shape[-1])).type_as(radiance)),
                                  dim=-1)
            radiance_token = self.RadianceToken(occ_input).unsqueeze(1)
            return torch.cat((radiance_token, occ_input), dim=1), radiance_token

        BRDFocc_inputs = {}
        BDRFradiance_tokens = {}
        for name, radiance in BRDF_radiances.items():
            BRDFocc_inputs[name], BDRFradiance_tokens[name] = prepare_brdf_occ_input(radiance)

        attn_mask = self.get_attn_mask(N_samples).type_as(BRDF_radiances['metallic'])
        #attn_mask = self.get_attn_mask(N_samples).type_as(
        #    metallic_radiance)  # Zhenyi Wan [2025/3/14] (1, N_samples+1, N_samples+1)     # Zhenyi Wan [2025/3/14] self.fuse_layer(radiance)(N_rand, N_samples, 8)

        # Zhenyi Wan [2025/3/14] BDRF occlusion Tranformer
        #metallic_occ = self.metallic_occu_transformer(metallicocc_input, mask0 = attn_mask)
        #roughness_occ = self.roughness_occu_transformer(roughnessocc_input, mask0 = attn_mask)
        #albedo_occ = self.albedo_occu_transformer(albedoocc_input, mask0 = attn_mask)
        #normals_occ = self.normals_occu_transformer(normalsocc_input, mask0 = attn_mask)

        brdf_occ_results = {
            name: self.brdf_occ_transformers[name](BRDFocc_inputs[name], mask0=attn_mask)
            for name in self.brdf_attributes
        }

        # Zhenyi Wan [2025/3/20] BRDF Decoder
        #output_metallic = self.metallic_decoder(metallic_occ[:,:1], metallic_occ[:,1:])
        #output_roughness = self.roughness_decoder(roughness_occ[:, :1], roughness_occ[:, 1:])
        #output_albedo = self.albedo_decoder(albedo_occ[:, :1], albedo_occ[:, 1:])
        #output_normals = self.normals_decoder(normals_occ[:, :1], normals_occ[:, 1:])

        brdf_outputs = {
            name: self.brdf_decoders[name](
                brdf_occ_results[name][:, :1],
                brdf_occ_results[name][:, 1:]
            )
            for name in self.brdf_attributes
        }

        # Zhenyi Wan [2025/3/20] BRDF weights
        #metallic_weight = self.metallic_decoder.atten_weight.squeeze()
        #roughness_weight = self.roughness_decoder.atten_weight.squeeze()
        #albedo_weight = self.albedo_decoder.atten_weight.squeeze()
        #normals_weight = self.normals_decoder.atten_weight.squeeze()

        brdf_weights = {
            name: self.brdf_decoders[name].atten_weight.squeeze()
            for name in self.brdf_attributes
        }

        # Zhenyi Wan [2025/3/20] BRDF-Buffer
        #metallic = torch.sigmoid(self.MaterialMLP(output_metallic))# Zhenyi Wan [2025/3/20] (N_rand, 1, 1)
        #roughness = torch.sigmoid(self.MaterialMLP(output_roughness))# Zhenyi Wan [2025/3/20] (N_rand, 1, 1)
        #albedo = torch.sigmoid(self.RadianceMLP(output_albedo))# Zhenyi Wan [2025/3/20] (N_rand, 1, 3)
        #normals = torch.sigmoid(self.RadianceMLP(output_normals))# Zhenyi Wan [2025/3/20] (N_rand, 1, 3)

        def process_brdf_output(name, output):
            mlp = self.MaterialMLP if name in ['metallic', 'roughness'] else self.RadianceMLP
            tensor = torch.sigmoid(mlp(output))
            tensor = tensor.squeeze(1)
            if ret_alpha:
                tensor = torch.cat([tensor, brdf_weights[name]], dim=1)
            return tensor

        BRDF_buffer = {
            name: process_brdf_output(name, brdf_outputs[name])
            for name in self.brdf_attributes
        }



        # Zhenyi Wan [2025/3/24] Squeeze the buffer tensor
        #if len(albedo.shape) == 3:
        #    albedo = albedo.squeeze()# Zhenyi Wan [2025/3/24] (N_rand, 3)
        #if len(normals.shape) == 3:
        #    normals = normals.squeeze()# Zhenyi Wan [2025/3/24] (N_rand, 3)
        #if len(metallic.shape) == 1:
        #    metallic = metallic.squeeze(1)# Zhenyi Wan [2025/3/24] (N_rand, 1)
        #if len(roughness.shape) == 1:
        #    roughness = roughness.squeeze(1)# Zhenyi Wan [2025/3/24] (N_rand, 1)

        #if ret_alpha is True:
        #    albedo = torch.cat([albedo, albedo_weight], dim=1)  # Zhenyi Wan [2025/3/24] (N_rand, N_samples+3)
        #    normals = torch.cat([normals, normals_weight], dim=1)  # Zhenyi Wan [2025/3/24] (N_rand, N_samples+3)
        #    metallic = torch.cat([metallic, metallic_weight], dim=1)  # Zhenyi Wan [2025/3/24] (N_rand, N_samples+1)
        #    roughness = torch.cat([roughness, roughness_weight], dim=1)  # Zhenyi Wan [2025/3/24] (N_rand, N_samples+1)

        # Zhenyi Wan [2025/3/24] BRDF outputs stored
        #BRDF_buffer = {
        #        "albedo": albedo,
        #        "normals": normals,
        #        "metallic": metallic,
        #        "roughness": roughness
        #    }
        return BRDF_buffer

    'LinGaoyuan_operation_20240919: create new function to test the combination of retr feature extractor + model_and_model_component projector + retr volume feature'
    'LinGaoyuan_20240930: retr model + retr feature extractor + feature volume'
    def forward_retr(self, point3D, ray_batch, source_imgs_feat, z_vals, mask, ray_d, ray_diff, fea_volume=None, ret_alpha=True):

        B, n_views, H, W, rgb_channel = ray_batch['src_rgbs'].shape  # LinGaoyuan_20240916: B = Batch NV = num_source_views
        N_rand, N_samples, _ = point3D.shape

        img_rgb_sampled = source_imgs_feat[:, :, :, :3]  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 3)
        img_feat_sampled = source_imgs_feat[:,:,:,3:]  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 32)
        dir_relative = ray_diff[:,:,:,:3]

        if fea_volume is not None:
            fea_volume_feat = grid_sample_3d(fea_volume, point3D[None,None,...].float())
            fea_volume_feat = rearrange(fea_volume_feat, "B C RN SN -> (B RN SN) C")

        # input_view = self.rgbfeat_fc(source_imgs_feat)  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 35) -> (N_rand, N_samples, n_views, 32)
        input_view = img_feat_sampled  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 32)
        input_view = rearrange(input_view, 'N_rand N_samples n_views C -> (N_rand N_samples) n_views C')

        if fea_volume is not None:
            input_view = torch.cat((fea_volume_feat.unsqueeze(1), input_view), dim=1)

        output_view = self.view_transformer(input_view, aliasing_filter=self.aliasing_filter, aliasing_filter_type=self.aliasing_filter_type)

        # output_view = output_view.permute(2,0,1,3)  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 32) -> (n_views, N_rand, N_samples, 32)

        input_occ = output_view[:,0,:]  # LinGaoyuan_20240916: (N_rand, N_samples, 1, 32)
        view_feature = output_view[:,1:,:]  # LinGaoyuan_20240916: (N_rand, N_samples, NV - 1, 32)
        # view_feature = output_view  # LinGaoyuan_20240916: (N_rand, N_samples, n_views, 32)
        view_feature = rearrange(view_feature, '(N_rand N_samples) n_views C -> N_rand N_samples n_views C', N_rand = N_rand, N_samples = N_samples)

        x_weight = torch.cat([view_feature, dir_relative], dim=-1)  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 35)
        x_weight = self.linear_radianceweight_1_softmax(x_weight)  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 1)
        if x_weight.dtype == torch.float32:
            x_weight[mask==0] = -1e9
        else:
            x_weight[mask==0] = -1e4
        weight = self.softmax(x_weight)  # LinGaoyuan_20240917: (N_rand, N_samples, n_views, 1)

        source_imgs_feat = rearrange(source_imgs_feat, "N_rand N_samples n_views C -> n_views C N_rand N_samples")
        radiance = (source_imgs_feat * rearrange(weight,"N_rand N_samples n_views 1 -> n_views 1 N_rand N_samples", N_rand=N_rand, N_samples = N_samples)).sum(axis=0)
        radiance = rearrange(radiance, "DimRGB N_rand N_samples -> N_rand N_samples DimRGB")

        input_occ = rearrange(input_occ, "(N_rand N_samples) C -> N_rand N_samples C", N_rand=N_rand, N_samples = N_samples)
        attn_mask = self.get_attn_mask(N_samples).type_as(radiance)
        input_occ = torch.cat((input_occ, self.fuse_layer(radiance), self.order_posenc(100 * z_vals.reshape(-1,z_vals.shape[-1])).type_as(radiance)), dim=-1)
        radiance_tokens = self.RadianceToken(input_occ).unsqueeze(1)
        input_occ = torch.cat((radiance_tokens, input_occ), dim=1)

        output_occ = self.occu_transformer(input_occ)

        output_ray = self.ray_transformer(output_occ[:,:1], output_occ[:,1:])
        weight = self.ray_transformer.atten_weight.squeeze()

        rgb = torch.sigmoid(self.RadianceMLP(output_ray))

        if len(rgb.shape) == 3:
            rgb = rgb.squeeze()

        if ret_alpha  is True:
            rgb = torch.cat([rgb,weight], dim=1)

        return rgb

    def forward_retr_original(self, point3D, ray_batch, source_imgs_feat, z_vals, fea_volume=None, ret_alpha=True):

        B, NV, H, W, rgb_channel = ray_batch['src_rgbs'].shape  # B = 1
        RN, SN, _ = point3D.shape

        target_img_pose = (ray_batch['camera'].squeeze())[-16:].reshape(-1, 4, 4)
        source_imgs_pose = (ray_batch['src_cameras'].squeeze())[:, -16:].reshape(-1, 4, 4)
        source_imgs_intrinsics = (ray_batch['src_cameras'].squeeze())[:, 2:18].reshape(-1, 4, 4)
        source_imgs_pose = source_imgs_intrinsics.bmm(torch.inverse(source_imgs_pose))

        'LinGaoyuan_operation_20240916: add a batch diemension to some variable, B = 1 in this code'
        point3D = point3D[None, ...]
        source_imgs_pose = source_imgs_pose[None, ...]
        source_imgs_feat = source_imgs_feat[None, ...]
        source_imgs_rgb = ray_batch['src_rgbs']

        vector_1 = (point3D - repeat(torch.inverse(target_img_pose)[:,:3,-1], "B DimX -> B 1 1 DimX"))
        vector_1 = repeat(vector_1, "B RN SN DimX -> B 1 RN SN DimX")
        # vector_2 = (point3D.unsqueeze(1) - repeat(source_imgs_pose[:,:,:3,-1], "B L DimX -> B L 1 1 DimX"))
        vector_2 = (point3D.unsqueeze(1) - repeat(torch.inverse(source_imgs_pose)[:, :, :3, -1], "B L DimX -> B L 1 1 DimX"))
        vector_1 = vector_1/torch.linalg.norm(vector_1, dim=-1, keepdim=True) # normalize to get direction
        vector_2 = vector_2/torch.linalg.norm(vector_2, dim=-1, keepdim=True)
        dir_relative = vector_1 - vector_2
        dir_relative = dir_relative.float()


        if fea_volume is not None:
            fea_volume_feat = grid_sample_3d(fea_volume, point3D.unsqueeze(1).float())
            fea_volume_feat = rearrange(fea_volume_feat, "B C RN SN -> (B RN SN) C")
        # -------- project points to feature map
        # B NV RN SN CN DimXYZ
        point3D = repeat(point3D, "B RN SN DimX -> B NV RN SN DimX", NV=NV).float()
        point3D = torch.cat([point3D, torch.ones_like(point3D[:,:,:,:,:1])], axis=4)

        # B NV 4 4 -> (B NV) 4 4
        points_in_pixel = torch.bmm(rearrange(source_imgs_pose, "B NV M_1 M_2 -> (B NV) M_1 M_2", M_1=4, M_2=4),
                                    rearrange(point3D, "B NV RN SN DimX -> (B NV) DimX (RN SN)"))

        points_in_pixel = rearrange(points_in_pixel, "(B NV) DimX (RN SN) -> B NV DimX RN SN", B=B, RN=RN)
        points_in_pixel = points_in_pixel[:, :, :3]

        # in 2D pixel coordinate
        mask_valid_depth = points_in_pixel[:,:,2]>0  #B NV RN SN
        mask_valid_depth = mask_valid_depth.float()
        points_in_pixel = points_in_pixel[:,:,:2] / points_in_pixel[:,:,2:3]

        img_feat_sampled, mask = grid_sample_2d(rearrange(source_imgs_feat, "B NV C H W -> (B NV) C H W"),
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"))
        img_rgb_sampled, _ = grid_sample_2d(rearrange(source_imgs_rgb, "B NV H W C -> (B NV) C H W"),
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"))

        mask = rearrange(mask, "(B NV) RN SN -> B NV RN SN", B=B)
        mask = mask * mask_valid_depth
        img_feat_sampled = rearrange(img_feat_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)
        img_rgb_sampled = rearrange(img_rgb_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)

        x = rearrange(img_feat_sampled, "B NV C RN SN -> (B RN SN) NV C")

        if fea_volume is not None:
            x = torch.cat((fea_volume_feat.unsqueeze(1), x), dim=1)

        x = self.view_transformer(x)

        x1 = rearrange(x, "B_RN_SN NV C -> NV B_RN_SN C")

        '''
        LinGaoyuan_operation_20240916: if use volume feature as the input in view transformer: set x=x1[0], set view_feature = x1[1:],
        if only use img_feat_sampled as the input for view transformer: no need to set x again, and set view_feature = x1
        '''
        if fea_volume is not None:
            x = x1[0] #reference
            view_feature = x1[1:]
            view_feature = rearrange(view_feature, "NV (B RN SN) C -> B RN SN NV C", B=B, RN=RN, SN=SN)
            dir_relative = rearrange(dir_relative, "B NV RN SN Dim3 -> B RN SN NV Dim3")

            x_weight = torch.cat([view_feature, dir_relative], axis=-1)
            x_weight = self.linear_radianceweight_1_softmax(x_weight)
            mask = rearrange(mask, "B NV RN SN -> B RN SN NV 1")

            if x_weight.dtype == torch.float32:
                x_weight[mask==0] = -1e9
            else:
                x_weight[mask==0] = -1e4
            weight = self.softmax(x_weight)
            radiance = (torch.cat((img_rgb_sampled,img_feat_sampled),dim=2) * rearrange(weight, "B RN SN L 1 -> B L 1 RN SN", B=B, RN=RN)).sum(axis=1)
            radiance = rearrange(radiance, "B DimRGB RN SN -> (B RN) SN DimRGB")

            # add positional encoding
            x = rearrange(x, "(B RN SN) C -> (B RN) SN C", RN=RN, B=B, SN=SN)
            attn_mask = self.get_attn_mask(SN).type_as(x)
            x = torch.cat(
                (x, self.fuse_layer(radiance), self.order_posenc(100 * z_vals.reshape(-1, z_vals.shape[-1])).type_as(x)),
                dim=-1)
        else:
            view_feature = x1
            view_feature = rearrange(view_feature, "NV (B RN SN) C -> B RN SN NV C", B=B, RN=RN, SN=SN)
            dir_relative = rearrange(dir_relative, "B NV RN SN Dim3 -> B RN SN NV Dim3")

            x_weight = torch.cat([view_feature, dir_relative], axis=-1)
            x_weight = self.linear_radianceweight_1_softmax(x_weight)
            mask = rearrange(mask, "B NV RN SN -> B RN SN NV 1")

            if x_weight.dtype == torch.float32:
                x_weight[mask == 0] = -1e9
            else:
                x_weight[mask == 0] = -1e4
            weight = self.softmax(x_weight)
            radiance = (torch.cat((img_rgb_sampled, img_feat_sampled), dim=2) * rearrange(weight,
                                                                                          "B RN SN L 1 -> B L 1 RN SN",
                                                                                          B=B, RN=RN)).sum(axis=1)
            radiance = rearrange(radiance, "B DimRGB RN SN -> (B RN) SN DimRGB")

            # add positional encoding

            attn_mask = self.get_attn_mask(SN).type_as(x)
            x = torch.cat((self.fuse_layer(radiance), self.order_posenc(100 * z_vals.reshape(-1, z_vals.shape[-1])).type_as(x)),
                dim=-1)

        radiance_tokens = self.RadianceToken(x).unsqueeze(1)
        x = torch.cat((radiance_tokens, x), dim=1)
        x = self.occu_transformer(x, mask0=attn_mask)

        # calculate weight using view transformers result
        x = self.ray_transformer(x[:, :1], x[:,1:])
        weights = self.ray_transformer.atten_weight.squeeze()

        rgb = torch.sigmoid(self.RadianceMLP(x))

        if len(rgb.shape) == 3:
            rgb = rgb.squeeze()

        if ret_alpha is True:
            rgb = torch.cat([rgb,weights], dim=1)

        return rgb


class ViewTokenNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_parameter('view_token', nn.Parameter(torch.randn([1,dim])))

    def forward(self, x):
        return torch.ones([len(x), 1]).type_as(x) * self.view_token