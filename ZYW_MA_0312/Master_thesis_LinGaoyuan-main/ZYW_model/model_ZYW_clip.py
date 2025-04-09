import torch
import os
from model_and_model_component.GNT_model_LinGaoyuan_clip import GNT
from model_and_model_component.GNT_feature_extractor import ResUNet

from ZYW_model.ZYW_ReTR_model_clip import ZYW_ReTR_model
from LinGaoyuan_function.ReTR_function.ReTR_feature_extractor import FPN_FeatureExtractor
from LinGaoyuan_function.ReTR_function.ReTR_feature_volume import FeatureVolume


def de_parallel(model):
    return model.module if hasattr(model, "module") else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class ZYW_GNTModel(object):
    '''
    LinGaoyuan_20240930: add dimension of shape code, if the model is used for building and street, the dim_clip_shape_code = 0, 
    and if model is used for car, dim_clip_shape_code = 128
    '''
    def __init__(self, args, load_opt=True, load_scheduler=True, dim_clip_shape_code=0):
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))

        'LinGaoyuan_operation_20240918: use ReTR model if args.use_retr_model is True, else use original model_and_model_component model'
        if self.args.use_retr_model is True:
            self.net_coarse = ZYW_ReTR_model(
                args,
                in_feat_ch=32,
                posenc_dim=3,
                viewenc_dim=3,
                ret_alpha=False,
                use_volume_feature=args.use_volume_feature,
                dim_clip_shape_code=dim_clip_shape_code,
            ).to(device)
        else:
            'LinGaoyuan_operation_20240830: set ret_alpha = True in order to always return depth prediction'
            # Zhenyi Wan [2025/3/26] we don't change the GNT model
            # create coarse GNT
            self.net_coarse = GNT(
                args,
                in_feat_ch=self.args.coarse_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                # ret_alpha=args.N_importance > 0,
                ret_alpha=True,
                dim_clip_shape_code=dim_clip_shape_code,
            ).to(device)
            # single_net - trains single network which can be used for both coarse and fine sampling
            if args.single_net:
                self.net_fine = None
            else:
                self.net_fine = GNT(
                    args,
                    in_feat_ch=self.args.fine_feat_dim,
                    posenc_dim=3 + 3 * 2 * 10,
                    viewenc_dim=3 + 3 * 2 * 10,
                    ret_alpha=True,
                ).to(device)

        'LinGaoyuan_operation_20240918: use retr feature extractor if args.use_retr_feature_extractor is True else use original model_and_model_component feature net'
        if self.args.use_retr_feature_extractor is True:
            self.retr_feature_extractor = FPN_FeatureExtractor(out_ch=32).to(device)
        else:
            # create feature extraction network
            self.feature_net = ResUNet(
                coarse_out_ch=self.args.coarse_feat_dim,
                fine_out_ch=self.args.fine_feat_dim,
                single_net=self.args.single_net,
            ).to(device)

        if self.args.use_volume_feature is True and self.args.use_retr_feature_extractor is True:
            self.retr_feature_volume = FeatureVolume(volume_reso=100).to(device)

        test_a = hasattr(self, 'retr_feature_volume')


        # optimizer and learning rate scheduler
        'LinGaoyuan_operation_20240918: add model parameters to learnable_params if the model is defined'
        learnable_params = list(self.net_coarse.parameters())
        if hasattr(self, 'retr_feature_extractor'):
            learnable_params += list(self.retr_feature_extractor.parameters())
        if hasattr(self, 'feature_net'):
            learnable_params += list(self.feature_net.parameters())
        if hasattr(self, 'retr_feature_volume'):
            learnable_params += list(self.retr_feature_volume.parameters())
        if hasattr(self, 'net_fine') and self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        # learnable_params = list(self.net_coarse.parameters())
        # learnable_params += list(self.feature_net.parameters())
        # if self.net_fine is not None:
        #     learnable_params += list(self.net_fine.parameters())

        'LinGaoyuan_operation_20240918: create optimizer according to different combination of model and feature extractor'

        # LinGaoyuan_20240918: original model_and_model_component model combination: model_and_model_component(coarse) + feature_net(model_and_model_component) + model_and_model_component(fine)
        if hasattr(self, 'net_fine') and self.net_fine is not None and hasattr(self, 'feature_net'):
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.net_fine.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )
        # LinGaoyuan_20240918: retr model + retr_feature_extractor + retr_feature_volume
        elif hasattr(self, 'retr_feature_extractor') and hasattr(self, 'retr_feature_volume') and self.args.use_retr_model is True:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.retr_feature_extractor.parameters(), "lr": args.lrate_retr_feature_extractor},
                    {"params": self.retr_feature_volume.parameters(), "lr": args.lrate_retr_feature_volume},
                ],
                lr=args.lrate_retr,
            )
        # LinGaoyuan_20240918: retr model + retr_feature_extractor
        elif hasattr(self, 'retr_feature_extractor') and self.args.use_retr_model is True:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.retr_feature_extractor.parameters(), "lr": args.lrate_retr_feature_extractor},
                ],
                lr=args.lrate_retr,
            )
        # LinGaoyuan_20240918: retr model + feature_net(model_and_model_component)
        elif hasattr(self, 'feature_net') and self.args.use_retr_model is True:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_retr,
            )
        # LinGaoyuan_20240918: original model_and_model_component model combination: model_and_model_component(coarse) + feature_net(model_and_model_component)
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse, device_ids=[args.local_rank], output_device=args.local_rank
            )

            # self.feature_net = torch.nn.parallel.DistributedDataParallel(
            #     self.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
            # )
            #
            # if self.net_fine is not None:
            #     self.net_fine = torch.nn.parallel.DistributedDataParallel(
            #         self.net_fine, device_ids=[args.local_rank], output_device=args.local_rank
            #     )

            'LinGaoyuan_operation_20240918: set the model to parallel if model is exist'
            if hasattr(self, 'retr_feature_extractor'):
                self.retr_feature_extractor = torch.nn.parallel.DistributedDataParallel(
                    self.retr_feature_extractor, device_ids=[args.local_rank], output_device=args.local_rank
                )
            if hasattr(self, 'feature_net'):
                self.feature_net = torch.nn.parallel.DistributedDataParallel(
                    self.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
                )
            if hasattr(self, 'retr_feature_volume'):
                self.retr_feature_volume = torch.nn.parallel.DistributedDataParallel(
                    self.retr_feature_volume, device_ids=[args.local_rank], output_device=args.local_rank
                )
            if hasattr(self, 'net_fine') and self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine, device_ids=[args.local_rank], output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        # self.feature_net.eval()
        # if self.net_fine is not None:
        #     self.net_fine.eval()
        'LinGaoyuan_operation_20240918: switch the model to eval mode if the model is defined'
        if hasattr(self, 'retr_feature_extractor'):
            self.retr_feature_extractor.eval()
        if hasattr(self, 'feature_net'):
            self.feature_net.eval()
        if hasattr(self, 'retr_feature_volume'):
            self.retr_feature_volume.eval()
        if hasattr(self, 'net_fine') and self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        # self.feature_net.train()
        # if self.net_fine is not None:
        #     self.net_fine.train()
        'LinGaoyuan_operation_20240918: switch the model to train mode if the model is defined'
        if hasattr(self, 'retr_feature_extractor'):
            self.retr_feature_extractor.train()
        if hasattr(self, 'feature_net'):
            self.feature_net.train()
        if hasattr(self, 'retr_feature_volume'):
            self.retr_feature_volume.train()
        if hasattr(self, 'net_fine') and self.net_fine is not None:
            self.net_fine.train()

    def save_model(self, filename):
        ''
        # to_save = {
        #     "optimizer": self.optimizer.state_dict(),
        #     "scheduler": self.scheduler.state_dict(),
        #     "net_coarse": de_parallel(self.net_coarse).state_dict(),
        #     "feature_net": de_parallel(self.feature_net).state_dict(),
        # }
        #
        # if self.net_fine is not None:
        #     to_save["net_fine"] = de_parallel(self.net_fine).state_dict()

        'LinGaoyuan_operation_20240918: define different to_save according to different combination of model'
        # LinGaoyuan_20240918: model_and_model_component(coarse) + feature_net(model_and_model_component) + model_and_model_component(fine)
        if hasattr(self, 'net_fine') and self.net_fine is not None and hasattr(self, 'feature_net'):
            to_save = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "net_coarse": de_parallel(self.net_coarse).state_dict(),
                "feature_net": de_parallel(self.feature_net).state_dict(),
            }
            to_save["net_fine"] = de_parallel(self.net_fine).state_dict()

        # LinGaoyuan_20240918: retr model + retr_feature_extractor + retr_feature_volume
        elif hasattr(self, 'retr_feature_extractor') and hasattr(self,'retr_feature_volume') and self.args.use_retr_model is True:
            to_save = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "net_coarse": de_parallel(self.net_coarse).state_dict(),
                "retr_feature_extractor": de_parallel(self.retr_feature_extractor).state_dict(),
                "retr_feature_volume": de_parallel(self.retr_feature_volume).state_dict(),
            }

        # LinGaoyuan_20240918: retr model + retr_feature_extractor
        elif hasattr(self, 'retr_feature_extractor') and self.args.use_retr_model is True:
            to_save = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "net_coarse": de_parallel(self.net_coarse).state_dict(),
                "retr_feature_extractor": de_parallel(self.retr_feature_extractor).state_dict(),
            }

        # LinGaoyuan_20240918: both model_and_model_component(coarse) + feature_net(model_and_model_component) and retr model + feature_net(model_and_model_component) have the same model structure: net_coarse + feature_net
        else:
            to_save = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "net_coarse": de_parallel(self.net_coarse).state_dict(),
                "feature_net": de_parallel(self.feature_net).state_dict(),
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
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        # self.net_coarse.load_state_dict(to_load["net_coarse"])
        # self.feature_net.load_state_dict(to_load["feature_net"])
        #
        # if self.net_fine is not None and "net_fine" in to_load.keys():
        #     self.net_fine.load_state_dict(to_load["net_fine"])

        'LinGaoyuan_operation_20240918: load ckpt according to different combination of model'
        # self.net_coarse always exist
        self.net_coarse.load_state_dict(to_load["net_coarse"])

        # LinGaoyuan_20240918: model_and_model_component(coarse) + feature_net(model_and_model_component) + model_and_model_component(fine)
        if hasattr(self, 'net_fine') and self.net_fine is not None and hasattr(self, 'feature_net'):
            self.feature_net.load_state_dict(to_load["feature_net"])

            if self.net_fine is not None and "net_fine" in to_load.keys():
                self.net_fine.load_state_dict(to_load["net_fine"])

        # LinGaoyuan_20240918: retr model + retr_feature_extractor + retr_feature_volume
        elif hasattr(self, 'retr_feature_extractor') and hasattr(self,'retr_feature_volume') and self.args.use_retr_model is True:
            self.retr_feature_extractor.load_state_dict(to_load['retr_feature_extractor'])
            self.retr_feature_volume.load_state_dict(to_load['retr_feature_volume'])

        # LinGaoyuan_20240918: retr model + retr_feature_extractor
        elif hasattr(self, 'retr_feature_extractor') and self.args.use_retr_model is True:
            self.retr_feature_extractor.load_state_dict(to_load['retr_feature_extractor'])

        # LinGaoyuan_20240918: both model_and_model_component(coarse) + feature_net(model_and_model_component) and retr model + feature_net(model_and_model_component) have the same model structure: net_coarse + feature_net
        else:
            self.feature_net.load_state_dict(to_load["feature_net"])


    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        'LinGaoyuan code: remove all the elements in ckpts which name contains sky_model'
        ckpts_copy = ckpts.copy()
        for i in ckpts:
            if i.split("/")[-1].find('sky_model') == 0:
                ckpts_copy.remove(i)
        ckpts = ckpts_copy

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0

        return step
