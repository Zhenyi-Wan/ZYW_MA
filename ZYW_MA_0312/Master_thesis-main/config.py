import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--rootdir",
        type=str,
        default="./",
        help="the path to the project root directory. Replace this path with yours!",
    )
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--distributed", action="store_true", help="if use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="rank for distributed training")
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )

    ########## dataset options ##########
    ## train and eval dataset
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="ibrnet_collected",
        help="the training dataset, should either be a single dataset, "
        'or multiple datasets connected with "+", for example, ibrnet_collected+llff+spaces',
    )
    parser.add_argument(
        "--dataset_weights",
        nargs="+",
        type=float,
        default=[],
        help="the weights for training datasets, valid when multiple datasets are used.",
    )
    parser.add_argument(
        "--train_scenes",
        nargs="+",
        default=[],
        help="optional, specify a subset of training scenes from training dataset",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="llff_test", help="the dataset to evaluate"
    )
    parser.add_argument(
        "--eval_scenes",
        nargs="+",
        default=[],
        help="optional, specify a subset of scenes from eval_dataset to evaluate",
    )

    parser.add_argument(
        "--image_H",
        type=int,
        default=900,
        help="Width of image",
    )

    parser.add_argument(
        "--image_W",
        type=int,
        default=1600,
        help="Height of image",
    )

    parser.add_argument(
        "--image_resize_H",
        type=int,
        default=450,
        help="Width of image",
    )

    parser.add_argument(
        "--image_resize_W",
        type=int,
        default=800,
        help="Height of image",
    )

    parser.add_argument(
        "--resize_image", action="store_true", help="whether or not to resize image"
    )


    ## others
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, "
        "useful for large datasets like deepvoxels or nerf_synthetic",
    )

    ########## model options ##########
    ## ray sampling options
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="uniform",
        help="how to sample pixels from images for training:" "uniform|center",
    )
    parser.add_argument(
        "--center_ratio", type=float, default=0.8, help="the ratio of center crop to keep"
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 16,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024 * 4,
        help="number of rays processed in parallel, decrease if running out of memory",
    )

    parser.add_argument(
        "--sample_with_prior_depth", action="store_true",
        help="sample 3D pts based on the prior value of depth"
    )

    ## model options
    parser.add_argument(
        "--coarse_feat_dim", type=int, default=32, help="2D feature dimension for coarse level"
    )
    parser.add_argument(
        "--fine_feat_dim", type=int, default=32, help="2D feature dimension for fine level"
    )
    parser.add_argument(
        "--num_source_views",
        type=int,
        default=10,
        help="the number of input source views for each target view",
    )
    parser.add_argument(
        "--rectify_inplane_rotation", action="store_true", help="if rectify inplane rotation"
    )
    parser.add_argument("--coarse_only", action="store_true", help="use coarse network only")
    parser.add_argument(
        "--anti_alias_pooling", type=int, default=1, help="if use anti-alias pooling"
    )
    parser.add_argument("--trans_depth", type=int, default=4, help="number of transformer layers")
    parser.add_argument("--netwidth", type=int, default=64, help="network intermediate dimension")
    parser.add_argument(
        "--single_net",
        type=bool,
        default=True,
        help="use single network for both coarse and/or fine sampling",
    )
    parser.add_argument(
        "--contraction_type", type=str, default=None, help="the type of unbounded contraction"
    )

    parser.add_argument(
        "--aliasing_filter", action="store_true", help="whether or not to use aliasing filter"
    )

    parser.add_argument(
        "--aliasing_filter_type", type=str, default=None, help="the type aliasing filter, filter bank or single filter"
    )

    ########## checkpoints ##########
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--no_load_opt", action="store_true", help="do not load optimizer when reloading"
    )
    parser.add_argument(
        "--no_load_scheduler", action="store_true", help="do not load scheduler when reloading"
    )

    ########### iterations & learning rate options ##########
    parser.add_argument("--n_iters", type=int, default=250000, help="num of iterations")

    parser.add_argument("--max_epochs", type=int, default=1000, help="num of max epoch")

    parser.add_argument(
        "--lrate_feature", type=float, default=1e-3, help="learning rate for feature extractor"
    )
    parser.add_argument("--lrate_gnt", type=float, default=5e-4, help="learning rate for model_and_model_component")
    parser.add_argument(
        "--lrate_decay_factor",
        type=float,
        default=0.5,
        help="decay learning rate by a factor every specified number of steps",
    )
    parser.add_argument(
        "--lrate_decay_steps",
        type=int,
        default=50000,
        help="decay learning rate by a factor every specified number of steps",
    )

    ########### loss coefficient ###########
    parser.add_argument("--lambda_rgb", type=float, default=0.5, help="loss coefficient for rgb loss")
    parser.add_argument("--lambda_depth", type=float, default=0.5, help="loss coefficient for depth loss")

    ########### depth prior update relevant variable ###########

    parser.add_argument(
        "--cov_criteria", action="store_true",
        help="whether to use the covariance of depth prediction to determine when update the prior depth value"
    )

    parser.add_argument(
        "--depth_loss_criteria", action="store_true",
        help="whether to use the loss of depth prediction to determine when update the prior depth value"
    )

    parser.add_argument(
        "--preset_depth_cov", type=float, default=50.0,
        help="preset value of depth covariance, when calculated depth cov <= preset value, the prior depth updating process will begin"
    )

    parser.add_argument(
        "--preset_depth_loss", type=float, default=50.0,
        help="preset value of depth loss, when calculated depth loss <= preset value, the prior depth updating process will begin"
    )

    parser.add_argument("--update_prior_depth_epochs", type=int, default=50,
                        help="after this epoch the prior depth array will be updated by depth prediction and depth loss will be given up")

    ########### sky model & style model ###########
    parser.add_argument(
        "--sky_model_type", type=str, default="mlp", help="the type of model to use for sky"
    )
    parser.add_argument("--lrate_sky_model", type=float, default=0.005, help="learning rate for sky model")
    parser.add_argument(
        "--lrate_decay_factor_sky_model",
        type=float,
        default=0.95,
        help="decay learning rate of sky model by a factor every specified number of steps",
    )
    parser.add_argument(
        "--lrate_decay_steps_sky_model",
        type=int,
        default=2000,
        help="decay learning rate of sky model by a factor every specified number of steps",
    )

    parser.add_argument("--lrate_sky_style_model", type=float, default=0.005, help="learning rate for sky style model")
    parser.add_argument(
        "--lrate_decay_factor_sky_style_model",
        type=float,
        default=0.95,
        help="decay learning rate o9f sky style model by a factor every specified number of steps",
    )
    parser.add_argument(
        "--lrate_decay_steps_sky_style_model",
        type=int,
        default=2000,
        help="decay learning rate of sky style model by a factor every specified number of steps",
    )

    ########## ReTR model #########
    parser.add_argument(
        "--use_volume_feature", action="store_true", help="whether or not to use feature extractor of ReTR"
    )

    parser.add_argument(
        "--use_retr_feature_extractor", action="store_true", help="whether or not to use volume feature extractor of ReTR"
    )

    parser.add_argument(
        "--use_retr_model", action="store_true", help="whether or not to use model of ReTR"
    )

    parser.add_argument("--lrate_retr", type=float, default=5e-4, help="learning rate for retr")

    parser.add_argument(
        "--lrate_retr_feature_volume", type=float, default=1e-3, help="learning rate for retr feature volume"
    )

    parser.add_argument(
        "--lrate_retr_feature_extractor", type=float, default=1e-3, help="learning rate for retr feature extractor"
    )

    ######### clip #########

    parser.add_argument("--lambda_shape_code", type=float, default=0.2, help="loss coefficient for clip loss in loss of shape code")

    parser.add_argument("--lambda_appearance_code", type=float, default=0.2, help="loss coefficient for clip loss in loss of appearance code")

    ######### BRDF #########
    parser.add_argument(
        "--BRDF_model", action="store_true", help="whether or not to use BRDF model"
    )

    parser.add_argument(
        "--use_NeILFPBR", action="store_true", help="whether or not to use NeILFPBR to rerender RGB"
    )

    parser.add_argument(
        "--use_NeROPBR", action="store_true", help="whether or not to use NeROPBR to rerender RGB"
    )

    ########## rendering options ##########
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance", type=int, default=64, help="number of important samples per ray"
    )
    parser.add_argument(
        "--inv_uniform", action="store_true", help="if True, will uniformly sample inverse depths"
    )
    parser.add_argument(
        "--det", action="store_true", help="deterministic sampling for coarse and fine samples"
    )
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="apply the trick to avoid fitting to white background",
    )
    parser.add_argument(
        "--render_stride",
        type=int,
        default=1,
        help="render with large stride for validation to save time",
    )

    parser.add_argument(
        "--N_samples_depth", type=int, default=64, help="number of samples per ray with prior depth"
    )

    ########## logging/saving options ##########
    parser.add_argument("--i_print", type=int, default=100, help="frequency of terminal printout")
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )

    parser.add_argument(
        "--update_prior_depth", action="store_true",
        help="determine whether to update the prior depth of train dataset and val dataset"
    )

    parser.add_argument(
        "--save_prior_depth", action="store_true",
        help="determine whether to save the prior depth of train dataset when the global step reach the i_weights"
    )

    parser.add_argument(
        "--i_prior_depth_update", type=int, default=10000, help="frequency of prior depth update"
    )

    ########## evaluation options ##########
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    return parser
