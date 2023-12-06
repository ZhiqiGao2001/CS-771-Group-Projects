import yaml

# DEFAULT defines the default params used for training / inference
# the parameters here will be overwritten if a yaml config is specified
DEFAULTS = {
    # default: single gpu
    "devices": ["cuda:0"],
    "model_name": "DDPM",
    # output folder that stores all log files and checkpoints
    "output_folder": None,
    "dataset": {
        "name": "MNIST",
        # training / testing splits
        "split": "train",
        "data_folder": None,
    },
    "loader": {
        # batch size
        "batch_size": 64,
        # number of workers for data fetching
        "num_workers": 2,
    },

    # unet architecture
    "model": {
        # shape of input image (C x H W)
        "img_shape": None,
        # number of timesteps in the diffusion process
        "timesteps": 500,
        # number of classes used for conditioning
        "num_classes": None,
        # base feature dimension in UNet
        "dim": 64,
        # condition dimension (embedding of the label) in UNet
        "context_dim": 64,
        # multiplier of feature dimensions in UNet
        # length of this list also specifies #blockes in UNet encoder/decoder
        # e.g., (1, 2, 4) -> 3 blocks with output dims of 1x, 2x, 4x w.r.t. base feature dim
        "dim_mults": (1, 2, 4),
    },
    # training config
    "train_cfg": {
        # learning rate
        "lr": 2e-4,
        # number of epochs in training
        "epochs": 30,
        # number of samples to draw when evaluating the model
        "num_eval_samples": 10,
    },
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    return config
