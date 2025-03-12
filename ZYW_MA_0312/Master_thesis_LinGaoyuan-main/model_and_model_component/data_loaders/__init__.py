from .LinGaoyuan_ma_nuscene import *
from .LinGaoyuan_ma_nuscene_define_dataset import *
from .LinGaoyuan_ma_nuscene_train_val import *

dataset_dict = {
    # "nuscene": NusceneDataset,
    # "nuscene_define_dataset": NusceneDataset_defined,
    "nuscene_train_val": NusceneDataset_train_val,
}
