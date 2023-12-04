from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import torch
# Configs
resume_path = './models/v1-5-pruned-emaonly-ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

import pickle
input_path = 'models/v1-5-pruned-emaonly_ini.ckpt'
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
#model = create_model('./models/cldm_v21.yaml').cpu()
# ファイルをバイナリ読み取りモードで開く
model = torch.load(input_path)
model_str = str(model)
with open("model_ini.txt", "w") as file:
    file.write(model_str)