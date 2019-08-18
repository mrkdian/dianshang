import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np

from pytorch_transformers import BertModel, load_tf_weights_in_bert, BertConfig, BertForPreTraining, BertTokenizer
from torchcrf import CRF

import json

data = json.load(open('statistic (2).json', mode='r'))
print(data)