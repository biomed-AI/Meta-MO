"""Loads a trained model checkpoint and makes predictions on a dataset."""
import torch
import numpy as np
import random

from chemprop.args import PredictArgs
from chemprop.train import make_predictions

if __name__ == '__main__':
    args = PredictArgs().parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    # this one is needed for torchtext random call (shuffled iterator)
    # in multi gpu it ensures datasets are read in the same order
    random.seed(0)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    
    make_predictions(args)