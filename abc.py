import argparse
import logging
import math
import os
import random

import numpy as np
import options.options as option
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from models import create_model
from utils import util

def main():
     # options
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
