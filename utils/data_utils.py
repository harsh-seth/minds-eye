import os
import torch

def save_checkpoint(model, save_path):
    """save model checkpoint"""
    dir_path = "/".join(save_path.split("/")[:-1])
    make_dir_path(dir_path)
    torch.save(model, save_path)        
    print("Checkpoint saved to {}".format(save_path))

def make_dir_path(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
