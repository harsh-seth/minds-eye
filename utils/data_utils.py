import os
import torch

def save_checkpoint(model, save_path):
    """save model checkpoint"""
    dir_path = save_path.split("/")[:-1].join("/")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(model, save_path)        
    print("Checkpoint saved to {}".format(save_path))
