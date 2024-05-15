import open3d as o3d
import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class Pix3dDataset():

    def __init__(self, objects = ['boookcase', 'desk']) -> None:
        pix3d_json_data = json.load(open("/Users/dhrumeen/Downloads/pix3d_full/pix3d.json"))

        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        data_dir = "../../pix3d_full" # Change as per the dataset folder location

        self.train_data = []

        for item in pix3d_json_data:
            if item["category"] in objects:
                folder_dir = os.path.join(data_dir, os.path.dirname(item['model']))
                object_name = os.path.basename(folder_dir)
                img_path = os.path.join(data_dir, "grid_images", f"{object_name}_grid.png")
                model_path = os.path.join(data_dir, item['model'])
                if os.path.exists(img_path) and os.path.exists(model_path):
                    self.train_data.append([img_path, model_path])
    

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        # pointcloud_data = np.array(np.load(self.train_data[idx][1]))
        pointcloud_data = o3d.io.read_triangle_mesh(self.train_data[idx][1])
        pointcloud_data = pointcloud_data.compute_vertex_normals()
        pointcloud_data = [
            torch.tensor(pointcloud_data.vertices, dtype=torch.float32), 
            torch.tensor(pointcloud_data.triangles, dtype=torch.int64)]
        img = Image.open(self.train_data[idx][0])
        img = self.transform(img)
        return img, pointcloud_data
    

if __name__=="__main__":
    pix3d_dataset = Pix3dDataset()
    dataloader = DataLoader(pix3d_dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        pointcloud = batch[1]
        images = batch[0]
        break
    print(images.shape)  # torch.Size([1, 3, 64, 64])
    print("****************************************")
    print(pointcloud[0].shape)  # torch.Size([1, 3694, 3])
    print(pointcloud[1].shape)  # torch.Size([1, 2166, 3])
    