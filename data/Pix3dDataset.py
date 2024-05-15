import open3d as o3d
import json
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class Pix3dDataset(Dataset):

    def __init__(self, objects = ['bookcase', 'desk'], data_dir="./pix3d_full") -> None:
        super().__init__()
        pix3d_json_data = json.load(open(f"{data_dir}/pix3d.json"))

        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.records = []

        print("Loading the Pix3D dataset...")
        for item in tqdm(pix3d_json_data):
            if item["category"] in objects:
                folder_dir = os.path.join(data_dir, os.path.dirname(item['model']))
                object_name = os.path.basename(folder_dir)
                img_path = os.path.join(data_dir, "grid_images", f"{object_name}_grid.png")
                model_path = os.path.join(data_dir, item['model'])
                if os.path.exists(img_path) and os.path.exists(model_path):
                    self.records.append([img_path, model_path])
        print(f"Prepared dataset of {len(self.records)} records.")
    

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        # pointcloud_data = np.array(np.load(self.records[idx][1]))
        pointcloud_data = o3d.io.read_triangle_mesh(self.records[idx][1])
        pointcloud_data = pointcloud_data.compute_vertex_normals()
        pointcloud_data = [
            torch.tensor(pointcloud_data.vertices, dtype=torch.float32), 
            torch.tensor(pointcloud_data.triangles, dtype=torch.int64)]
        img = Image.open(self.records[idx][0])
        img = self.transform(img)
        return img, pointcloud_data
    

if __name__=="__main__":
    base_path = "/mnt/d/harsh/Downloads/pix3d_full"
    pix3d_dataset = Pix3dDataset(
        data_dir=base_path
    )
    dataloader = DataLoader(pix3d_dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        pointcloud = batch[1]
        images = batch[0]
        break
    print(images.shape)  # torch.Size([1, 3, 64, 64])
    print("****************************************")
    print(pointcloud[0].shape)  # torch.Size([1, 3694, 3])
    print(pointcloud[1].shape)  # torch.Size([1, 2166, 3])
    