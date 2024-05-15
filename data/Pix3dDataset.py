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

    def __init__(self, objects = ['bookcase', 'desk'], data_dir="./pix3d", max_records=None, max_vertices=4000, max_triangles=750) -> None:
        super().__init__()
        self.max_vertices=max_vertices
        self.max_triangles=max_triangles
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
            if max_records and len(self.records) >= max_records:
                break
        print(f"Prepared dataset of {len(self.records)} records.")
    

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        pointcloud_data = o3d.io.read_triangle_mesh(self.records[idx][1])
        pointcloud_data = pointcloud_data.compute_vertex_normals()
        
        vertices = np.asarray(pointcloud_data.vertices)
        while vertices.shape[0] < self.max_vertices: # padding with repeats
            difference = self.max_vertices - vertices.shape[0]
            vertices = np.vstack([vertices, vertices[:difference]])
        vertices = vertices[:self.max_vertices]
        
        triangles = np.asarray(pointcloud_data.triangles)
        while triangles.shape[0] < self.max_triangles: # padding with repeats
            difference = self.max_triangles - triangles.shape[0]
            triangles = np.vstack([triangles, triangles[:difference]])
        triangles = triangles[:self.max_triangles]
        
        pointcloud_data = [
            torch.tensor(vertices, dtype=torch.float32), 
            torch.tensor(triangles, dtype=torch.int64)
        ]
        
        img = Image.open(self.records[idx][0])
        img = self.transform(img)
        return img, pointcloud_data
    

if __name__=="__main__":
    base_path = "./pix3d"
    pix3d_dataset = Pix3dDataset(
        data_dir=base_path,
        max_records=200
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
    