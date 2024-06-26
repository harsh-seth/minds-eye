import argparse
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as transforms

from PIL import Image

# Given an image grid of views of a subject, generate a corresponding 3D model of the subject
def generate3DModelFromImageGrid(image, vertices_model, triangles_model, max_vertices=5000):
    vertices_model.eval()
    triangles_model.eval()

    image_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
    ])

    image = image_transform(image).unsqueeze(0) # creating a batch of size 1

    with torch.no_grad():
        vertices = vertices_model(image)[0]
        triangles = triangles_model(image)[0]

    triangles = (triangles * max_vertices).int()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.numpy().astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.numpy().astype(np.float64))

    o3d.visualization.draw_geometries([mesh])

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="CLI for running inference on Mind's Eye")

    parser.add_argument('--image_path', type=str,
                        help='Path to image file')
    
    parser.add_argument('--vertices_checkpoint', type=int,
                        help='Checkpoint number to use for the Vertices model')
    
    parser.add_argument('--triangles_checkpoint', type=int,
                        help='Checkpoint number to use for the Triangles model')
    
    parser.add_argument('--results_dir', type=str, default="results",
                        help='Path to the results folder')
    
    parser.add_argument('--max_vertices', type=int, default=5000,
                        help='Needs to be the same number as used during training')

    parsed_arguments = parser.parse_args()
    return parsed_arguments

if __name__ == '__main__':
    args = parse_command_line_arguments()
    vertices_model = torch.load(f'{args.results_dir}/vertices/checkpoints/checkpoint-{args.vertices_checkpoint}.pth', map_location=lambda storage, location: storage)
    triangle_model = torch.load(f'{args.results_dir}/triangles/checkpoints/checkpoint-{args.triangles_checkpoint}.pth', map_location=lambda storage, location: storage)

    img = Image.open(args.image_path)
    generate3DModelFromImageGrid(image=img, vertices_model=vertices_model, triangles_model=triangle_model, max_vertices=args.max_vertices)
