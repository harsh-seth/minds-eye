import argparse
import numpy as np
import open3d as o3d
import torch

from PIL import Image

from model.MindsEye import MindsEye

# Given an image grid of views of a subject, generate a corresponding 3D model of the subject
def generate3DModelFromImageGrid(image, vertices_model, triangles_model):
    vertices_model.eval()
    triangles_model.eval()

    with torch.no_grad():
        vertices = vertices_model(image)
        triangles = triangles_model(image)

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

    parsed_arguments = parser.parse_args()
    return parsed_arguments

if __name__ == '__main__':
    args = parse_command_line_arguments()
    vertices_model = torch.load(f'{args.results_dir}/vertices/checkpoints/checkpoint-{args.vertex_checkpoint}.pth', map_location=lambda storage, location: storage)
    triangle_model = torch.load(f'{args.results_dir}/triangles/checkpoints/checkpoint-{args.triangles_checkpoint}.pth', map_location=lambda storage, location: storage)

    img = Image.open(args.image_path)
    generate3DModelFromImageGrid(image=img, vertices_model=vertices_model, triangles_model=triangle_model)
