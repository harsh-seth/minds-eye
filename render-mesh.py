import open3d as o3d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
args = parser.parse_args()

def loadAndRenderMesh(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    visualize(mesh)

def visualize(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

loadAndRenderMesh(args.filename)
