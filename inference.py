import open3d as o3d
import numpy as np
from model.MindsEye import MindsEye

# Given an image grid of views of a subject, generate a corresponding 3D model of the subject
def generate3DModelFromImageGrid(img):
    vertices_model = MindsEye()
    triangles_model = MindsEye(generate_triangles=True)

    vertices = vertices_model(img)
    triangles = triangles_model(img)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.numpy().astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.numpy().astype(np.float64))

    o3d.visualization.draw_geometries([mesh])
