from torch import nn

from model.ImageGridEncoder import ImageGridEncoder
from model.MeshDecoder import MeshDecoder

class MindsEye(nn.Module):
    def __init__(self, generate_triangles=False, num_vertices=4000, num_triangles=750):
        super().__init__()

        self.imageGridEncoder = ImageGridEncoder()
        self.meshDecoder = MeshDecoder(generate_triangles=generate_triangles, num_vertices=num_vertices, num_triangles=num_triangles)
    
    def forward(self, x):
        # (N, 3, 64, 64)
        x = self.imageGridEncoder(x)
        # (N, 512)
        x = self.meshDecoder(x)
        # (N, num_vertices, 3) / (N, num_triangles, 3)
        return x
