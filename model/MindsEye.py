from torch import nn

from model.ImageGridEncoder import ImageGridEncoder

class MindsEye(nn.Module):
    def __init__(self):
        super().__init__()

        self.imageGridEncoder = ImageGridEncoder()
    
    def forward(self, x):
        # (N, 3, 64, 64)
        x = self.imageGridEncoder(x)
        # (N, 3, ?, ?)
        return x
