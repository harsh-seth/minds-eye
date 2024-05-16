import torch
import torch.nn as nn
from einops import rearrange

class MeshDecoder(nn.Module):

    def __init__(self, embedding_dim=512, generate_triangles=False, num_vertices=3000, num_triangles=750):
        super().__init__()
        self.generate_triangles = generate_triangles
        self.num_vertices=num_vertices

        self.sig = nn.Sigmoid()

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.comb_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)

        mlp_ratio = 8
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embedding_dim * mlp_ratio), embedding_dim),
        )

        self.lin1 = nn.Linear(embedding_dim, num_triangles if generate_triangles else num_vertices)

        self.convTranspose1 = nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=1, bias=True)

    def forward(self, x):
       # (N, 512)
       # Generating 3D features from combined features with self-attention
       normed_x = self.norm1(x)
       x = x + self.comb_attn(normed_x, normed_x, normed_x, need_weights=False)[0]
       x = x + self.mlp(self.norm2(x))
       # (N, 512)
       x = self.lin1(x)
       # (N, num_vertices) / (N, num_triangles)
       # Generating 3D features in 3 channels
       x = x.unsqueeze(1).unsqueeze(1) # required to allow ConvTranspose operation
       # (N, 1, 1 num_vertices) / (N, 1, 1, num_triangles)
       x = self.sig(self.convTranspose1(x))
       # (N, 3, 1 num_vertices) / (N, 3, 1, num_triangles)
       # Scale values to become valid entries
       if self.generate_triangles:
           x = torch.floor(x*self.num_vertices)
       else:
           x = x*2 - 1
        # (N, num_vertices) / (N, num_triangles)
       x = rearrange(x, 'N c 1 d -> N d c')
       # (N, 3, num_vertices) / (N, 3, num_triangles)
       return x
