import torch
import torch.nn as nn

class ImageGridEncoder(nn.Module):

    def __init__(self, in_channels=3, grid_n=3, embedding_dim=512, in_size=64, device='cuda'):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=grid_n*grid_n, kernel_size=5, padding=2, bias=True)
        self.conv2d_2 = nn.Conv2d(in_channels=grid_n*grid_n, out_channels=in_channels, kernel_size=5, padding=2, bias=True)

        self.relu = nn.ReLU()
        self.flat = nn.Flatten()


        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.5)
        self.dropout_3 = nn.Dropout(0.2)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        
        mlp_ratio = 20
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels * (in_size//2)**2, embedding_dim*mlp_ratio),
            nn.ReLU(),
            nn.Linear(embedding_dim*mlp_ratio, embedding_dim)
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.img_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        self.pos_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)

        mlp_ratio = 2
        self.mlp2 = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embedding_dim * mlp_ratio), embedding_dim),
        )

        self.positional_embeddings = nn.Parameter(torch.randn(embedding_dim))  # Learnable positional embeddings

    def forward(self, x):
       # (N, 3, 64, 64)
       # Convoluting a grid of KxK images to a stack of (KxK) "images"
       x = self.relu(self.conv2d_1(x))
       x = self.dropout_1(x)
       # (N, 9, 64, 64)
       # Convoluting a (KxK) stack of "images" to a "single image"
       x = self.relu(self.conv2d_2(x))
       x = self.dropout_2(x)
       # (N, 3, 64, 64)
       # Reducing "resolution" of the "single image", also lose "noise"/"empty values" from representation, ALSO help with ignoring redundant/repeated points!
       x = self.maxpool_1(x)
       # (N, 3, 32, 32)
       # Flattening the "single image" into a vector to feed into a transformer
       x = self.flat(x)
       # (N, 3072)
       x = self.mlp1(x)
       # (N, 512)
       # Encoding image features with self-attention
       normed_x = self.norm1(x)
       x = x + self.img_attn(normed_x, normed_x, normed_x, need_weights=False)[0]
       x = x + self.mlp2(self.norm2(x))
       # (N, 512)
       # Combining image features with learnable position features with cross-attention (query: pos; key, val: img)
       # x = self.positional_embeddings + self.pos_attn(self.norm3(self.positional_embeddings), x, x, need_weights=False)[0] # Need to give positional_embeddings the dimension!
       x = self.positional_embeddings + x
       # (N, 512)
       return x
