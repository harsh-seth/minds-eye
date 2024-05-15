import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class ImageGridEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding=1, bias=True)
        self.conv2d_2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=1, bias=True)

        self.relu = nn.ReLU()

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)

        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.5)
        self.dropout_3 = nn.Dropout(0.2)

        self.flat = nn.Flatten()

        self.fc_1 = nn.Linear(20480, 10240)
        self.fc_2 = nn.Linear(10240, 1024)

        self.dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base")

        self.positional_embeddings = nn.Parameter(torch.randn((1024*768), 256))  # Learnable positional embeddings

        self.vit_preprocessor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224")


    def forward(self, x):
       # 3x64x64
       x = self.relu(self.conv2d_1(x))
       x = self.dropout_1(x)
       # 10*64*64 to 20*64*64
       x = self.relu(self.conv2d_2(x))
       x = self.dropout_2(x)
       # 20*64*64 to 20*32*32
       x = self.maxpool_1(x)
       # 20*32*32 to 20480
       x = self.flat(x)
       x = self.relu(self.fc_1(x))
       x = self.dropout_3(x)
       x = self.fc_2(x)
       x = self.dino_processor(x)
       x = self.dino_model(x)
       x = torch.cat((x, self.positional_embeddings), dim=1)
       x = self.vit_model(**self.vit_preprocessor(x))
       return x
