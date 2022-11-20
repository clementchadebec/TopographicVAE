import torch
from torch import nn
import numpy as np

def MLP_Encoder(s_dim, n_cin, n_hw):
    model = nn.Sequential(
                nn.Conv2d(n_cin, s_dim*3,
                    kernel_size=n_hw, stride=1, padding=0),
                nn.ReLU(True),
                nn.Conv2d(s_dim*3, s_dim*2,
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Conv2d(s_dim*2, s_dim*2,
                    kernel_size=1, stride=1, padding=0))
    return model

def MLP_Decoder(s_dim, n_cout, n_hw):
    model = nn.Sequential(
                nn.ConvTranspose2d(s_dim, s_dim*2, 
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(s_dim*2, s_dim*3, 
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(s_dim*3, n_cout, 
                    kernel_size=n_hw, stride=1, padding=0),
                nn.Sigmoid())
    return model

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)


### Define paper encoder network
class Encoder_ColorMNIST(nn.Module):
    def __init__(self, s_dim, n_cin, n_hw) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_cin*n_hw*n_hw, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, s_dim*2)
        )

    def forward(self, x):
        return self.model(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, 1, 1)

class Decoder_ColorMNIST(nn.Module):
   
    def __init__(self, s_dim, n_cout, n_hw) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_cout*n_hw*n_hw),
            nn.Sigmoid()
        )
        self.output_dim = (n_cout, n_hw, n_hw)

    def forward(self, z):
        return self.model(z.reshape(z.shape[0], -1)).reshape((z.shape[0],)+self.output_dim)


class Encoder_Chairs(nn.Module):

    def __init__(self, s_dim, n_cin, n_hw):
        nn.Module.__init__(self)

       

        layers = nn.Sequential(
            nn.Conv2d(n_cin, 16, 4, 2, padding=1),
            nn.Conv2d(16, 32, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.LeakyReLU(),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),

        )

        self.layers = layers
        self.embedding = nn.Linear(128 * 4 * 4, s_dim*2)
        self.input_dim = (n_cin, n_hw, n_hw)
        
    def forward(self, x: torch.Tensor):
        
        out = self.layers(x.reshape((-1,) + self.input_dim))
        return self.embedding(out.reshape(-1, 128*4*4)).reshape(x.shape[0], -1, 1, 1)


class Decoder_Chairs(nn.Module):
    def __init__(self, s_dim, n_cout, n_hw):
        nn.Module.__init__(self)

        self.output_dim = (n_cout, n_hw, n_hw)

        self.fc = nn.Linear(s_dim, 128 * 4 * 4)

        layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, n_cout, 4, 2, padding=1)
            #nn.Sigmoid()
        )   

        self.layers = layers

    def forward(self, z: torch.Tensor):
       
        out = self.fc(z.reshape(z.shape[0], -1)).reshape(z.shape[0], 128, 4, 4)
        return self.layers(out)
