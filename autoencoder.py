
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """ Class to encode the EEG signal and combine channel information"""

    def __init__(self, n_channels: int=128):
        self.n_channels = n_channels
        super(Autoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 64), stride=(1, 16))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(64, 1), stride=(16, 1))

        # Deconvolution
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=(64, 1), stride=(16, 1))
        self.deconv2 = nn.ConvTranspose2d(8, 1, kernel_size=(1, 64), stride=(1, 16))

    
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        x = F.relu(self.conv2(x))

        # Deconvolution
        x = F.relu(self.deconv1(x))
        x = nn.Upsample(scale_factor=(1, 2))(x)
        x = F.relu(self.deconv2(x))
        
        return x

if __name__ == "__main__":
    # Create an instance of the autoencoder
    autoencoder = Autoencoder()
    # Create a random tensor
    x = torch.rand((1, 1, 128, 512))
    # Pass the tensor through the autoencoder
    y = autoencoder(x)
    # Print the shape of the output
    print(y.shape)
