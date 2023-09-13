"""
Deep Convolutional Autoencoder
Bryan S. Oliveira
TCC - PUC/2023
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
  def __init__(self):
    super(ConvAutoencoder, self).__init__()

    self.device = torch.device(
      "cuda" if torch.cuda.is_available() 
      else "cpu"
    )
    
    # Encoder
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 16, 3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, 7),
      nn.ReLU()
    )
    
    # Decoder
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(64, 32, 7),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.Sigmoid()
    )

    self.encoder.to(self.device)
    self.decoder.to(self.device)

  def forward(self, x):
    x = self.encoder(x)
    latent_space = x
    x = self.decoder(x)

    return x