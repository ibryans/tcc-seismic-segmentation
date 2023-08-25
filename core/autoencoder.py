"""
Deep Convolutional Autoencoder
Bryan S. Oliveira
TCC - PUC/2023
"""

import torch
import torch.nn as nn



class Encoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, activation_function=nn.ReLU()):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
        activation_function,
        nn.Conv2d(out_channels, out_channels, 3, padding=1), 
        activation_function,
        nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (16, 16)
        activation_function,
        nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
        activation_function,
        nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (8, 8)
        activation_function,
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        activation_function,
        nn.Flatten(),
        nn.Linear(4*out_channels*8*8, latent_dim),
        activation_function
    )

  def forward(self, x):
    x = x.view(-1, 3, 32, 32)
    output = self.net(x)
    return output



class Decoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, activation_function=nn.ReLU()):
    super().__init__()

    self.out_channels = out_channels

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 4*out_channels*8*8),
        activation_function
    )

    self.conv = nn.Sequential(
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (8, 8)
        activation_function,
        nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, 
                           stride=2, output_padding=1), # (16, 16)
        activation_function,
        nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
        activation_function,
        nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, 
                           stride=2, output_padding=1), # (32, 32)
        activation_function,
        nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
        activation_function,
        nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
    )

  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, 4*self.out_channels, 8, 8)
    output = self.conv(output)
    return output


# Classe principal do Autoencoder
class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()

    #  configuring device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('~ Rodando na GPU')
    else:
        device = torch.device('cpu')
        print('~ Rodando no Processador')

    self.encoder = encoder
    self.encoder.to(device)

    self.decoder = decoder
    self.decoder.to(device)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded