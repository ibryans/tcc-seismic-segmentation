import os
import torch 
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from data.data_loader import split_train, SeismicDataset
from core.autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt


# Processador ou GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)

print('🔹 Device configurado -> ' + str(device))

# Constantes de treinamento
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

# Pré-processamento (transformações) das imagens
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Gera os splits de treino
split_train()

print('🔹 Gerou splits de treino')

# Cria o dataset de treino
train_dataset = SeismicDataset(transforms=transform, split='train')
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print('🔹 Gerou o dataset completo de treino \n')

# Criando o modelo
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print('🔹 Começando o treinamento...\n')
train_loss = []



# Processo de treinamento
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for data in train_loader:
        images = data.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        print("🔹 Predict realizado!")

        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss = running_loss / len(train_loader)
    train_loss.append(loss)

    print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, NUM_EPOCHS, loss))





def test_image_reconstruction(image, predict):
    figure = plt.figure()
    ax1 = figure.add_subplot(121)
    ax2 = figure.add_subplot(122)
    sim1 = ax1.imshow(image.cpu().detach().numpy().transpose((1,2,0)), cmap = 'gray')
    sim2 = ax2.imshow(predict.cpu().detach().numpy().transpose((1,2,0)), cmap = 'gray')
    plt.show()