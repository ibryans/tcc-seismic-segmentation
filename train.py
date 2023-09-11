import os
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from data.data_loader import create_dataset
from core.autoencoder import Autoencoder, Encoder, Decoder


# Processador ou GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)

# Constantes de treinamento
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

# Pré-processamento (transformações) das imagens
transform = transforms.Compose([
    transforms.ToTensor(),
])


train_set, test_set = create_dataset()

train_loader = DataLoader(
    train_set, 
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_set, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

# Criando o modelo
model = Autoencoder(Encoder, Decoder)
criterion = nn.MSELoss()
train_loss = []
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Processo de treinamento
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for data in train_loader:
        images, _ = data
        images = images.to(device)
        images = images.view(images.size(0), -1)   # analisar

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss = running_loss / len(train_loader)
    # train_loss.append(loss)

    print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss))






        

def train(net, trainloader, NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss))
        if epoch % 5 == 0:
            save_decoded_image(outputs.cpu().data, epoch)
    return train_loss
def test_image_reconstruction(net, testloader):
     for batch in testloader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = net(img)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(outputs, 'fashionmnist_reconstruction.png')
        break