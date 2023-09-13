from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
import torch
import numpy as np
import os


# Função pra criar os slices de treino (inlines e crosslines)
def split_train():
    seismic = np.load('data/datasets/f3_block/train/train_labels.npy')
    irange, crange, depth = seismic.shape

    # Pega a lista de inlines
    inline_list = list(range(irange))
    inline_list = ['i_' + str(inline) for inline in inline_list]

    # Pega a lista de crosslines
    # crossline_list = list(range(crange))
    # crossline_list = ['x_' + str(crossline) for crossline in crossline_list]

    # Junta os dois
    train_list = inline_list # + crossline_list

    # Gera uma lista de inlines e crosslines aleatórios
    train_list, test_list = train_test_split(train_list, train_size=0.99, shuffle=True)

    # Salva os slices num arquivo
    file_object = open('data/datasets/f3_block/splits/train.txt', 'w')
    file_object.write('\n'.join(train_list))
    file_object.close()



# Classe que implementa o Dataset de sísmicas
class SeismicDataset(Dataset):

    def __init__(self, transforms, split='train'):
        self.split = split
        self.seismic = np.load('data/datasets/f3_block/train/train_seismic.npy')
        self.transforms = transforms
        self.mean = 0.0009996710808862074

        # Pegando os slices de treinamento
        splits = 'data/datasets/f3_block/splits/train.txt'
        file_list = tuple(open(splits, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.slices = file_list


    def __len__(self):
        return len(self.slices)
    

    def __getitem__(self, index):
        slice_name = self.slices[index]
        direction, number = slice_name.split(sep='_')
        slice_number = int(number)

        # Se for um inline
        if direction == 'i':
            image = self.seismic[slice_number, :, :].transpose((1,0))
        elif direction == 'x':
            image = self.seismic[:, slice_number, :].transpose((1,0))

        image = self.transform(image)

        return image



    def transform(self, image):
        # to be in the BxCxHxW that PyTorch uses: 
        if len(image.shape) == 2:
            image = np.expand_dims(image,0)

        image = torch.from_numpy(image).float()
                
        return image