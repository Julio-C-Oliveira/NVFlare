import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Arquitetura de Rede Neural Convolucional Leve (SimpleCNN) para Aprendizado Federado.

    Papel Federado:
    Esta rede é serializada em `state_dict` e transmitida entre o Servidor (orquestrador)
    e os Clientes (sites). Ela é projetada sem camadas de Normalização em Lote (BatchNorm2d)
    para evitar a divergência de estatísticas locais (running_mean e running_var) entre clientes
    com distribuições de dados heterogêneas (não-IID).
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Camada convolucional 1: 3 canais de entrada (RGB), 16 mapas de mapas de características, kernel 3x3, padding 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Camada convolucional 2: 16 canais de entrada, 32 mapas de características, kernel 3x3, padding 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Subamostragem (Pooling): reduz as dimensões espaciais (largura e altura) pela metade (2x2)
        self.pool = nn.MaxPool2d(2, 2)
        # Camada totalmente conectada (densa): mapeia 32 mapas de 8x8 (após 2 max poolings) para 128 neurônios
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        # Camada de saída: 128 neurônios para 10 classes (CIFAR-10)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Bloco convolucional 1: Conv -> ReLU -> MaxPool (dimensão 32x32 -> 16x16)
        x = self.pool(F.relu(self.conv1(x)))
        # Bloco convolucional 2: Conv -> ReLU -> MaxPool (dimensão 16x16 -> 8x8)
        x = self.pool(F.relu(self.conv2(x)))
        # Achata a representação espacial (32 * 8 * 8 = 2048 elementos por exemplo)
        x = torch.flatten(x, 1)
        # Camada densa oculta com ativação ReLU
        x = F.relu(self.fc1(x))
        # Saída linear (logits de classificação para 10 classes)
        x = self.fc2(x)
        return x