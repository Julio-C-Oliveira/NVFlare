import argparse
import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter
from src.model import SimpleCNN


def partition_cifar10_dirichlet(train_dataset, client_id, num_clients, alpha=100.0):
    """
    Particiona os índices do dataset CIFAR-10 usando a distribuição de Dirichlet (Não-IID).
    Se alpha for alto (ex: 100.0), aproxima-se de uma distribuição IID uniforme.
    """
    np.random.seed(42)  # Semente fixa para garantir partição reprodutível

    labels = np.array(train_dataset.targets)
    num_classes = 10
    client_indices = []

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_k_indices = np.split(idx_k, splits)
        if client_id < len(client_k_indices):
            client_indices.extend(client_k_indices[client_id])

    return Subset(train_dataset, client_indices)


def main():
    parser = argparse.ArgumentParser(description="NVFlare Client Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Caminho absoluto para o diretório de dados CIFAR-10")
    parser.add_argument("--epochs", type=int, default=1, help="Épocas locais de treino por rodada")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do mini-batch")
    parser.add_argument("--alpha", type=float, default=100.0, help="Parâmetro Alpha da distribuição de Dirichlet")
    parser.add_argument("--num_clients", type=int, default=2, help="Número total de clientes na simulação")
    parser.add_argument("--mu", type=float, default=0.0, help="Coeficiente Proximal para FedProx (0.0 = FedAvg)")
    parser.add_argument("--max_batches", type=int, default=40, help="Limite de mini-batches processados por época")

    args = parser.parse_args()

    # Validação de argumentos antes de inicializar o NVFlare
    if args.mu < 0.0:
        parser.error("--mu deve ser >= 0.0")

    # 1. Inicializa o cliente NVFlare e o SummaryWriter
    flare.init()
    writer = SummaryWriter()

    site_name = flare.get_site_name()
    try:
        client_id = int(site_name.split("-")[-1]) - 1
    except Exception:
        client_id = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Transformações padronizadas para CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Carrega dados de treino e teste sem download (download=False) a partir de data_dir
    data_path = os.path.abspath(args.data_dir)
    full_train = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
    local_data = partition_cifar10_dirichlet(full_train, client_id, args.num_clients, alpha=args.alpha)
    train_loader = DataLoader(local_data, batch_size=args.batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[{site_name}] Pronto. Amostras locais: {len(local_data)} | Alpha: {args.alpha}")

    # 2. Ciclo principal de treinamento federado
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        current_round = input_model.current_round

        # Atualiza o modelo local com os parâmetros recebidos do servidor
        net.load_state_dict(input_model.params)

        # Captura cópia dos parâmetros globais para o termo de penalidade do FedProx (se mu > 0.0)
        global_params = copy.deepcopy(list(net.parameters()))

        # Treinamento Local
        net.train()
        for _ in range(args.epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                if batch_idx >= args.max_batches:
                    break

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = net(x)
                loss = criterion(outputs, y)

                # Penalidade proximal do FedProx
                if args.mu > 0.0:
                    proximal_term = sum(((p - p0) ** 2).sum() for p, p0 in zip(net.parameters(), global_params))
                    loss += (args.mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()

        # 3. Avaliação no conjunto de teste completo
        net.eval()
        test_loss, test_correct, test_samples = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                loss = criterion(outputs, y)

                test_loss += loss.item() * x.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == y).item()
                test_samples += x.size(0)

        epoch_test_loss = test_loss / test_samples
        epoch_test_acc = test_correct / test_samples
        print(f"[{site_name}] Rodada {current_round}: Test Loss = {epoch_test_loss:.4f} | Test Acc = {epoch_test_acc:.4f}")

        # Grava métricas no TensorBoard por rodada
        writer.add_scalar("test/loss", epoch_test_loss, current_round)
        writer.add_scalar("test/accuracy", epoch_test_acc, current_round)

        # 4. Envia o modelo treinado e métricas de volta ao servidor
        output_model = flare.FLModel(
            params=net.state_dict(),
            metrics={
                "accuracy": epoch_test_acc,
                "test_loss": epoch_test_loss
            }
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()