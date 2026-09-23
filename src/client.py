import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter
from src.model import SimpleCNN

def partition_cifar10_dirichlet(train_dataset, client_id, num_clients, alpha=0.5):
    """
    Particiona os índices do dataset baseado na distribuição de Dirichlet (Não-IID).
    Se alpha for alto (ex: 100), aproxima-se do formato IID.
    """
    labels = np.array(train_dataset.targets)
    num_classes = 10
    client_indices = []

    np.random.seed(42)  # Semente fixa para reprodutibilidade
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_k_indices = np.split(idx_k, splits)
        client_indices.extend(client_k_indices[client_id])

    return Subset(train_dataset, client_indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=100.0, help="Alpha de Dirichlet: 100=IID, 0.1=Não-IID")
    parser.add_argument("--num_clients", type=int, default=2)
    args = parser.parse_args()

    # 1. Inicializa o cliente NVFlare e o rastreador de métricas (TensorBoard)
    flare.init()
    writer = SummaryWriter()
    site_name = flare.get_site_name()
    
    # Extrai o ID numérico do site (ex: "site-1" -> índice 0)
    try:
        client_id = int(site_name.split("-")[-1]) - 1
    except Exception:
        client_id = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Preparação dos dados: Treino local e Teste global
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Dataset de treino particionado
    full_train = datasets.CIFAR10(root="/home/julio/documents/github/NVFlare/data", train=True, download=False, transform=transform)
    local_data = partition_cifar10_dirichlet(full_train, client_id, args.num_clients, alpha=args.alpha)
    train_loader = DataLoader(local_data, batch_size=args.batch_size, shuffle=True)

    # Dataset de teste (CIFAR-10 de teste: 10.000 imagens)
    test_dataset = datasets.CIFAR10(root="/home/julio/documents/github/NVFlare/data", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[{site_name}] Pronto. Amostras locais: {len(local_data)} | Alpha: {args.alpha}")

    # 2. Ciclo de Rodadas Federadas
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        current_round = input_model.current_round
        net.load_state_dict(input_model.params)

        # Treinamento Local
        net.train()
        for _ in range(args.epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = net(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        # 3. Avaliação no Conjunto de Teste
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

        # Grava apenas as métricas de teste no TensorBoard
        writer.add_scalar("test/loss", epoch_test_loss, current_round)
        writer.add_scalar("test/accuracy", epoch_test_acc, current_round)

        # 4. Transmite tensores e métricas para o servidor
        output_model = flare.FLModel(
            params=net.state_dict(),
            metrics={
                "accuracy": epoch_test_acc,  # Utilizado pelo servidor para avaliar o melhor checkpoint
                "test_loss": epoch_test_loss
            }
        )
        flare.send(output_model)

if __name__ == "__main__":
    main()