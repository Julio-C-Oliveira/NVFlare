import argparse
from src.model import SimpleCNN
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

def main():
    parser = argparse.ArgumentParser(description="Orquestrador Federado NVFlare")
    parser.add_argument("--clients", type=int, default=2, help="Número de clientes simulados")
    parser.add_argument("--rounds", type=int, default=5, help="Número de rodadas globais")
    parser.add_argument("--epochs", type=int, default=1, help="Épocas locais por rodada")
    parser.add_argument("--alpha", type=float, default=100.0, help="Grau de heterogeneidade Dirichlet")
    args = parser.parse_args()

    job_name = f"nvflare_cifar10_alpha_{args.alpha}"
    train_args = f"--epochs {args.epochs} --alpha {args.alpha} --num_clients {args.clients}"

    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=args.clients,
        num_rounds=args.rounds,
        model=SimpleCNN(),
        train_script="src/client.py",
        train_args=train_args
    )

    # Habilita telemetria integrada com TensorBoard
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    from torchvision import datasets, transforms

    print("Verificando integridade dos dados...")
    datasets.CIFAR10(root="./data", train=True, download=True)
    datasets.CIFAR10(root="./data", train=False, download=True)

    print(f"\nIniciando simulação federada com {args.clients} nós...")
    print(f"Configuração: {args.rounds} rodadas | Dirichlet alpha: {args.alpha}\n")

    env = SimEnv(num_clients=args.clients)
    run = recipe.execute(env)

    print("\nExecução finalizada com sucesso.")
    print(f"Status do Job: {run.get_status()}")
    print(f"Resultados e logs: {run.get_result()}")

if __name__ == "__main__":
    main()