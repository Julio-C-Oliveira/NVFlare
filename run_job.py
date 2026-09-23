import argparse
import os
import sys
import torch
from torchvision import datasets

from src.model import SimpleCNN
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def main():
    parser = argparse.ArgumentParser(description="Orquestrador Federado NVFlare")
    parser.add_argument("--clients", type=int, default=2, help="Número de clientes simulados")
    parser.add_argument("--rounds", type=int, default=4, help="Número de rodadas globais de agregação")
    parser.add_argument("--alpha", type=float, default=100.0, help="Grau de heterogeneidade Dirichlet (100.0=IID, 0.1=Não-IID)")
    parser.add_argument("--mu", type=float, default=0.0, help="Coeficiente Proximal para FedProx (0.0 = FedAvg)")
    parser.add_argument("--max_batches", type=int, default=40, help="Limite de mini-batches processados por época local")

    args = parser.parse_args()

    # Validação rigorosa de argumentos
    if args.clients < 1:
        print("Erro: --clients deve ser >= 1", file=sys.stderr)
        sys.exit(1)
    if args.rounds < 1:
        print("Erro: --rounds deve ser >= 1", file=sys.stderr)
        sys.exit(1)
    if args.alpha <= 0.0:
        print("Erro: --alpha deve ser > 0.0", file=sys.stderr)
        sys.exit(1)
    if args.mu < 0.0:
        print("Erro: --mu deve ser >= 0.0", file=sys.stderr)
        sys.exit(1)
    if args.max_batches < 1:
        print("Erro: --max_batches deve ser >= 1", file=sys.stderr)
        sys.exit(1)

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    train_script_path = os.path.join(project_root, "src", "client.py")

    print("Verificando integridade dos dados...")
    try:
        datasets.CIFAR10(root=data_dir, train=True, download=True)
        datasets.CIFAR10(root=data_dir, train=False, download=True)
    except Exception as e:
        print(f"Erro ao carregar ou baixar CIFAR-10: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Iniciando simulação federada com {args.clients} nós...")
    print(f"Configuração: {args.rounds} rodadas | Dirichlet alpha: {args.alpha} | mu: {args.mu} | max_batches: {args.max_batches}")

    job_name = f"fl_cifar10_alpha{str(args.alpha).replace('.', '_')}_mu{str(args.mu).replace('.', '_')}"
    train_args = f"--data_dir {data_dir} --alpha {args.alpha} --mu {args.mu} --max_batches {args.max_batches} --num_clients {args.clients}"

    recipe = FedAvgRecipe(
        name=job_name,
        model=SimpleCNN(),
        min_clients=args.clients,
        num_rounds=args.rounds,
        train_script=train_script_path,
        train_args=train_args,
        key_metric="accuracy",
        key_metric_mode="max",
    )

    add_experiment_tracking(recipe, tracking_type="tensorboard")
    env = SimEnv(num_clients=args.clients)

    try:
        run = recipe.execute(env)
    except Exception as e:
        print(f"Erro durante a execução do NVFlare SimEnv: {e}", file=sys.stderr)
        sys.exit(1)

    # Extração e salvamento do modelo global treinado
    result_dir = run.get_result()
    ckpt_path = os.path.join(result_dir, "server", "simulate_job", "app_server", "best_FL_global_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(result_dir, "server", "simulate_job", "app_server", "FL_global_model.pt")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_weights = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        global_model = SimpleCNN()
        global_model.load_state_dict(model_weights)
        save_path = os.path.join(project_root, "global_model.pt")
        torch.save(global_model.state_dict(), save_path)
        print(f"Modelo global salvo em: {save_path}")
    else:
        print(f"Aviso: Checkpoint do modelo não foi encontrado em {ckpt_path}", file=sys.stderr)

    print("TensorBoard: tensorboard --logdir=/tmp/nvflare/simulation/ --load_fast=false")


if __name__ == "__main__":
    main()