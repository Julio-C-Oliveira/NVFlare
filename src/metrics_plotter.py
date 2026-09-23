import os
import re
import glob
import matplotlib.pyplot as plt


def get_job_name(alpha: float, mu: float) -> str:
    """Gera o nome do job no padrão usado pelo run_job.py."""
    alpha_str = str(alpha).replace('.', '_')
    mu_str = str(mu).replace('.', '_')
    return f"fl_cifar10_alpha{alpha_str}_mu{mu_str}"


def parse_client_accuracies(job_name: str, base_dir: str = "/tmp/nvflare/simulation"):
    """
    Lê os arquivos log.txt de cada site-<N> no diretório do job
    e extrai as acurácias de teste por rodada.

    Retorna um dicionário no formato:
    {
        'site-1': {0: 0.1805, 1: 0.2102, ...},
        'site-2': {0: 0.1912, 1: 0.2234, ...}
    }
    """
    job_dir = os.path.join(base_dir, job_name)
    if not os.path.exists(job_dir):
        print(f"Aviso: Diretório do job não encontrado: {job_dir}")
        return {}

    # Busca diretórios de sites (site-1, site-2, ...)
    site_dirs = glob.glob(os.path.join(job_dir, "site-*"))
    site_dirs.sort()

    client_metrics = {}
    pattern = re.compile(r"\[(site-\d+)\] Rodada (\d+): Test Loss = ([\d\.]+) \| Test Acc = ([\d\.]+)")

    for site_dir in site_dirs:
        site_name = os.path.basename(site_dir)
        log_file = os.path.join(site_dir, "log.txt")

        if not os.path.exists(log_file):
            continue

        site_accs = {}
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    s_name, round_idx, loss_val, acc_val = match.groups()
                    site_accs[int(round_idx)] = float(acc_val)

        if site_accs:
            client_metrics[site_name] = site_accs

    return client_metrics


def plot_client_accuracies(job_name: str, title: str = None, base_dir: str = "/tmp/nvflare/simulation", ax=None):
    """
    Plota o gráfico de acurácia por cliente ao longo das rodadas.
    Pode plotar em um eixo Matplotlib específico (ax) ou criar uma figura nova.
    """
    metrics = parse_client_accuracies(job_name, base_dir=base_dir)

    if not metrics:
        print(f"Nenhum dado encontrado para o job: {job_name}")
        return

    show_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        show_fig = True

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for idx, (site_name, round_dict) in enumerate(sorted(metrics.items())):
        rounds = sorted(round_dict.keys())
        accs = [round_dict[r] for r in rounds]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax.plot(
            rounds,
            accs,
            label=f"Cliente {idx + 1} ({site_name})",
            marker=marker,
            linewidth=2.2,
            markersize=7,
            color=color
        )

        # Adiciona os valores nas pontas dos marcadores
        for r, acc in zip(rounds, accs):
            ax.annotate(
                f"{acc * 100:.1f}%",
                (r, acc),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=9,
                fontweight='bold',
                color=color
            )

    plot_title = title if title else f"Acurácia por Cliente - {job_name}"
    ax.set_title(plot_title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Rodada Global", fontsize=10)
    ax.set_ylabel("Acurácia de Teste", fontsize=10)

    # Formatação do eixo X e Y
    all_rounds = sorted(list(set(r for s in metrics.values() for r in s.keys())))
    ax.set_xticks(all_rounds)
    ax.set_xticklabels([f"Rodada {r}" for r in all_rounds])
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    if show_fig:
        plt.tight_layout()
        plt.show()


def plot_grid_comparison(experiments: list, base_dir: str = "/tmp/nvflare/simulation"):
    """
    Plota um grid comparativo de experimentos (ex: 2x2).
    `experiments` deve ser uma lista de dicionários no formato:
    [
        {"alpha": 0.5, "mu": 0.01, "title": "Non-IID (α=0.5) | FedProx (μ=0.01)"},
        {"alpha": 0.5, "mu": 0.0,  "title": "Non-IID (α=0.5) | FedAvg (μ=0.0)"},
        {"alpha": 100.0, "mu": 0.01, "title": "IID (α=100) | FedProx (μ=0.01)"},
        {"alpha": 100.0, "mu": 0.0,  "title": "IID (α=100) | FedAvg (μ=0.0)"}
    ]
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, exp in enumerate(experiments):
        alpha = exp.get("alpha", 100.0)
        mu = exp.get("mu", 0.0)
        title = exp.get("title", f"Alpha {alpha} | Mu {mu}")

        job_name = get_job_name(alpha, mu)
        plot_client_accuracies(job_name, title=title, base_dir=base_dir, ax=axes_flat[idx])

    plt.suptitle("Comparativo de Acurácia dos Clientes entre Cenários Federados", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
