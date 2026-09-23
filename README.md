# Pipeline Didático de Aprendizado Federado com PyTorch e NVFlare

Este repositório contém um **Pipeline Didático de Aprendizado Federado (Federated Learning - FL)** projetado para sessões práticas e laboratórios (duração estimada: 40 a 50 minutos). Utilizando **PyTorch** e o **NVIDIA NVFlare 2.5+**, os alunos aprendem os conceitos fundamentais de simulação federada, agregação via **FedAvg** e **FedProx**, e o impacto da heterogeneidade de dados (distribuição Não-IID via amostragem de Dirichlet).

---

## 🎯 Ordem Pedagógica Recomendada

Para extrair o máximo valor didático desta atividade, siga a ordem abaixo:

1. **Passo 1: Fundamentos & Arquitetura** (`notebooks/01_fundamentos_nvflare_pytorch.ipynb`)
   - Abra o notebook no Jupyter local ou no Google Colab.
   - Explore a arquitetura do NVFlare (**Bloco A**), a compatibilidade com múltiplos frameworks (**Bloco B**) e a anatomia das camadas PyTorch, funções de ativação e otimizadores (**Bloco C**).

2. **Passo 2: Execução & Simulação Interativa** (`notebooks/02_execucao_simulacao_nvflare.ipynb`)
   - Execução interativa da **Job Recipe API (`FedAvgRecipe`)** e do simulador **NVFlare (`SimEnv`)**.
   - Análise gráfica da heterogeneidade estatística via amostragem de Dirichlet ($Dir(\alpha)$) e comparação entre **FedAvg** e **FedProx**.

3. **Passo 3: Pipeline via Linha de Comando (CLI)** (`run_job.py`)
   - Execute o orquestrador federado via CLI e observe a agregação em tempo real dos clientes locais.
   - Abra o **TensorBoard** para acompanhar a evolução das curvas de perda e acurácia.

---

## 🛠️ Pré-requisitos e Instalação

- **Python 3.10+**
- **Virtualenv** `.venv`

```bash
# 1. Clonar o repositório e entrar no diretório
git clone https://github.com/julio/NVFlare.git && cd NVFlare

# 2. Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar as dependências do projeto
pip install -r requirements.txt
```

---

## 🚀 Cenários de Execução Rápida (Prontos para Uso)

### Cenário 1: Aprendizado Federado IID Padrão (MVP)
Executa 4 rodadas de agregação federada com 2 clientes e distribuição de dados IID ($\alpha = 100.0$).
```bash
python run_job.py
```

### Cenário 2: Comparação de Heterogeneidade (Não-IID Severo)
Simula uma alta heterogeneidade de dados entre os clientes ($\alpha = 0.1$).
```bash
python run_job.py --alpha 0.1
```

### Cenário 3: Mitigação de Deriva de Pesos com FedProx
Aplica o algoritmo **FedProx** com penalidade proximal ($\mu = 0.01$) para estabilizar o treino em ambiente não-IID.
```bash
python run_job.py --alpha 0.1 --mu 0.01
```

### Cenário 4: Simulação Multi-Cliente e Controle de Mini-Batches
Executa uma simulação com 3 clientes federados, 6 rodadas e limite de 20 mini-batches por época local.
```bash
python run_job.py --clients 3 --rounds 6 --alpha 0.5 --mu 0.01 --max_batches 20
```

---

## 📊 Visualização de Métricas com TensorBoard

Após disparar qualquer simulação federada, execute o seguinte comando no terminal para iniciar o servidor do TensorBoard:

```bash
tensorboard --logdir=/tmp/nvflare/simulation/ --load_fast=false
```

Acesse [http://localhost:6006](http://localhost:6006) em seu navegador para visualizar as curvas de `test/loss` e `test/accuracy` por cliente.

---

## 💾 Inspeção e Recarregamento do Modelo Global

Ao término da simulação, o checkpoint do modelo global otimizado é salvo automaticamente em `./global_model.pt`. Para inspecionar os parâmetros do modelo via Python:

```python
import torch
from src.model import SimpleCNN

# Instancia a arquitetura e carrega os pesos treinados
model = SimpleCNN()
model.load_state_dict(torch.load("global_model.pt"))

print("Chaves do state_dict do modelo global:")
for key, tensor in model.state_dict().items():
    print(f" - {key}: {tensor.shape}")
```

---

## 📓 Execução no Google Colab

O notebook interativo está preparado para rodar em instâncias gratuitas de CPU do Google Colab:
- Acesse `notebooks/01_fundamentos_nvflare_pytorch.ipynb`.
- Selecione **Ambiente de Execução** -> **Executar tudo** (`Ctrl + F9`).
- Tempo total estimado de execução: $\le 10$ minutos.

---

## 📁 Estrutura do Repositório

```
.
├── notebooks/
│   ├── 01_fundamentos_nvflare_pytorch.ipynb   # Notebook de Fundamentos (Blocos A, B, C)
│   └── 02_execucao_simulacao_nvflare.ipynb   # Notebook de Execução & Simulação NVFlare
├── src/
│   ├── model.py                               # Arquitetura SimpleCNN (sem BatchNorm)
│   └── client.py                              # Script do cliente NVFlare (Treino local & FedProx)
├── run_job.py                                 # Orquestrador CLI da simulação federada
├── requirements.txt                           # Dependências do projeto
├── .gitignore                                 # Regras de exclusão do Git
└── README.md                                  # Documentação principal
```
