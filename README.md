

# Regressão de Propriedades de Concreto com Fibras (ML e DNN)

Este projeto aplica e compara diferentes modelos de Machine Learning (ML) e Deep Learning (DNN) para prever as propriedades de resistência residual de concreto reforçado com fibras.

O foco principal é a previsão da resistência residual à tração do concreto (`fR,1`).

## Conteúdo do Repositório

O projeto é composto por dois notebooks Jupyter que exploram abordagens distintas de regressão, além do arquivo de dependências:

1.  **`ml_regressao.ipynb`**
      * Implementa modelos de **Machine Learning**.
      * Compara o desempenho de um **Random Forest Regressor** e uma **Regressão Linear Simples** (implementada via Keras).
      * O modelo Random Forest obteve uma métrica R² de aproximadamente **0.809**.
2.  **`dnn_regressao.ipynb`**
      * Treina uma **Deep Neural Network (DNN)** para a tarefa de regressão.
      * Foca na interpretabilidade, gerando uma **Equação Linear Aproximada** em **unidades originais** a partir das previsões da DNN.
3.  **`requirements.txt`**
      * Lista todas as bibliotecas e dependências necessárias para executar os notebooks.

## Conjunto de Dados

Os notebooks esperam que o arquivo de dados esteja no mesmo diretório, sob o nome **`teste banco de dados.xlsx`**.

### Variáveis

| Tipo | Variável | Unidade/Descrição |
| :--- | :--- | :--- |
| **Features (Entrada)** | `fck (resistência)` | Resistência do Concreto [MPa] |
| **Features (Entrada)** | `l (comprimento)` | Comprimento do corpo de prova/fibra [mm] |
| **Features (Entrada)** | `d (diâmetro)` | Diâmetro do corpo de prova/fibra [mm] |
| **Features (Entrada)** | `l/d (fator de forma)` | Fator de forma |
| **Features (Entrada)** | `Teor de fibra (%)` | Teor de fibra [%] |
| **Features (Entrada)** | `N (ganchos)` | Número de ganchos (na fibra) |
| **Target (Saída)** | `fR,1 (N/mm²) (experimental)` | Resistência Residual à Tração (Alvo principal) |
| **Target Alternativo** | `fR,3 (N/mm²) (experimental)` | Resistência Residual à Tração (Alvo secundário/alternativo) |

## Pré-requisitos

Para replicar os resultados, você precisará de um ambiente Python com as seguintes bibliotecas:

  * `pandas` (Manipulação de dados)
  * `numpy` (Computação numérica)
  * `scikit-learn` (Modelos de ML e pré-processamento)
  * `tensorflow` (Modelos DNN e Keras)
  * `matplotlib` e `seaborn` (Visualização de dados)
  * `openpyxl` (Para ler o arquivo de dados `.xlsx`)
  * `jupyter` e `ipykernel` (Para rodar os notebooks)

