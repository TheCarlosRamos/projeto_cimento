
# Regressão de Resistência Residual em Concreto com Fibras (FRC)

A análise utiliza modelos de Regressão Linear (Ridge), Random Forest e Redes Neurais Profundas (DNN) para determinar a influência de parâmetros do concreto e da fibra (como teor de fibra, $f_{ck}$, comprimento e diâmetro) no desempenho do material.

-----

## Tecnologias Utilizadas

  * **Python**
  * **Pandas** e **NumPy** (Manipulação de Dados)
  * **Matplotlib** e **Seaborn** (Visualização de Dados)
  * **Scikit-learn (sklearn)** (Random Forest, Regressão Ridge, Pré-processamento, Avaliação)
  * **TensorFlow/Keras** (Rede Neural Profunda - DNN)
  * **Jupyter Notebook** (Ambiente de Desenvolvimento)

-----

## Estrutura do Projeto

  * `regressao_concreto_fibras.ipynb.fixed`: O *notebook* principal contendo todo o código de importação, pré-processamento, análise exploratória, modelagem (DNN, Ridge), treinamento e avaliação.
  * **Dados:** Os dados de entrada consistem em propriedades do concreto e da fibra.
  * `modelo_fR1.h5`: Modelo Keras (DNN) treinado para prever o $f_{R,1}$.
  * `modelo_fR3.h5`: Modelo Keras (DNN) treinado para prever o $f_{R,3}$.
  * `scaler_X.pkl`: Objeto `StandardScaler` treinado para as variáveis de entrada.
  * `scaler_y.pkl`: Objeto `StandardScaler` treinado para a variável alvo ($f_{R,1}$ ou $f_{R,3}$).

-----

## Metodologia e Resultados Principais

### 1\. Variáveis de Entrada (Features)

O modelo utiliza as seguintes características para prever a resistência residual:

  * **$f_{ck}$ (MPa):** Resistência à compressão do concreto.
  * **$l$ (mm):** Comprimento da fibra.
  * **$d$ (mm):** Diâmetro da fibra.
  * **$l/d$:** Fator de forma (relação comprimento/diâmetro).
  * **Teor\_fibra ($\%$):** Teor volumétrico de fibra.
  * **N\_ganchos:** Número de ganchos (para fibras de aço).

### 2\. Modelagem e Desempenho

| Modelo | Métrica | Valor | Observação |
| :--- | :--- | :--- | :--- |
| **Random Forest** | $R^2$ | $0.8095$ | Melhor capacidade de ajuste geral. |
| **Regressão Ridge** | $R^2$ | $0.7489$ | Base para a equação linear interpretável. |
| **Equação Linear** | $R^2$ | $0.7869$ | Excelente ajuste para um modelo interpretável. |
| **Equação Linear** | MAE | $0.8894$ | Erro Médio Absoluto em N/mm². |

### 3\. Equação Linear Estimada (Regressão Ridge)

O modelo de Regressão Ridge resulta em uma equação, que é altamente interpretável e pode ser usada para cálculo manual (termos em unidades originais):

-----

## Como Executar o Projeto

1.  **Pré-requisitos:** Instale as bibliotecas Python necessárias:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras joblib
    ```
2.  **Execução:** Abra o arquivo `regressao_concreto_fibras.ipynb.fixed` em um ambiente Jupyter (JupyterLab, VS Code, Google Colab).
3.  **Análise:** Execute as células sequencialmente para:
      * Carregar e inspecionar os dados.
      * Visualizar a Análise Exploratória de Dados (AED) e a matriz de correlação.
      * Treinar os modelos DNN para $f_{R,1}$ e $f_{R,3}$.
      * Avaliar e salvar os modelos e *scalers*.
      * Fazer uma previsão com dados de um novo concreto.

-----
