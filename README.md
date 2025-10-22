
# Análise e Regressão de Concreto com Fibras


Este notebook Jupyter tem como objetivo principal aplicar técnicas de Machine Learning tradicionais (Regressão Múltipla) para **derivar uma equação explícita e interpretável** que preveja a resistência residual à tração na flexão do concreto com fibras ($f_{R,1}$).

O foco é na **interpretabilidade linear** (através dos coeficientes da Regressão Ridge) em detrimento da complexidade preditiva de modelos de Deep Learning (DNN).

-----

## Objetivo do Projeto

1.  **Previsão de $f_{R,1}$:** Prever a Resistência Residual ($f_{R,1}$) do Concreto Reforçado com Fibras (FRC) com base em suas características físicas.
2.  **Equação Explicita:** Utilizar a Regressão Ridge (regularizada) para obter os coeficientes que formam a equação linear de previsão mais robusta.
3.  **Comparação:** Usar o Random Forest como um modelo não-linear de *benchmark* para avaliar o potencial preditivo máximo dos dados.

-----

## Tecnologias e Dependências

O notebook foi desenvolvido em Python e requer as seguintes bibliotecas:

  * **`pandas`** e **`numpy`**: Para manipulação e cálculo de dados.
  * **`scikit-learn` (`sklearn`)**: Contém os algoritmos de Machine Learning (`RidgeCV`, `RandomForestRegressor`) e ferramentas de pré-processamento (`StandardScaler`).
  * **`matplotlib`** e **`seaborn`**: Para visualização e análise de dados.

Você pode instalar todas as dependências usando `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

-----

## Estrutura e Dados

### Variáveis de Entrada (Features - $X$)

O modelo utiliza as seguintes características do concreto e das fibras:

| Variável | Descrição |
| :--- | :--- |
| `fck_MPa` | Resistência à compressão do concreto (fck). |
| `l_mm` | Comprimento da fibra. |
| `d_mm` | Diâmetro da fibra. |
| `l_d` | Fator de forma da fibra (Razão l/d). |
| `Teor_fibra_percent` | Teor de volume de fibras (em porcentagem). |
| `N_ganchos` | Indicador da presença de ganchos ou forma de ancoragem da fibra. |

### Variável de Saída (Alvo - $y$)

  * `fR1_experimental`: Resistência Residual do Concreto (em $\text{N/mm}^2$ ou $\text{MPa}$) no CMOD de $0.5 \text{ mm}$.

-----

## Como Executar o Notebook

1.  **Abra o Notebook:** Inicie o JupyterLab ou Jupyter Notebook e abra o arquivo `regressao_concreto_fibras_final.ipynb`.
2.  **Substitua os Dados:** Substitua o dicionário de dados de exemplo no Bloco 2 pelos seus dados completos (recomenda-se carregar um arquivo `.csv` ou `.xlsx`).
3.  **Execução Sequencial:** Execute todas as células em ordem (pode usar `Run -> Run All Cells`).
4.  **Análise:**
      * Verifique os resultados de $R^2$ para Random Forest e Ridge (o primeiro deve ser maior).
      * Analise os coeficientes da Regressão Ridge para interpretar a influência de cada variável na equação linear.



O notebook gera as seguintes saídas principais:

  * **Resultados de ML:** Métricas de avaliação (MAE e $R^2$) para Random Forest e Regressão Ridge no conjunto de teste.
  * **Coeficientes da Equação:** Lista dos coeficientes de regressão que definem a equação de previsão.
  * **Gráficos:** Visualização da comparação entre os valores reais ($y_{true}$) e os valores previstos ($y_{hat}$) pela equação final.
