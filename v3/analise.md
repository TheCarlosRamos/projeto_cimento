# RELATÓRIO TÉCNICO: MÉTODOS, RESULTADOS E COMPARAÇÃO  
**Projeto: Previsão de fR,1 em Concreto com Fibras**

---

## 1. **MÉTODOS UTILIZADOS**

### 1.1. **Pré-processamento e Preparação de Dados**
- **Fonte:** `teste banco de dados.xlsx` (281 amostras brutas)
- **Etapas:**
  1. Detecção automática do cabeçalho (linha contendo “Tipo do concreto”)
  2. Remoção de colunas irrelevantes (ex.: “Unnamed”)
  3. Conversão para numérico e tratamento de valores nulos (8 amostras removidas)
  4. Normalização das features (StandardScaler) para modelos sensíveis à escala
  5. Divisão treino/teste (80/20) com `random_state=42`

### 1.2. **Modelos Implementados**

| Método                          | Tipo           | Objetivo                                     | Configurações Principais                              |
|----------------------------------|----------------|----------------------------------------------|-------------------------------------------------------|
| **DNN (Deep Neural Network)**   | Rede Neural    | Prever fR,1 e extrair equação linear aproximada | 3 camadas (128, 64, 1), Dropout(0.2), Adam(lr=0.001), 200 épocas |
| **Regressão Linear (RL)**       | Linear         | Baseline simples                              | `LinearRegression()` (scikit-learn)                   |
| **Gradient Boosting (GB)**      | Ensemble       | Regressão de alta precisão                    | `n_estimators=300, learning_rate=0.05, max_depth=3`   |
| **BaggingClassifier**           | Classificação  | Classificar fR,1 como High/Low (binário)      | `BaggingClassifier()` (melhor classificador)          |
| **LazyPredict (AutoML)**        | Multi-modelos  | Comparação automática de 26+ modelos          | `LazyClassifier` e `LazyRegressor`                    |
| **SHAP Analysis**               | Interpretação  | Explicar contribuições das features           | `TreeExplainer` (Gradient Boosting)                   |

### 1.3. **Métricas de Avaliação**
- **Regressão:** R², MAE (Mean Absolute Error), RMSE (Root Mean Square Error)
- **Classificação:** Acurácia, F1-Score, Acurácia Balanceada
- **Validação:** Validação cruzada 5-fold, split treino/validação/teste

---

## 2. **RESULTADOS POR MÉTODO**

### 2.1. **DNN + Equação Linear Aproximada**
```
Equação extraída:
fR,1 ≈ 2.65228 + 
(0.06408 × fck) + 
(0.00680 × l) + 
(-1.04693 × d) + 
(0.03438 × l/d) + 
(0.18683 × Teor de fibra) + 
(0.46285 × N)
```
**Desempenho:**
- **R²:** 0.774
- **MAE:** 0.506
- **RMSE:** 1.408
- **Observação:** Modelo interpretável, com erro médio de ±0.5 N/mm².

### 2.2. **Regressão Linear (Baseline)**
- **R²:** 0.702
- **MAE:** 1.187
- **RMSE:** 1.539
- **Observação:** Performance inferior, mas útil como baseline.

### 2.3. **Gradient Boosting Regressor**
- **R²:** 0.890
- **MAE:** 0.336
- **RMSE:** 0.881
- **Validação Cruzada (R² médio):** 0.872 ± 0.089
- **Observação:** Melhor desempenho geral, robusto a outliers.

### 2.4. **Classificação (High/Low fR,1)**
**Melhor modelo:** `BaggingClassifier`
- **Acurácia:** 87.72%
- **F1-Score:** 87.71%
- **Acurácia Balanceada:** 87.68%
- **Distribuição das classes:** Low_fR1 (141), High_fR1 (140)

### 2.5. **LazyPredict – Top 5 Modelos (Regressão)**
| Modelo                     | R²     | MAE   | RMSE  |
|----------------------------|--------|-------|-------|
| GradientBoostingRegressor  | 0.890  | 0.336 | 0.881 |
| ExtraTreesRegressor        | 0.880  | 0.342 | 0.901 |
| RandomForestRegressor      | 0.875  | 0.359 | 0.917 |
| XGBRegressor               | 0.868  | 0.371 | 0.932 |
| HistGradientBoostingRegressor | 0.865 | 0.373 | 0.939 |

### 2.6. **SHAP Analysis (Gradient Boosting)**
**Ordem de importância das features:**
1. **Teor de fibra (%)** – 67%
2. **fck (MPa)** – 15%
3. **l/d (fator de forma)** – 8%
4. **d (diâmetro)** – 5%
5. **l (comprimento)** – 3%
6. **N (ganchos)** – 2%

---

## 3. **COMPARAÇÃO DOS MÉTODOS**

### 3.1. **Desempenho em Regressão (fR,1)**
| Método                     | R²     | MAE   | RMSE  | Interpretabilidade | Tempo de Treinamento |
|----------------------------|--------|-------|-------|-------------------|----------------------|
| Gradient Boosting          | **0.890** | **0.336** | **0.881** | Média (com SHAP)  | Moderado             |
| DNN + Equação Aproximada   | 0.774  | 0.506  | 1.408  | **Alta**           | Alto                 |
| Regressão Linear           | 0.702  | 1.187  | 1.539  | **Alta**           | Baixo                |
| ExtraTreesRegressor        | 0.880  | 0.342  | 0.901  | Baixa              | Moderado             |
| XGBRegressor               | 0.868  | 0.371  | 0.932  | Média (com SHAP)  | Moderado             |

### 3.2. **Classificação (High/Low fR,1)**
| Modelo                | Acurácia | F1-Score | Balanced Accuracy |
|-----------------------|----------|----------|-------------------|
| BaggingClassifier     | **87.72%** | **87.71%** | **87.68%**        |
| ExtraTreesClassifier  | 85.96%   | 85.96%   | 86.02%            |
| SGDClassifier         | 85.96%   | 85.94%   | 85.90%            |
| LinearSVC             | 85.96%   | 85.78%   | 85.78%            |
| LogisticRegression    | 85.96%   | 85.78%   | 85.78%            |

### 3.3. **Análise de Correlação com fR,1**
| Feature              | Correlação (r) | Classificação |
|----------------------|----------------|---------------|
| Teor de fibra (%)    | **0.8833**     | Alta positiva |
| fck (MPa)            | 0.4379         | Média positiva |
| l/d                  | 0.2833         | Baixa positiva |
| N (ganchos)          | 0.0898         | Baixa positiva |
| l (comprimento)      | -0.0213        | Baixa negativa |
| d (diâmetro)         | -0.2810        | Baixa negativa |

---

## 4. **ANÁLISE CRÍTICA E OBSERVAÇÕES**

### 4.1. **Pontos Fortes por Método**
- **Gradient Boosting:** Melhor precisão geral, robustez a outliers, boa generalização.
- **DNN + Equação Linear:** Equação interpretável em unidades originais, boa relação precisão/explicabilidade.
- **BaggingClassifier:** Alta acurácia em classificação binária, estável.
- **SHAP:** Explicabilidade clara das contribuições das features.

### 4.2. **Limitações**
- **DNN:** Requer normalização, maior tempo de treinamento, risco de overfitting sem tuning fino.
- **Regressão Linear:** Não captura relações não-lineares (ex.: interação entre fibra e fck).
- **LightGBM:** Avisos de “no further splits” sugerem possível overfitting ou dados insuficientes.
- **fR,3:** Modelos performaram pior (R² = 0.732), indicando maior complexidade ou ruído.

### 4.3. **Impacto da Remoção de Outliers**
- **Gradient Boosting (com remoção de 20% outliers):**
  - **MAE caiu de 0.644 para 0.336**
  - **R² subiu de 0.838 para 0.890**
- **Regressão Linear (com remoção):**
  - **R² subiu de 0.702 para 0.810**
- Conclusão: A qualidade dos dados é crítica para modelos de regressão.

---

## 5. **RECOMENDAÇÕES PARA USO PRÁTICO**

### 5.1. **Cenário: Prioridade em Precisão**
- **Use Gradient Boosting** se a precisão é o foco principal.
- **Aplique SHAP** para explicar previsões individuais.
- **Remova outliers** (20% piores erros) para melhorar robustez.

### 5.2. **Cenário: Prioridade em Interpretabilidade**
- **Use a Equação Linear da DNN** para cálculos manuais ou explicativos.
- **Complemente com Regressão Linear** para análises estatísticas simples.

### 5.3. **Cenário: Classificação Binária (High/Low)**
- **Use BaggingClassifier** para decisões baseadas em limites (ex.: aceitar/rejeitar lote).

### 5.4. **Para Pesquisa e Desenvolvimento**
- **Colete mais dados** para melhorar a generalização.
- **Experimente redes neurais mais profundas** com regularização.
- **Inclua variáveis contextuais** (tipo de fibra, idade, cura).

---

## 6. **CONCLUSÃO FINAL**

- **Gradient Boosting** é o método mais preciso para prever fR,1 (R² = 0.890).
- **DNN + Equação Linear** oferece um bom equilíbrio entre precisão e interpretabilidade.
- **Teor de fibra** é o fator mais influente (67% da importância, r = 0.88).
- **Remoção de outliers** melhora significativamente todos os modelos.
- **Classificação binária** é viável com acurácia >87%.
- **SHAP** é essencial para explicar previsões de modelos complexos.

---


Relatório gerado em: 2026-01-06  
Baseado na análise de: `project_cimento.pdf`
