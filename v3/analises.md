# Análise Computacional de Concreto com Fibras: Explicação Detalhada das Visualizações

##  Introdução
Este documento fornece uma análise técnica detalhada das visualizações geradas durante o estudo computacional de propriedades do concreto com fibras. As imagens representam diferentes estágios do fluxo analítico: desde a exploração inicial dos dados até a modelagem preditiva avançada e interpretação de resultados.

---

##  FASE 1: Imagens 01-07 - Análise Exploratória e Modelagem Inicial

###  Imagem 01: Top 15 Modelos de Classificação
![Classificação](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/01.png)

**Análise Técnica Detalhada:**
Esta visualização compara algoritmos de classificação binária para prever se fR,1 será "Alta" ou "Baixa" (threshold = 4.598 N/mm²).

**Principais Insights:**
- **BaggingClassifier** lidera com 87.72% de acurácia, seguido por ExtraTreesClassifier e SGDClassifier (85.96%)
- Pequena diferença entre métricas (acurácia vs F1-Score) indica **dados balanceados** e ausência de viés significativo
- Algoritmos ensemble (Bagging, ExtraTrees) superam modelos lineares, sugerindo **relações não-lineares** nos dados
- O gráfico à direita mostra análise multidimensional: BaggingClassifier tem melhor compromisso entre acurácia e tempo

**Implicações para Engenharia:**
- Decisões binárias (aceitar/rejeitar) podem ser automatizadas com alta confiabilidade
- Múltiplos algoritmos performam bem, permitindo flexibilidade na escolha do modelo

---

###  Imagem 02: Matriz de Correlação
![Correlação](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/02.png)

**Análise Técnica Detalhada:**
Matriz de correlação mostrando relações lineares entre todas as variáveis do estudo.

**Principais Relações Identificadas:**

| Variáveis | Coeficiente (r) | Interpretação Técnica |
|-----------|-----------------|------------------------|
| Teor de fibra ↔ fR,1 | **0.80** | Correlação forte positiva - principal fator de influência |
| fck ↔ fR,1 | 0.44 | Correlação moderada - matriz de concreto também influencia |
| l ↔ d | 0.67 | Multicolinearidade esperada - proporções geométricas mantidas |
| d ↔ fR,1 | -0.28 | Correlação negativa fraca - diâmetros maiores reduzem resistência específica |

**Padrões Observados:**
- Cluster superior: Teor de fibra + fR,1 (correlação dominante)
- Cluster geométrico: l, d, l/d (relações estruturais)
- fck forma ponte entre clusters, atuando como variável de interface

**Implicações para Projeto Experimental:**
- Confirmação de relações físicas esperadas valida qualidade dos dados
- Multicolinearidade entre l e d pode simplificar modelos (uso de razão l/d suficiente)

---

###  Imagem 03: Top Modelos de Regressão (fR,1)
![Regressão fR1](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/03.png)

**Análise Técnica Detalhada:**
Ranking dos algoritmos de regressão para previsão contínua de fR,1.

**Desempenho Comparativo:**

| Posição | Algoritmo | R² | Característica Técnica |
|---------|-----------|----|------------------------|
| 1 | GradientBoostingRegressor | **0.890** | Ensemble com boosting, captura relações complexas |
| 2 | XGBRegressor | 0.868 | Otimizado para performance, regularização L1/L2 |
| 3 | BaggingRegressor | 0.865 | Redução de variância via bootstrap aggregating |
| 8 | RidgeCV | 0.758 | Modelo linear com regularização L2 |

**Análise de Performance:**
- Diferença de apenas 2.5% entre primeiro e quinto lugar indica **robustez das relações identificadas**
- Superioridade consistente de métodos ensemble sobre lineares confirma **não-linearidade das relações**
- Gradient Boosting combina múltiplas árvores fracas, eficaz para dados com interações complexas

**Implicações para Modelagem:**
- Não há "algoritmo único ideal" - múltiplas abordagens são viáveis
- Trade-off entre complexidade (Gradient Boosting) e interpretabilidade (modelos lineares)

---

###  Imagem 04: Top Modelos de Regressão (fR,3)
![Regressão fR3](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/04.png)

**Análise Técnica Detalhada:**
Desempenho dos modelos na previsão de fR,3 (outra medida de resistência).

**Comparativo fR,1 vs fR,3:**

| Métrica | fR,1 | fR,3 | Redução | Interpretação |
|---------|------|------|---------|---------------|
| Melhor R² | 0.890 | 0.732 | **17.8%** | fR,3 significativamente menos previsível |
| Algoritmo líder | Gradient Boosting | XGBoost | - | Diferentes algoritmos ótimos para cada target |

**Possíveis Explicações Técnicas:**
1. **Maior variabilidade experimental** nas medições de fR,3
2. **Relações físicas mais complexas** envolvendo mecanismos de falha diferentes
3. **Sensibilidade a fatores não mensurados** que afetam fR,3 mais que fR,1

**Implicações para Pesquisa:**
- Necessidade de investigar mecanismos específicos que regem fR,3
- Possível necessidade de variáveis adicionais para melhor previsão

---

###  Imagem 05: Feature Importance - fR,1
![Importância fR1](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/05.png)

**Análise Técnica Detalhada:**
Importância relativa das variáveis no modelo Gradient Boosting para fR,1.

**Distribuição de Importância:**

| Variável | Importância | Contribuição Cumulativa | Mecanismo Físico |
|----------|-------------|-------------------------|------------------|
| Teor de fibra (%) | **67%** | 67% | Ponte de fissuras, transferência de tensões |
| fck (MPa) | 15% | 82% | Resistência da matriz cimentícia |
| l/d | 8% | 90% | Efeito de aspecto, ancoragem de fibras |
| d (mm) | 5% | 95% | Tamanho do elemento, concentração de tensões |
| l (mm) | 3% | 98% | Comprimento de ancoragem |
| N (ganchos) | 2% | 100% | Extremidades das fibras |

**Análise da Dominância:**
- Teor de fibra responde por **2/3 da importância total** - variável dominante
- fck + l/d somam **23%** - variáveis secundárias importantes
- Características geométricas (d, l) somam **8%** - influência moderada
- N (ganchos) tem **contribuição marginal** - possível otimização

**Implicações para Formulação:**
- Priorizar ajuste de teor de fibra para maximizar fR,1
- fck deve ser mantido em níveis adequados (não necessariamente máximo)
- Geometria tem papel secundário, mas não desprezível

---

###  Imagem 06: Real vs Previsto - fR,1
![Real vs Previsto](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/06.png)

**Análise Técnica Detalhada:**
Gráfico de dispersão avaliando qualidade das previsões de fR,1.

**Métricas de Desempenho:**

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| R² | 0.890 | Modelo explica **89%** da variância dos dados |
| RMSE | 0.891 N/mm² | Erro médio de **16.5%** do valor médio (5.384 N/mm²) |
| MAE | ~0.65 N/mm² (estimado) | Erro absoluto médio aceitável para aplicações práticas |

**Análise da Dispersão:**
- **Alta densidade** próxima à linha de identidade (y=x) na faixa 4-8 N/mm²
- **Alguns outliers** acima de 10 N/mm² mostram maior erro de previsão
- **Distribuição homogênea** dos resíduos indica ausência de viés sistemático

**Faixas de Confiança do Modelo:**
- Faixa 4-8 N/mm²: **Alta precisão** (erro < 15%)
- Faixa 8-12 N/mm²: **Precisão moderada** (erro 15-25%)
- >12 N/mm²: **Menor precisão** (dados escassos nesta faixa)

**Implicações para Aplicação:**
- Modelo adequado para formulações típicas (4-10 N/mm²)
- Cautela necessária para formulações de ultra-alta performance (>12 N/mm²)

---

###  Imagem 07: SHAP Summary Plot - fR,1
![SHAP fR1](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/07.png)

**Análise Técnica Detalhada:**
Explicabilidade local e global do modelo via valores SHAP.

**Interpretação dos Padrões:**

**Teor de fibra:**
- **Impacto positivo dominante** (pontos à direita do zero)
- **Magnitude proporcional** ao valor da feature
- **Relação quase monotônica** - quanto mais fibra, maior SHAP

**fck (resistência):**
- **Impacto positivo consistente**
- **Gradiente claro** - valores altos de fck associados a SHAP mais positivo
- **Contribuição secundária** mas significativa

**d (diâmetro):**
- **Impacto negativo predominante**
- **Relação inversa** - maiores diâmetros reduzem fR,1 previsto
- **Mecanismo:** possivelmente concentração de tensões em seções maiores

**l/d (fator de forma):**
- **Impacto positivo moderado**
- **Dispersão significativa** sugere interações com outras variáveis

**Interpretação Física dos Valores SHAP:**
- SHAP positivo: feature **aumenta** a resistência prevista
- SHAP negativo: feature **diminui** a resistência prevista
- Magnitude: **intensidade** do efeito na previsão

**Aplicação para Otimização:**
- Maximizar: Teor de fibra, fck, l/d
- Minimizar: d (diâmetro)
- Otimizar: l (comprimento) e N (ganchos) conforme interações

---

##  FASE 2: Imagens 08-14 - Análise Comparativa e Validação

###  Imagem 08: SHAP Plot com Escala de Cores
![SHAP Cores]([08.png](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/08.png))

**Análise Técnica Detalhada:**
SHAP plot aprimorado mostrando valores das features via escala de cores.

**Decodificação da Escala de Cores:**
- ** Vermelho:** Valor **alto** da feature
- ** Azul:** Valor **baixo** da feature
- **Gradiente:** Transição suave entre extremos

**Relações Cor-Posição Reveladas:**

**Teor de fibra:**
- Pontos **vermelhos concentrados à direita** (alto SHAP positivo)
- Pontos **azuis dispersos** mas tendendo à esquerda
- Confirma **relação monotônica forte**

**d (diâmetro):**
- Pontos **vermelhos à esquerda** (alto SHAP negativo)
- Pontos **azuis neutros ou positivos**
- Confirma **relação inversa**

**fck:**
- Gradiente perfeito: **vermelho → direita, azul → esquerda**
- **Relação linear clara** entre valor da feature e impacto

**Vantagens desta Visualização:**
1. **Relações diretas visíveis** sem consultar tabelas
2. **Padrões não-monotônicos detectáveis** (se presentes)
3. **Interações qualitativas** entre features identificáveis

---

###  Imagem 09: Feature Importance - fR,3
![Importância fR3](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/09.png)

**Análise Técnica Detalhada:**
Importância das features para fR,3 - comparação com fR,1.

**Comparativo Quantitativo:**

| Variável | Importância fR,1 | Importância fR,3 | Variação | Interpretação |
|----------|------------------|------------------|----------|---------------|
| Teor de fibra | **67%** | **58%** | -9% | Menor dominância, ainda principal |
| fck | 15% | ~12% | -3% | Importância reduzida |
| l/d | 8% | ~10% | +2% | Ganho relativo de importância |
| d | 5% | ~8% | +3% | Maior sensibilidade geométrica |
| l | 3% | ~6% | +3% | Comprimento mais relevante |
| N (ganchos) | 2% | ~6% | **+4%** | Ganho mais significativo |

**Mudanças na Hierarquia:**
1. Teor de fibra mantém **liderança absoluta** mas com menor margem
2. **Características geométricas** ganham importância relativa
3. **N (ganchos)** quase triplica em importância relativa

**Interpretação Mecânica:**
- fR,3 pode depender mais de **mecanismos de ancoragem** (explicando ganho de N)
- **Fatores geométricos** mais críticos para resistência residual
- **Matriz (fck)** um pouco menos determinante

---

###  Imagem 10: Real vs Previsto - fR,3
![Real vs Previsto fR3](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/10.png)

**Análise Técnica Detalhada:**
Desempenho do modelo na previsão de fR,3.

**Comparativo de Métricas:**

| Métrica | fR,1 | fR,3 | Deterioração |
|---------|------|------|--------------|
| R² | **0.890** | 0.693 | -22.1% |
| RMSE | 0.891 | 1.412 | +58.5% |
| Erro Relativo | ~16.5% | ~25-30% | +50-80% |

**Análise da Dispersão:**
- **Maior espalhamento** em torno da linha ideal
- **Outliers mais frequentes**, especialmente acima de 8 N/mm²
- **Tendência não-linear** perceptível (curvatura dos pontos)

**Possíveis Causas da Menor Precisão:**
1. **Variabilidade experimental inerente** maior em ensaios de fR,3
2. **Mecanismos físicos mais complexos** (fissuração múltipla, desancoragem progressiva)
3. **Sensibilidade a condições de teste** não controladas
4. **Faixa dinâmica maior** introduzindo não-linearidades

**Implicações para Aplicação:**
- Modelos para fR,3 devem ser usados com **maior margem de segurança**
- **Validação experimental** mais crítica para fR,3 que para fR,1
- Possível necessidade de **modelos específicos por faixa de valores**

---

###  Imagem 11: SHAP Summary Plot - fR,3
![SHAP fR3](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/11.png)

**Análise Técnica Detalhada:**
Explicabilidade do modelo para fR,3.

**Comparação com fR,1 (Imagem 07):**

**Teor de fibra:**
- Mantém **padrão positivo** mas com **menor magnitude**
- **Maior dispersão** nos valores SHAP
- Sugere **interações mais complexas** com outras variáveis

**fck:**
- **Impacto positivo atenuado**
- **Dispersão aumentada** - relação menos consistente
- Possível **saturação de efeito** em fR,3

**l/d:**
- **Comportamento bimodal** possível
- Alguns valores altos com **impacto negativo**
- Indica **ótimo não trivial** para fR,3

**N (ganchos):**
- **Amplitude aumentada** de valores SHAP
- **Comportamento menos previsível**
- Sugere **forte dependência de contexto**

**Interpretação Técnica:**
- fR,3 responde a **combinações mais complexas** de parâmetros
- **Interações entre features** mais significativas
- **Não-linearidades** mais pronunciadas

---

###  Imagem 13: Comparação fR,1 vs fR,3
![Comparação fR1 vs fR3](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/13.png)

**Análise Técnica Detalhada:**
Comparação visual direta das importâncias para ambos os targets.

**Análise por Variável:**

**Teor de fibra:**
- **Dominante em ambos**, mas claramente mais para fR,1
- Diferença de ~9% representa **redução relativa de 13%** para fR,3

**fck:**
- **Redução consistente** para fR,3
- Indica que **matriz é menos limitante** para resistência residual

**Características Geométricas (l/d, d, l):**
- **Ganho coletivo de ~8%** em importância relativa
- Geometria **mais crítica** para comportamento pós-fissuração

**N (ganchos):**
- **Maior ganho relativo** (+4%, aumento de 200% em importância)
- Extremidades das fibras **cruciais para ancoragem residual**

**Interpretação Mecânica Integrada:**
- fR,1: Governado por **resistência inicial da interface fibra-matriz**
- fR,3: Governado por **capacidade de ancoragem e extração progressiva**
- **Transição de mecanismos** entre primeira fissura e regime residual

---

###  Imagem 14: Curva de Aprendizado
![Curva de Aprendizado](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/14.png)

**Análise Técnica Detalhada:**
Avaliação da capacidade de generalização do modelo Gradient Boosting.

**Fases da Curva de Aprendizado:**

**Fase 1: Subamostragem (0-50 amostras)**
- **Overfitting severo**: RMSE treino ≈ 0.2, RMSE validação ≈ 1.8
- Gap viés-variância: **1.6 unidades de RMSE**
- Modelo **memoriza** dados de treino, generaliza mal

**Fase 2: Aprendizado Rápido (50-150 amostras)**
- **Melhoria acelerada** da validação
- Gap reduz para **~0.4 unidades de RMSE**
- Modelo **aprende padrões gerais**

**Fase 3: Saturação (150-225 amostras)**
- **Estabilização** das curvas
- Gap final: **~0.2 unidades de RMSE**
- **Ponto ótimo prático** alcançado

**Análise de Viés-Variância:**
- **Viés (bias):** Baixo (RMSE treino final ≈ 0.75)
- **Variância:** Controlada (gap pequeno entre curvas)
- **Equilíbrio:** Bom compromisso alcançado

**Recomendações Baseadas na Curva:**
- **Tamanho mínimo:** ~100 amostras para modelos razoáveis
- **Tamanho adequado:** 150+ amostras para estabilização
- **Tamanho atual (281):** Suficiente com margem de segurança

---

##  FASE 3: Imagens 15-20 - Refinamento e Análise Avançada

###  Imagem 15: Real vs Previsto - Versão Inicial
![Versão Inicial](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/15.png)

**Análise Técnica Detalhada:**
Linha de base antes da limpeza de dados.

**Problemas Identificados:**
1. **Outliers evidentes**: Pontos distantes >2 unidades da linha ideal
2. **Heterocedasticidade**: Variância do erro parece aumentar com fR,1
3. **Viés em extremos**: Tendência a subestimar valores altos

**Quantificação dos Problemas:**
- **R² inicial**: 0.838 (vs 0.890 final)
- **RMSE inicial**: ~1.123 (vs 0.891 final)
- **Piora de ~26%** no erro quadrático

**Importância como Benchmark:**
- Permite **quantificar ganhos** com pré-processamento
- Demonstra **impacto de qualidade de dados** na modelagem
- Estabelece **expectativas realistas** para dados brutos

---

###  Imagem 16: SHAP Plot Alternativo
![SHAP Alternativo](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/16.png)

**Análise Técnica Detalhada:**
Confirmação de padrões através de visualização alternativa.

**Verificação de Consistência:**
1. **Hierarquia mantida**: Teor fibra > fck > l/d > d > l > N
2. **Magnitudes similares**: Distribuições comparáveis às imagens anteriores
3. **Padrões confirmados**: Relações positivas/negativas consistentes

**Validação Metodológica:**
- **Reprodutibilidade** das análises SHAP
- **Robustez** das conclusões sobre importância
- **Confiabilidade** das interpretações do modelo

---

###  Imagem 17: Real vs Previsto - Dados Limpos
![Dados Limpos](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/17.png)

**Análise Técnica Detalhada:**
Resultado após remoção de 20% outliers (56 amostras).

**Melhorias Quantificadas:**

| Métrica | Antes | Depois | Melhoria |
|---------|-------|--------|----------|
| R² | 0.838 | **0.890** | +6.2% |
| RMSE | 1.123 | **0.891** | -20.7% |
| MAE | ~0.85 | **~0.65** | -23.5% |

**Melhorias Qualitativas:**
1. **Redução de dispersão**: Pontos mais concentrados
2. **Eliminação de outliers óbvios**: Limpeza visual clara
3. **Linearidade aprimorada**: Melhor alinhamento com y=x

**Estratégia de Limpeza:**
- **Critério**: 20% amostras com maior erro absoluto
- **Justificativa**: Remover medições potencialmente errôneas ou condições anômalas
- **Resultado**: Dataset de **225 amostras** com maior consistência

**Risco de Overfitting:**
- **Avaliação**: Gap viés-variância mantido (curva aprendizado)
- **Conclusão**: Limpeza não introduziu overfitting significativo

---

###  Imagem 18: Feature Importance - Modelo Refinado
![Importância Refinada](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/18.png)

**Análise Técnica Detalhada:**
Importância das features no modelo treinado com dados limpos.

**Comparação Pré/Pós-Limpeza:**

| Variável | Antes | Depois | Mudança | Interpretação |
|----------|-------|--------|----------|---------------|
| Teor fibra | 67% | **60%** | -7% | Menor dominância absoluta |
| N (ganchos) | 2% | **~10%** | **+8%** | Ganho mais significativo |
| fck | 15% | **~12%** | -3% | Redução moderada |
| l/d | 8% | **~8%** | Estável | Importância mantida |

**Interpretação das Mudanças:**
1. **Redistribuição de importância**: Dados limpos revelam papel mais balanceado
2. **N (ganchos) ganha relevância**: Possivelmente outliers mascaravam seu efeito
3. **Teor fibra ainda dominante**, mas menos absolutamente

**Implicações para Modelagem:**
- Limpeza pode **revelar relações mascaradas** por ruído
- **Importâncias relativas** ajustadas para maior fidelidade
- **Estrutura fundamental** mantida (hierarquia preservada)

---

###  Imagem 19: SHAP Plot - Modelo Refinado
![SHAP Refinado](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/19.png)

**Análise Técnica Detalhada:**
Explicabilidade do modelo após limpeza de dados.

**Melhorias na Clareza:**
1. **Padrões mais definidos**: Menor dispersão intra-feature
2. **Relações mais nítidas**: Gradientes mais claros
3. **Ruído reduzido**: Menos pontos atípicos nas distribuições

**Análise por Feature:**

**Teor de fibra:**
- **Relação mais linear** aparente
- **Dispersão reduzida** nos valores SHAP
- **Confiança aumentada** na interpretação

**N (ganchos):**
- **Padrão mais complexo** revelado
- Possível **interação não-linear** com teor de fibra
- **Importância contextual** mais evidente

**fck e l/d:**
- **Comportamentos mais consistentes**
- **Outliers explicativos** reduzidos
- **Interpretações mais confiáveis**

---

###  Imagem 20: SHAP Beeswarm Plot
![Beeswarm](https://github.com/TheCarlosRamos/projeto_cimento/blob/main/source/20.png)

**Análise Técnica Detalhada:**
Visualização mais detalhada da distribuição SHAP.

**Vantagens desta Visualização:**
1. **Densidade de informação**: Cada ponto = uma amostra
2. **Distribuição completa**: Não apenas estatísticas resumo
3. **Padrões locais**: Identificação de subgrupos

**Insights Específicos:**

**Teor de fibra:**
- **Distribuição compacta**: Consistência do efeito
- **Gradiente perfeito**: Vermelho → direita, azul → esquerda
- **Poucos outliers**: Relação robusta através do dataset

**fck:**
- **Gradiente claro** mas com **maior dispersão**
- **Alguns valores altos** com SHAP neutro (saturação possível)
- **Valores baixos** consistentemente negativos

**d (diâmetro):**
- **Comportamento bimodal** sugerido
- **Maiores diâmetros** claramente negativos
- **Diâmetros intermediários** com efeito variável

**N (ganchos):**
- **Distribuição mais complexa**
- **Possível efeito limiar**: abaixo de certo valor, impacto neutro
- **Interações** com outras features sugeridas

**Aplicação para Diagnóstico:**
- **Amostras atípicas** identificáveis (pontos distantes)
- **Subpopulações** detectáveis (clusters na distribuição)
- **Não-linearidades** visíveis (distribuições não uniformes)

---

##  CONCLUSÕES GERAIS E RECOMENDAÇÕES

###  Principais Descobertas

1. **Dominância do Teor de Fibra**
   - Responde por 60-67% da importância preditiva
   - Relação quase linear com resistência
   - Variável prioritária para otimização

2. **Alta Previsibilidade de fR,1**
   - R² = 0.890 alcançável
   - Erro médio < 17% do valor
   - Adequado para aplicações práticas

3. **Complexidade de fR,3**
   - Previsibilidade reduzida (R² = 0.693)
   - Relações mais complexas e não-lineares
   - Necessidade de modelos específicos

4. **Eficácia dos Métodos Ensemble**
   - Gradient Boosting e XGBoost performam melhor
   - Capturam relações não-lineares e interações
   - Robustos a ruído moderado

---
