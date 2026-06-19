# CAP 3 (IR ALÉM) — Classificação de Grãos de Trigo com Machine Learning

Atividade **Ir Além** da Fase 4 (FIAP): *"Da Terra ao Código — Automatizando a
Classificação de Grãos com Machine Learning"*, usando **Scikit-Learn** e a
metodologia **CRISP-DM**. Autor: **Douglas Felício da Silva**.

## 🎯 Objetivo
Automatizar a classificação de variedades de grãos de trigo (**Kama, Rosa, Canadian**)
a partir de 7 medidas físicas, substituindo a inspeção manual em cooperativas.

## 📊 Dataset
**Seeds Dataset** — UCI Machine Learning Repository (id 236):
210 amostras, 3 variedades (70 cada), 7 atributos: área, perímetro, compacidade,
comprimento e largura do núcleo, coeficiente de assimetria e comprimento do sulco.

## 🧪 O que o notebook faz (CRISP-DM)
1. **Entendimento do Negócio** — contexto da cooperativa.
2. **Entendimento dos Dados** — estatísticas (média/mediana/desvio), histogramas,
   boxplots, pairplot, matriz de correlação, checagem de valores ausentes.
3. **Preparação** — split treino/teste 70/30 estratificado + padronização (Pipeline).
4. **Modelagem** — 5 algoritmos: **KNN, SVM, Random Forest, Naive Bayes, Regressão Logística**.
5. **Avaliação** — acurácia, precisão, recall, F1 e **matrizes de confusão**; comparação.
6. **Otimização** — **GridSearchCV** (validação cruzada estratificada) nos 3 melhores.
7. **Interpretação** — importância das variáveis e insights de negócio.

## 📈 Resultados (reais)
| Modelo | Acurácia (teste) |
|--------|-----------------|
| **Random Forest** | **~0,92** |
| KNN | ~0,87 |
| SVM | ~0,87 |
| Regressão Logística | ~0,86 |
| Naive Bayes | ~0,83 |

Após **GridSearchCV**, o **SVM** atingiu ~0,96 de acurácia em validação cruzada.

## 📂 Estrutura
```
notebooks/Classificacao_Graos_Seeds.ipynb   # entregável principal
notebooks/Classificacao_Graos_Seeds.html    # versão para visualização
src/classificacao_graos.py                   # mesma análise como script
data/seeds_dataset.csv                        # dataset
README.md · COMO_EXECUTAR.txt · requirements.txt
```

## ▶️ Como executar
Ver `COMO_EXECUTAR.txt` (abrir o HTML, ou `jupyter notebook`, ou `python src/classificacao_graos.py`).

---
*Douglas Felício da Silva — FIAP, Fase 4 (Ir Além).*
