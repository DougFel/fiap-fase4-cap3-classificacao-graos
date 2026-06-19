"""
CAP 3 (IR ALEM) - Classificacao de graos de trigo (Seeds Dataset, UCI) com Scikit-Learn.
Versao em script da analise do notebook (CRISP-DM): EDA, 5 classificadores,
metricas + matrizes de confusao e otimizacao com GridSearchCV.

Uso:  python src/classificacao_graos.py
Saidas: graficos em assets/ e metricas no terminal.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "seeds_dataset.csv")
ASSETS = os.path.join(BASE, "assets"); os.makedirs(ASSETS, exist_ok=True)
RS = 42
FEATURES = ["area","perimetro","compacidade","comprimento_nucleo",
            "largura_nucleo","coef_assimetria","comprimento_sulco"]
NOMES = {1:"Kama", 2:"Rosa", 3:"Canadian"}
sns.set_theme(style="whitegrid", palette="viridis")


def main():
    df = pd.read_csv(DATA)
    df["variedade_nome"] = df["variedade"].map(NOMES)
    print("Dimensoes:", df.shape, "| nulos:", int(df.isna().sum().sum()))
    print("\nEstatisticas (media/mediana/desvio):")
    print(df[FEATURES].agg(["mean","median","std"]).T.round(3))

    # EDA: correlacao
    plt.figure(figsize=(8,6))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="viridis", square=True)
    plt.title("Matriz de correlacao"); plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "correlacao.png"), dpi=110); plt.close()

    X, y = df[FEATURES].values, df["variedade"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=RS)

    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RS),
        "Naive Bayes": GaussianNB(),
        "Regressao Logistica": LogisticRegression(max_iter=1000, random_state=RS),
    }
    pipe = lambda m: Pipeline([("sc", StandardScaler()), ("clf", m)])
    preds, linhas = {}, []
    for nome, m in modelos.items():
        p = pipe(m); p.fit(Xtr, ytr); pr = p.predict(Xte); preds[nome] = pr
        linhas.append({"Modelo":nome,
                       "Acuracia":round(accuracy_score(yte,pr),4),
                       "Precisao":round(precision_score(yte,pr,average="macro"),4),
                       "Recall":round(recall_score(yte,pr,average="macro"),4),
                       "F1":round(f1_score(yte,pr,average="macro"),4)})
    tab = pd.DataFrame(linhas).sort_values("Acuracia", ascending=False)
    print("\n=== Comparacao dos classificadores ===")
    print(tab.to_string(index=False))

    # matrizes de confusao
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    for ax, (nome, pr) in zip(axes, preds.items()):
        cm = confusion_matrix(yte, pr, labels=[1,2,3])
        ConfusionMatrixDisplay(cm, display_labels=list(NOMES.values())).plot(
            ax=ax, cmap="Greens", colorbar=False)
        ax.set_title(nome)
    plt.tight_layout(); plt.savefig(os.path.join(ASSETS, "matrizes_confusao.png"), dpi=110); plt.close()

    melhor = tab.iloc[0]["Modelo"]
    print(f"\nMelhor modelo base: {melhor}")
    print(classification_report(yte, preds[melhor], target_names=list(NOMES.values())))

    # GridSearch
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    grades = {
        "SVM": (pipe(SVC(random_state=RS)),
                {"clf__C":[0.1,1,10,100], "clf__gamma":["scale",0.01,0.1,1]}),
        "KNN": (pipe(KNeighborsClassifier()),
                {"clf__n_neighbors":[3,5,7,9,11], "clf__weights":["uniform","distance"]}),
        "Random Forest": (pipe(RandomForestClassifier(random_state=RS)),
                {"clf__n_estimators":[100,200,300], "clf__max_depth":[None,5,10]}),
    }
    print("\n=== Otimizacao (GridSearchCV) ===")
    for nome,(pp,g) in grades.items():
        gs = GridSearchCV(pp, g, cv=cv, scoring="accuracy", n_jobs=-1); gs.fit(Xtr, ytr)
        acc = accuracy_score(yte, gs.predict(Xte))
        print(f"{nome:<16} CV={gs.best_score_:.4f}  teste={acc:.4f}  {gs.best_params_}")
    print("\nGraficos salvos em assets/.")


if __name__ == "__main__":
    main()
