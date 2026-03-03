# 🏠 House Price Predictor (Ames Housing) — Regression + Streamlit + SHAP

Aplicação de **Ciência de Dados / Machine Learning** para **prever preços de imóveis** com base no dataset **Ames Housing**.  
O projeto inclui **pipeline reprodutível**, **validação**, **modelo final com XGBoost**, e **explicabilidade com SHAP**, além de um **app em Streamlit** para uso por pessoas não técnicas.

---

## 🎯 Objetivo
Prever o preço de venda (`SalePrice`) de imóveis utilizando regressão supervisionada, com foco em:
- boas práticas de ML (pipelines, validação, prevenção de leakage)
- interpretabilidade (SHAP)
- entrega em formato de produto (Streamlit)

---

## 🧠 Abordagem (resumo técnico)

### Dataset
- **Ames Housing** (Kaggle)  
- Variáveis numéricas e categóricas (tabular)

### Pré-processamento
- tratamento de missing estruturais
- imputação (numéricas: mediana, categóricas: mais frequente)
- one-hot encoding para categorias
- clipping por quantis (winsorization) para reduzir impacto de outliers

### Modelagem
- baseline: Ridge Regression
- modelo final: **XGBoost Regressor**
- target transform: `log1p(SalePrice)` e retorno com `expm1`

### Explicabilidade
- **SHAP** (TreeExplainer)
- explicação por predição no app (impacto estimado em $)

---

## 📊 Resultados (holdout 80/20)
- **MAE:** ~$13,600  
- **RMSE:** ~$22,844  
- **R²:** ~0.9349  

> *As métricas foram calculadas na escala original de preço (após `expm1`).*

---

## 🖥️ App (Streamlit)
O app permite:
- input simplificado (campos principais)
- **preço estimado**
- **faixa estimada** (±8% como aproximação do erro médio)
- explicação SHAP em **impacto aproximado em dólares**
- aba “Sobre o modelo” com métricas e importância global

---

## 📂 Estrutura do Projeto
imoveis-price-predictor/
├── app/
│ └── streamlit_app.py
├── models/
│ └── xgb_final.pkl
├── raw/
│ └── train.csv
├── src/
│ ├── data.py
│ ├── train.py
│ └── transformers.py
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_modeling.ipynb
│ └── 03_explainability.ipynb
├── requirements.txt
└── README.md

---

## ⚙️ Como rodar localmente

### 1) Criar e ativar ambiente virtual
Windows (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\activate

