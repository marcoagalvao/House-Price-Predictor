# 🏠 House Price Predictor — Ames Housing

Aplicação completa de **Ciência de Dados / Machine Learning** para prever preços de imóveis utilizando o dataset **Ames Housing**.

O projeto inclui:

- Pipeline reprodutível com Scikit-learn
- Modelo final com XGBoost
- Transformação logarítmica do target
- Validação adequada
- Interpretabilidade com SHAP
- Aplicação interativa com Streamlit

---

## 🎯 Objetivo

Prever o preço de venda (`SalePrice`) de imóveis utilizando regressão supervisionada, aplicando boas práticas de modelagem e entregando o resultado em formato de produto (aplicação web).

---

## 🧠 Abordagem Técnica

### 📊 Dataset
- Ames Housing Dataset (Kaggle)
- Dados tabulares com variáveis numéricas e categóricas

### 🔄 Pré-processamento
- Tratamento de missing estruturais
- Imputação:
  - Numéricas → mediana
  - Categóricas → valor mais frequente
- One-hot encoding para variáveis categóricas
- Clipping por quantis para reduzir impacto de outliers
- Engenharia de variáveis (ex: idade da casa)

### 📈 Modelagem
- Baseline: Ridge Regression
- Modelo final: **XGBoost Regressor**
- Target transform:
  - Treino → `log1p(SalePrice)`
  - Previsão → `expm1()`

### 🔎 Interpretabilidade
- SHAP (TreeExplainer)
- Impacto das variáveis exibido no app
- Impacto estimado em dólares para melhor compreensão

---

## 📊 Resultados (Holdout 80/20)

- **MAE:** ~$13,600  
- **RMSE:** ~$22,844  
- **R²:** ~0.9349  

*Métricas calculadas na escala original de preço.*

---

## 🖥️ Aplicação Web (Streamlit)

A aplicação permite:

- Inserção simplificada de características do imóvel
- Preço estimado
- Faixa estimada de valor (±8%)
- Explicação individual da previsão com SHAP
- Aba técnica com métricas e importância global do modelo

---

## 📂 Estrutura do Projeto

```
imoveis-price-predictor/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── xgb_final.pkl
│
├── raw/
│   └── train.csv
│
├── src/
│   ├── data.py
│   ├── train.py
│   └── transformers.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Como Rodar Localmente

### 1️⃣ Criar ambiente virtual

Windows:
```
python -m venv .venv
.\.venv\Scripts\activate
```

Mac/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2️⃣ Instalar dependências

```
pip install -r requirements.txt
```

---

### 3️⃣ Rodar o aplicativo

Na raiz do projeto:

```
streamlit run app/streamlit_app.py
```

---

## 🚀 Deploy no Streamlit Cloud

1. Subir o projeto para o GitHub
2. Acessar Streamlit Community Cloud
3. Conectar o repositório
4. Definir o arquivo principal:

```
app/streamlit_app.py
```

5. Deploy
🚀 Acesse o app online:
[https://house-price-predictor-marcoagalvao.streamlit.app/]
---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib

---

## 📈 Próximas Melhorias

- Quantile regression (P10 / P50 / P90)
- Intervalo de confiança mais robusto
- Monitoramento de drift
- Dockerização
- CI/CD
- Testes automatizados

---

## 📌 Status do Projeto

✅ Modelagem validada  
✅ App funcional  
✅ Explicabilidade implementada  
✅ Performance otimizada com cache  

Projeto finalizado em nível portfólio profissional.

---

## 👨‍💻 Autor

Projeto desenvolvido para fins de portfólio em Ciência de Dados.
