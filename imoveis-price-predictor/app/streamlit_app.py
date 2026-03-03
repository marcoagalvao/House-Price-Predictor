import sys
from pathlib import Path
import time

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# permitir importar src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_raw_data, handle_structural_missing, add_features


# =========================
# Cache (Performance)
# =========================
@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "xgb_final.pkl")


@st.cache_data
def load_reference_data():
    df_ = load_raw_data()
    df_ = handle_structural_missing(df_)
    df_ = add_features(df_)
    return df_


@st.cache_resource
def get_shap_explainer():
    # Modelo é fixo no app -> não precisa passar parâmetro (evita hashing de Pipeline)
    model_pipeline = load_model()
    xgb_model = model_pipeline.named_steps["model"]
    return shap.TreeExplainer(xgb_model)


# =========================
# App
# =========================
st.set_page_config(page_title="Preditor de Preço de Imóveis", layout="wide")

st.title("🏠 Predição de Preço de Imóveis")
st.write("Preencha as características do imóvel abaixo:")

# carregar modelo e dataset (cache)
model = load_model()
df = load_reference_data()

# Tabs
tab_pred, tab_about = st.tabs(["🏠 Previsão", "ℹ️ Sobre o modelo"])

# =========================
# Helpers de nomes amigáveis
# =========================
num_name_map = {
    "Overall Qual": "Qualidade Geral",
    "TotalSF": "Área total",
    "Gr Liv Area": "Área útil",
    "HouseAge": "Idade do imóvel",
    "Garage Cars": "Vagas na garagem",
    "Fireplaces": "Lareiras",
    "Lot Area": "Área do lote",
    "Overall Cond": "Condição geral",
    "Year Remod/Add": "Ano da reforma",
}

cat_name_map = {
    "Neighborhood": "Bairro",
    "Central Air": "Ar central",
    "Paved Drive": "Entrada pavimentada",
    "Sale Condition": "Condição de venda",
    "Functional": "Funcionalidade",
    "Exterior 1st": "Revestimento externo",
    "Condition 1": "Condição 1",
}


def format_feature(raw: str, user_values: dict) -> str:
    raw = raw.replace("num__", "").replace("cat__", "")

    # Numéricas
    if raw in num_name_map:
        label = num_name_map[raw]
        if raw in user_values:
            val = user_values[raw]
            if raw in ["TotalSF", "Gr Liv Area", "Lot Area"]:
                return f"{label} ({val} m²)"
            if raw == "HouseAge":
                return f"{label} ({val} anos)"
            return f"{label} ({val})"
        return label

    # One-hot categóricas
    if "_" in raw:
        base, value = raw.split("_", 1)
        base_label = cat_name_map.get(base, base)

        if value in ["Y", "N"]:
            yn = "Sim" if value == "Y" else "Não"
            return f"{base_label}: {yn}"

        return f"{base_label}: {value}"

    return raw


# =========================
# TAB: PREVISÃO
# =========================
with tab_pred:
    show_explain = st.checkbox("Mostrar explicação (SHAP)", value=True)

    col1, col2 = st.columns(2)

    with col1:
        overall_qual = st.slider("Qualidade Geral", 1, 10, 5)
        total_sf = st.number_input("Área Total (m²)", 300, 10000, 1500)
        garage_cars = st.slider("Vagas na Garagem", 0, 5, 2)
        fireplaces = st.slider("Lareiras", 0, 3, 1)

    with col2:
        house_age = st.slider("Idade da Casa", 0, 150, 20)
        central_air = st.selectbox("Ar Condicionado Central?", ["Sim", "Não"])
        neighborhood = st.selectbox("Bairro", sorted(df["Neighborhood"].unique()))

    # construir DataFrame de input
    input_dict = {
        "Overall Qual": overall_qual,
        "TotalSF": total_sf,
        "Garage Cars": garage_cars,
        "Fireplaces": fireplaces,
        "HouseAge": house_age,
        "Central Air": "Y" if central_air == "Sim" else "N",
        "Neighborhood": neighborhood,
    }

    input_df = pd.DataFrame([input_dict])

    # completar colunas faltantes com valores padrão (moda)
    for col in df.columns:
        if col not in input_df.columns and col != "SalePrice":
            input_df[col] = df[col].mode()[0]

    # mesmas transformações do treino
    input_df = handle_structural_missing(input_df)
    input_df = add_features(input_df)

    # valores que o usuário realmente informou
    user_values = {
        "Overall Qual": overall_qual,
        "TotalSF": total_sf,
        "Garage Cars": garage_cars,
        "Fireplaces": fireplaces,
        "HouseAge": house_age,
        "Central Air": "Sim" if central_air == "Sim" else "Não",
        "Neighborhood": neighborhood,
    }

    if st.button("Estimar Preço"):
        t0 = time.perf_counter()

        # ----------- PREVISÃO -----------
        pred_log = model.predict(input_df)
        pred = np.expm1(pred_log)
        final_price = float(pred[0])

        t_pred = time.perf_counter() - t0

        st.success(f"💰 Preço estimado: ${final_price:,.2f}")

        # faixa simples (±8%)
        error_pct = 0.08
        lower = final_price * (1 - error_pct)
        upper = final_price * (1 + error_pct)
        st.info(f"📊 Faixa estimada: ${lower:,.2f} - ${upper:,.2f}")

        st.caption(f"⏱️ Tempo de previsão: {t_pred*1000:.0f} ms")

        # ----------- EXPLICAÇÃO (opcional) -----------
        if show_explain:
            with st.spinner("Calculando explicação (SHAP)..."):
                t1 = time.perf_counter()

                preprocessor = model.named_steps["prep"]
                explainer = get_shap_explainer()

                X_transformed = preprocessor.transform(input_df)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()

                shap_values = explainer.shap_values(X_transformed)
                feature_names = model[:-1].get_feature_names_out()

                # SHAP completo
                shap_all = pd.DataFrame({
                    "feature": feature_names,
                    "impact": shap_values[0]
                })

                shap_all["impact_dollar"] = shap_all["impact"] * final_price
                shap_all["abs_dollar"] = shap_all["impact_dollar"].abs()

                # filtrar apenas campos que o usuário escolheu (inclui one-hot)
                user_feature_bases = {
                    "Overall Qual",
                    "TotalSF",
                    "HouseAge",
                    "Garage Cars",
                    "Fireplaces",
                    "Central Air",
                    "Neighborhood",
                }

                def is_user_feature(f: str) -> bool:
                    f = f.replace("num__", "").replace("cat__", "")
                    base = f.split("_", 1)[0]
                    return base in user_feature_bases

                shap_user = shap_all[shap_all["feature"].apply(is_user_feature)].copy()
                shap_user = shap_user.sort_values("abs_dollar", ascending=False).head(7)

                st.subheader("🧩 Impacto dos campos que você preencheu (estimado em $)")

                for _, row in shap_user.iterrows():
                    label = format_feature(row["feature"], user_values=user_values)
                    val = float(row["impact_dollar"])
                    if val > 0:
                        st.write(f"⬆️ **{label}** aumentou aproximadamente ${val:,.0f}")
                    else:
                        st.write(f"⬇️ **{label}** reduziu aproximadamente ${abs(val):,.0f}")

                with st.expander("🔎 Ver top fatores gerais do modelo"):
                    top_global = shap_all.sort_values("abs_dollar", ascending=False).head(5)

                    for _, row in top_global.iterrows():
                        label = format_feature(row["feature"], user_values=user_values)
                        val = float(row["impact_dollar"])
                        if val > 0:
                            st.write(f"⬆️ **{label}** aumentou aproximadamente ${val:,.0f}")
                        else:
                            st.write(f"⬇️ **{label}** reduziu aproximadamente ${abs(val):,.0f}")

                t_shap = time.perf_counter() - t1
                st.caption(f"⏱️ Tempo SHAP: {t_shap:.2f} s")


# =========================
# TAB: SOBRE O MODELO
# =========================
with tab_about:
    st.header("ℹ️ Sobre o modelo")

    st.markdown(
        """
**Objetivo:** prever o preço de imóveis (Ames Housing) usando regressão com Machine Learning.  
**Modelo:** XGBoost Regressor  
**Target:** `log1p(SalePrice)` (na saída, convertemos com `expm1`)  
**Pré-processamento:** imputação + one-hot encoding + clipping por quantis (outliers)  
**Explicabilidade:** SHAP (impacto por variável)
"""
    )

    st.subheader("📊 Métricas (holdout 80/20)")
    # Valores que você já obteve
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", "$13,600")
    c2.metric("RMSE", "$22,844")
    c3.metric("R²", "0.9349")

    st.subheader("🗂 Dataset")
    st.write("Ames Housing (Kaggle).")
    st.write(f"Linhas: {df.shape[0]}  |  Colunas: {df.shape[1]} (inclui SalePrice)")

    st.subheader("🛠 Stack")
    st.markdown(
        """
- Python
- Pandas / NumPy
- Scikit-learn (pipelines)
- XGBoost
- SHAP
- Streamlit
"""
    )

    st.subheader("🌍 Importância global (XGBoost)")
    try:
        xgb_model = model.named_steps["model"]
        feature_names = model[:-1].get_feature_names_out()
        importances = getattr(xgb_model, "feature_importances_", None)

        if importances is None:
            st.warning("Este modelo não expôs feature_importances_.")
        else:
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False).head(15)

            fig, ax = plt.subplots()
            ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
            ax.set_xlabel("Importância")
            ax.set_ylabel("Feature")
            st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.warning(f"Não foi possível renderizar o gráfico de importância: {e}")