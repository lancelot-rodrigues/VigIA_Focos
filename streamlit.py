# streamlit_app.py

import streamlit as st
# --- CONFIGURAÇÃO DA PÁGINA DEVE SER A PRIMEIRA COISA STREAMLIT ---
st.set_page_config(
    layout="wide",
    page_title="Soluções IA FIAP - Incêndios",
    page_icon="🔥"
)

import pandas as pd
import joblib
import os
import numpy as np
import traceback
import warnings
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# Suprimir Warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", message=".*is_categorical_dtype is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*is_sparse is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray", category=UserWarning, module="shap.explainers._tree")
warnings.filterwarnings("ignore", message="Trying to unpickle estimator StandardScaler from version", category=UserWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator SimpleImputer from version", category=UserWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator TargetEncoder from version", category=UserWarning)
warnings.filterwarnings("ignore", message="'M' is deprecated and will be removed in a future version, please use 'ME' instead.", category=FutureWarning)


# Importar funções do seu projeto DEPOIS de st.set_page_config
from utils.preprocessing_utils import aplicar_engenharia_features_bloco3
from rag_assistant_tab import display_rag_chat_tab, ROBOT_NAME
from utils.rpa_data_collector import fetch_and_process_inpe_daily_data

FEATURE_DESCRIPTIONS = {
    # FWI-like Proxies
    "ffmc_proxy": "Proxy do FFMC (Fine Fuel Moisture Code): Umidade de combustíveis finos e de rápida secagem. Valores altos = material seco.",
    "dmc_proxy": "Proxy do DMC (Duff Moisture Code): Umidade de camadas orgânicas do solo (média profundidade). Valores altos = secura.",
    "dc_proxy": "Proxy do DC (Drought Code): Índice de seca profunda. Valores altos = seca prolongada.",
    "isi_proxy": "Proxy do ISI (Initial Spread Index): Taxa de propagação inicial do fogo (sem influência de combustíveis variáveis).",
    "bui_proxy": "Proxy do BUI (Buildup Index): Quantidade total de combustível disponível para combustão.",
    "fwi_proxy_final": "Proxy do FWI (Fire Weather Index): Índice numérico final da intensidade do fogo esperada.",

    # Dias Secos
    "dias_secos_consecutivos": "Número de dias consecutivos com precipitação < 0.1mm (prioritariamente INMET).",

    # Features de Interação
    "dias_secos_X_umidade_baixa_lag1": "Interação: Dias secos vs. (100 - Umidade Mínima do dia anterior). Alto = período seco + baixa umidade recente.",
    "dias_secos_X_risco_fogo_lag1": "Interação: Dias secos vs. Risco de Fogo (média dos focos) do dia anterior. Alto = período seco + risco alto recente.",

    # Lags de Variáveis Meteorológicas e de Focos (INMET e Originais)
    "precip_total_dia_mm_lag1": "Precipitação INMET (mm) do dia anterior (D-1).",
    "precip_total_dia_mm_lag3": "Precipitação INMET (mm) de 3 dias atrás (D-3).",
    "precip_total_dia_mm_lag7": "Precipitação INMET (mm) de 7 dias atrás (D-7).",
    "temp_media_dia_C_lag1": "Temperatura média INMET (°C) do dia anterior (D-1).",
    "temp_media_dia_C_lag3": "Temperatura média INMET (°C) de 3 dias atrás (D-3).",
    "temp_media_dia_C_lag7": "Temperatura média INMET (°C) de 7 dias atrás (D-7).",
    "temp_max_dia_C_lag1": "Temperatura máxima INMET (°C) do dia anterior (D-1).",
    "temp_max_dia_C_lag3": "Temperatura máxima INMET (°C) de 3 dias atrás (D-3).",
    "temp_max_dia_C_lag7": "Temperatura máxima INMET (°C) de 7 dias atrás (D-7).",
    "temp_min_dia_C_lag1": "Temperatura mínima INMET (°C) do dia anterior (D-1).",
    "temp_min_dia_C_lag3": "Temperatura mínima INMET (°C) de 3 dias atrás (D-3).",
    "temp_min_dia_C_lag7": "Temperatura mínima INMET (°C) de 7 dias atrás (D-7).",
    "umidade_media_dia_perc_lag1": "Umidade relativa média INMET (%) do dia anterior (D-1).",
    "umidade_media_dia_perc_lag3": "Umidade relativa média INMET (%) de 3 dias atrás (D-3).",
    "umidade_media_dia_perc_lag7": "Umidade relativa média INMET (%) de 7 dias atrás (D-7).",
    "umidade_min_dia_perc_lag1": "Umidade relativa mínima INMET (%) do dia anterior (D-1).",
    "umidade_min_dia_perc_lag3": "Umidade relativa mínima INMET (%) de 3 dias atrás (D-3).",
    "umidade_min_dia_perc_lag7": "Umidade relativa mínima INMET (%) de 7 dias atrás (D-7).",
    "vento_vel_media_dia_ms_lag1": "Velocidade média do vento INMET (m/s) do dia anterior (D-1).",
    "vento_vel_media_dia_ms_lag3": "Velocidade média do vento INMET (m/s) de 3 dias atrás (D-3).",
    "vento_vel_media_dia_ms_lag7": "Velocidade média do vento INMET (m/s) de 7 dias atrás (D-7).",
    "numero_dias_sem_chuva_first_lag1": "Nº dias sem chuva (base focos) do dia anterior (D-1).",
    "numero_dias_sem_chuva_first_lag3": "Nº dias sem chuva (base focos) de 3 dias atrás (D-3).",
    "numero_dias_sem_chuva_first_lag7": "Nº dias sem chuva (base focos) de 7 dias atrás (D-7).",
    "precipitacao_sum_lag1": "Precipitação (base focos, mm) do dia anterior (D-1).",
    "precipitacao_sum_lag3": "Precipitação (base focos, mm) de 3 dias atrás (D-3).",
    "precipitacao_sum_lag7": "Precipitação (base focos, mm) de 7 dias atrás (D-7).",
    "risco_fogo_mean_lag1": "Risco de fogo médio (base focos) do dia anterior (D-1).",
    "risco_fogo_mean_lag3": "Risco de fogo médio (base focos) de 3 dias atrás (D-3).",
    "risco_fogo_mean_lag7": "Risco de fogo médio (base focos) de 7 dias atrás (D-7).",

    # Lags de FRP
    "frp_max_lag1": "FRP Máximo (focos) do dia anterior (D-1).",
    "frp_max_lag3": "FRP Máximo (focos) de 3 dias atrás (D-3).",
    "frp_max_lag7": "FRP Máximo (focos) de 7 dias atrás (D-7).",
    "frp_mean_lag1": "FRP Médio (focos) do dia anterior (D-1).",
    "frp_mean_lag3": "FRP Médio (focos) de 3 dias atrás (D-3).",
    "frp_mean_lag7": "FRP Médio (focos) de 7 dias atrás (D-7).",
    "frp_sum_lag1": "Soma do FRP (focos) do dia anterior (D-1).",
    "frp_sum_lag3": "Soma do FRP (focos) de 3 dias atrás (D-3).",
    "frp_sum_lag7": "Soma do FRP (focos) de 7 dias atrás (D-7).",

    # Médias Móveis (Rollings) de Variáveis Meteorológicas e de Focos
    "precip_total_dia_mm_roll_mean3": "Média da precipitação INMET (mm) dos últimos 3 dias (D-1 a D-3).",
    "precip_total_dia_mm_roll_mean7": "Média da precipitação INMET (mm) dos últimos 7 dias (D-1 a D-7).",
    "precip_total_dia_mm_roll_mean15": "Média da precipitação INMET (mm) dos últimos 15 dias (D-1 a D-15).",
    "temp_media_dia_C_roll_mean3": "Média da temp. média INMET (°C) dos últimos 3 dias.",
    "temp_media_dia_C_roll_mean7": "Média da temp. média INMET (°C) dos últimos 7 dias.",
    "temp_media_dia_C_roll_mean15": "Média da temp. média INMET (°C) dos últimos 15 dias.",
    "temp_max_dia_C_roll_mean3": "Média da temp. máxima INMET (°C) dos últimos 3 dias.",
    "temp_max_dia_C_roll_mean7": "Média da temp. máxima INMET (°C) dos últimos 7 dias.",
    "temp_max_dia_C_roll_mean15": "Média da temp. máxima INMET (°C) dos últimos 15 dias.",
    "temp_min_dia_C_roll_mean3": "Média da temp. mínima INMET (°C) dos últimos 3 dias.",
    "temp_min_dia_C_roll_mean7": "Média da temp. mínima INMET (°C) dos últimos 7 dias.",
    "temp_min_dia_C_roll_mean15": "Média da temp. mínima INMET (°C) dos últimos 15 dias.",
    "umidade_media_dia_perc_roll_mean3": "Média da umidade média INMET (%) dos últimos 3 dias.",
    "umidade_media_dia_perc_roll_mean7": "Média da umidade média INMET (%) dos últimos 7 dias.",
    "umidade_media_dia_perc_roll_mean15": "Média da umidade média INMET (%) dos últimos 15 dias.",
    "umidade_min_dia_perc_roll_mean3": "Média da umidade mínima INMET (%) dos últimos 3 dias.",
    "umidade_min_dia_perc_roll_mean7": "Média da umidade mínima INMET (%) dos últimos 7 dias.",
    "umidade_min_dia_perc_roll_mean15": "Média da umidade mínima INMET (%) dos últimos 15 dias.",
    "vento_vel_media_dia_ms_roll_mean3": "Média da vel. do vento INMET (m/s) dos últimos 3 dias.",
    "vento_vel_media_dia_ms_roll_mean7": "Média da vel. do vento INMET (m/s) dos últimos 7 dias.",
    "vento_vel_media_dia_ms_roll_mean15": "Média da vel. do vento INMET (m/s) dos últimos 15 dias.",
    "numero_dias_sem_chuva_first_roll_mean3": "Média do nº dias sem chuva (focos) dos últimos 3 dias.",
    "numero_dias_sem_chuva_first_roll_mean7": "Média do nº dias sem chuva (focos) dos últimos 7 dias.",
    "numero_dias_sem_chuva_first_roll_mean15": "Média do nº dias sem chuva (focos) dos últimos 15 dias.",
    "precipitacao_sum_roll_mean3": "Média da precipitação (focos, mm) dos últimos 3 dias.",
    "precipitacao_sum_roll_mean7": "Média da precipitação (focos, mm) dos últimos 7 dias.",
    "precipitacao_sum_roll_mean15": "Média da precipitação (focos, mm) dos últimos 15 dias.",
    "risco_fogo_mean_roll_mean3": "Média do risco de fogo (focos) dos últimos 3 dias.",
    "risco_fogo_mean_roll_mean7": "Média do risco de fogo (focos) dos últimos 7 dias.",
    "risco_fogo_mean_roll_mean15": "Média do risco de fogo (focos) dos últimos 15 dias.",

    # Médias Móveis (Rollings) de FRP
    "frp_max_roll_mean3": "Média do FRP Máximo diário (focos) dos últimos 3 dias.",
    "frp_max_roll_mean7": "Média do FRP Máximo diário (focos) dos últimos 7 dias.",
    "frp_max_roll_mean15": "Média do FRP Máximo diário (focos) dos últimos 15 dias.",
    "frp_mean_roll_mean3": "Média do FRP Médio diário (focos) dos últimos 3 dias.",
    "frp_mean_roll_mean7": "Média do FRP Médio diário (focos) dos últimos 7 dias.",
    "frp_mean_roll_mean15": "Média do FRP Médio diário (focos) dos últimos 15 dias.",
    "frp_sum_roll_mean3": "Média da Soma diária do FRP (focos) dos últimos 3 dias.",
    "frp_sum_roll_mean7": "Média da Soma diária do FRP (focos) dos últimos 7 dias.",
    "frp_sum_roll_mean15": "Média da Soma diária do FRP (focos) dos últimos 15 dias.",

    # MapBiomas (Proporções) - Adicionei todas que mencionei antes como exemplo
    "mapbiomas_class_0_prop": "Proporção: Não Observado (MapBiomas).",
    "mapbiomas_class_3_prop": "Proporção: Formação Florestal (MapBiomas).",
    "mapbiomas_class_4_prop": "Proporção: Formação Savânica (MapBiomas).",
    "mapbiomas_class_5_prop": "Proporção: Mangue (MapBiomas).",
    "mapbiomas_class_9_prop": "Proporção: Floresta Plantada (MapBiomas).",
    "mapbiomas_class_11_prop": "Proporção: Área Úmida Natural Não Florestal (MapBiomas).",
    "mapbiomas_class_12_prop": "Proporção: Formação Campestre (MapBiomas).",
    "mapbiomas_class_13_prop": "Proporção: Outra Formação Natural Não Florestal (MapBiomas).",
    "mapbiomas_class_15_prop": "Proporção: Pastagem (MapBiomas).",
    "mapbiomas_class_20_prop": "Proporção: Lavoura Temporária (MapBiomas).",
    "mapbiomas_class_21_prop": "Proporção: Mosaico de Agricultura e Pastagem (MapBiomas).",
    "mapbiomas_class_23_prop": "Proporção: Praia, Duna e Areal (MapBiomas).",
    "mapbiomas_class_24_prop": "Proporção: Área de Vegetação Urbana (MapBiomas).",
    "mapbiomas_class_25_prop": "Proporção: Outras Áreas Não Vegetadas (MapBiomas).",
    "mapbiomas_class_29_prop": "Proporção: Afloramento Rochoso (MapBiomas).",
    "mapbiomas_class_30_prop": "Proporção: Mineração (MapBiomas).",
    "mapbiomas_class_31_prop": "Proporção: Aquicultura (MapBiomas).",
    "mapbiomas_class_32_prop": "Proporção: Salgado / Salinas (MapBiomas).",
    "mapbiomas_class_33_prop": "Proporção: Rio, Lago e Oceano (MapBiomas).",
    "mapbiomas_class_39_prop": "Proporção: Soja (MapBiomas).",
    "mapbiomas_class_41_prop": "Proporção: Outras Lavouras Temporárias (MapBiomas).",
    "mapbiomas_class_46_prop": "Proporção: Café (MapBiomas).",
    "mapbiomas_class_47_prop": "Proporção: Citrus (MapBiomas).",
    "mapbiomas_class_48_prop": "Proporção: Outras Lavouras Perenes (MapBiomas).",
    "mapbiomas_class_49_prop": "Proporção: Área de Descanso (Pousio) (MapBiomas).",
    "mapbiomas_class_50_prop": "Proporção: Silvicultura (PINUS, EUCALIPTUS) (MapBiomas).",
    "mapbiomas_class_62_prop": "Proporção: Campo Alagado e Área Pantanosa (MapBiomas).",

    # Encodada
    "municipio_encoded": "Representação numérica do município (Target Encoded).",
}

# --- Caminhos e Constantes (PREVISÃO) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PRED_MAX_LAG_DAYS = 30
PRED_DECISION_THRESHOLD = 0.140

# --- Funções de Carregamento (PREVISÃO) ---
@st.cache_resource
def load_prediction_model_artifacts():
    try:
        model = joblib.load(os.path.join(MODEL_ARTIFACTS_DIR, 'modelo_incendio_lgbm_recall_focus.joblib'))
        scaler = joblib.load(os.path.join(MODEL_ARTIFACTS_DIR, 'scaler_incendio_recall_focus.joblib'))
        target_encoder = joblib.load(os.path.join(MODEL_ARTIFACTS_DIR, 'target_encoder_municipio_recall_focus.joblib'))
        imputers_dict = joblib.load(os.path.join(MODEL_ARTIFACTS_DIR, 'imputadores_finais_recall_focus.joblib'))
        model_columns = joblib.load(os.path.join(MODEL_ARTIFACTS_DIR, 'colunas_modelo_final_lista_recall_focus.joblib'))
        return model, scaler, target_encoder, imputers_dict, model_columns
    except Exception: return (None,) * 5

@st.cache_data
def load_prediction_dataset():
    try:
        csv_path = os.path.join(DATA_DIR, 'dataset_SP_2022_completo_com_inmet.csv')
        df = pd.read_csv(csv_path, sep=';', decimal='.')
        if 'data' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['data']):
             df['data'] = pd.to_datetime(df['data'], errors='coerce')
        precip_cols = ['precip_total_dia_mm', 'precipitacao_sum']
        for p_col in precip_cols:
            if p_col in df.columns and df[p_col].dtype == 'object':
                df[p_col] = df[p_col].astype(str).str.replace(',', '.', regex=False)
                df[p_col] = pd.to_numeric(df[p_col], errors='coerce')
        return df
    except Exception: return None

# --- Inicialização de variáveis de estado (PREVISÃO) ---
pred_model, pred_scaler, pred_target_encoder, pred_imputers, pred_model_columns = (None,) * 5
pred_full_dataset = None
pred_load_status = "error"

if 'prediction_module_initialized_v9' not in st.session_state: # Nova chave
    st.session_state.pred_model, st.session_state.pred_scaler, \
    st.session_state.pred_target_encoder, st.session_state.pred_imputers, \
    st.session_state.pred_model_columns = load_prediction_model_artifacts()
    st.session_state.pred_full_dataset = load_prediction_dataset()
    st.session_state.pred_load_status = "success" if all(v is not None for v in [
        st.session_state.pred_model, st.session_state.pred_full_dataset,
        st.session_state.pred_scaler, st.session_state.pred_target_encoder,
        st.session_state.pred_imputers, st.session_state.pred_model_columns
    ]) else "error"
    st.session_state.prediction_module_initialized_v9 = True

pred_model = st.session_state.get('pred_model', pred_model)
pred_scaler = st.session_state.get('pred_scaler', pred_scaler)
pred_target_encoder = st.session_state.get('pred_target_encoder', pred_target_encoder)
pred_imputers = st.session_state.get('pred_imputers', pred_imputers)
pred_model_columns = st.session_state.get('pred_model_columns', pred_model_columns)
pred_full_dataset = st.session_state.get('pred_full_dataset', pred_full_dataset)
pred_load_status = st.session_state.get('pred_load_status', "error")

# --- Título e Navegação Sidebar ---
# st.sidebar.image(os.path.join(BASE_DIR, "fiap_logo.png"), width=150) # Descomente se tiver o logo
st.sidebar.title("Soluções IA FIAP")
st.sidebar.markdown("---")
st.sidebar.markdown("Navegue pelos módulos:")

# --- Gerenciamento da Aba Ativa com Sidebar ---
aba_pred_key = "🔥 Previsão de Risco"
aba_desc_key = "📊 Análise Descritiva"
aba_rag_key = f"💬 Assistente {ROBOT_NAME}"
aba_sobre_key = "ℹ️ Sobre os Projetos"
opcoes_sidebar = [aba_pred_key, aba_desc_key, aba_rag_key, aba_sobre_key]

if 'active_main_tab_v8' not in st.session_state: # Nova chave
    st.session_state.active_main_tab_v8 = opcoes_sidebar[0]

aba_selecionada_pelo_usuario = st.sidebar.radio(
    "Módulos:", opcoes_sidebar,
    index=opcoes_sidebar.index(st.session_state.active_main_tab_v8),
    key="main_nav_radio_v8",
    label_visibility="collapsed"
)

if aba_selecionada_pelo_usuario != st.session_state.active_main_tab_v8:
    st.session_state.active_main_tab_v8 = aba_selecionada_pelo_usuario
    st.rerun()

st.title(st.session_state.active_main_tab_v8) # Título principal da página


# --- Exibição Condicional do Conteúdo da Aba ---

if st.session_state.active_main_tab_v8 == aba_pred_key:
    if pred_load_status != "success" or pred_full_dataset is None:
        st.error("Módulo de previsão não carregado ou dataset ausente.")
    else:
        st.markdown("Selecione município e data para prever o risco de incêndio.")
        municipios_lista_pred = sorted(pred_full_dataset['municipio'].unique())
        col1_pred_tab, col2_pred_tab = st.columns(2)
        selected_municipio_pred = col1_pred_tab.selectbox("Município:", municipios_lista_pred, key="sb_mun_pred_final_v6")
        
        default_min_date_pred = pd.to_datetime("2022-01-01"); default_max_date_pred = pd.to_datetime("2022-12-31")
        min_dt_pred = pred_full_dataset['data'].min() if pred_full_dataset['data'].dropna().shape[0]>0 else default_min_date_pred
        max_dt_pred = pred_full_dataset['data'].max() if pred_full_dataset['data'].dropna().shape[0]>0 else default_max_date_pred
        
        selected_date_input_pred = col2_pred_tab.date_input("Data:", min_value=min_dt_pred, max_value=max_dt_pred, value=min_dt_pred, key="di_data_pred_final_v6")
        selected_date_pred = pd.to_datetime(selected_date_input_pred)

        if st.button("🔎 Gerar Previsão", type="primary", key="btn_gerar_pred_final_v6"):
            if not selected_municipio_pred or not selected_date_pred: st.warning("Selecione município e data.")
            else:
                with st.spinner(f"Calculando para {selected_municipio_pred}, {selected_date_pred.strftime('%d/%m/%Y')}..."):
                    hist_start_dt_pred = selected_date_pred - pd.Timedelta(days=PRED_MAX_LAG_DAYS)
                    data_for_features_pred = pred_full_dataset[
                        (pred_full_dataset['municipio'] == selected_municipio_pred) &
                        (pred_full_dataset['data'] >= hist_start_dt_pred) &
                        (pred_full_dataset['data'] <= selected_date_pred)
                    ].copy().sort_values(by='data')

                    if data_for_features_pred.empty or data_for_features_pred[data_for_features_pred['data'] == selected_date_pred].empty:
                        st.error(f"Dados históricos insuficientes para {selected_municipio_pred} na data.")
                    else:
                        try:
                            df_engineered_pred = aplicar_engenharia_features_bloco3(data_for_features_pred)
                            df_to_predict = df_engineered_pred[df_engineered_pred['data'] == selected_date_pred].copy()
                            if df_to_predict.empty: st.error("Eng. de features não produziu dados.")
                            else:
                                for col_name, imputer_obj in pred_imputers.items():
                                    if col_name not in df_to_predict.columns: df_to_predict[col_name] = imputer_obj.statistics_[0]
                                    else: df_to_predict.loc[:, col_name] = imputer_obj.transform(df_to_predict[[col_name]])
                                if 'municipio' in df_to_predict: df_to_predict['municipio_encoded'] = pred_target_encoder.transform(df_to_predict[['municipio']])
                                else: raise ValueError("'municipio' ausente.")
                                df_model_input = pd.DataFrame(columns=pred_model_columns, index=df_to_predict.index)
                                for col_m in pred_model_columns:
                                    if col_m in df_to_predict.columns: df_model_input[col_m] = df_to_predict[col_m]
                                    else: df_model_input[col_m] = 0
                                features_orig_shap = df_model_input.copy()
                                scaled_arr_pred = pred_scaler.transform(features_orig_shap)
                                scaled_df_model = pd.DataFrame(scaled_arr_pred, columns=pred_model_columns, index=features_orig_shap.index)
                                proba_fire = pred_model.predict_proba(scaled_df_model)[:, 1][0]
                                final_pred = 1 if proba_fire >= PRED_DECISION_THRESHOLD else 0
                                st.markdown("---"); st.subheader("Resultado da Previsão:")
                                color,icon,status = ("red","🚨","RISCO") if final_pred==1 else ("green","✅","BAIXO RISCO")
                                st.markdown(f"<h4 style='color:{color};'>{icon} {status} DE INCÊNDIO</h4>", unsafe_allow_html=True)
                                st.markdown(f"**Probabilidade de Incêndio:** {proba_fire*100:.2f}%"); st.caption(f"Limiar: {PRED_DECISION_THRESHOLD*100:.1f}%")
                                with st.expander("🔎 Detalhes das Features e Importância (SHAP)"):
                                    try:
                                        explainer = shap.TreeExplainer(pred_model); shap_vals_inst = explainer.shap_values(scaled_df_model)
                                        shap_pos = shap_vals_inst[1][0] if isinstance(shap_vals_inst, list) else shap_vals_inst[0]
                                        df_shap = pd.DataFrame({'F': features_orig_shap.columns, 'V': features_orig_shap.iloc[0].values, 'S': shap_pos})
                                        df_shap['Abs_S'] = df_shap['S'].abs(); df_shap = df_shap.sort_values('Abs_S', ascending=False).drop('Abs_S', axis=1)
                                        ch = st.columns((.4,.3,.3)); ch[0].markdown("**Feature**"); ch[1].markdown("**Valor**"); ch[2].markdown("**SHAP**"); st.markdown("---")
                                        for _,r in df_shap.head(15).iterrows():
                                            d=FEATURE_DESCRIPTIONS.get(r['F'],"-"); cr=st.columns((.4,.3,.3))
                                            cr[0].markdown(f"<span title='{d}'>{r['F']}</span>", unsafe_allow_html=True)
                                            cr[1].text(f"{r['V']:.2f}"); sc="red" if r['S']>.001 else "green" if r['S']<-.001 else "gray"
                                            cr[2].markdown(f"<span style='color:{sc};'>{r['S']:+.3f}</span>", unsafe_allow_html=True)
                                    except Exception as e_s: st.warning(f"SHAP: {e_s}")
                        except Exception as e_p: st.error(f"Erro previsão: {e_p}"); st.text(traceback.format_exc())

elif st.session_state.active_main_tab_v8 == aba_desc_key: # << Use a chave correta aqui
    # st.header(f"📊 {aba_desc_key}") # Opcional, já que st.title mostra o nome da aba
    if pred_load_status == "success" and pred_full_dataset is not None:
        st.markdown("### Análise Descritiva dos Dados de Incêndio")
        st.write("Explore estatísticas e visualizações sobre focos de incêndio em SP (2022 e recentes do INPE).")
        st.markdown("---")

        # Seção para Análise Histórica (seu CSV de ML para SP 2022)
        st.subheader("📈 Análise Histórica (SP 2022)")
        
        # Contagem Mensal de Focos de Incêndio (2022) - GRÁFICO DE BARRAS
        st.markdown("##### Contagem Mensal de Focos de Incêndio (2022)")
        if 'ocorreu_incendio' in pred_full_dataset.columns and 'data' in pred_full_dataset.columns:
            monthly_fire_counts_2022 = pred_full_dataset[pred_full_dataset['ocorreu_incendio'] == 1].set_index('data').resample('ME').size()
            monthly_fire_counts_2022.index = monthly_fire_counts_2022.index.strftime('%Y-%m')
            if not monthly_fire_counts_2022.empty:
                st.bar_chart(monthly_fire_counts_2022, color="#FF4B4B", height=300) # MUDADO PARA BAR_CHART
                if st.checkbox("Mostrar dados tabulares mensais (2022)", key="cb_monthly_data_2022_v3"):
                    st.dataframe(monthly_fire_counts_2022.reset_index().rename(columns={'index':'Mês', 0:'Nº de Focos'})) # Ajustado nome da coluna data
            else:
                st.info("Não há dados de ocorrência de incêndios para exibir o gráfico mensal de 2022.")
        else:
            st.warning("Colunas 'ocorreu_incendio' ou 'data' não encontradas para gerar gráfico temporal de 2022.")

        # Distribuição por Bioma (Histórico 2022) - GRÁFICO DE PIZZA AJUSTADO
        if 'bioma' in pred_full_dataset.columns and 'ocorreu_incendio' in pred_full_dataset.columns:
            st.markdown("##### Distribuição de Focos por Bioma (2022)")
            focos_por_bioma_2022 = pred_full_dataset[pred_full_dataset['ocorreu_incendio'] == 1]['bioma'].value_counts()
            if not focos_por_bioma_2022.empty:
                fig_b_22, ax_b_22 = plt.subplots(figsize=(4, 3)) # Tamanho ajustado para caber legenda ao lado
                
                # Cores para o texto
                text_color = 'white'
                
                wedges, texts, autotexts = ax_b_22.pie(
                    focos_por_bioma_2022,
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 0.5},
                    pctdistance=0.80, # Posição dos percentuais
                    textprops={'color': text_color, 'fontsize': 7} # Cor e tamanho dos percentuais
                    # Removido 'labels' daqui para usar legenda
                )
                ax_b_22.set_ylabel('')
                
                # Adicionar Legenda com texto branco
                # bbox_to_anchor ajusta a posição da legenda para fora do gráfico
                # ncol define o número de colunas da legenda
                legend = ax_b_22.legend(
                    wedges, focos_por_bioma_2022.index,
                    title="Biomas",
                    loc="center left",
                    bbox_to_anchor=(0.95, 0, 0.5, 1), # Posição à direita do gráfico
                    fontsize=8,
                    labelcolor=text_color, # Cor do texto da legenda
                    frameon=False # Remove a moldura da legenda
                )
                plt.setp(legend.get_title(), color=text_color, fontsize=9) # Cor do título da legenda

                fig_b_22.patch.set_alpha(0.0)
                ax_b_22.patch.set_alpha(0.0)
                
                st.pyplot(fig_b_22, use_container_width=True) # use_container_width=True pode ajudar a ajustar
            else:
                st.info("Sem dados de bioma para focos de 2022.")
        st.markdown("---")
    else:
        st.warning("Dados históricos de 2022 não disponíveis para análise.")


    # Seção para Monitoramento de Queimadas Atuais (INPE)
    st.subheader("🛰️ Monitoramento de Queimadas Atuais em SP (Fonte: INPE)")
    
    if 'inpe_dados_recentes' not in st.session_state: st.session_state.inpe_dados_recentes = pd.DataFrame()
    if 'inpe_data_last_updated' not in st.session_state: st.session_state.inpe_data_last_updated = None

    cols_rpa_ctrl_desc = st.columns([1,3]) # Renomeada variável para evitar conflito
    num_dias_rpa_desc_val = cols_rpa_ctrl_desc[0].number_input("Analisar focos dos últimos (dias):", 1, 30, 7, key="rpa_num_dias_desc_v6") # Nova chave
    
    if cols_rpa_ctrl_desc[1].button(f"🔄 Buscar Dados INPE (SP - {num_dias_rpa_desc_val} dias)", key="rpa_btn_fetch_desc_v6"): # Nova chave
        with st.spinner(f"Buscando dados do INPE..."):
            st.session_state.inpe_dados_recentes = fetch_and_process_inpe_daily_data(
                num_past_days=int(num_dias_rpa_desc_val), target_estado="Sao Paulo" # Usar "Sao Paulo" para filtro correto
            )
            if not st.session_state.inpe_dados_recentes.empty:
                st.session_state.inpe_data_last_updated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                st.success(f"Dados INPE atualizados: {len(st.session_state.inpe_dados_recentes)} focos para SP.")
            else: st.warning("Nenhum foco recente encontrado para SP (INPE) ou falha na busca.")
    
    if not st.session_state.inpe_dados_recentes.empty:
        df_inpe_rec_desc = st.session_state.inpe_dados_recentes # Renomeada variável para evitar conflito
        df_inpe_rec_desc['data'] = pd.to_datetime(df_inpe_rec_desc['data'])

        last_upd_str_desc = st.session_state.inpe_data_last_updated if st.session_state.inpe_data_last_updated else "N/A"
        st.caption(f"Dados INPE para SP. Atualizado nesta sessão: {last_upd_str_desc}")
        
        # Contagem de Focos por Dia (Recente) - GRÁFICO DE BARRAS
        st.markdown(f"##### Contagem de Focos Diários em SP (Últimos {num_dias_rpa_desc_val} Dias - INPE)")
        focos_dia_rpa_desc = df_inpe_rec_desc.groupby(df_inpe_rec_desc['data'].dt.date).size().sort_index()
        if not focos_dia_rpa_desc.empty:
            focos_dia_rpa_desc.index = pd.to_datetime(focos_dia_rpa_desc.index).strftime('%d/%m')
            st.bar_chart(focos_dia_rpa_desc, color="#FF8C00", height=300) # GRÁFICO DE BARRAS
            
            limiar_alerta_rpa = st.slider("Limiar de alerta (focos/dia):", 1, 100, 20, key="slider_alerta_rpa_v3") # Nova chave
            dias_criticos_rpa_desc = focos_dia_rpa_desc[focos_dia_rpa_desc > limiar_alerta_rpa]
            if not dias_criticos_rpa_desc.empty:
                st.error(f"🚨 ALERTA: Dias com mais de {limiar_alerta_rpa} focos (INPE):")
                for d, c in dias_criticos_rpa_desc.items(): st.markdown(f"- {d}: **{c} focos**")
            else: st.success(f"Nenhum dia com mais de {limiar_alerta_rpa} focos no período recente.")
        else: st.info("Sem dados de focos por dia (INPE) para o período.")

        # Distribuição por Bioma (Recente) - GRÁFICO DE PIZZA AJUSTADO
        if 'bioma' in df_inpe_rec_desc.columns:
            st.markdown("##### Distribuição Focos Recentes por Bioma (INPE)")
            focos_bioma_recente_desc = df_inpe_rec_desc['bioma'].value_counts()
            if not focos_bioma_recente_desc.empty:
                fig_br_desc, ax_br_desc = plt.subplots(figsize=(4, 3)) # TAMANHO AJUSTADO

                text_color_rec = 'white'
                wedges_rec, texts_rec, autotexts_rec = ax_br_desc.pie(
                    focos_bioma_recente_desc, autopct='%1.1f%%', startangle=120,
                    wedgeprops={'edgecolor':'white', 'linewidth':0.5}, pctdistance=0.80,
                    textprops={'color': text_color_rec, 'fontsize': 7},
                    colors=plt.cm.Pastel1.colors
                )
                ax_br_desc.set_ylabel('')

                legend_rec = ax_br_desc.legend(
                    wedges_rec, focos_bioma_recente_desc.index,
                    title="Biomas (Recente)",
                    loc="center left",
                    bbox_to_anchor=(0.95, 0, 0.5, 1),
                    fontsize=8,
                    labelcolor=text_color_rec,
                    frameon=False
                )
                plt.setp(legend_rec.get_title(), color=text_color_rec, fontsize=9)
                
                fig_br_desc.patch.set_alpha(0.0)
                ax_br_desc.patch.set_alpha(0.0)
                st.pyplot(fig_br_desc, use_container_width=True)
            else:
                st.info("Sem dados de bioma para focos recentes do INPE.")
        
        # Mapa e Tabela (como antes, mas pode adicionar colunas se desejar)
        if 'latitude' in df_inpe_rec_desc.columns and st.checkbox("Mostrar mapa de focos recentes (INPE)", key="cb_mapa_rpa_v3"): # Nova chave
            df_mapa_rpa_desc = df_inpe_rec_desc.dropna(subset=['latitude', 'longitude'])
            if not df_mapa_rpa_desc.empty: st.map(df_mapa_rpa_desc[['latitude', 'longitude']])
        
        if st.checkbox("Mostrar tabela detalhada dos focos recentes (INPE)", key="rpa_cb_table_desc_v4"): # Nova chave
            cols_disp_inpe_desc = ['datahora_gmt', 'municipio', 'estado', 'bioma', 'latitude', 'longitude', 'satelite', 'risco_fogo', 'frp', 'dias_sem_chuva']
            cols_pres_inpe_desc = [c for c in cols_disp_inpe_desc if c in df_inpe_rec_desc.columns]
            st.dataframe(df_inpe_rec_desc[cols_pres_inpe_desc])
    else:
        st.info("Clique no botão para buscar dados recentes do INPE e ver análises.")

elif st.session_state.active_main_tab_v8 == aba_rag_key: # Atenção à chave
    display_rag_chat_tab()

elif st.session_state.active_main_tab_v8 == aba_sobre_key: # Use sua chave de session_state correta
    # st.header(f"ℹ️ {aba_sobre_key}") # Opcional, pois st.title já mostra
    st.markdown("### Sobre a Plataforma VigIA Focos") # Use o nome que você escolher
    st.markdown("""
    Bem-vindo à **VigIA Focos**, uma plataforma integrada desenvolvida com o objetivo de auxiliar na prevenção, monitoramento e orientação em situações de incêndios. Esta solução combina três componentes principais para oferecer um suporte abrangente:

    1.  **Módulo de Previsão de Risco de Incêndio:**
        *   Utiliza um modelo de Machine Learning para estimar a probabilidade de ocorrência de focos de incêndio. Atualmente, o modelo está treinado e validado com dados históricos detalhados do estado de São Paulo para o ano de 2022, mas foi projetado com a perspectiva de expansão para abranger novos períodos e regiões, incorporando dados atualizados para previsões futuras.

    2.  **Assistente Virtual Inteligente (VigIA):**
        *   Um chatbot conversacional, chamado VigIA, que emprega técnicas de Inteligência Artificial Generativa com Arquitetura RAG (Retrieval-Augmented Generation). VigIA busca informações atualizadas na web para fornecer orientações e respostas personalizadas para diferentes perfis de usuários – **Vítimas**, **Moradores** e **Familiares** – em contextos de incêndio, focando em segurança, prevenção e formas de auxílio.

    3.  **Painel de Análise Descritiva:**
        *   Oferece uma visualização detalhada de dados sobre incêndios através de gráficos e tabelas interativas. Esta seção inclui:
            *   Uma análise histórica baseada nos dados utilizados pelo modelo de Machine Learning (SP, 2022).
            *   Um monitoramento de queimadas atuais, com dados atualizados do portal TerraBrasilis/INPE, permitindo o acompanhamento de focos recentes.

    A plataforma **VigIA Focos** representa um esforço para aplicar tecnologias de Inteligência Artificial na mitigação dos impactos causados por incêndios, oferecendo ferramentas tanto para antecipação de riscos quanto para orientação em momentos críticos.
    """)
    st.markdown("---")