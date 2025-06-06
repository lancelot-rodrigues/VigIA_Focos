# utils/preprocessing_utils.py
import pandas as pd
import numpy as np

# ==============================================================================
# BLOCO 2: DEFINIÇÃO DE FUNÇÕES AUXILIARES (Extraído de incendio_ml.py)
# ==============================================================================

def criar_lags_features(df, group_col, target_cols, lags, data_col='data'):
    df_lagged = df.copy()
    df_lagged[data_col] = pd.to_datetime(df_lagged[data_col])
    df_lagged = df_lagged.sort_values(by=[group_col, data_col])
    for col in target_cols:
        if col not in df_lagged.columns:
            # print(f"  Aviso em criar_lags_features: Coluna '{col}' não encontrada. Pulando.") #debug
            continue
        for lag in lags:
            df_lagged[f'{col}_lag{lag}'] = df_lagged.groupby(group_col)[col].shift(lag)
    return df_lagged

def criar_rolling_mean_features(df, group_col, target_cols, windows, data_col='data', min_periods=1):
    df_rolled = df.copy()
    df_rolled[data_col] = pd.to_datetime(df_rolled[data_col])
    df_rolled = df_rolled.sort_values(by=[group_col, data_col])
    for col in target_cols:
        if col not in df_rolled.columns:
            # print(f"  Aviso em criar_rolling_mean_features: Coluna '{col}' não encontrada. Pulando.") #debug
            continue
        for window in windows:
            # O shift(1) aqui significa que a média móvel usa dados ATÉ O DIA ANTERIOR.
            df_rolled[f'{col}_roll_mean{window}'] = df_rolled.groupby(group_col)[col].shift(1).rolling(window=window, min_periods=min_periods).mean()
    return df_rolled

def calcular_dias_secos_consecutivos(df, group_col, precip_col, data_col='data', limiar_chuva=0.1):
    df_sorted = df.copy()
    df_sorted[data_col] = pd.to_datetime(df_sorted[data_col])
    df_sorted = df_sorted.sort_values(by=[group_col, data_col])

    if precip_col not in df_sorted.columns:
        print(f"AVISO em dias_secos: Coluna '{precip_col}' não encontrada. Coluna 'dias_secos_consecutivos' será preenchida com 0.")
        df_sorted['dias_secos_consecutivos'] = 0
        return df_sorted

    # ASSUMINDO QUE A IMPUTAÇÃO INICIAL JÁ TRATOU NaNs EM precip_col COM 0.0
    # Se ainda houver NaNs aqui, significa que a imputação inicial não cobriu este caso
    # ou a coluna de precipitação não passou pela imputação inicial.
    # Para robustez, podemos manter um fillna aqui, mas com 0 (para dia seco se NaN).
    df_sorted['dia_seco_flag'] = (df_sorted[precip_col].fillna(0.0) <= limiar_chuva).astype(int)
    
    df_sorted['bloco_id'] = (df_sorted['dia_seco_flag'] != df_sorted.groupby(group_col)['dia_seco_flag'].shift(1)).cumsum()
    df_sorted['dias_secos_consecutivos'] = df_sorted.groupby([group_col, 'bloco_id'])['dia_seco_flag'].cumsum()
    df_sorted.loc[df_sorted['dia_seco_flag'] == 0, 'dias_secos_consecutivos'] = 0
    df_sorted.drop(columns=['dia_seco_flag', 'bloco_id'], inplace=True, errors='ignore')
    return df_sorted

def ensure_series_for_fwi(df_series_or_none, fallback_value, index_ref):
    """
    Garante que uma série Pandas seja retornada com o índice de referência e NaNs preenchidos.
    Se df_series_or_none for None ou uma série vazia, retorna uma nova série com o fallback_value.
    Se for uma série, preenche NaNs com fallback_value. Assume que se for uma série,
    ela já tem o índice correto ou pode ser reindexada (o reindex foi removido aqui
    para simplificar, assumindo que as colunas .get() já alinham os índices ou são None).
    """
    if isinstance(df_series_or_none, pd.Series):
        # Se a série existe mas pode ter NaNs (ex: lags nos primeiros dias do subset)
        return df_series_or_none.fillna(fallback_value)
    # Se a série não existe (df_model.get retornou None)
    return pd.Series([fallback_value] * len(index_ref), index=index_ref, dtype=float)


# ==============================================================================
# FUNÇÃO PRINCIPAL DE ENGENHARIA DE FEATURES (Adaptado do Bloco 3)
# ==============================================================================
def aplicar_engenharia_features_bloco3(df_historico_municipio, max_lag_days_config=None): # max_lag_days_config é opcional aqui
    """
    Aplica a engenharia de features do Bloco 3 do script de ML incendio_ml.py.
    df_historico_municipio: DataFrame contendo dados do município selecionado
                               e os N dias anteriores necessários para lags/rollings.
                               Deve estar ordenado por 'data' ascendentemente e já filtrado para um único município.
    """
    if df_historico_municipio.empty:
        print("DataFrame de histórico do município está vazio. Nenhuma feature será criada.")
        return df_historico_municipio

    # A variável no script original era df_model, vamos usar o mesmo nome aqui para consistência interna
    df_model = df_historico_municipio.copy()
    # print(f"Iniciando Engenharia de Features para Streamlit. Linhas recebidas: {len(df_model)}") # Para debug no console do Streamlit

    # --- Imputação Inicial (do Bloco 3) ---
    colunas_para_imputacao_inicial = [
        'numero_dias_sem_chuva_first', 'precipitacao_sum', 'risco_fogo_mean',
        'frp_sum', 'frp_mean', 'frp_max',
        'precip_total_dia_mm', 'temp_media_dia_C', 'temp_max_dia_C',
        'temp_min_dia_C', 'umidade_media_dia_perc',
        'umidade_min_dia_perc', 'vento_vel_media_dia_ms'
    ]
    # print("Imputando NaNs iniciais...")
    for col in colunas_para_imputacao_inicial:
        if col in df_model.columns and df_model[col].isnull().any():
            fill_val = np.nan
            # Tenta usar mediana do subset atual (df_model)
            if 'precip' in col or 'frp' in col: fill_val = 0.0
            elif df_model[col].notnull().any(): # Apenas calcula mediana se houver algum valor não-NaN
                 if 'temp' in col: fill_val = df_model[col].median()
                 elif 'umidade' in col: fill_val = df_model[col].median()
                 elif 'vento' in col: fill_val = df_model[col].median()
                 else: fill_val = df_model[col].median()
            
            # Fallback para valores fixos se a mediana for NaN (ex: coluna toda NaN no subset)
            if pd.isna(fill_val):
                # print(f"  Coluna {col} teve mediana NaN, usando fallback fixo.") #debug
                if 'precip' in col or 'frp' in col: fill_val = 0.0
                elif 'temp' in col: fill_val = 25.0
                elif 'umidade' in col: fill_val = 60.0
                elif 'vento' in col: fill_val = 2.0
                else: fill_val = 0.0
            df_model[col] = df_model[col].fillna(fill_val)

    # --- Criação de Lags e Rollings (do Bloco 3) ---
    lags_a_criar = [1, 3, 7]
    windows_a_criar = [3, 7, 15]
    cols_meteo_inmet_para_lagroll = [
        'precip_total_dia_mm', 'temp_media_dia_C', 'temp_max_dia_C',
        'temp_min_dia_C', 'umidade_media_dia_perc',
        'umidade_min_dia_perc', 'vento_vel_media_dia_ms'
    ]
    cols_focos_originais_para_lagroll = [
        'numero_dias_sem_chuva_first', 'risco_fogo_mean', 'precipitacao_sum'
    ]
    cols_frp_base_para_lagroll = ['frp_sum', 'frp_mean', 'frp_max']
    colunas_para_lags_e_rollings_final = []
    for c_list in [cols_meteo_inmet_para_lagroll, cols_focos_originais_para_lagroll, cols_frp_base_para_lagroll]:
        for col_base in c_list:
            if col_base in df_model.columns: colunas_para_lags_e_rollings_final.append(col_base)
    colunas_para_lags_e_rollings_final = sorted(list(set(colunas_para_lags_e_rollings_final)))
    # print(f"Colunas para Lags/Rollings (Streamlit): {colunas_para_lags_e_rollings_final}")

    if colunas_para_lags_e_rollings_final:
        # 'municipio' é o group_col. df_model já é filtrado para um município e ordenado por data.
        df_model = criar_lags_features(df_model, 'municipio', colunas_para_lags_e_rollings_final, lags_a_criar)
        df_model = criar_rolling_mean_features(df_model, 'municipio', colunas_para_lags_e_rollings_final, windows_a_criar)
        # print(f"Features de lag/rolling criadas (Streamlit).")

    # --- Cálculo de Dias Secos (do Bloco 3) ---
    col_precip_dias_secos = 'precip_total_dia_mm' if 'precip_total_dia_mm' in df_model.columns else 'precipitacao_sum'
    if col_precip_dias_secos in df_model.columns:
        df_model = calcular_dias_secos_consecutivos(df_model, 'municipio', col_precip_dias_secos)
        # if 'dias_secos_consecutivos' in df_model.columns:
            # print("Feature 'dias_secos_consecutivos' criada (Streamlit).")

    # --- Criação de Features de Interação (do Bloco 3) ---
    # Para fillna nas features de interação, usar a mediana da coluna defasada DENTRO do subset histórico atual
    # é mais robusto do que um valor fixo, mas ainda pode ser problemático se houver poucos dados.
    umidade_lag1_median_fallback = df_model['umidade_min_dia_perc_lag1'].median() if 'umidade_min_dia_perc_lag1' in df_model.columns and df_model['umidade_min_dia_perc_lag1'].notnull().any() else 70.0
    risco_fogo_lag1_median_fallback = df_model['risco_fogo_mean_lag1'].median() if 'risco_fogo_mean_lag1' in df_model.columns and df_model['risco_fogo_mean_lag1'].notnull().any() else 0.5

    if 'dias_secos_consecutivos' in df_model.columns and 'umidade_min_dia_perc_lag1' in df_model.columns:
        df_model['dias_secos_X_umidade_baixa_lag1'] = df_model['dias_secos_consecutivos'] * \
            (100 - df_model['umidade_min_dia_perc_lag1'].fillna(umidade_lag1_median_fallback))
        # print("Feature 'dias_secos_X_umidade_baixa_lag1' criada (Streamlit).")
    # else:
        # print("Não foi possível criar 'dias_secos_X_umidade_baixa_lag1': colunas dependentes ausentes (Streamlit).")

    if 'dias_secos_consecutivos' in df_model.columns and 'risco_fogo_mean_lag1' in df_model.columns:
        df_model['dias_secos_X_risco_fogo_lag1'] = df_model['dias_secos_consecutivos'] * \
            df_model['risco_fogo_mean_lag1'].fillna(risco_fogo_lag1_median_fallback)
        # print("Feature 'dias_secos_X_risco_fogo_lag1' criada (Streamlit).")
    # else:
        # print("Não foi possível criar 'dias_secos_X_risco_fogo_lag1': colunas dependentes ausentes (Streamlit).")

    # --- Cálculo das FWI-like Proxies (do Bloco 3) ---
    idx_ref = df_model.index

    temp_max_fwi = ensure_series_for_fwi(df_model.get('temp_max_dia_C_lag1'), 25.0, idx_ref)
    umidade_min_fwi = ensure_series_for_fwi(df_model.get('umidade_min_dia_perc_lag1'), 70.0, idx_ref)
    vento_fwi = ensure_series_for_fwi(df_model.get('vento_vel_media_dia_ms_lag1'), 2.0, idx_ref)
    temp_media_fwi = ensure_series_for_fwi(df_model.get('temp_media_dia_C_lag1'), 20.0, idx_ref)
    umidade_media_fwi = ensure_series_for_fwi(df_model.get('umidade_media_dia_perc_lag1'), 60.0, idx_ref)

    precip_lag1_col_name = f'{col_precip_dias_secos}_lag1' if col_precip_dias_secos in df_model.columns else None
    precip_roll7_col_name = f'{col_precip_dias_secos}_roll_mean7' if col_precip_dias_secos in df_model.columns else None
    precip_roll15_col_name = f'{col_precip_dias_secos}_roll_mean15' if col_precip_dias_secos in df_model.columns else None

    precip_lag1_fwi = ensure_series_for_fwi(df_model.get(precip_lag1_col_name) if precip_lag1_col_name else None, 0.0, idx_ref)
    precip_roll7_fwi = ensure_series_for_fwi(df_model.get(precip_roll7_col_name) if precip_roll7_col_name else None, 0.0, idx_ref)
    precip_roll15_fwi = ensure_series_for_fwi(df_model.get(precip_roll15_col_name) if precip_roll15_col_name else None, 0.0, idx_ref)
    dias_secos_fwi = ensure_series_for_fwi(df_model.get('dias_secos_consecutivos'), 0.0, idx_ref)

    # Cálculo das FWI proxies
    df_model['ffmc_proxy'] = (temp_max_fwi / 30.0) + ((100.1 - umidade_min_fwi) / 50.0) - (precip_lag1_fwi / 10.0)
    df_model['dmc_proxy'] = (temp_media_fwi / 20.0) + ((100.1 - umidade_media_fwi) / 60.0) - (precip_roll7_fwi / 5.0)
    df_model['dc_proxy'] = (dias_secos_fwi / 10.0) - (precip_roll15_fwi / 20.0)

    # Fill NaNs nas proxies intermediárias antes de usá-las em outras
    df_model['ffmc_proxy'] = df_model['ffmc_proxy'].fillna(0) # Usar um valor que faça sentido como 'sem risco'
    df_model['dmc_proxy'] = df_model['dmc_proxy'].fillna(0)
    df_model['dc_proxy'] = df_model['dc_proxy'].fillna(0)

    df_model['isi_proxy'] = vento_fwi * np.exp(0.05 * df_model['ffmc_proxy'])
    df_model['bui_proxy'] = (0.8 * df_model['dmc_proxy'] + 0.2 * df_model['dc_proxy']).clip(lower=0)

    df_model['isi_proxy'] = df_model['isi_proxy'].fillna(0)
    df_model['bui_proxy'] = df_model['bui_proxy'].fillna(0)
    
    fwi_b_calc = 0.1 * df_model['isi_proxy'] * df_model['bui_proxy']
    # Evitar log de zero ou negativo
    safe_fwi_b_calc = np.maximum(fwi_b_calc, 1e-9)
    df_model['fwi_proxy_final'] = np.where(fwi_b_calc <=1, fwi_b_calc, np.exp(2.72 * (0.434 * np.log(safe_fwi_b_calc))**0.647 ) )
    df_model['fwi_proxy_final'] = df_model['fwi_proxy_final'].fillna(0)

    # print("Features FWI-like proxies criadas/atualizadas (Streamlit).")
    # print(f"Engenharia de Features Concluída (Streamlit). Colunas resultantes: {df_model.columns.tolist()}")
    return df_model