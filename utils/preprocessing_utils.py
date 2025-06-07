# ==============================================================================
# ETAPA 1: IMPORTAÇÕES
# ==============================================================================
import pandas as pd
import numpy as np

# ==============================================================================
# ETAPA 2: FUNÇÕES AUXILIARES DE ENGENHARIA DE FEATURES
# ==============================================================================

def criar_lags_features(df, group_col, target_cols, lags, data_col='data'):
    """Cria features defasadas (lags) para colunas alvo, agrupadas por uma coluna de referência."""
    df_lagged = df.copy()
    df_lagged[data_col] = pd.to_datetime(df_lagged[data_col])
    df_lagged = df_lagged.sort_values(by=[group_col, data_col])
    for col in target_cols:
        if col not in df_lagged.columns: continue
        for lag in lags:
            df_lagged[f'{col}_lag{lag}'] = df_lagged.groupby(group_col)[col].shift(lag)
    return df_lagged

def criar_rolling_mean_features(df, group_col, target_cols, windows, data_col='data', min_periods=1):
    """Cria features de média móvel (rolling mean) para colunas alvo."""
    df_rolled = df.copy()
    df_rolled[data_col] = pd.to_datetime(df_rolled[data_col])
    df_rolled = df_rolled.sort_values(by=[group_col, data_col])
    for col in target_cols:
        if col not in df_rolled.columns: continue
        for window in windows:
            # O shift(1) garante que a média móvel use dados até o dia anterior, evitando data leakage.
            df_rolled[f'{col}_roll_mean{window}'] = df_rolled.groupby(group_col)[col].shift(1).rolling(window=window, min_periods=min_periods).mean()
    return df_rolled

def calcular_dias_secos_consecutivos(df, group_col, precip_col, data_col='data', limiar_chuva=0.1):
    """Calcula o número de dias consecutivos com precipitação abaixo de um limiar."""
    df_sorted = df.copy()
    df_sorted[data_col] = pd.to_datetime(df_sorted[data_col])
    df_sorted = df_sorted.sort_values(by=[group_col, data_col])

    if precip_col not in df_sorted.columns:
        df_sorted['dias_secos_consecutivos'] = 0
        return df_sorted

    df_sorted['dia_seco_flag'] = (df_sorted[precip_col].fillna(0.0) <= limiar_chuva).astype(int)
    df_sorted['bloco_id'] = (df_sorted['dia_seco_flag'] != df_sorted.groupby(group_col)['dia_seco_flag'].shift(1)).cumsum()
    df_sorted['dias_secos_consecutivos'] = df_sorted.groupby([group_col, 'bloco_id'])['dia_seco_flag'].cumsum()
    df_sorted.loc[df_sorted['dia_seco_flag'] == 0, 'dias_secos_consecutivos'] = 0
    df_sorted.drop(columns=['dia_seco_flag', 'bloco_id'], inplace=True, errors='ignore')
    return df_sorted

def ensure_series_for_fwi(df_series_or_none, fallback_value, index_ref):
    """Garante que uma série Pandas exista e esteja preenchida, usando um valor padrão se necessário."""
    if isinstance(df_series_or_none, pd.Series):
        return df_series_or_none.fillna(fallback_value)
    return pd.Series([fallback_value] * len(index_ref), index=index_ref, dtype=float)


# ==============================================================================
# ETAPA 3: FUNÇÃO PRINCIPAL DE ENGENHARIA DE FEATURES
# ==============================================================================
def aplicar_engenharia_features_bloco3(df_historico_municipio, max_lag_days_config=None):
    """Orquestra a aplicação de todas as etapas de engenharia de features em um DataFrame de dados históricos."""
    if df_historico_municipio.empty:
        return df_historico_municipio

    df_model = df_historico_municipio.copy()

    # ETAPA 3.1: IMPUTAÇÃO INICIAL DE DADOS AUSENTES
    # Preenche valores nulos em colunas-chave com a mediana ou valores fixos para garantir a robustez dos cálculos.
    colunas_para_imputacao_inicial = [
        'numero_dias_sem_chuva_first', 'precipitacao_sum', 'risco_fogo_mean', 'frp_sum', 'frp_mean', 'frp_max',
        'precip_total_dia_mm', 'temp_media_dia_C', 'temp_max_dia_C', 'temp_min_dia_C', 'umidade_media_dia_perc',
        'umidade_min_dia_perc', 'vento_vel_media_dia_ms'
    ]
    for col in colunas_para_imputacao_inicial:
        if col in df_model.columns and df_model[col].isnull().any():
            fill_val = np.nan
            if 'precip' in col or 'frp' in col: fill_val = 0.0
            elif df_model[col].notnull().any():
                 if 'temp' in col: fill_val = df_model[col].median()
                 elif 'umidade' in col: fill_val = df_model[col].median()
                 elif 'vento' in col: fill_val = df_model[col].median()
                 else: fill_val = df_model[col].median()
            
            if pd.isna(fill_val):
                if 'precip' in col or 'frp' in col: fill_val = 0.0
                elif 'temp' in col: fill_val = 25.0
                elif 'umidade' in col: fill_val = 60.0
                elif 'vento' in col: fill_val = 2.0
                else: fill_val = 0.0
            df_model[col] = df_model[col].fillna(fill_val)

    # ETAPA 3.2: CRIAÇÃO DE FEATURES TEMPORAIS (LAGS E MÉDIAS MÓVEIS)
    # Gera features baseadas em dados de dias anteriores para capturar tendências e padrões temporais.
    lags_a_criar = [1, 3, 7]
    windows_a_criar = [3, 7, 15]
    cols_meteo_inmet_para_lagroll = ['precip_total_dia_mm', 'temp_media_dia_C', 'temp_max_dia_C', 'temp_min_dia_C', 'umidade_media_dia_perc', 'umidade_min_dia_perc', 'vento_vel_media_dia_ms']
    cols_focos_originais_para_lagroll = ['numero_dias_sem_chuva_first', 'risco_fogo_mean', 'precipitacao_sum']
    cols_frp_base_para_lagroll = ['frp_sum', 'frp_mean', 'frp_max']
    colunas_para_lags_e_rollings_final = []
    for c_list in [cols_meteo_inmet_para_lagroll, cols_focos_originais_para_lagroll, cols_frp_base_para_lagroll]:
        for col_base in c_list:
            if col_base in df_model.columns: colunas_para_lags_e_rollings_final.append(col_base)
    colunas_para_lags_e_rollings_final = sorted(list(set(colunas_para_lags_e_rollings_final)))

    if colunas_para_lags_e_rollings_final:
        df_model = criar_lags_features(df_model, 'municipio', colunas_para_lags_e_rollings_final, lags_a_criar)
        df_model = criar_rolling_mean_features(df_model, 'municipio', colunas_para_lags_e_rollings_final, windows_a_criar)

    # ETAPA 3.3: CÁLCULO DE DIAS SECOS CONSECUTIVOS
    col_precip_dias_secos = 'precip_total_dia_mm' if 'precip_total_dia_mm' in df_model.columns else 'precipitacao_sum'
    if col_precip_dias_secos in df_model.columns:
        df_model = calcular_dias_secos_consecutivos(df_model, 'municipio', col_precip_dias_secos)

    # ETAPA 3.4: CRIAÇÃO DE FEATURES DE INTERAÇÃO
    # Combina features existentes para modelar relações não-lineares, como o efeito combinado de dias secos e baixa umidade.
    umidade_lag1_median_fallback = df_model['umidade_min_dia_perc_lag1'].median() if 'umidade_min_dia_perc_lag1' in df_model.columns and df_model['umidade_min_dia_perc_lag1'].notnull().any() else 70.0
    risco_fogo_lag1_median_fallback = df_model['risco_fogo_mean_lag1'].median() if 'risco_fogo_mean_lag1' in df_model.columns and df_model['risco_fogo_mean_lag1'].notnull().any() else 0.5

    if 'dias_secos_consecutivos' in df_model.columns and 'umidade_min_dia_perc_lag1' in df_model.columns:
        df_model['dias_secos_X_umidade_baixa_lag1'] = df_model['dias_secos_consecutivos'] * (100 - df_model['umidade_min_dia_perc_lag1'].fillna(umidade_lag1_median_fallback))

    if 'dias_secos_consecutivos' in df_model.columns and 'risco_fogo_mean_lag1' in df_model.columns:
        df_model['dias_secos_X_risco_fogo_lag1'] = df_model['dias_secos_consecutivos'] * df_model['risco_fogo_mean_lag1'].fillna(risco_fogo_lag1_median_fallback)

    # ETAPA 3.5: CÁLCULO DAS PROXIES DO ÍNDICE DE RISCO DE FOGO (FWI)
    # Calcula aproximações dos componentes do FWI (Fire Weather Index) usando os dados meteorológicos disponíveis.
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

    df_model['ffmc_proxy'] = (temp_max_fwi / 30.0) + ((100.1 - umidade_min_fwi) / 50.0) - (precip_lag1_fwi / 10.0)
    df_model['dmc_proxy'] = (temp_media_fwi / 20.0) + ((100.1 - umidade_media_fwi) / 60.0) - (precip_roll7_fwi / 5.0)
    df_model['dc_proxy'] = (dias_secos_fwi / 10.0) - (precip_roll15_fwi / 20.0)
    df_model['ffmc_proxy'] = df_model['ffmc_proxy'].fillna(0)
    df_model['dmc_proxy'] = df_model['dmc_proxy'].fillna(0)
    df_model['dc_proxy'] = df_model['dc_proxy'].fillna(0)
    df_model['isi_proxy'] = vento_fwi * np.exp(0.05 * df_model['ffmc_proxy'])
    df_model['bui_proxy'] = (0.8 * df_model['dmc_proxy'] + 0.2 * df_model['dc_proxy']).clip(lower=0)
    df_model['isi_proxy'] = df_model['isi_proxy'].fillna(0)
    df_model['bui_proxy'] = df_model['bui_proxy'].fillna(0)
    
    fwi_b_calc = 0.1 * df_model['isi_proxy'] * df_model['bui_proxy']
    safe_fwi_b_calc = np.maximum(fwi_b_calc, 1e-9)
    df_model['fwi_proxy_final'] = np.where(fwi_b_calc <=1, fwi_b_calc, np.exp(2.72 * (0.434 * np.log(safe_fwi_b_calc))**0.647 ) )
    df_model['fwi_proxy_final'] = df_model['fwi_proxy_final'].fillna(0)

    # Retorna o DataFrame com todas as novas features engenheiradas.
    return df_model