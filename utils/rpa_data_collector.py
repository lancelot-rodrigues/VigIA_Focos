# utils/rpa_data_collector.py
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import streamlit as st # Usado apenas para st.info/warning/error dentro desta função para feedback direto
import os
from bs4 import BeautifulSoup # Mantido caso precise no futuro, mas não usado para nomes de arquivo fixos
import traceback
from unidecode import unidecode as unidecode_function # Para normalizar strings com acentos

# URL base para os dados DIÁRIOS do BRASIL
BASE_INPE_DIARIO_URL = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/diario/Brasil/"

# Mapa de renomeação de colunas para padronização interna
COL_RENAME_MAP_DIARIO = {
    'lat': 'latitude',
    'lon': 'longitude',
    'data_hora_gmt': 'datahora_gmt_original', # Renomeia para preservar original se necessário
    'numero_dias_sem_chuva': 'dias_sem_chuva',
    # Adicione outros mapeamentos se seus nomes de coluna forem diferentes dos esperados
}

def fetch_and_process_inpe_daily_data(num_past_days=7, target_estado="SP"): # Aumentei padrão para 7 dias
    """
    Busca os arquivos CSV diários de focos de queimadas do INPE para o Brasil,
    concatena os dados dos últimos 'num_past_days', limpa e filtra para um estado específico.
    """
    print(f"RPA_COLLECTOR: Buscando dados diários INPE para últimos {num_past_days} dias (Estado Alvo: {target_estado}).")
    all_daily_data = []
    
    # Itera para trás a partir de D-1 (dados de hoje geralmente não disponíveis no mesmo dia)
    # Tenta buscar um pouco mais de dias para garantir que temos dados suficientes.
    # Se num_past_days=1, tentará D-1, D-2. Se num_past_days=7, tentará D-1 a D-8.
    for i in range(1, num_past_days + 2): 
        process_date = datetime.now() - timedelta(days=i)
        date_str_for_filename = process_date.strftime("%Y%m%d")
        filename = f"focos_diario_br_{date_str_for_filename}.csv"
        
        # Construção de URL mais robusta
        file_url = BASE_INPE_DIARIO_URL.rstrip('/') + '/' + filename
        
        print(f"  Tentando baixar: {file_url}")
        try:
            response = requests.get(file_url, timeout=20) # Timeout para a requisição
            response.raise_for_status() # Levanta erro para status HTTP ruins (4xx, 5xx)
            
            # Tentar decodificar e ler o CSV
            try:
                csv_content_str = response.content.decode('utf-8')
                df_day = pd.read_csv(StringIO(csv_content_str), sep=',', encoding='utf-8')
            except UnicodeDecodeError:
                print(f"    Falha UTF-8 para {filename}, tentando latin1...")
                csv_content_str = response.content.decode('latin1')
                df_day = pd.read_csv(StringIO(csv_content_str), sep=',', encoding='latin1')
            except pd.errors.ParserError as pe:
                print(f"    Erro de parsing no Pandas para {filename}: {pe}. Arquivo pode estar malformatado. Pulando.")
                continue # Pula para o próximo arquivo/dia

            # print(f"    Dados de {date_str_for_filename} carregados. Shape: {df_day.shape}. Colunas: {df_day.columns.tolist()}") # Debug
            df_day['data_arquivo_original'] = process_date.date() # Data de referência do arquivo
            all_daily_data.append(df_day)
            
            # Se já coletamos dados suficientes para o período solicitado
            if len(all_daily_data) >= num_past_days:
                print(f"    Coletados dados de {len(all_daily_data)} dias. Parando a busca.")
                break

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 404:
                print(f"    Arquivo {filename} não encontrado (404). Isso é comum para dias muito recentes.")
            else:
                print(f"    Erro HTTP para {filename}: {http_err}")
        except requests.exceptions.RequestException as e: # Outros erros de request (timeout, DNS, etc.)
            print(f"    Erro de request para {filename}: {e}")
        except pd.errors.EmptyDataError: # Se o CSV estiver vazio
            print(f"    Arquivo {filename} está vazio.")
        except Exception as e: # Outros erros inesperados
            print(f"    Erro inesperado processando {filename}: {e}")
            print(traceback.format_exc()) # Imprime o traceback completo para depuração

    if not all_daily_data:
        st.warning(f"Não foi possível baixar nenhum dado de focos de incêndio dos últimos ~{num_past_days} dias do INPE.")
        return pd.DataFrame()

    df_total = pd.concat(all_daily_data, ignore_index=True)
    print(f"RPA_COLLECTOR: Total de {len(df_total)} focos brutos carregados dos arquivos diários combinados.")

    # --- Limpeza e Tratamento ---
    df_total.rename(columns=COL_RENAME_MAP_DIARIO, inplace=True, errors='ignore')

    # Normalizar e converter coluna de data/hora
    datetime_col_to_use = 'datahora_gmt_original' if 'datahora_gmt_original' in df_total.columns else None
    if not datetime_col_to_use: # Fallback para o nome original se a renomeação não ocorreu
        datetime_col_to_use = 'data_hora_gmt' if 'data_hora_gmt' in df_total.columns else None

    if datetime_col_to_use:
        df_total['datahora_gmt'] = pd.to_datetime(df_total[datetime_col_to_use], errors='coerce')
        df_total.dropna(subset=['datahora_gmt'], inplace=True)
        # Criar coluna 'data' (apenas data, sem hora) para agrupamentos e uso na UI
        # Esta data será baseada no GMT. Para precisão local, ajustes de fuso seriam necessários.
        df_total['data'] = df_total['datahora_gmt'].dt.normalize()
    else:
        st.error("Coluna de data/hora ('data_hora_gmt' ou 'datahora_gmt_original') crucial não encontrada após concatenação.")
        return pd.DataFrame()
        
    # Converter outras colunas para numérico
    cols_to_numeric = ['latitude', 'longitude', 'dias_sem_chuva', 'precipitacao', 'risco_fogo', 'frp']
    for col_num in cols_to_numeric:
        if col_num in df_total.columns:
            df_total[col_num] = pd.to_numeric(df_total[col_num], errors='coerce')

    # Normalizar e filtrar por estado
    final_df = pd.DataFrame() # Inicializa como DataFrame vazio
    if target_estado and 'estado' in df_total.columns:
        # Normaliza a coluna 'estado' no DataFrame para comparação robusta
        df_total['estado_normalizado'] = df_total['estado'].astype(str).apply(
            lambda x: unidecode_function(x).strip().upper() if pd.notnull(x) else ""
        )
        # Normaliza o 'target_estado' da mesma forma
        target_estado_normalizado = unidecode_function(target_estado).strip().upper()
        
        # print(f"  Valores únicos em 'estado_normalizado' (amostra): {df_total['estado_normalizado'].unique()[:10]}") # Debug
        # print(f"  Filtrando por '{target_estado_normalizado}'")

        df_estado_filtrado = df_total[df_total['estado_normalizado'] == target_estado_normalizado].copy()
        print(f"  Filtrado para o estado '{target_estado_normalizado}'. Shape: {df_estado_filtrado.shape}")
        
        if df_estado_filtrado.empty and not df_total.empty:
             st.info(f"Nenhum foco encontrado para o estado de '{target_estado}' nos dados baixados dos últimos {num_past_days} dias.")
        final_df = df_estado_filtrado
    elif 'estado' not in df_total.columns:
        print(f"  AVISO: Coluna 'estado' não encontrada. Retornando dados do Brasil se `target_estado` não for None e houver dados.")
        final_df = df_total # Ou pode optar por retornar DataFrame vazio se o estado for obrigatório
    else: # Se target_estado for None ou vazio
        print(f"  Nenhum filtro de estado aplicado (target_estado='{target_estado}'). Usando dados do Brasil.")
        final_df = df_total
    
    if final_df.empty:
        st.info(f"Nenhum foco de incêndio encontrado para '{target_estado if target_estado else 'Brasil'}' no período analisado.")
    else:
        # Ordenar por data/hora mais recente para exibição
        final_df.sort_values(by='datahora_gmt', ascending=False, inplace=True)
        
    return final_df

if __name__ == '__main__':
    print("Testando fetch_and_process_inpe_daily_data para SP, últimos 3 dias...")
    dados_sp = fetch_and_process_inpe_daily_data(num_past_days=3, target_estado="SP")
    if not dados_sp.empty:
        print("\nDados Recentes de Incêndio (SP - Diário):")
        print(dados_sp[['data', 'municipio', 'latitude', 'longitude', 'estado', 'estado_normalizado']].head())
        print(f"\nTotal de focos para SP nos últimos dias: {len(dados_sp)}")
        if 'municipio' in dados_sp.columns: print("\nFocos por município em SP:\n", dados_sp['municipio'].value_counts().head())
        if 'data' in dados_sp.columns: print("\nFocos por data em SP:\n", dados_sp['data'].value_counts().sort_index())
    else: print("Nenhum dado recente encontrado para SP no teste.")

    print("\nTestando para o Brasil inteiro, últimos 1 dia...")
    dados_br = fetch_and_process_inpe_daily_data(num_past_days=1, target_estado=None)
    if not dados_br.empty:
        print(f"\nTotal de focos para Brasil no último dia: {len(dados_br)}")
        print(dados_br[['data', 'estado', 'municipio']].head())