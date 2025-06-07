# ==============================================================================
# ETAPA 1: IMPORTAÇÕES E CONSTANTES
# ==============================================================================
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import streamlit as st
import os
import traceback
from unidecode import unidecode as unidecode_function

# URL base para os arquivos CSV diários de focos de queimadas do INPE.
BASE_INPE_DIARIO_URL = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/diario/Brasil/"

# Mapeamento para renomear colunas para um padrão interno consistente.
COL_RENAME_MAP_DIARIO = {
    'lat': 'latitude',
    'lon': 'longitude',
    'data_hora_gmt': 'datahora_gmt_original',
    'numero_dias_sem_chuva': 'dias_sem_chuva',
}

# ==============================================================================
# ETAPA 2: FUNÇÃO DE COLETA E PROCESSAMENTO DE DADOS
# ==============================================================================
def fetch_and_process_inpe_daily_data(num_past_days=7, target_estado="SP"):
    """Busca, combina e processa os dados diários de focos de queimadas do INPE."""
    print(f"RPA_COLLECTOR: Iniciando busca de dados INPE para {num_past_days} dias (Alvo: {target_estado}).")
    all_daily_data = []
    
    # ETAPA 2.1: DOWNLOAD DOS DADOS DIÁRIOS
    # Itera sobre os dias anteriores para baixar os arquivos CSV correspondentes.
    for i in range(1, num_past_days + 2): 
        process_date = datetime.now() - timedelta(days=i)
        date_str_for_filename = process_date.strftime("%Y%m%d")
        filename = f"focos_diario_br_{date_str_for_filename}.csv"
        file_url = BASE_INPE_DIARIO_URL.rstrip('/') + '/' + filename
        
        print(f"  Tentando baixar: {file_url}")
        try:
            response = requests.get(file_url, timeout=20)
            response.raise_for_status()
            
            # Tenta decodificar o conteúdo com UTF-8 e fallback para latin1.
            try:
                csv_content_str = response.content.decode('utf-8')
                df_day = pd.read_csv(StringIO(csv_content_str), sep=',', encoding='utf-8')
            except UnicodeDecodeError:
                csv_content_str = response.content.decode('latin1')
                df_day = pd.read_csv(StringIO(csv_content_str), sep=',', encoding='latin1')
            except pd.errors.ParserError as pe:
                print(f"    Erro de parsing no Pandas para {filename}: {pe}. Pulando.")
                continue

            df_day['data_arquivo_original'] = process_date.date()
            all_daily_data.append(df_day)
            
            # Interrompe a busca se já coletou dados para o número de dias solicitado.
            if len(all_daily_data) >= num_past_days:
                print(f"    Coleta concluída para o período solicitado.")
                break

        # Trata possíveis erros durante o download e processamento do arquivo.
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 404:
                print(f"    Arquivo {filename} não encontrado (404).")
            else:
                print(f"    Erro HTTP para {filename}: {http_err}")
        except requests.exceptions.RequestException as e:
            print(f"    Erro de request para {filename}: {e}")
        except pd.errors.EmptyDataError:
            print(f"    Arquivo {filename} está vazio.")
        except Exception as e:
            print(f"    Erro inesperado processando {filename}: {e}\n{traceback.format_exc()}")

    # ETAPA 2.2: CONCATENAÇÃO E LIMPEZA
    # Combina todos os DataFrames diários em um único DataFrame.
    if not all_daily_data:
        st.warning(f"Não foi possível baixar nenhum dado de focos de incêndio dos últimos {num_past_days} dias do INPE.")
        return pd.DataFrame()

    df_total = pd.concat(all_daily_data, ignore_index=True)
    print(f"RPA_COLLECTOR: {len(df_total)} focos brutos carregados dos arquivos combinados.")

    df_total.rename(columns=COL_RENAME_MAP_DIARIO, inplace=True, errors='ignore')

    # ETAPA 2.3: TRATAMENTO DE TIPOS DE DADOS
    # Converte colunas de data e numéricas para os tipos corretos.
    datetime_col_to_use = 'datahora_gmt_original' if 'datahora_gmt_original' in df_total.columns else 'data_hora_gmt'
    if datetime_col_to_use in df_total.columns:
        df_total['datahora_gmt'] = pd.to_datetime(df_total[datetime_col_to_use], errors='coerce')
        df_total.dropna(subset=['datahora_gmt'], inplace=True)
        df_total['data'] = df_total['datahora_gmt'].dt.normalize()
    else:
        st.error("Coluna de data/hora crucial não encontrada após concatenação.")
        return pd.DataFrame()
        
    cols_to_numeric = ['latitude', 'longitude', 'dias_sem_chuva', 'precipitacao', 'risco_fogo', 'frp']
    for col_num in cols_to_numeric:
        if col_num in df_total.columns:
            df_total[col_num] = pd.to_numeric(df_total[col_num], errors='coerce')

    # ETAPA 2.4: FILTRAGEM E FINALIZAÇÃO
    # Filtra os dados pelo estado alvo, normalizando os nomes para uma comparação robusta.
    final_df = pd.DataFrame()
    if target_estado and 'estado' in df_total.columns:
        df_total['estado_normalizado'] = df_total['estado'].astype(str).apply(
            lambda x: unidecode_function(x).strip().upper() if pd.notnull(x) else ""
        )
        target_estado_normalizado = unidecode_function(target_estado).strip().upper()
        df_estado_filtrado = df_total[df_total['estado_normalizado'] == target_estado_normalizado].copy()
        print(f"  Filtrado para o estado '{target_estado_normalizado}'. Shape: {df_estado_filtrado.shape}")
        
        if df_estado_filtrado.empty and not df_total.empty:
             st.info(f"Nenhum foco encontrado para o estado de '{target_estado}' nos dados baixados.")
        final_df = df_estado_filtrado
    else:
        final_df = df_total
    
    if final_df.empty:
        st.info(f"Nenhum foco de incêndio encontrado para '{target_estado if target_estado else 'Brasil'}' no período.")
    else:
        final_df.sort_values(by='datahora_gmt', ascending=False, inplace=True)
        
    return final_df

# Bloco para execução de teste do script de forma independente.
if __name__ == '__main__':
    print("Testando fetch_and_process_inpe_daily_data para SP, últimos 3 dias...")
    dados_sp = fetch_and_process_inpe_daily_data(num_past_days=3, target_estado="SP")
    if not dados_sp.empty:
        print("\nDados Recentes de Incêndio (SP - Diário):")
        print(dados_sp[['data', 'municipio', 'latitude', 'longitude', 'estado']].head())
    else:
        print("Nenhum dado recente encontrado para SP no teste.")