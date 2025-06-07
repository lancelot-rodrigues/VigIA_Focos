# Plataforma VigIA Focos - Soluções IA para Incêndios (FIAP)

Bem-vindo à **VigIA Focos**! Esta plataforma unifica três soluções de Inteligência Artificial desenvolvidas para auxiliar na prevenção, monitoramento e orientação em situações de incêndios. O projeto foi concebido no contexto da Global Solution da FIAP, abrangendo conhecimentos das disciplinas de Front End & Mobile Development, Generative AI (com foco em RAG), e AI for RPA.

**Repositório GitHub:** [https://github.com/lancelot-rodrigues/VigIA_Focos](https://github.com/lancelot-rodrigues/VigIA_Focos)

**Link do site:** [https://vigia-focos.streamlit.app/](https://vigia-focos.streamlit.app/)

## Índice

1.  [Visão Geral do Projeto](#visão-geral-do-projeto)
2.  [Tecnologias Utilizadas](#tecnologias-utilizadas)
3.  [Instruções de Configuração e Execução](#instruções-de-configuração-e-execução)
    *   [Pré-requisitos](#pré-requisitos)
    *   [Configuração do Ambiente](#configuração-do-ambiente)
    *   [Executando a Aplicação](#executando-a-aplicação)
4.  [Funcionalidades por Módulo (Disciplina)](#funcionalidades-por-módulo-disciplina)
    *   [Módulo 1: Previsão de Risco de Incêndio (Front End & Mobile)](#módulo-1-previsão-de-risco-de-incêndio-front-end--mobile)
    *   [Módulo 2: Assistente Virtual VigIA (Generative AI)](#módulo-2-assistente-virtual-vigia-generative-ai)
    *   [Módulo 3: Monitoramento de Queimadas (RPA)](#módulo-3-monitoramento-de-queimadas-rpa)
5.  [Relatórios Detalhados](#relatórios-detalhados)
6.  [Autores](#autores)

---

## 1. Visão Geral do Projeto

A plataforma **VigIA Focos** é uma aplicação web interativa construída com Streamlit que integra:
*   Um modelo preditivo de Machine Learning para risco de incêndios.
*   Un assistente virtual (chatbot) chamado VigIA, baseado em RAG, para orientações.
*   Um painel de análise descritiva com dados históricos e monitoramento de dados recentes de queimadas.

O objetivo é fornecer uma ferramenta multifacetada para usuários interessados na temática de incêndios, desde a previsão de riscos até a obtenção de informações seguras e atualizadas. Os artefatos e código principal da aplicação Streamlit estão localizados dentro da pasta `streamlit_app/`.

---

## 2. Tecnologias Utilizadas

*   **Linguagem:** Python 3.11+
*   **Interface Web:** Streamlit
*   **Machine Learning (Previsão):** Pandas, NumPy, Scikit-learn, LightGBM, SHAP, Joblib, Category Encoders.
*   **Assistente Virtual (VigIA - RAG):**
    *   LLM: Google Gemini (via `google-generativeai`)
    *   Embeddings: Google Embedding Model (`models/embedding-001`)
    *   Busca Web: DuckDuckGo (`duckduckgo-search`)
    *   Extração de Conteúdo Web: Newspaper3k
    *   Chunking: LangChain (`RecursiveCharacterTextSplitter`)
    *   Busca Vetorial: FAISS (`faiss-cpu`)
*   **Monitoramento de Queimadas (RPA):** Pandas, Requests, BeautifulSoup4.
*   **Visualização de Dados:** Matplotlib, Streamlit charts.
*   **Outras:** NLTK, Unidecode.

---

## 3. Instruções de Configuração e Execução

### Pré-requisitos

*   Python 3.11 ou superior instalado.
*   (Opcional, mas recomendado) Um ambiente virtual Python (ex: venv, Conda).

### Configuração do Ambiente

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/lancelot-rodrigues/VigIA_Focos.git
    cd VigIA_Focos
    ```

2.  **Crie e Ative um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instale as Dependências:**
    A partir da pasta raiz do projeto (`VigIA_Focos`):
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as API Keys:**
    *   Navegue até a pasta `Incendio/`.
    *   Crie uma subpasta chamada `.streamlit` (se não existir).
    *   Dentro de `Incendio/.streamlit/`, crie um arquivo chamado `secrets.toml`.
    *   Adicione sua chave da API do Google GenAI:
        ```toml
        # Incendio/.streamlit/secrets.toml
        GOOGLE_API_KEY = "SUA_CHAVE_API_DO_GOOGLE_GEMINI_AQUI"
        ```
    *   **IMPORTANTE:** O arquivo `Incendio/.streamlit/secrets.toml` **NÃO DEVE SER COMMITADO** ao Git. Ele está incluído no `.gitignore`.

### Executando a Aplicação

1.  A partir da pasta raiz do projeto (`VigIA_Focos`), execute o comando:
    ```bash
    streamlit run streamlit.py 
    ```
    *(Se o seu arquivo principal dentro da pasta `Incendio/` tiver outro nome, ajuste o comando acima de acordo).*

2.  A aplicação abrirá automaticamente no seu navegador web.

---

## 4. Funcionalidades por Módulo (Disciplina)

A plataforma está dividida em módulos acessíveis pela barra lateral de navegação:

### Módulo 1: Previsão de Risco de Incêndio (Front End & Mobile)

*   **Aba:** "🔥 Previsão de Risco de Incêndio"
*   **Funcionalidade:** Permite ao usuário selecionar um município de São Paulo e uma data (dentro de 2022) para obter uma previsão da probabilidade de ocorrência de incêndio. Apresenta interpretabilidade do modelo usando SHAP.
*   **Relatório Detalhado:** Consulte `relatorios_e_documentacao/Relatorio_FrontEnd_Mobile.pdf`.

### Módulo 2: Assistente Virtual VigIA (Generative AI

*   **Aba:** "💬 Assistente VigIA"
*   **Funcionalidade:** Chatbot interativo que fornece orientações sobre incêndios, adaptadas ao perfil do usuário (Vítima, Morador, Familiar), utilizando busca na web em tempo real para enriquecer as respostas.
*   **Relatório Detalhado:** (A ser adicionado: `relatorios_e_documentacao/Relatorio_Generative_AI.pdf`).

### Módulo 3: Monitoramento de Queimadas (RPA)

*   **Aba:** "📊 Análise Descritiva" (Seção "Monitoramento de Queimadas Atuais em SP (Fonte: INPE)")
*   **Funcionalidade:** Automatiza a coleta, processamento e apresentação de dados recentes de focos de queimada do INPE para o estado de São Paulo, permitindo análises e alertas.
*   **Relatório Detalhado:** (A ser adicionado: `relatorios_e_documentacao/Relatorio_RPA_IA.pdf`).

---

## 5. Relatórios Detalhados

Para uma análise mais aprofundada de cada módulo, incluindo a metodologia, desenvolvimento, desafios e resultados, consulte os relatórios individuais localizados na pasta `relatorios_e_documentacao/` deste repositório.

---

## 6. Autores

*   Lancelot Chagas Rodrigues / 554707
*   Ana Carolina Martins da Silva / 555762
*   Kauan Alves Batista / 555082


---
