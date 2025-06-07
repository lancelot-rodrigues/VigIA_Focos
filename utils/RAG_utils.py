# ==============================================================================
# ETAPA 1: IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==============================================================================
import streamlit as st
import google.generativeai as genai
import numpy as np
import os
import time
import nltk
import traceback

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from duckduckgo_search import DDGS
from newspaper import Article, ArticleException
import faiss

# Define uma chave para o estado da sessão para garantir que os pacotes NLTK sejam baixados apenas uma vez.
NLTK_PACKAGES_DOWNLOADED_KEY_V3 = "nltk_packages_downloaded_for_rag_v3"
if NLTK_PACKAGES_DOWNLOADED_KEY_V3 not in st.session_state:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        st.session_state[NLTK_PACKAGES_DOWNLOADED_KEY_V3] = True
        print("RAG_UTILS: Pacotes NLTK ('punkt', 'wordnet') prontos.")
    except Exception as e:
        st.session_state[NLTK_PACKAGES_DOWNLOADED_KEY_V3] = False
        print(f"RAG_UTILS: ERRO no download de pacotes NLTK: {e}")

# Define o nome do modelo de embedding a ser usado.
EMBEDDING_MODEL_RAG_DIRECT = "models/embedding-001"

# ==============================================================================
# ETAPA 2: FUNÇÕES DO PIPELINE DE RAG (Recuperação, Processamento, Busca)
# ==============================================================================

def get_google_embedding_direct(text_to_embed: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float] | None:
    """Gera um vetor de embedding para um texto usando a API do Google GenAI."""
    try:
        # Ignora textos muito curtos ou inválidos para evitar erros na API.
        if not text_to_embed or not isinstance(text_to_embed, str) or len(text_to_embed.strip()) < 3:
            return None
        # Chama a API para converter o texto em um vetor numérico (embedding).
        result = genai.embed_content(model=EMBEDDING_MODEL_RAG_DIRECT,
                                     content=text_to_embed,
                                     task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"ERRO (RAG_utils): Falha ao gerar embedding para texto (task: {task_type}): {e}")
        return None

def discover_urls_ddg(query: str, max_results: int = 5, region: str = 'br-pt') -> list[str]:
    """Utiliza a busca do DuckDuckGo para encontrar URLs relevantes para a consulta do usuário."""
    print(f"RAG_UTILS: Buscando URLs via DDG para: '{query}'")
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region=region, safesearch='moderate', max_results=max_results)
            if results: urls = [r.get('href') for r in results if r.get('href')]
        print(f"  DDG encontrou {len(urls)} URLs.")
    except Exception as e:
        print(f"  ERRO na busca DDG para '{query}': {e}")
    return list(set(urls))

def extract_content_from_urls_newspaper(urls: list[str], timeout: int = 10) -> list[Document]:
    """Extrai o conteúdo principal de uma lista de URLs usando a biblioteca Newspaper3k."""
    print(f"RAG_UTILS: Extraindo conteúdo de {len(urls)} URLs com Newspaper3k...")
    langchain_docs = []
    if not st.session_state.get(NLTK_PACKAGES_DOWNLOADED_KEY_V3, False):
        print("AVISO (RAG_utils): Newspaper3k pode falhar (NLTK não pronto).")

    # Itera sobre as URLs, baixa e processa o texto de cada artigo.
    for url in set(urls):
        if not url: continue
        try:
            article = Article(url, request_timeout=timeout, fetch_images=False, language='pt')
            article.download()
            time.sleep(0.25)
            article.parse()
            # Adiciona o documento se ele contiver texto substancial.
            if article.text and len(article.text.strip()) > 250:
                langchain_docs.append(Document(page_content=article.text,
                                            metadata={"source": url, "title": article.title or "N/A"}))
        except Exception as e:
            print(f"  ERRO no Newspaper3k em {url}: {type(e).__name__}")
    print(f"  Newspaper3k extraiu {len(langchain_docs)} documentos válidos.")
    return langchain_docs

def split_langchain_documents(langchain_docs: list[Document], chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Document]:
    """Divide documentos longos em pedaços menores (chunks) para processamento."""
    if not langchain_docs: return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=len, add_start_index=True,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = splitter.split_documents(langchain_docs)
    print(f"  Documentos divididos em {len(chunks)} chunks textuais.")
    return chunks

# ==============================================================================
# ETAPA 3: FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO DA RECUPERAÇÃO DE CONTEXTO
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False, max_entries=15)
def retrieve_web_context_for_rag(user_query: str, num_search_results: int = 4, num_top_chunks_to_retrieve: int = 4) -> list[str]:
    """Orquestra o pipeline completo de RAG: busca, extração, divisão, embedding e recuperação de similaridade."""
    print(f"\nRAG_UTILS: Iniciando recuperação web para query: '{user_query}'")
    
    # Executa as etapas sequenciais do pipeline de recuperação.
    discovered_urls = discover_urls_ddg(user_query, max_results=num_search_results)
    if not discovered_urls:
        return ["Não encontrei fontes relevantes na web para esta pergunta específica neste momento."]

    extracted_docs_lc = extract_content_from_urls_newspaper(discovered_urls)
    if not extracted_docs_lc:
        return ["Não consegui extrair informações utilizáveis das fontes encontradas para esta pergunta."]

    chunked_docs_lc = split_langchain_documents(extracted_docs_lc)
    if not chunked_docs_lc:
        return ["O conteúdo encontrado não pôde ser dividido em partes para análise."]

    # Gera embeddings para cada chunk de texto extraído.
    print(f"  Gerando embeddings para {len(chunked_docs_lc)} chunks de documentos...")
    chunk_embeddings_list = []
    valid_chunks_for_faiss = []
    for doc_chunk in chunked_docs_lc:
        embedding = get_google_embedding_direct(doc_chunk.page_content, task_type="RETRIEVAL_DOCUMENT")
        if embedding:
            chunk_embeddings_list.append(embedding)
            valid_chunks_for_faiss.append(doc_chunk)

    if not chunk_embeddings_list:
        return ["Não foi possível processar o conteúdo da web para busca (falha nos embeddings)."]

    # Constrói um índice FAISS em memória para busca rápida de vetores.
    embeddings_array = np.array(chunk_embeddings_list).astype(np.float32)
    faiss.normalize_L2(embeddings_array) # Normaliza os vetores para busca de similaridade de cosseno.
    
    print(f"  Construindo índice FAISS (IP) com {embeddings_array.shape[0]} vetores...")
    try:
        dimension = embeddings_array.shape[1]
        index_in_memory = faiss.IndexFlatIP(dimension)
        index_in_memory.add(embeddings_array)
        print(f"  Índice FAISS (IP) criado com {index_in_memory.ntotal} vetores.")
    except Exception as e:
        print(f"  ERRO CRÍTICO ao construir índice FAISS: {e}\n{traceback.format_exc()}")
        return ["Ocorreu um problema técnico ao indexar as informações da web."]

    # Gera o embedding para a consulta do usuário.
    print(f"  Gerando embedding para a query do usuário: '{user_query}'")
    query_embedding_vector = get_google_embedding_direct(user_query, task_type="RETRIEVAL_QUERY")
    if not query_embedding_vector:
        return ["Não foi possível processar sua pergunta para busca (falha no embedding da query)."]

    query_embedding_np = np.array(query_embedding_vector).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_embedding_np) # Normaliza o vetor da query.

    # Realiza a busca no índice FAISS para encontrar os chunks mais relevantes.
    print(f"  Buscando no índice FAISS (IP) por chunks similares (k={num_top_chunks_to_retrieve})...")
    try:
        distances, indices = index_in_memory.search(query_embedding_np, num_top_chunks_to_retrieve)
        
        relevant_texts = []
        for L_actual_idx in indices[0]:
            if 0 <= L_actual_idx < len(valid_chunks_for_faiss):
                retrieved_doc = valid_chunks_for_faiss[L_actual_idx]
                relevant_texts.append(retrieved_doc.page_content)
            
        if not relevant_texts:
            return ["Encontrei algumas informações na web, mas nada que parecesse altamente relevante para sua pergunta específica após análise."]
        
        print(f"RAG_UTILS: Recuperados {len(relevant_texts)} chunks relevantes via FAISS (IP).")
        return relevant_texts

    except Exception as e:
        print(f"  ERRO durante a busca FAISS (IP): {e}\n{traceback.format_exc()}")
        return ["Ocorreu um problema ao buscar informações detalhadas na base de conhecimento da web."]