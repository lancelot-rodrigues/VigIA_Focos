# utils/RAG_utils.py

import streamlit as st
import google.generativeai as genai
import numpy as np
import os
import time
import nltk
import traceback
# import json # Não usado para salvar/carregar chunks estáticos neste fluxo

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from duckduckgo_search import DDGS
from newspaper import Article, ArticleException
import faiss # pip install faiss-cpu

# --- Configurações Iniciais e Download NLTK ---
NLTK_PACKAGES_DOWNLOADED_KEY_V3 = "nltk_packages_downloaded_for_rag_v3"
if NLTK_PACKAGES_DOWNLOADED_KEY_V3 not in st.session_state:
    try:
        print("RAG_UTILS: Verificando/Baixando NLTK ('punkt', 'wordnet')...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        st.session_state[NLTK_PACKAGES_DOWNLOADED_KEY_V3] = True
        print("RAG_UTILS: NLTK pronto.")
    except Exception as e:
        st.session_state[NLTK_PACKAGES_DOWNLOADED_KEY_V3] = False
        print(f"RAG_UTILS: ERRO NLTK: {e}")

EMBEDDING_MODEL_RAG_DIRECT = "models/embedding-001" # Modelo do Google para embeddings

# --- FUNÇÕES DO PIPELINE RAG DINÂMICO ---

def get_google_embedding_direct(text_to_embed: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float] | None:
    """Gera embedding para um texto usando genai.embed_content diretamente."""
    try:
        if not text_to_embed or not isinstance(text_to_embed, str) or len(text_to_embed.strip()) < 3:
            # print(f"DEBUG (RAG_utils): Texto inválido/curto para embedding: '{str(text_to_embed)[:30]}'")
            return None
        # A configuração da API genai.configure() é feita no script principal da aba (rag_assistant_tab.py)
        result = genai.embed_content(model=EMBEDDING_MODEL_RAG_DIRECT,
                                     content=text_to_embed,
                                     task_type=task_type) # Importante: RETRIEVAL_DOCUMENT para docs, RETRIEVAL_QUERY para queries
        return result['embedding']
    except Exception as e:
        print(f"ERRO (RAG_utils): Ao gerar embedding para '{str(text_to_embed)[:30]}...' (task: {task_type}): {e}")
        return None

def discover_urls_ddg(query: str, max_results: int = 5, region: str = 'br-pt') -> list[str]:
    """Usa DuckDuckGo Search para encontrar URLs relevantes."""
    print(f"RAG_UTILS: DDG Buscando URLs para: '{query}' (max: {max_results}, region: {region})")
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region=region, safesearch='moderate', max_results=max_results)
            if results: urls = [r.get('href') for r in results if r.get('href')]
        print(f"  DDG encontrou {len(urls)} URLs.")
    except Exception as e: print(f"  ERRO na busca DDG para '{query}': {e}")
    return list(set(urls))

def extract_content_from_urls_newspaper(urls: list[str], timeout: int = 10) -> list[Document]:
    """Extrai texto principal de URLs usando Newspaper3k e retorna como Documentos LangChain."""
    print(f"RAG_UTILS: Newspaper3k extraindo de {len(urls)} URLs...")
    langchain_docs = []
    if not st.session_state.get(NLTK_PACKAGES_DOWNLOADED_KEY_V3, False):
        print("AVISO (RAG_utils): Newspaper3k pode falhar (NLTK não pronto).")

    for url in set(urls): # Evita re-processar a mesma URL
        if not url: continue
        try:
            article = Article(url, request_timeout=timeout, fetch_images=False, language='pt') # Assume português
            article.download()
            time.sleep(0.25) # Cortesia para os servidores
            article.parse()
            if article.text and len(article.text.strip()) > 250: # Considerar texto mínimo
                langchain_docs.append(Document(page_content=article.text,
                                            metadata={"source": url, "title": article.title or "N/A"}))
        except Exception as e: print(f"  ERRO Newspaper3k em {url}: {type(e).__name__}") # Mostra apenas o tipo do erro para ser mais conciso
    print(f"  Newspaper3k extraiu {len(langchain_docs)} documentos válidos.")
    return langchain_docs

def split_langchain_documents(langchain_docs: list[Document], chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Document]:
    """Divide uma lista de Documentos LangChain em chunks menores."""
    if not langchain_docs: return []
    # print(f"RAG_UTILS: Chunking {len(langchain_docs)} docs (size:{chunk_size}, overlap:{chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=len, add_start_index=True,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""] # Tenta quebras semânticas primeiro
    )
    chunks = splitter.split_documents(langchain_docs)
    print(f"  Documentos divididos em {len(chunks)} chunks textuais.")
    return chunks

@st.cache_data(ttl=1800, show_spinner=False, max_entries=15) # Cacheia por 30 min, para até 15 queries diferentes
def retrieve_web_context_for_rag(user_query: str, num_search_results: int = 4, num_top_chunks_to_retrieve: int = 4) -> list[str]:
    """
    Orquestra a busca na web, extração, chunking, embedding e RECUPERAÇÃO FAISS (com IndexFlatIP).
    """
    print(f"\nRAG_UTILS: Iniciando recuperação web para query: '{user_query}'")
    
    discovered_urls = discover_urls_ddg(user_query, max_results=num_search_results)
    if not discovered_urls:
        return ["Não encontrei fontes relevantes na web para esta pergunta específica neste momento."]

    extracted_docs_lc = extract_content_from_urls_newspaper(discovered_urls)
    if not extracted_docs_lc:
        return ["Não consegui extrair informações utilizáveis das fontes encontradas para esta pergunta."]

    chunked_docs_lc = split_langchain_documents(extracted_docs_lc)
    if not chunked_docs_lc:
        return ["O conteúdo encontrado não pôde ser dividido em partes para análise."]

    print(f"  Gerando embeddings para {len(chunked_docs_lc)} chunks da web (documentos)...")
    chunk_embeddings_list = []
    valid_chunks_for_faiss = [] # Para manter os Documentos LangChain correspondentes aos embeddings válidos
    for i, doc_chunk in enumerate(chunked_docs_lc):
        embedding = get_google_embedding_direct(doc_chunk.page_content, task_type="RETRIEVAL_DOCUMENT")
        if embedding:
            chunk_embeddings_list.append(embedding)
            valid_chunks_for_faiss.append(doc_chunk)
        # if (i + 1) % 10 == 0 and i > 0: time.sleep(0.3) # Delay menor

    if not chunk_embeddings_list:
        print("  Nenhum embedding gerado para os chunks da web.")
        return ["Não foi possível processar o conteúdo da web para busca (falha nos embeddings)."]

    embeddings_array = np.array(chunk_embeddings_list).astype(np.float32)
    
    # Normalizar embeddings para IndexFlatIP (similaridade de cosseno)
    faiss.normalize_L2(embeddings_array)
    
    print(f"  Construindo índice FAISS (IP) em memória com {embeddings_array.shape[0]} vetores (normalizados)...")
    try:
        dimension = embeddings_array.shape[1]
        index_in_memory = faiss.IndexFlatIP(dimension) # Usando Produto Interno (IP) para similaridade de cosseno
        index_in_memory.add(embeddings_array) # Adiciona os embeddings normalizados
        print(f"  Índice FAISS (IP) em memória criado. Total de vetores: {index_in_memory.ntotal}")
    except Exception as e:
        print(f"  ERRO CRÍTICO ao construir índice FAISS: {e}\n{traceback.format_exc()}")
        return ["Ocorreu um problema técnico ao indexar as informações da web."]

    print(f"  Gerando embedding para a query do usuário: '{user_query}'")
    query_embedding_vector = get_google_embedding_direct(user_query, task_type="RETRIEVAL_QUERY")
    if not query_embedding_vector:
        return ["Não foi possível processar sua pergunta para busca (falha no embedding da query)."]

    query_embedding_np = np.array(query_embedding_vector).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_embedding_np) # Normalizar o embedding da query também

    print(f"  Buscando no índice FAISS (IP) (k={num_top_chunks_to_retrieve})...")
    try:
        # Com IndexFlatIP, D são os produtos internos (maior é melhor).
        # O método .search ainda os retorna de uma forma que "menor distância" é melhor (ex: 1 - cosseno ou -cosseno).
        # Mas FAISS lida com isso internamente para retornar os vizinhos mais próximos no espaço do cosseno.
        distances, indices = index_in_memory.search(query_embedding_np, num_top_chunks_to_retrieve)
        
        relevant_texts = []
        # print(f"  Índices FAISS (IP) recuperados: {indices[0]}") # Debug
        # print(f"  Distâncias/Similaridades FAISS (IP) recuperadas: {distances[0]}") # Debug

        for i_loop_idx, L_actual_idx in enumerate(indices[0]):
            if 0 <= L_actual_idx < len(valid_chunks_for_faiss): # L_actual_idx é o índice em valid_chunks_for_faiss
                retrieved_doc = valid_chunks_for_faiss[L_actual_idx]
                text_content = retrieved_doc.page_content
                # source_url = retrieved_doc.metadata.get("source", "Fonte desconhecida") # Opcional para adicionar ao contexto
                # title = retrieved_doc.metadata.get("title", "Título desconhecido")
                relevant_texts.append(text_content)
                # print(f"    Chunk Relevante {i_loop_idx+1} (Dist/Sim: {distances[0][i_loop_idx]:.4f}): {text_content[:100]}...")
            # else: print(f"  Aviso: Índice FAISS {L_actual_idx} fora do range.")
            
        if not relevant_texts:
            print("  Nenhum chunk relevante encontrado após busca FAISS (IP).")
            return ["Encontrei algumas informações na web, mas nada que parecesse altamente relevante para sua pergunta específica após análise."]
        
        print(f"RAG_UTILS: Recuperados {len(relevant_texts)} chunks relevantes via FAISS (IP).")
        return relevant_texts

    except Exception as e:
        print(f"  ERRO durante a busca FAISS (IP): {e}\n{traceback.format_exc()}")
        return ["Ocorreu um problema ao buscar informações detalhadas na base de conhecimento da web."]