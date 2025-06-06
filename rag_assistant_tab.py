# rag_assistant_tab.py

import streamlit as st
import google.generativeai as genai # Import completo
import re
import os
import traceback

# Importar a função principal de recuperação do RAG_utils
from utils.RAG_utils import retrieve_web_context_for_rag, NLTK_PACKAGES_DOWNLOADED_KEY_V3

# --- Configurações ---
ROBOT_NAME = "VigIA"

# Carregamento seguro da API Key
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
except Exception:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "SUA_CHAVE_DE_API_GOOGLE_AQUI_NAO_CONFIGURADA"

# Configuração do genai e do modelo LLM
llm = None
if GOOGLE_API_KEY and GOOGLE_API_KEY != "SUA_CHAVE_DE_API_GOOGLE_AQUI_NAO_CONFIGURADA":
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        GENERATION_MODEL_NAME = "gemini-1.5-pro-latest"
        llm = genai.GenerativeModel(GENERATION_MODEL_NAME)
        # print(f"DEBUG (rag_assistant_tab): LLM ({GENERATION_MODEL_NAME}) inicializado.")
    except Exception as e:
        print(f"DEBUG (rag_assistant_tab): Erro na inicialização do GenAI: {traceback.format_exc()}")
else:
    print(f"AVISO (rag_assistant_tab): {ROBOT_NAME} - GOOGLE_API_KEY não configurada.")


def generate_rag_response_with_google_api_tab(user_profile, user_query, retrieved_contexts_list):
    if not llm:
        context_preview = retrieved_contexts_list[0][:70] if retrieved_contexts_list else "sem contexto disponível"
        return f"({ROBOT_NAME} - MODO OFFLINE: LLM não inicializado) Resposta para '{user_query}'. Contexto: '{context_preview}...'"

    context_str = "\n\n---\n\n".join(retrieved_contexts_list)
    
    # COLE SEU SYSTEM PROMPT REFINADO E COMPLETO AQUI (como na mensagem anterior)
    prompt = f"""SYSTEM:
Você é {ROBOT_NAME}, um assistente virtual altamente especializado, calmo, preciso e empático, dedicado a fornecer orientação durante desastres naturais, com FOCO EXCLUSIVO EM INCÊNDIOS.
Seu objetivo principal é ajudar usuários a se manterem seguros e informados sobre situações de incêndio.

PRINCÍPIOS FUNDAMENTAIS DA SUA RESPOSTA:
1.  **Prioridade Absoluta ao Contexto:** Baseie TODAS as suas respostas PRIORITARIAMENTE e FUNDAMENTALMENTE nas "INFORMAÇÕES DE CONTEXTO RECUPERADAS" fornecidas abaixo. NÃO use conhecimento externo ao que foi explicitamente fornecido aqui.
2.  **Honestidade sobre Limitações:** Se as informações recuperadas não responderem diretamente à pergunta, afirme isso de forma clara (ex: "Com base nas informações encontradas na web, não localizei detalhes específicos sobre X."). Não invente.
3.  **Segurança em Primeiro Lugar:** Sempre priorize a segurança do usuário nas suas orientações.
4.  **Sem Conselhos Profissionais Definitivos:** Não forneça conselhos médicos ou legais complexos. Em vez disso, direcione para profissionais qualificados ou para os serviços de emergência mencionados no contexto.
5.  **Proibição de Invenção:** É CRUCIAL que você não invente detalhes, nomes de locais, contatos, procedimentos ou nomes de organizações que não estejam explicitamente nas "INFORMAÇÕES DE CONTEXTO RECUPERADAS". Se não estiver no contexto, não existe para você.
6.  **Conselhos Gerais com Cautela:** Se, após afirmar que a informação específica não foi encontrada no contexto, for apropriado e seguro oferecer um conselho geral sobre o tema, deixe MUITO CLARO que este é um conselho geral e não baseado no contexto específico recuperado para a pergunta.

PERFIL DO USUÁRIO: {user_profile}

INFORMAÇÕES DE CONTEXTO RECUPERADAS (SUA ÚNICA FONTE DE CONHECIMENTO PARA ESTA RESPOSTA):
---
{context_str}
---

PERGUNTA DO USUÁRIO:
{user_query}

INSTRUÇÕES ADICIONAIS PARA O TOM E CONTEÚDO, CONFORME O PERFIL:
-   **Para Vítimas:** Respostas curtas, acionáveis, foco na segurança imediata. Use linguagem simples e imperativa. Priorize evacuação se no contexto. Conforte, mas seja direto sobre perigos.
-   **Para Moradores:** Informativo e preventivo. Explique riscos e preparação. Use alertas/sazonalidade do contexto.
-   **Para Familiares:** Empático. Forneça informações concretas do contexto (canais oficiais, abrigos). Não crie falsas esperanças.
-   **Para todos:** Se o contexto citar contatos de emergência (193 Bombeiros, 199 Defesa Civil), reforce-os se relevante. Lembre de seguir autoridades e procurar ajuda profissional.

RESPOSTA CUIDADOSA DE {ROBOT_NAME}:"""
    # FIM DO LOCAL PARA COLAR O PROMPT
    
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        
        # ***** SAFETY SETTINGS CORRIGIDAS *****
        safety_settings_corrected = [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            # A categoria HARM_CATEGORY_CIVIC_INTEGRITY é válida, mas vamos começar com as 4 principais.
            # Se o erro persistir, podemos tentar adicionar outras ou simplificar.
        ]
        
        response = llm.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings_corrected
        )
        
        if response.parts:
            return response.text
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             block_reason_msg = response.prompt_feedback.block_reason
             # safety_ratings_msg = response.prompt_feedback.safety_ratings if response.prompt_feedback.safety_ratings else "N/A" # Debug
             # print(f"AVISO (rag_assistant_tab): Resposta bloqueada. Razão: {block_reason_msg}, Safety Ratings: {safety_ratings_msg}")
             return f"Desculpe, {ROBOT_NAME} não pôde responder devido a restrições de conteúdo (Razão: {block_reason_msg})."
        else:
            # print(f"AVISO (rag_assistant_tab): Resposta da API vazia ou em formato inesperado. {response}")
            return f"Desculpe, {ROBOT_NAME} não conseguiu gerar uma resposta no momento (API não retornou texto útil)."

    except Exception as e:
        print(f"DEBUG (rag_assistant_tab): Erro na chamada GenAI na função generate_rag_response: {traceback.format_exc()}")
        return f"Desculpe, {ROBOT_NAME} teve um problema técnico ao gerar a resposta. Detalhe: {type(e).__name__}"

def display_rag_chat_tab():
    # ... (código de header, verificações de API KEY, LLM, NLTK como na última versão) ...
    st.header(f"Assistente Virtual {ROBOT_NAME}")
    st.markdown(f"Olá! Sou {ROBOT_NAME}, seu assistente para orientações sobre **incêndios**.")

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "SUA_CHAVE_DE_API_GOOGLE_AQUI_NAO_CONFIGURADA":
        st.error(f"{ROBOT_NAME} não pode operar: Chave Google API não configurada.")
        return
    if not llm:
         st.error(f"{ROBOT_NAME} não inicializado (modelo de linguagem). Verifique config da API.")
         return
    if not st.session_state.get(NLTK_PACKAGES_DOWNLOADED_KEY_V3, False): # Verifica flag NLTK
        st.warning(f"{ROBOT_NAME} pode ter dificuldades com conteúdo web (NLTK). Veja console.")

    session_prefix = f"{ROBOT_NAME}_chat_v5.1_" # Novo prefixo
    if f"{session_prefix}messages" not in st.session_state: st.session_state[f"{session_prefix}messages"] = []
    if f"{session_prefix}user_profile" not in st.session_state: st.session_state[f"{session_prefix}user_profile"] = None
    if f"{session_prefix}profile_prompted" not in st.session_state: st.session_state[f"{session_prefix}profile_prompted"] = False
    
    # ... (loop para exibir mensagens e lógica de seleção de perfil como na última versão) ...
    for message in st.session_state[f"{session_prefix}messages"]:
        avatar_icon = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    current_user_profile = st.session_state[f"{session_prefix}user_profile"]

    if not current_user_profile:
        if not st.session_state[f"{session_prefix}profile_prompted"]:
            greeting = (f"Para começar, qual seu perfil?\n\n1. **Vítima**\n2. **Morador**\n3. **Familiar**\n\nDigite o número ou nome.")
            st.session_state[f"{session_prefix}messages"].append({"role": "assistant", "content": greeting})
            with st.chat_message("assistant", avatar="🤖"): st.markdown(greeting)
            st.session_state[f"{session_prefix}profile_prompted"] = True
        
        profile_val = st.chat_input("Seu perfil:", key=f"{session_prefix}prof_in_v10") # Nova chave
        if profile_val:
            # Lógica de seleção de perfil (COLE SUA LÓGICA FUNCIONAL AQUI)
            # Exemplo da última vez:
            st.session_state[f"{session_prefix}messages"].append({"role": "user", "content": profile_val})
            # with st.chat_message("user", avatar="👤"): st.markdown(profile_val) # O loop acima já mostra
            raw_prof_in = profile_val.strip(); num_prof_chk="".join(filter(str.isdigit,raw_prof_in)); txt_prof_chk=raw_prof_in.lower(); sel_prof_txt=None
            if num_prof_chk == "1" or bool(re.search(r'\b(vitima|vítima)\b', txt_prof_chk)): st.session_state[f"{session_prefix}user_profile"], sel_prof_txt = "Vítima", "Vítima"
            elif num_prof_chk == "2" or bool(re.search(r'\bmorador(a)?\b', txt_prof_chk)): st.session_state[f"{session_prefix}user_profile"], sel_prof_txt = "Morador", "Morador"
            elif num_prof_chk == "3" or bool(re.search(r'\bfamiliar(es)?\b', txt_prof_chk)): st.session_state[f"{session_prefix}user_profile"], sel_prof_txt = "Familiar", "Familiar"
            else:
                resp_txt = f"Desculpe, não reconheci '{raw_prof_in}'. Use 1, 2, 3 ou Vítima, Morador, Familiar."
                st.session_state[f"{session_prefix}messages"].append({"role": "assistant", "content": resp_txt})
            if st.session_state[f"{session_prefix}user_profile"]:
                confirm_txt = f"Entendido. Perfil: **{sel_prof_txt}**. Como posso ajudar sobre incêndios?"
                st.session_state[f"{session_prefix}messages"].append({"role": "assistant", "content": confirm_txt})
            st.rerun() # Rerun para atualizar a UI e mostrar o input de pergunta ou erro
            
    elif current_user_profile: # Lógica de chat normal
        query_val = st.chat_input(f"Pergunte a {ROBOT_NAME} (Perfil: {current_user_profile}) ou digite 'mudar perfil'", key=f"{session_prefix}q_in_v10_{current_user_profile.lower()}")
        if query_val:
            st.session_state[f"{session_prefix}messages"].append({"role": "user", "content": query_val})
            # with st.chat_message("user", avatar="👤"): st.markdown(query_val) # O loop acima já mostra
            
            norm_query = query_val.lower().strip()
            change_keywords = ["mudar perfil", "trocar perfil", "outro perfil", "mudar usuario", "trocar usuario", "resetar perfil"]
            if any(keyword in norm_query for keyword in change_keywords):
                st.session_state[f"{session_prefix}user_profile"] = None
                st.session_state[f"{session_prefix}profile_prompted"] = False
                st.session_state[f"{session_prefix}messages"].append({"role": "assistant", "content": "Ok, seu perfil foi redefinido. Por favor, selecione novamente."})
                st.rerun()
            else:
                with st.spinner(f"{ROBOT_NAME} pesquisando..."):
                    retrieved_contexts = retrieve_web_context_for_rag(user_query=query_val)
                    assistant_response = generate_rag_response_with_google_api_tab(current_user_profile, query_val, retrieved_contexts)
                st.session_state[f"{session_prefix}messages"].append({"role": "assistant", "content": assistant_response})
                st.rerun() # Rerun para mostrar a resposta do assistente e o input de chat abaixo

    # Forçar scroll para baixo (JavaScript hack)
    if st.session_state.get(f"{session_prefix}messages"): # Verifica se há mensagens antes de tentar rolar
        js_scroll_to_bottom = """
        <script>
            setTimeout(function() {
                const chatContainer = window.parent.document.querySelector('section.main .block-container');
                if (chatContainer) {
                    chatContainer.scrollTo(0, chatContainer.scrollHeight);
                } else {
                    window.scrollTo(0, document.body.scrollHeight);
                }
            }, 150);
        </script>
        """
        # O seletor do chatContainer pode precisar de ajuste fino dependendo da estrutura exata do Streamlit DOM
        st.components.v1.html(js_scroll_to_bottom, height=0, scrolling=False)

if __name__ == "__main__":
    display_rag_chat_tab()