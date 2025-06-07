# ==============================================================================
# ETAPA 1: IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==============================================================================
import streamlit as st
import google.generativeai as genai
import re
import os
import traceback

from utils.RAG_utils import retrieve_web_context_for_rag, NLTK_PACKAGES_DOWNLOADED_KEY_V3

ROBOT_NAME = "VigIA"

# Tenta carregar a chave de API do Google de forma segura a partir dos segredos do Streamlit ou variáveis de ambiente.
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
except Exception:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Define um valor padrão caso a chave de API não seja encontrada.
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "SUA_CHAVE_DE_API_GOOGLE_AQUI_NAO_CONFIGURADA"

# Configura a biblioteca do Google e inicializa o modelo generativo se a chave de API for válida.
llm = None
if GOOGLE_API_KEY and GOOGLE_API_KEY != "SUA_CHAVE_DE_API_GOOGLE_AQUI_NAO_CONFIGURADA":
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        GENERATION_MODEL_NAME = "gemini-1.5-pro-latest"
        llm = genai.GenerativeModel(GENERATION_MODEL_NAME)
    except Exception as e:
        print(f"ERRO (rag_assistant_tab): Falha na inicialização do GenAI: {traceback.format_exc()}")
else:
    print(f"AVISO (rag_assistant_tab): {ROBOT_NAME} - GOOGLE_API_KEY não configurada.")


# ==============================================================================
# ETAPA 2: GERAÇÃO DA RESPOSTA COM RAG (Retrieval-Augmented Generation)
# ==============================================================================
def generate_rag_response_with_google_api_tab(user_profile, user_query, retrieved_contexts_list):
    """Gera uma resposta do assistente usando o contexto recuperado da web e um prompt estruturado."""
    # Retorna uma mensagem de modo offline se o modelo de linguagem não foi inicializado.
    if not llm:
        context_preview = retrieved_contexts_list[0][:70] if retrieved_contexts_list else "sem contexto disponível"
        return f"({ROBOT_NAME} - MODO OFFLINE: LLM não inicializado) Resposta para '{user_query}'. Contexto: '{context_preview}...'"

    # Concatena os múltiplos contextos recuperados em uma única string.
    context_str = "\n\n---\n\n".join(retrieved_contexts_list)
    
    # Monta o prompt detalhado para o modelo, definindo seu papel, regras e a fonte de informação.
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
    
    try:
        # Define os parâmetros de geração, como a temperatura para controlar a criatividade da resposta.
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        
        # Define as configurações de segurança para filtrar conteúdo potencialmente prejudicial.
        safety_settings_corrected = [
            {"category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
        
        # Envia o prompt para a API do Google e aguarda a geração da resposta.
        response = llm.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings_corrected
        )
        
        # Processa a resposta da API, tratando casos de sucesso, bloqueio ou erro.
        if response.parts:
            return response.text
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             block_reason_msg = response.prompt_feedback.block_reason
             return f"Desculpe, {ROBOT_NAME} não pôde responder devido a restrições de conteúdo (Razão: {block_reason_msg})."
        else:
            return f"Desculpe, {ROBOT_NAME} não conseguiu gerar uma resposta no momento (API não retornou texto útil)."

    except Exception as e:
        print(f"ERRO (rag_assistant_tab): Falha na chamada GenAI: {traceback.format_exc()}")
        return f"Desculpe, {ROBOT_NAME} teve um problema técnico ao gerar a resposta. Detalhe: {type(e).__name__}"


# ==============================================================================
# ETAPA 3: RENDERIZAÇÃO E LÓGICA DA INTERFACE DE CHAT
# ==============================================================================
def display_rag_chat_tab():
    """Renderiza a aba de chat do assistente virtual no Streamlit."""
    st.header(f"Assistente Virtual {ROBOT_NAME}")
    st.markdown(f"Olá! Sou {ROBOT_NAME}, seu assistente para orientações sobre **incêndios**.")

    # Realiza verificações prévias para garantir que os componentes essenciais estão funcionando.
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "SUA_CHAVE_DE_API_GOOGLE_AQUI_NAO_CONFIGURADA":
        st.error(f"{ROBOT_NAME} não pode operar: Chave Google API não configurada.")
        return
    if not llm:
         st.error(f"{ROBOT_NAME} não inicializado (modelo de linguagem). Verifique config da API.")
         return
    if not st.session_state.get(NLTK_PACKAGES_DOWNLOADED_KEY_V3, False):
        st.warning(f"{ROBOT_NAME} pode ter dificuldades com conteúdo web (NLTK). Veja console.")

    # Inicializa o estado da sessão para armazenar o histórico de mensagens e o perfil do usuário.
    session_prefix = f"{ROBOT_NAME}_chat_v5.1_"
    if f"{session_prefix}messages" not in st.session_state: st.session_state[f"{session_prefix}messages"] = []
    if f"{session_prefix}user_profile" not in st.session_state: st.session_state[f"{session_prefix}user_profile"] = None
    if f"{session_prefix}profile_prompted" not in st.session_state: st.session_state[f"{session_prefix}profile_prompted"] = False
    
    # Exibe todas as mensagens do histórico do chat na interface.
    for message in st.session_state[f"{session_prefix}messages"]:
        avatar_icon = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    current_user_profile = st.session_state[f"{session_prefix}user_profile"]

    # ==========================================================================
    # ETAPA 4: GERENCIAMENTO DO FLUXO DE CONVERSA (SELEÇÃO DE PERFIL E CHAT)
    # ==========================================================================
    
    # Se o perfil do usuário ainda não foi definido, solicita a seleção.
    if not current_user_profile:
        if not st.session_state[f"{session_prefix}profile_prompted"]:
            greeting = (f"Para começar, qual seu perfil?\n\n1. **Vítima**\n2. **Morador**\n3. **Familiar**\n\nDigite o número ou nome.")
            st.session_state[f"{session_prefix}messages"].append({"role": "assistant", "content": greeting})
            with st.chat_message("assistant", avatar="🤖"): st.markdown(greeting)
            st.session_state[f"{session_prefix}profile_prompted"] = True
        
        # Captura e processa a entrada do usuário para definir o perfil.
        profile_val = st.chat_input("Seu perfil:", key=f"{session_prefix}prof_in_v10")
        if profile_val:
            st.session_state[f"{session_prefix}messages"].append({"role": "user", "content": profile_val})
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
            st.rerun()
            
    # Se um perfil já foi definido, entra no fluxo de chat normal.
    elif current_user_profile:
        query_val = st.chat_input(f"Pergunte a {ROBOT_NAME} (Perfil: {current_user_profile}) ou digite 'mudar perfil'", key=f"{session_prefix}q_in_v10_{current_user_profile.lower()}")
        if query_val:
            st.session_state[f"{session_prefix}messages"].append({"role": "user", "content": query_val})
            
            norm_query = query_val.lower().strip()
            change_keywords = ["mudar perfil", "trocar perfil", "outro perfil", "mudar usuario", "trocar usuario", "resetar perfil"]
            
            # Verifica se o usuário deseja trocar de perfil.
            if any(keyword in norm_query for keyword in change_keywords):
                st.session_state[f"{session_prefix}user_profile"] = None
                st.session_state[f"{session_prefix}profile_prompted"] = False
                st.session_state[f"{session_prefix}messages"].append({"role": "assistant", "content": "Ok, seu perfil foi redefinido. Por favor, selecione novamente."})
                st.rerun()
            # Caso contrário, processa a pergunta do usuário.
            else:
                with st.spinner(f"{ROBOT_NAME} pesquisando..."):
                    retrieved_contexts = retrieve_web_context_for_rag(user_query=query_val)
                    assistant_response = generate_rag_response_with_google_api_tab(current_user_profile, query_val, retrieved_contexts)
                st.session_state[f"{session_prefix}messages"].append({"role": "assistant", "content": assistant_response})
                st.rerun()

    # ==========================================================================
    # ETAPA 5: AJUSTES FINAIS DA INTERFACE (AUTO-SCROLL)
    # ==========================================================================

    # Injeta um script JavaScript para rolar a tela automaticamente para a mensagem mais recente.
    if st.session_state.get(f"{session_prefix}messages"):
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
        st.components.v1.html(js_scroll_to_bottom, height=0, scrolling=False)

# Ponto de entrada para executar a aba de chat.
if __name__ == "__main__":
    display_rag_chat_tab()