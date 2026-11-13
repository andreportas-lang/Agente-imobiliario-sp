import os
import json
import re
import streamlit as st
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. ESTRUTURAS DE DADOS (ESTADO)
# ==========================================
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    parametros: dict
    texto_busca_bruta: str         
    resultados_estruturados: str   
    analise_financeira: str        
    tentativas_busca: int
    erro_busca: bool
    erro_extracao: str # NOVO: Para capturar erros da API na tela

# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================
def instanciar_llm(provedor: str, modelo: str, api_key: str):
    if not api_key:
        raise ValueError(f"Chave de API do {provedor} ausente!")
    if provedor == "OpenAI":
        return ChatOpenAI(model=modelo, api_key=api_key, temperature=0)
    elif provedor == "Anthropic":
        # Removido o temperature=0 pois a API da Anthropic passou a rejeitá-lo como "input extra"
        return ChatAnthropic(model=modelo, api_key=api_key, max_tokens=1024)

def limpar_numero(valor, padrao=0):
    """Garante que qualquer lixo em texto (ex: 'R$ 1.200.000', '1.2Mi') vire um inteiro."""
    if isinstance(valor, (int, float)): return int(valor)
    if isinstance(valor, str):
        numeros_extraidos = re.sub(r'[^\d]', '', valor)
        return int(numeros_extraidos) if numeros_extraidos else padrao
    return padrao

# ==========================================
# 3. NÓS DO GRAFO
# ==========================================
def node_extrator(state: AgentState):
    """Extrai os parâmetros do texto do usuário com memória e tratamento anti-quebra."""
    state["erro_extracao"] = "" # Reseta os erros anteriores
    
    llm = instanciar_llm(
        st.session_state.provedor_fast, 
        st.session_state.fast_model, 
        st.session_state.api_key
    )
    
    ultima_msg = state["messages"][-1]["content"]
    
    # Puxa os dados antigos da memória
    params_atuais = state.get("parametros", {})
    if not params_atuais: 
        params_atuais = {"bairro": "Não definido", "valor_max": 0, "entrada": 0}
        
    sys_msg = SystemMessage(content=f"""
    Extraia dados da nova mensagem do usuário para atualizar os parâmetros de busca.
    Parâmetros atuais gravados na memória: {json.dumps(params_atuais, ensure_ascii=False)}
    
    Regras:
    - Se a nova mensagem informar um bairro, valor_max ou entrada novos, atualize.
    - Se a nova mensagem não tiver essas informações, MANTENHA os dados atuais.
    
    Retorne APENAS um JSON estrito no formato abaixo (use números inteiros puros para valores):
    {{
      "bairro": "nome do bairro",
      "valor_max": 1200000,
      "entrada": 300000
    }}
    """)
    
    try:
        resposta = llm.invoke([sys_msg, HumanMessage(content=ultima_msg)])
        texto_bruto = resposta.content.strip()
        
        # Puxa apenas o que tem cara de JSON, ignorando textos ao redor
        match = re.search(r'\{.*\}', texto_bruto, re.DOTALL)
        texto_json = match.group(0) if match else texto_bruto
            
        parametros_dict = json.loads(texto_json)
        
        # Salva o estado fazendo uma mescla com os dados antigos e limpando números
        state["parametros"] = {
            "bairro": parametros_dict.get("bairro", params_atuais.get("bairro", "Não definido")),
            "valor_max": limpar_numero(parametros_dict.get("valor_max"), params_atuais.get("valor_max", 0)),
            "entrada": limpar_numero(parametros_dict.get("entrada"), params_atuais.get("entrada", 0))
        }
        
    except Exception as e:
        # SE DER ERRO NA API (ex: falta de créditos), ELE MOSTRA AQUI
        state["erro_extracao"] = str(e)
        state["parametros"] = params_atuais
        
    state["tentativas_busca"] = state.get("tentativas_busca", 0) + 1
    return state

def node_buscador_web(state: AgentState):
    """Busca com DuckDuckGo evitando os gatilhos de bloqueio."""
    bairro = state["parametros"].get("bairro", "")
    valor_max = state["parametros"].get("valor_max", 0)
    
    if bairro == "Não definido" or valor_max == 0:
        state["erro_busca"] = True
        state["texto_busca_bruta"] = ""
        return state

    query = f"apartamento {bairro} São Paulo comprar site:vivareal.com.br"
    
    try:
        buscador = DuckDuckGoSearchResults(max_results=5)
        resultados = buscador.invoke(query)
        
        if not resultados or len(resultados) < 20:
            state["erro_busca"] = True
            state["texto_busca_bruta"] = ""
        else:
            state["texto_busca_bruta"] = resultados
            state["erro_busca"] = False
            
    except Exception as e:
        state["erro_busca"] = True
        state["texto_busca_bruta"] = ""
        
    return state

def node_validador(state: AgentState):
    return state

def node_analista(state: AgentState):
    """Lê a pesquisa web e emite o laudo."""
    llm = instanciar_llm(
        st.session_state.provedor_reasoning, 
        st.session_state.reasoning_model, 
        st.session_state.api_key
    )
    
    sys_prompt = SystemMessage(content="""
    Você é um consultor imobiliário. Extraia imóveis do texto bruto e crie uma TABELA MARKDOWN.
    Abaixo, faça uma ANÁLISE FINANCEIRA. Separe a Tabela da Análise pela string exata "---SEPARADOR---".
    """)
    user_prompt = HumanMessage(content=f"Dados: {state['parametros']}\nTexto Web: {state['texto_busca_bruta']}")
    
    resposta = llm.invoke([sys_prompt, user_prompt]).content
    partes = resposta.split("---SEPARADOR---")
    
    state["resultados_estruturados"] = partes[0].strip() if len(partes) > 0 else "Falha ao gerar tabela."
    state["analise_financeira"] = partes[1].strip() if len(partes) > 1 else resposta
    state["messages"].append({"role": "assistant", "content": "Busca concluída! Os resultados estão no painel ao lado."})
    
    return state

# ==========================================
# 4. ORQUESTRAÇÃO DO GRAFO
# ==========================================
def roteador_correcao(state: AgentState) -> str:
    if state.get("erro_extracao"):
        state["messages"].append({"role": "assistant", "content": f"🚨 Erro na API do LLM: {state['erro_extracao']}"})
        return END
    if state["erro_busca"]:
        state["messages"].append({"role": "assistant", "content": "Não encontrei resultados ou faltam dados (Bairro e Valor). Pode detalhar?"})
        return END
    return "node_analista"

builder = StateGraph(AgentState)
builder.add_node("node_extrator", node_extrator)
builder.add_node("node_buscador_web", node_buscador_web)
builder.add_node("node_validador", node_validador)
builder.add_node("node_analista", node_analista)

builder.add_edge(START, "node_extrator")
builder.add_edge("node_extrator", "node_buscador_web")
builder.add_edge("node_buscador_web", "node_validador")
builder.add_conditional_edges("node_validador", roteador_correcao)
builder.add_edge("node_analista", END)

graph = builder.compile()

# ==========================================
# 5. FRONT-END (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Consultor Imobiliário Inteligente", layout="wide")

if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [], "parametros": {}, "texto_busca_bruta": "", 
        "resultados_estruturados": "", "analise_financeira": "", 
        "tentativas_busca": 0, "erro_busca": False, "erro_extracao": ""
    }
    
# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuração de IA")
    
    provedor = st.radio("Escolha o Provedor LLM:", ["OpenAI", "Anthropic"])
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    if provedor == "OpenAI":
        st.session_state.api_key = openai_key
        st.session_state.provedor_fast = "OpenAI"
        st.session_state.fast_model = "gpt-4o-mini"
        st.session_state.provedor_reasoning = "OpenAI"
        st.session_state.reasoning_model = "gpt-4o"
    else:
        st.session_state.api_key = anthropic_key
        st.session_state.provedor_fast = "Anthropic"
        st.session_state.fast_model = "claude-3-5-haiku-20241022"
        st.session_state.provedor_reasoning = "Anthropic"
        st.session_state.reasoning_model = "claude-3-5-sonnet-20241022"
        
    st.markdown("---")
    st.subheader("🛠️ PAINEL DE DEPURAÇÃO")
    st.info("Veja aqui se o robô está salvando corretamente na memória em tempo real:")
    st.json(st.session_state.graph_state.get("parametros", {}))
    
    if st.session_state.graph_state.get("erro_extracao"):
        st.error(f"ERRO DA API:\n{st.session_state.graph_state['erro_extracao']}")
    if st.session_state.graph_state.get("erro_busca"):
        st.warning("A busca web falhou ou não encontrou o bairro/valor.")

# --- LAYOUT PRINCIPAL ---
col_chat, col_results = st.columns([1, 1.2], gap="large")

with col_chat:
    st.subheader("Interação")
    chat_container = st.container(height=600)
    
    with chat_container:
        for msg in st.session_state.graph_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ex: Procuro apê em Moema até R$ 1.2Mi"):
        if not st.session_state.api_key:
            st.error("Chave de API ausente.")
            st.stop()
            
        st.session_state.graph_state["messages"].append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Processando..."):
                    novo_estado = graph.invoke(st.session_state.graph_state)
                    st.session_state.graph_state = novo_estado
                    st.markdown(novo_estado["messages"][-1]["content"])
                    st.rerun()

with col_results:
    st.subheader("Painel de Análise Estruturada")
    state = st.session_state.graph_state
    
    if state["resultados_estruturados"]:
        st.markdown("#### 📋 Oportunidades")
        st.markdown(state["resultados_estruturados"])
        st.markdown("#### 💰 Laudo Financeiro")
        st.info(state["analise_financeira"])
    else:
        st.markdown("<div style='color: gray; margin-top: 100px; text-align: center;'>Aguardando parâmetros válidos no chat...</div>", unsafe_allow_html=True)
