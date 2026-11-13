import os
import json
import streamlit as st
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# Integrações LangChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. ESTRUTURAS DE DADOS (PYDANTIC E ESTADO)
# ==========================================
class ParametrosBusca(BaseModel):
    """Extrai os parâmetros de busca de imóveis fornecidos pelo usuário na conversa."""
    bairro: str = Field(description="Bairro de São Paulo desejado. Retorne 'Não definido' se não informado.")
    valor_max: int = Field(description="Valor máximo do imóvel em reais. Use 0 se não informado.")
    entrada: int = Field(description="Valor disponível para entrada em reais. Use 0 se não informado.")

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    parametros: dict
    texto_busca_bruta: str         # Resultado bruto do DuckDuckGo
    resultados_estruturados: str   # Tabela em Markdown gerada pelo LLM pesado
    analise_financeira: str        # Laudo gerado pelo LLM pesado
    tentativas_busca: int
    erro_busca: bool

# ==========================================
# 2. FUNÇÕES AUXILIARES DE MODELOS
# ==========================================
def instanciar_llm(provedor: str, modelo: str, api_key: str):
    """Instancia o modelo correto (OpenAI ou Anthropic) via LangChain."""
    if not api_key:
        raise ValueError(f"Chave de API do {provedor} não configurada ou ausente no secrets.toml!")
    
    if provedor == "OpenAI":
        return ChatOpenAI(model=modelo, api_key=api_key, temperature=0)
    elif provedor == "Anthropic":
        # A Anthropic exige o limite de tokens explícito para evitar Bad Request
        return ChatAnthropic(model=modelo, api_key=api_key, temperature=0, max_tokens=1024)

# ==========================================
# 3. NÓS DO GRAFO
# ==========================================
def node_extrator(state: AgentState):
    """Usa um modelo rápido para extrair a intenção e os parâmetros da mensagem do usuário retornando JSON puro."""
    llm = instanciar_llm(
        st.session_state.provedor_fast, 
        st.session_state.fast_model, 
        st.session_state.api_key
    )
    
    ultima_msg = state["messages"][-1]["content"]
    
    # Prompt explícito pedindo JSON. Isso funciona em qualquer modelo perfeitamente.
    sys_msg = SystemMessage(content="""
    Extraia as informações do pedido do usuário e retorne APENAS um JSON válido.
    Não adicione texto antes ou depois do JSON. Não use blocos de código (```json).
    Use a seguinte estrutura:
    {
      "bairro": "Nome do bairro de SP. Se não informado, retorne 'Não definido'",
      "valor_max": inteiro representando o valor máximo em reais (ex: 900000). Se não informado, retorne 0,
      "entrada": inteiro representando o valor de entrada em reais (ex: 150000). Se não informado, retorne 0
    }
    """)
    
    try:
        # Chama o LLM pedindo a resposta em texto bruto
        resposta = llm.invoke([sys_msg, HumanMessage(content=ultima_msg)])
        texto_limpo = resposta.content.strip().replace("```json", "").replace("```", "")
        
        # Converte a string JSON para um dicionário Python
        parametros_dict = json.loads(texto_limpo)
        
        # Opcional: passa pelo Pydantic para validar os tipos
        validado = ParametrosBusca(**parametros_dict)
        state["parametros"] = validado.model_dump()
        
    except Exception as e:
        # Fallback de segurança se o LLM falhar em gerar o JSON ou o usuário falar algo sem sentido
        print(f"Falha ao extrair JSON: {e}")
        state["parametros"] = {"bairro": "Não definido", "valor_max": 0, "entrada": 0}
        
    state["tentativas_busca"] = state.get("tentativas_busca", 0) + 1
    return state

def node_buscador_web(state: AgentState):
    """Faz uma busca real na web usando DuckDuckGo em vez de API de imobiliária."""
    bairro = state["parametros"].get("bairro", "")
    valor_max = state["parametros"].get("valor_max", 0)
    
    if bairro == "Não definido" or valor_max == 0:
        state["erro_busca"] = True
        state["texto_busca_bruta"] = ""
        return state

    # Query focada em pegar fragmentos de sites imobiliários conhecidos
    query = f"apartamento {bairro} São Paulo comprar até R$ {valor_max} site:vivareal.com.br OR site:quintoandar.com.br"
    
    buscador = DuckDuckGoSearchResults(max_results=5)
    resultados = buscador.invoke(query)
    
    state["texto_busca_bruta"] = resultados
    state["erro_busca"] = len(resultados) < 20 # Considera erro se não retornar texto suficiente
    return state

def node_validador(state: AgentState):
    """Nó de passagem para avaliação condicional."""
    return state

def node_analista(state: AgentState):
    """Usa o modelo de raciocínio pesado para ler a web, extrair imóveis e gerar o laudo."""
    llm = instanciar_llm(
        st.session_state.provedor_reasoning, 
        st.session_state.reasoning_model, 
        st.session_state.api_key
    )
    
    sys_prompt = SystemMessage(content="""
    Você é um consultor imobiliário de São Paulo.
    Analise o texto bruto de uma busca na web e extraia possíveis anúncios reais que se encaixem no perfil do usuário.
    
    Regras estritas de formatação:
    1. Crie uma TABELA MARKDOWN com os imóveis (Bairro, m², Preço, Condição estimada baseada na descrição).
    2. Abaixo da tabela, faça uma ANÁLISE FINANCEIRA comparando À Vista, Financiamento (Tabela SAC estimando a parcela inicial) e Consórcio, considerando o valor de entrada fornecido.
    3. Traga um insight sobre o custo de oportunidade entre comprar reformado vs. para reformar.
    
    Separe a Tabela da Análise Financeira usando EXATAMENTE a string "---SEPARADOR---".
    """)
    
    user_prompt = HumanMessage(content=f"""
    Parâmetros do Cliente: {state['parametros']}
    
    Resultados Brutos da Busca na Web:
    {state['texto_busca_bruta']}
    """)
    
    resposta = llm.invoke([sys_prompt, user_prompt]).content
    
    # Separar Tabela e Análise
    partes = resposta.split("---SEPARADOR---")
    state["resultados_estruturados"] = partes[0].strip() if len(partes) > 0 else "Não foi possível estruturar a tabela."
    state["analise_financeira"] = partes[1].strip() if len(partes) > 1 else resposta
    
    # Atualiza o chat confirmando o sucesso
    state["messages"].append({
        "role": "assistant", 
        "content": f"Busca concluída na região de {state['parametros'].get('bairro')}! Os resultados estruturados e o laudo financeiro já estão disponíveis no painel ao lado."
    })
    return state

# ==========================================
# 4. ORQUESTRAÇÃO DO GRAFO
# ==========================================
def roteador_correcao(state: AgentState) -> str:
    # Se faltam parâmetros ou a busca na web falhou
    if state["erro_busca"]:
        state["messages"].append({
            "role": "assistant", 
            "content": "Não consegui encontrar resultados ou faltaram dados (preciso do Bairro e Valor Máximo). Pode me detalhar mais?"
        })
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
        "tentativas_busca": 0, "erro_busca": False
    }

# --- BARRA LATERAL: CONFIGURAÇÕES E CHAVES VIA SECRETS ---
with st.sidebar:
    st.header("Configuração de IA")
    
    provedor = st.radio("Escolha o Provedor LLM:", ["OpenAI", "Anthropic"])
    
    # Tenta carregar as chaves do .streamlit/secrets.toml de forma segura
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    # Roteamento de modelos e verificação de chaves
    if provedor == "OpenAI":
        st.session_state.api_key = openai_key
        st.session_state.provedor_fast = "OpenAI"
        st.session_state.fast_model = "gpt-4o-mini"
        st.session_state.provedor_reasoning = "OpenAI"
        st.session_state.reasoning_model = "gpt-4o"
    else:
        st.session_state.api_key = anthropic_key
        st.session_state.provedor_fast = "Anthropic"
        st.session_state.fast_model = "claude-3-haiku-20240307"
        st.session_state.provedor_reasoning = "Anthropic"
        st.session_state.reasoning_model = "claude-3-5-sonnet-20241022"
        
    if st.session_state.api_key:
        st.success(f"✅ Chave da {provedor} carregada com segurança!")
    else:
        st.error(f"⚠️ Chave da {provedor} não encontrada no arquivo secrets.toml!")
        
    st.markdown("---")
    st.caption("A extração de dados usa o modelo rápido (Haiku/Mini). A leitura da web e raciocínio financeiro usam o modelo pesado (Sonnet/GPT-4o).")

# --- LAYOUT PRINCIPAL (DUAS COLUNAS) ---
col_chat, col_results = st.columns([1, 1.2], gap="large")

with col_chat:
    st.subheader("Interação")
    chat_container = st.container(height=600)
    
    # Renderiza mensagens anteriores
    with chat_container:
        for msg in st.session_state.graph_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Recebe novo input
    if prompt := st.chat_input("Ex: Procuro apê em Moema até R$ 1.2Mi, tenho 300k de entrada"):
        if not st.session_state.api_key:
            st.error("Chave de API ausente. Verifique o arquivo .streamlit/secrets.toml!")
            st.stop()
            
        st.session_state.graph_state["messages"].append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(f"Extraindo dados com {st.session_state.fast_model} e buscando imóveis..."):
                    # Executa o Grafo
                    novo_estado = graph.invoke(st.session_state.graph_state)
                    st.session_state.graph_state = novo_estado
                    # Exibe a resposta final do grafo no chat
                    st.markdown(novo_estado["messages"][-1]["content"])
                    st.rerun()

with col_results:
    st.subheader("Painel de Análise Estruturada")
    state = st.session_state.graph_state
    
    if state["resultados_estruturados"]:
        st.markdown("#### 📋 Oportunidades Encontradas na Web")
        st.markdown(state["resultados_estruturados"])
        
        st.markdown("#### 💰 Laudo Financeiro e Estratégia")
        st.info(state["analise_financeira"])
        
        with st.expander("🛠️ Ver dados brutos extraídos do DuckDuckGo"):
            st.text(state["texto_busca_bruta"])
    else:
        st.markdown(
            "<div style='text-align: center; color: gray; margin-top: 100px;'>"
            "Aguardando os detalhes da busca no chat para gerar o painel de imóveis..."
            "</div>", 
            unsafe_allow_html=True
        )
