import re
import json
import streamlit as st

from typing import TypedDict, List, Dict

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from duckduckgo_search import DDGS

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

# ============================================================
# CONFIGURAÇÕES
# ============================================================

BAIRROS_SP = [
    "moema",
    "itaim",
    "pinheiros",
    "vila mariana",
    "brooklin",
    "campo belo",
    "perdizes",
    "saúde",
    "tatuapé",
    "morumbi"
]

# ============================================================
# PYDANTIC
# ============================================================

class PropertySearch(BaseModel):

    bairro: str = Field(default="Não definido")
    valor_max: int = Field(default=0)
    entrada: int = Field(default=0)

# ============================================================
# ESTADO DO GRAFO
# ============================================================

class AgentState(TypedDict):

    messages: List[Dict[str, str]]

    parametros: dict

    texto_busca_bruta: str

    resultados_estruturados: str

    analise_financeira: str

    tentativas_busca: int

    erro_busca: bool

    erro_extracao: str

# ============================================================
# UTILITÁRIOS
# ============================================================

def instanciar_llm(
    provedor: str,
    modelo: str,
    api_key: str
):

    if not api_key:
        raise ValueError(
            f"API Key do {provedor} não encontrada."
        )

    common = {
        "timeout": 30,
        "max_retries": 2
    }

    if provedor == "OpenAI":

        return ChatOpenAI(
            model=modelo,
            api_key=api_key,
            temperature=0,
            **common
        )

    if provedor == "Anthropic":

        return ChatAnthropic(
            model=modelo,
            api_key=api_key,
            max_tokens=1024,
            **common
        )

    raise ValueError(
        f"Provedor inválido: {provedor}"
    )

# ============================================================
# EXTRAÇÃO LOCAL
# ============================================================

def detectar_bairro(texto: str):

    texto_lower = texto.lower()

    for bairro in BAIRROS_SP:

        if bairro in texto_lower:
            return bairro.title()

    return None


def detectar_valor(texto: str):

    texto = texto.lower()

    match = re.search(
        r'(\d+(?:[\.,]\d+)?)\s*(milhões|milhão|mi|mil)',
        texto
    )

    if not match:
        return None

    numero = float(
        match.group(1)
        .replace(".", "")
        .replace(",", ".")
    )

    unidade = match.group(2)

    if unidade == "mil":
        return int(numero * 1000)

    return int(numero * 1000000)

# ============================================================
# NÓ EXTRATOR
# ============================================================

def node_extrator(state: AgentState):

    state["erro_extracao"] = ""

    ultima_msg = state["messages"][-1]["content"]

    params_atuais = state.get(
        "parametros",
        {
            "bairro": "Não definido",
            "valor_max": 0,
            "entrada": 0
        }
    )

    try:

        bairro_local = detectar_bairro(
            ultima_msg
        )

        valor_local = detectar_valor(
            ultima_msg
        )

        if bairro_local or valor_local:

            state["parametros"] = {
                "bairro":
                    bairro_local
                    or params_atuais["bairro"],

                "valor_max":
                    valor_local
                    or params_atuais["valor_max"],

                "entrada":
                    params_atuais["entrada"]
            }

            return state

        llm = instanciar_llm(
            st.session_state.provedor_fast,
            st.session_state.fast_model,
            st.session_state.api_key
        )

        extractor = llm.with_structured_output(
            PropertySearch
        )

        response = extractor.invoke(
            ultima_msg
        )

        novos = response.model_dump()

        state["parametros"] = {
            "bairro":
                novos["bairro"]
                if novos["bairro"]
                != "Não definido"
                else params_atuais["bairro"],

            "valor_max":
                novos["valor_max"]
                or params_atuais["valor_max"],

            "entrada":
                novos["entrada"]
                or params_atuais["entrada"]
        }

        state["tentativas_busca"] = (
            state.get("tentativas_busca", 0) + 1
        )

    except Exception as e:

        state["erro_extracao"] = str(e)

    return state

# ============================================================
# NÓ BUSCA WEB
# ============================================================

def node_buscador_web(state: AgentState):

    bairro = state["parametros"].get(
        "bairro",
        "Não definido"
    )

    if bairro == "Não definido":

        state["erro_busca"] = True

        return state

    query = (
        f"apartamento à venda "
        f"{bairro} São Paulo "
        f"vivareal OR zapimoveis"
    )

    try:

        resultados = []

        with DDGS() as ddgs:

            busca = list(
                ddgs.text(
                    query,
                    max_results=10
                )
            )

        for item in busca:

            resultados.append(
                {
                    "titulo":
                        item.get("title", ""),

                    "url":
                        item.get("href", ""),

                    "descricao":
                        item.get("body", "")
                }
            )

        state["texto_busca_bruta"] = json.dumps(
            resultados,
            ensure_ascii=False,
            indent=2
        )

        state["erro_busca"] = (
            len(resultados) == 0
        )

    except Exception:

        state["erro_busca"] = True

    return state

# ============================================================
# NÓ VALIDADOR
# ============================================================

def node_validador(state: AgentState):

    return state

# ============================================================
# NÓ ANALISTA
# ============================================================

def node_analista(state: AgentState):

    try:

        llm = instanciar_llm(
            st.session_state.provedor_reasoning,
            st.session_state.reasoning_model,
            st.session_state.api_key
        )

        sys_prompt = SystemMessage(
            content="""
Você é um consultor imobiliário.

REGRAS:

1. NÃO invente imóveis.

2. Utilize APENAS imóveis presentes na busca.

3. Caso os dados sejam insuficientes diga:
Nenhum imóvel encontrado.

4. Gere uma tabela markdown.

5. Depois escreva exatamente:

---SEPARADOR---

6. Em seguida faça uma análise financeira.
"""
        )

        user_prompt = HumanMessage(
            content=f"""
Parâmetros:

{json.dumps(state["parametros"], ensure_ascii=False)}

Resultados Web:

{state["texto_busca_bruta"]}
"""
        )

        resposta = llm.invoke(
            [sys_prompt, user_prompt]
        ).content

        partes = resposta.split(
            "---SEPARADOR---"
        )

        state["resultados_estruturados"] = (
            partes[0].strip()
            if len(partes) > 0
            else "Nenhum resultado."
        )

        state["analise_financeira"] = (
            partes[1].strip()
            if len(partes) > 1
            else ""
        )

        state["messages"].append(
            {
                "role": "assistant",
                "content":
                "✅ Busca concluída."
            }
        )

    except Exception as e:

        state["resultados_estruturados"] = (
            "Erro ao gerar relatório."
        )

        state["analise_financeira"] = str(e)

    return state

# ============================================================
# ROTEADOR
# ============================================================

def roteador(state: AgentState):

    if state.get("erro_extracao"):

        state["messages"].append(
            {
                "role": "assistant",
                "content":
                f"Erro de API: {state['erro_extracao']}"
            }
        )

        return END

    if state.get("erro_busca", False):

        state["messages"].append(
            {
                "role": "assistant",
                "content":
                "Não encontrei resultados. "
                "Informe um bairro válido."
            }
        )

        return END

    return "node_analista"

# ============================================================
# GRAFO
# ============================================================

builder = StateGraph(AgentState)

builder.add_node(
    "node_extrator",
    node_extrator
)

builder.add_node(
    "node_buscador_web",
    node_buscador_web
)

builder.add_node(
    "node_validador",
    node_validador
)

builder.add_node(
    "node_analista",
    node_analista
)

builder.add_edge(
    START,
    "node_extrator"
)

builder.add_edge(
    "node_extrator",
    "node_buscador_web"
)

builder.add_edge(
    "node_buscador_web",
    "node_validador"
)

builder.add_conditional_edges(
    "node_validador",
    roteador
)

builder.add_edge(
    "node_analista",
    END
)

graph = builder.compile()

# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Consultor Imobiliário IA",
    layout="wide"
)

if "graph_state" not in st.session_state:

    st.session_state.graph_state = {
        "messages": [],
        "parametros": {},
        "texto_busca_bruta": "",
        "resultados_estruturados": "",
        "analise_financeira": "",
        "tentativas_busca": 0,
        "erro_busca": False,
        "erro_extracao": ""
    }

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Modelos")

    provedor = st.radio(
        "Provedor",
        ["OpenAI", "Anthropic"]
    )

    if provedor == "OpenAI":

        st.session_state.api_key = (
            st.secrets.get(
                "OPENAI_API_KEY",
                ""
            )
        )

        st.session_state.provedor_fast = "OpenAI"

        st.session_state.fast_model = (
            "gpt-4o-mini"
        )

        st.session_state.provedor_reasoning = (
            "OpenAI"
        )

        st.session_state.reasoning_model = (
            "gpt-4o"
        )

    else:

        st.session_state.api_key = (
            st.secrets.get(
                "ANTHROPIC_API_KEY",
                ""
            )
        )

        st.session_state.provedor_fast = (
            "Anthropic"
        )

        st.session_state.fast_model = (
            "claude-3-5-haiku-latest"
        )

        st.session_state.provedor_reasoning = (
            "Anthropic"
        )

        st.session_state.reasoning_model = (
            "claude-3-7-sonnet-latest"
        )

    st.divider()

    st.subheader("Memória")

    st.json(
        st.session_state.graph_state.get(
            "parametros",
            {}
        )
    )

# ============================================================
# LAYOUT
# ============================================================

col_chat, col_resultados = st.columns(
    [1, 1.2]
)

# ============================================================
# CHAT
# ============================================================

with col_chat:

    st.subheader("Chat")

    for msg in st.session_state.graph_state[
        "messages"
    ]:

        with st.chat_message(
            msg["role"]
        ):
            st.markdown(
                msg["content"]
            )

    prompt = st.chat_input(
        "Ex: apartamento em Moema até 1.2 mi"
    )

    if prompt:

        st.session_state.graph_state[
            "messages"
        ].append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.spinner(
            "Consultando..."
        ):

            novo_estado = graph.invoke(
                st.session_state.graph_state
            )

            st.session_state.graph_state = (
                novo_estado
            )

            st.rerun()

# ============================================================
# RESULTADOS
# ============================================================

with col_resultados:

    st.subheader(
        "Análise Estruturada"
    )

    estado = (
        st.session_state.graph_state
    )

    if estado[
        "resultados_estruturados"
    ]:

        st.markdown(
            estado[
                "resultados_estruturados"
            ]
        )

        st.markdown(
            "### Análise Financeira"
        )

        st.info(
            estado[
                "analise_financeira"
            ]
        )

    else:

        st.info(
            "Aguardando consulta."
        )
