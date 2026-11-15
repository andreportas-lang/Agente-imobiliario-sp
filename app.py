"""
Agente Imobiliário — Streamlit + grafo de execução (OpenAI ou Claude)

Fluxo (grafo):
  extract ─► [conversa] ─► answer ─► END
          ├► [faltam dados] ─► ask ─► END
          └► queries ─► search(paralelo) ─► rank(heurística) ─► enrich(paralelo)
                       ─► select(1 chamada LLM) ─► [<3 opções e 1ª tentativa] ─► queries (busca relaxada)
                                                └► compose ─► END
Só 2 nós usam LLM no caminho feliz (extract e select), ambos com modelo barato
e poucos tokens. Busca, ranqueamento, filtros e resposta final são código puro.

Secrets esperados (Streamlit Community Cloud > Settings > Secrets):
  OPENAI_API_KEY = "sk-..."
  ANTHROPIC_API_KEY = "sk-ant-..."
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import time
from urllib.parse import urlparse

import requests
import streamlit as st

st.set_page_config(page_title="Agente Imobiliário", page_icon="🏠", layout="wide")

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
SITES = {
    "QuintoAndar": "quintoandar.com.br",
    "Loft": "loft.com.br",
    "Viva Real": "vivareal.com.br",
    "Tamaras": "tamaras.com.br",
    "ZAP Imóveis": "zapimoveis.com.br",
    "Imovelweb": "imovelweb.com.br",
    "Chaves na Mão": "chavesnamao.com.br",
    "OLX": "olx.com.br",
}
DEFAULT_MODELS = {"OpenAI": "gpt-4o-mini", "Claude": "claude-haiku-5-5"}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
CACHE_TTL = 3600


@st.cache_resource
def _cache_store():
    return {}


CACHE = _cache_store()


def cached(key, fn, ttl=CACHE_TTL):
    hit = CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1], True
    val = fn()
    CACHE[key] = (time.time(), val)
    return val, False


def secret(name):
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    return v or os.environ.get(name)


# ----------------------------------------------------------------------------
# Mini-motor de grafo (nós, arestas fixas e condicionais, trace p/ debug)
# ----------------------------------------------------------------------------
class Graph:
    def __init__(self, entry):
        self.nodes, self.edges, self.entry = {}, {}, entry

    def node(self, name, fn):
        self.nodes[name] = fn
        return self

    def edge(self, a, b):  # b: nome do nó, "END" ou função(state)->nome
        self.edges[a] = b
        return self

    def run(self, state, on_step=None, max_steps=20):
        cur = self.entry
        for _ in range(max_steps):
            if cur == "END":
                break
            if on_step:
                on_step(cur)
            t0 = time.perf_counter()
            note = ""
            try:
                note = self.nodes[cur](state) or ""
                nxt = self.edges.get(cur, "END")
                nxt = nxt(state) if callable(nxt) else nxt
            except Exception as e:  # erro em um nó -> vai para "fail"
                state["errors"].append(f"[{cur}] {type(e).__name__}: {e}")
                note, nxt = f"ERRO: {e}", "fail"
                if cur == "fail":
                    nxt = "END"
            state["trace"].append(
                {"nó": cur, "ms": round((time.perf_counter() - t0) * 1000), "info": str(note)[:200], "próximo": nxt}
            )
            cur = nxt
        return state


# ----------------------------------------------------------------------------
# LLM (OpenAI ou Claude) com contagem de tokens para debug
# ----------------------------------------------------------------------------
def llm(state, node, system, user, max_tokens=600, as_json=False):
    provider, model = state["provider"], state["model"]
    t0 = time.perf_counter()
    if provider == "OpenAI":
        key = secret("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY não encontrada nos secrets.")
        from openai import OpenAI

        kw = {"response_format": {"type": "json_object"}} if as_json else {}
        r = OpenAI(api_key=key).chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kw,
        )
        text = r.choices[0].message.content or ""
        stop = r.choices[0].finish_reason
        tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
    else:
        key = secret("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY não encontrada nos secrets.")
        import anthropic

        r = anthropic.Anthropic(api_key=key).messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        stop = f"{r.stop_reason} | blocos={[getattr(b, 'type', '?') for b in r.content]}"
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
    state["llm_log"].append(
        {"nó": node, "modelo": f"{provider}:{model}", "tokens_in": tin, "tokens_out": tout,
         "max_tokens": max_tokens, "parada": stop, "ms": round((time.perf_counter() - t0) * 1000),
         "enviado_system": system, "enviado_user": user, "resposta": text[:1500]}
    )
    if not text.strip():
        raise RuntimeError(
            f"Resposta vazia do LLM (parada={stop}; tokens_out={tout}/{max_tokens}). "
            "Provável: o modelo gastou o limite de tokens 'pensando'. Aumente max_tokens ou troque o modelo."
        )
    return text


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"LLM não retornou JSON: {text[:200]}")
    return json.loads(m.group(0))


# ----------------------------------------------------------------------------
# Utilidades (regex, busca, fetch)
# ----------------------------------------------------------------------------
PRICE_RE = re.compile(r"R\$\s?([\d]{1,3}(?:\.\d{3})+|\d+)(?:,\d{2})?")
AREA_RE = re.compile(r"(\d{2,4})(?:[.,]\d+)?\s?(?:m²|m2|metros)", re.I)


def brl(v):
    return "—" if v is None else "R$ " + f"{v:,.0f}".replace(",", ".")


def first_price(text):
    vals = [int(m.replace(".", "")) for m in PRICE_RE.findall(text or "")]
    vals = [v for v in vals if v >= 300]
    return vals[0] if vals else None


def first_area(text):
    vals = [int(m) for m in AREA_RE.findall(text or "")]
    vals = [v for v in vals if 10 <= v <= 2000]
    return vals[0] if vals else None


def site_name(url):
    host = urlparse(url).netloc.replace("www.", "")
    for name, dom in SITES.items():
        if dom in host:
            return name
    return host


def ddg(query, n=6):
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    with DDGS() as d:
        return list(d.text(query, region="br-pt", max_results=n))


def fetch_meta(url):
    r = requests.get(url, headers=UA, timeout=6)
    html = r.text[:400_000]
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    desc = re.search(r'<meta[^>]+(?:og:description|name="description")[^>]+content="([^"]+)"', html, re.I)
    body = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", html, flags=re.S | re.I)
    body = re.sub(r"\s+", " ", body)[:3000]
    return {
        "status": r.status_code,
        "title": (title.group(1).strip() if title else "")[:160],
        "desc": (desc.group(1).strip() if desc else "")[:300],
        "preco": first_price(body),
        "area": first_area(body),
    }


# ----------------------------------------------------------------------------
# Nós do grafo
# ----------------------------------------------------------------------------
def n_extract(s):
    prof = s["profile"]
    msg = s["user_msg"]
    system = "Você extrai critérios de busca imobiliária no Brasil. Responda SOMENTE com JSON válido."
    user = (
        f"Perfil atual: {json.dumps(prof, ensure_ascii=False)}\n"
        f"Mensagem do usuário: {msg}\n\n"
        "Retorne JSON com as chaves: intent ('busca' se o usuário informa/ajusta critérios ou pede nova busca; "
        "'conversa' se é dúvida/comentário sobre opções já mostradas), bairro, cidade, valor_max (número em reais; "
        "'3 mil' = 3000), finalidade ('alugar' ou 'comprar' ou null), area_m2 (número), quartos (número), vagas (número), "
        "extras (texto curto com outras preferências, ex: aceita pet, andar alto). Use null para o que não foi informado."
    )
    try:
        data = parse_json(llm(s, "extract", system, user, max_tokens=1500, as_json=True))
    except Exception as e:
        s["errors"].append(f"[extract] fallback regex ({e})")
        data = {"intent": "busca", "valor_max": first_price(msg), "area_m2": first_area(msg)}
    for k in ["bairro", "cidade", "valor_max", "finalidade", "area_m2", "quartos", "vagas", "extras"]:
        v = data.get(k)
        if v not in (None, "", 0):
            prof[k] = v
    prof.setdefault("cidade", s["cidade_default"])
    s["intent"] = data.get("intent", "busca")
    if s["intent"] == "conversa" and not s["top3"] and not any(prof.get(k) for k in ("bairro", "valor_max")):
        s["intent"] = "busca"
    s["missing"] = [lbl for k, lbl in [("bairro", "bairro"), ("valor_max", "valor máximo"), ("area_m2", "tamanho (m²)")]
                    if not prof.get(k)]
    return f"intent={s['intent']} missing={s['missing']}"


def r_after_extract(s):
    if s["intent"] == "conversa":
        return "answer"
    return "ask" if s["missing"] else "queries"


def n_ask(s):
    falta = ", ".join(s["missing"])
    s["reply"] = (f"Para buscar, ainda preciso de: **{falta}**. "
                  "Pode também me contar quartos, vagas, se aceita pet ou outras preferências.")
    return "pediu dados faltantes (sem LLM)"


def n_answer(s):
    ctx = json.dumps(s["top3"], ensure_ascii=False)[:4000]
    hist = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in s["history"][-6:])
    system = ("Você é um corretor de imóveis prestativo. Responda em português, curto e direto, usando apenas as "
              "opções fornecidas. Se faltar informação, diga que precisa confirmar no anúncio.")
    s["reply"] = llm(s, "answer", system, f"Opções atuais: {ctx}\nHistórico:\n{hist}\nPergunta: {s['user_msg']}",
                     max_tokens=1500)
    return "resposta de acompanhamento"


def n_queries(s):
    p, mode = s["profile"], s["attempt"]
    fin = p.get("finalidade") or ("alugar" if p["valor_max"] <= 20000 else "comprar")
    p["_finalidade_inferida"] = not s["profile"].get("finalidade")
    p["finalidade"] = fin
    base = f"{fin} apartamento {p['bairro']} {p['cidade']}"
    extras = f" {p['quartos']} quartos" if p.get("quartos") else ""
    qs = []
    if mode == 0:
        for name in s["sites"]:
            qs.append((name, f"site:{SITES[name]} {base} {int(p['area_m2'])} m²{extras}"))
        qs.append(("Web", f"{base} {int(p['area_m2'])} m² até R$ {int(p['valor_max'])}{extras}"))
    else:  # busca relaxada: sem tamanho, sem restrição de quartos
        for name in s["sites"]:
            qs.append((name, f"site:{SITES[name]} {base}"))
        qs.append(("Web", f"imóveis {fin} {p['bairro']} {p['cidade']} até R$ {int(p['valor_max'])}"))
    s["queries"] = qs
    s["attempt"] += 1
    return f"{len(qs)} queries (modo {'relaxado' if mode else 'estrito'})"


def n_search(s):
    def one(item):
        name, q = item
        res, hit = cached("ddg:" + q, lambda: ddg(q))
        return name, q, res, hit

    hits = 0
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for name, q, res, hit in ex.map(one, s["queries"]):
            hits += hit
            s["search_log"].append({"query": q, "resultados": len(res), "cache": hit})
            for r in res:
                url = r.get("href", "")
                if url and url not in s["raw"]:
                    s["raw"][url] = {"url": url, "fonte": site_name(url), "titulo": r.get("title", ""),
                                     "trecho": r.get("body", "")}
    return f"{len(s['raw'])} URLs únicas ({hits} queries do cache)"


def n_rank(s):
    p = s["profile"]
    bairro = p["bairro"].lower()
    cands = []
    for c in s["raw"].values():
        text = f"{c['titulo']} {c['trecho']}"
        c["preco"], c["area"] = first_price(text), first_area(text)
        score = 0
        score += 2 if "/imovel/" in c["url"] or "/imoveis/" in c["url"] else 0
        score += 2 if bairro in (text + c["url"]).lower() else 0
        score += 1 if c["preco"] else 0
        score += 1 if c["area"] else 0
        if c["preco"] and c["preco"] > p["valor_max"] * 1.15:
            score -= 5
        if c["area"] and c["area"] < p["area_m2"] * 0.8:
            score -= 3
        c["score"] = score
        cands.append(c)
    cands.sort(key=lambda c: c["score"], reverse=True)
    s["cands"] = [c for c in cands if c["score"] > -3][:8]
    return f"{len(s['cands'])} candidatos (de {len(cands)})"


def n_enrich(s):
    todo = [c for c in s["cands"] if not (c.get("preco") and c.get("area")) and "meta" not in c]

    def one(c):
        try:
            meta, _ = cached("meta:" + c["url"], lambda: fetch_meta(c["url"]))
            return c, meta
        except Exception as e:
            return c, {"erro": str(e)[:80]}

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for c, meta in ex.map(one, todo):
            c["meta"] = meta
            c["preco"] = c.get("preco") or meta.get("preco")
            c["area"] = c.get("area") or meta.get("area")
            c["trecho"] = (c["trecho"] + " " + meta.get("desc", ""))[:500]
    ok = sum(1 for c in todo if "erro" not in c["meta"])
    return f"{ok}/{len(todo)} páginas lidas"


def n_select(s):
    p = s["profile"]
    if not s["cands"]:
        s["top3"] = []
        return "sem candidatos"
    compact = [{"id": i, "fonte": c["fonte"], "titulo": c["titulo"][:120], "trecho": c["trecho"][:350],
                "preco_detectado": c.get("preco"), "area_detectada": c.get("area")}
               for i, c in enumerate(s["cands"])]
    crit = {k: v for k, v in p.items() if not k.startswith("_")}
    system = ("Você é um corretor especialista. Escolha até 3 anúncios que melhor atendem aos critérios, SOMENTE entre "
              "os candidatos fornecidos. Nunca invente dados: use null quando não souber. Responda só JSON.")
    user = (f"Critérios: {json.dumps(crit, ensure_ascii=False)}\nCandidatos: {json.dumps(compact, ensure_ascii=False)}\n\n"
            'Formato: {"opcoes":[{"id":int,"titulo":str,"preco":number|null,"area_m2":number|null,"quartos":number|null,'
            '"aderencia":0-100,"motivo":"1-2 frases","alertas":["itens a confirmar/divergências"]}]} '
            "ordenado por aderência. Ignore páginas de listagem genérica sem imóvel específico, a menos que não haja melhor.")
    data = parse_json(llm(s, "select", system, user, max_tokens=4000, as_json=True))
    out = []
    for o in data.get("opcoes", [])[:3]:
        i = o.get("id")
        if isinstance(i, int) and 0 <= i < len(s["cands"]):  # link vem do candidato real, nunca do LLM
            c = s["cands"][i]
            out.append({**o, "url": c["url"], "fonte": c["fonte"]})
    s["top3"] = out
    return f"{len(out)} opções escolhidas"


def r_after_select(s):
    return "queries" if len(s["top3"]) < 3 and s["attempt"] < 2 else "compose"


def n_compose(s):
    p, n = s["profile"], len(s["top3"])
    if n == 0:
        s["reply"] = ("Não encontrei anúncios aderentes agora. Tente ampliar o valor, o tamanho ou o bairro "
                      "(ou ative o modo debug para ver o que as buscas retornaram).")
        return "nenhuma opção"
    s["reply"] = (f"Encontrei **{n} opç{'ão' if n == 1 else 'ões'}** em **{p['bairro']}/{p['cidade']}** "
                  f"para **{p['finalidade']}**, até {brl(p['valor_max'])} e ~{int(p['area_m2'])} m². "
                  "Veja os detalhes e links ao lado →")
    if p.get("_finalidade_inferida"):
        s["reply"] += f"\n\n_Assumi que você quer **{p['finalidade']}** pelo valor informado; me diga se for o contrário._"
    if n < 3:
        s["reply"] += "\n\n_Só achei menos de 3 opções confiáveis, mesmo após ampliar a busca._"
    s["reply"] += "\n\n⚠️ Preços e disponibilidade vêm de busca pública: confirme no anúncio."
    return f"{n} opções"


def n_fail(s):
    s["reply"] = "Tive um problema ao processar seu pedido. Ative o **modo debug** na barra lateral para ver o erro."
    return "falha"


def build_graph():
    g = Graph(entry="extract")
    g.node("extract", n_extract).edge("extract", r_after_extract)
    g.node("ask", n_ask).edge("ask", "END")
    g.node("answer", n_answer).edge("answer", "END")
    g.node("queries", n_queries).edge("queries", "search")
    g.node("search", n_search).edge("search", "rank")
    g.node("rank", n_rank).edge("rank", "enrich")
    g.node("enrich", n_enrich).edge("enrich", "select")
    g.node("select", n_select).edge("select", r_after_select)
    g.node("compose", n_compose).edge("compose", "END")
    g.node("fail", n_fail).edge("fail", "END")
    return g


LABELS = {"extract": "Entendendo o pedido", "queries": "Montando buscas", "search": "Buscando nos sites",
          "rank": "Filtrando candidatos", "enrich": "Lendo anúncios", "select": "Escolhendo as 3 melhores",
          "answer": "Respondendo", "ask": "Verificando dados", "compose": "Finalizando", "fail": "Tratando erro"}

# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("profile", {})
ss.setdefault("top3", [])
ss.setdefault("last_state", None)

with st.sidebar:
    st.header("⚙️ Configurações")
    provider = st.radio("Provedor de IA", ["OpenAI", "Claude"], horizontal=True)
    model = st.text_input("Modelo", DEFAULT_MODELS[provider], key=f"model_{provider}",
                          help="Use um modelo rápido/barato; o grafo faz só 2 chamadas por busca.")
    cidade = st.text_input("Cidade padrão", "São Paulo")
    sites = st.multiselect("Sites", list(SITES), default=list(SITES)[:5])
    debug = st.toggle("🐞 Modo debug", value=False)
    if st.button("🗑️ Nova conversa", use_container_width=True):
        for k in ("messages", "profile", "top3", "last_state"):
            ss.pop(k, None)
        st.rerun()
    if debug and st.button("Limpar cache de buscas", use_container_width=True):
        CACHE.clear()
        st.toast("Cache limpo")
    if debug:
        st.caption("Chaves nos secrets")
        st.write("OpenAI:", "✅" if secret("OPENAI_API_KEY") else "❌", " Claude:",
                 "✅" if secret("ANTHROPIC_API_KEY") else "❌")

st.title("🏠 Agente Imobiliário")
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("💬 Conversa")
    box = st.container(height=520)
    prompt = st.chat_input("Ex.: apartamento em Pinheiros, até R$ 4.500, 60 m², aceita pet")

with box:
    if not ss.messages:
        st.info("Informe **bairro**, **valor** e **tamanho** do imóvel. Quanto mais detalhes, melhor a busca.")
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            state = {
                "user_msg": prompt, "history": list(ss.messages), "profile": dict(ss.profile), "top3": ss.top3,
                "provider": provider, "model": model.strip() or DEFAULT_MODELS[provider],
                "cidade_default": cidade, "sites": sites or list(SITES)[:3], "attempt": 0,
                "raw": {}, "cands": [], "queries": [], "search_log": [], "llm_log": [], "trace": [], "errors": [],
                "reply": "", "intent": "", "missing": [],
            }
            with st.status("Trabalhando...", expanded=debug) as status:
                build_graph().run(state, on_step=lambda n: status.update(label=LABELS.get(n, n) + "..."))
                status.update(label="Pronto", state="error" if state["errors"] else "complete", expanded=False)
            st.markdown(state["reply"])
        ss.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": state["reply"]}]
        ss.profile = {k: v for k, v in state["profile"].items()}
        if state["intent"] == "busca" and not state["missing"]:
            ss.top3 = state["top3"]
        ss.last_state = state

with right:
    st.subheader("🏘️ Melhores opções")
    if not ss.top3:
        st.caption("As 3 melhores opções aparecerão aqui.")
    for i, o in enumerate(ss.top3, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {o.get('titulo', 'Imóvel')}**  \n`{o['fonte']}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Preço", brl(o.get("preco")))
            c2.metric("Área", f"{o['area_m2']} m²" if o.get("area_m2") else "—")
            c3.metric("Quartos", o.get("quartos") or "—")
            score = int(o.get("aderencia") or 0)
            st.progress(min(max(score, 0), 100) / 100, text=f"Aderência: {score}%")
            st.write(o.get("motivo", ""))
            for a in o.get("alertas") or []:
                st.caption(f"⚠️ {a}")
            st.link_button("Ver anúncio ↗", o["url"], use_container_width=True)

    if debug and ss.last_state:
        s = ss.last_state
        st.divider()
        st.subheader("🐞 Debug")
        if s["errors"]:
            for e in s["errors"]:
                st.error(e)
        tabs = st.tabs(["Grafo", "LLM", "Buscas", "Candidatos", "Estado"])
        with tabs[0]:
            st.dataframe(s["trace"], use_container_width=True)
            st.caption(f"Tempo total: {sum(t['ms'] for t in s['trace'])} ms")
        with tabs[1]:
            st.caption(f"Tokens: in={sum(x['tokens_in'] for x in s['llm_log'])} "
                       f"out={sum(x['tokens_out'] for x in s['llm_log'])}")
            for x in s["llm_log"]:
                with st.expander(f"{x['nó']} · {x['modelo']} · {x['ms']} ms"):
                    st.json(x)
        with tabs[2]:
            st.dataframe(s["search_log"], use_container_width=True)
        with tabs[3]:
            st.json([{k: v for k, v in c.items() if k != "trecho"} for c in s["cands"]], expanded=False)
        with tabs[4]:
            st.json({"perfil": s["profile"], "intent": s["intent"], "faltando": s["missing"],
                     "tentativas": s["attempt"]}, expanded=False)
