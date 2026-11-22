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
import html as htmllib
import unicodedata
from pathlib import Path
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
    "Tamaras": "maramores.com.br",
    "ZAP Imóveis": "zapimoveis.com.br",
    "Imovelweb": "imovelweb.com.br",
    "Chaves na Mão": "chavesnamao.com.br",
    "OLX": "olx.com.br",
    "Pilar Home": "pilarhomes.com.br",
    "Thura": "thura.com.br",  # domínio não verificado: confirme/edite na barra lateral
}
DEFAULT_MODELS = {"OpenAI": "gpt-4o-mini", "Claude": "claude-haiku-5-5"}
MODEL_OPTIONS = {  # modelos oferecidos na lista (há também a opção "Outro" para digitar qualquer ID)
    "OpenAI": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
    "Claude": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5"],
}
OTHER = "Outro (digitar ID)"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
CACHE_TTL = 3600
# US$ por 1M de tokens (entrada, saída). Confira no site do provedor; dá para ajustar na barra lateral.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
}
SEARCH_FEE = 0.01  # US$ por busca web nativa (OpenAI e Anthropic: ~US$ 10 / 1.000 buscas)
USAGE_FILE = Path(__file__).with_name("usage_log.json")
AREA_TOL_DOWN = 10   # tolerância máxima (m²) abaixo do tamanho pedido
PRICE_TOL = 0.10     # tolerância de 10% acima do valor máximo
# Só aceitamos URLs que apontam para UM imóvel (não páginas de busca/listagem)
LISTING_RE = {
    "quintoandar.com.br": r"/imovel/\d+",
    "loft.com.br": r"/imovel/",
    "vivareal.com.br": r"/imovel/",
    "zapimoveis.com.br": r"/imovel/",
    "imovelweb.com.br": r"/propriedades/",
    "chavesnamao.com.br": r"/imovel/",
    "olx.com.br": r"-\d{7,}(?:[/?#]|$)",
    "pilarhomes.com.br": r"/imovel/|/(imoveis|apartamento|apartamentos|unidade|aluguel|locacao|venda)/[^?#]*\d{3,}",
    "thura.com.br": r"/imovel/|/(imoveis|apartamento|apartamentos|unidade|aluguel|locacao|venda)/[^?#]*\d{3,}",
    "maramores.com.br": r"/imovel/|/(imoveis|casa|casas|apartamento|apartamentos|unidade|aluguel|locacao|venda)/[^?#]*\d{3,}",
}
SITE_PATH = {"quintoandar.com.br": "/imovel", "loft.com.br": "/imovel", "vivareal.com.br": "/imovel",
             "zapimoveis.com.br": "/imovel", "chavesnamao.com.br": "/imovel", "imovelweb.com.br": "/propriedades"}


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


def secret_info(name):
    try:
        v, src = st.secrets.get(name), "st.secrets"
    except Exception:
        v, src = None, ""
    if not v:
        v, src = os.environ.get(name), "variável de ambiente"
    if isinstance(v, str):
        v = v.strip().strip("\"'").strip()  # remove espaços/aspas coladas por engano
    return (v, src) if v else (None, "")


def secret(name):
    return secret_info(name)[0]


def hint(e):
    """Dica amigável para erros comuns de API."""
    code = getattr(e, "status_code", None)
    msg = str(e)
    if code == 401 or "401" in msg or "AuthenticationError" in type(e).__name__:
        return (" → CHAVE RECUSADA (401): chave inválida, revogada ou de outro projeto/organização. "
                "Veja 'Diagnóstico das chaves' na barra lateral.")
    if code == 429 or "RateLimit" in type(e).__name__:
        return " → limite de uso/crédito atingido (429)."
    return ""


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
                state["errors"].append(f"[{cur}] {type(e).__name__}: {e}{hint(e)}")
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
def call_raw(api, **kw):
    """Chama api.with_raw_response.create para obter também os headers (limites de uso restantes)."""
    raw = api.with_raw_response.create(**kw)
    return raw.parse(), dict(raw.headers)


def rl_parse(provider, hdr):
    h = {k.lower(): v for k, v in (hdr or {}).items()}
    if provider == "OpenAI":
        keys = {"requisições restantes": "x-ratelimit-remaining-requests", "limite de requisições": "x-ratelimit-limit-requests",
                "tokens restantes": "x-ratelimit-remaining-tokens", "limite de tokens": "x-ratelimit-limit-tokens",
                "renova em": "x-ratelimit-reset-tokens"}
    else:
        keys = {"requisições restantes": "anthropic-ratelimit-requests-remaining",
                "limite de requisições": "anthropic-ratelimit-requests-limit",
                "tokens restantes": "anthropic-ratelimit-tokens-remaining",
                "limite de tokens": "anthropic-ratelimit-tokens-limit", "renova em": "anthropic-ratelimit-tokens-reset"}
    return {k: h[v] for k, v in keys.items() if h.get(v) is not None}


def llm(state, node, system, user, max_tokens=600, as_json=False):
    provider, model = state["provider"], state["model"]
    t0 = time.perf_counter()
    if provider == "OpenAI":
        key = secret("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY não encontrada nos secrets.")
        from openai import OpenAI

        kw = {"response_format": {"type": "json_object"}} if as_json else {}
        r, hdr = call_raw(OpenAI(api_key=key).chat.completions,
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

        r, hdr = call_raw(anthropic.Anthropic(api_key=key).messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        stop = f"{r.stop_reason} | blocos={[getattr(b, 'type', '?') for b in r.content]}"
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
    state["ratelimit"] = rl_parse(provider, hdr)
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


def llm_web(state, node, system, user, domains, max_tokens=4000):
    """Chamada com busca web NATIVA do provedor (usada só como fallback do DuckDuckGo)."""
    provider, model = state["provider"], state["model"]
    t0 = time.perf_counter()
    if provider == "OpenAI":
        key = secret("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY não encontrada nos secrets.")
        from openai import OpenAI

        client = OpenAI(api_key=key)
        kw = dict(model=model, instructions=system, input=user, max_output_tokens=max_tokens)
        try:
            r, hdr = call_raw(client.responses, tools=[{"type": "web_search"}], **kw)
        except Exception:  # modelos antigos só aceitam a versão preview da ferramenta
            r, hdr = call_raw(client.responses, tools=[{"type": "web_search_preview"}], **kw)
        text = r.output_text or ""
        nsearch = sum(1 for o in r.output if getattr(o, "type", "") == "web_search_call")
        stop = str(r.status)
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
    else:
        key = secret("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY não encontrada nos secrets.")
        import anthropic

        r, hdr = call_raw(anthropic.Anthropic(api_key=key).messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3,
                    **({"allowed_domains": domains} if domains else {})}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        nsearch = getattr(getattr(r.usage, "server_tool_use", None), "web_search_requests", 0) or 0
        stop = f"{r.stop_reason} | blocos={[getattr(b, 'type', '?') for b in r.content]}"
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
    state["ratelimit"] = rl_parse(provider, hdr)
    state["llm_log"].append(
        {"nó": node, "modelo": f"{provider}:{model}", "tokens_in": tin, "tokens_out": tout,
         "max_tokens": max_tokens, "web_searches": nsearch, "parada": stop,
         "ms": round((time.perf_counter() - t0) * 1000), "enviado_system": system, "enviado_user": user,
         "resposta": text[:1500]}
    )
    if not text.strip():
        raise RuntimeError(f"Busca nativa retornou vazio (parada={stop}; tokens_out={tout}/{max_tokens}).")
    return text


def num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


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


def prices(text):
    out = []
    for m in PRICE_RE.finditer(text or ""):
        before = text[max(0, m.start() - 14):m.start()].lower()
        if any(w in before for w in ("condom", "iptu", "taxa")):  # ignora condomínio/IPTU
            continue
        v = int(m.group(1).replace(".", ""))
        if v >= 300:
            out.append(v)
    return out


def first_price(text):
    pr = prices(text)
    return pr[0] if pr else None


def ok_price(v, fin):
    """Descarta valores implausíveis (ex.: preço de venda numa busca de aluguel)."""
    if not v:
        return None
    if fin == "alugar":
        return v if 300 <= v <= 80_000 else None
    return v if v >= 40_000 else None


def is_listing_url(url):
    u = urlparse(url)
    host, path = u.netloc.replace("www.", ""), u.path
    for dom, pat in LISTING_RE.items():
        if dom in host:
            return bool(re.search(pat, path))
    return bool(re.search(r"/(imovel|imoveis|propriedades?|anuncio|ad)/.*\d{4,}|-\d{6,}(?:[/?#]|$)", path))


def norm(t):
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")


def city_confirmed(c, p):
    """A cidade pedida precisa aparecer na URL, no título ou na descrição (não no corpo/rodapé da página)."""
    meta = c.get("meta") or {}
    hay = "-" + norm(f"{c['url']} {c['titulo']} {c['trecho']} {meta.get('title', '')} {meta.get('desc', '')}") + "-"
    return f"-{norm(p['cidade'])}-" in hay


TIPO_PRIORIDADE = ["casa de vila", "casa de condomínio", "sobrado", "cobertura", "studio/kitnet", "apartamento", "casa",
                   "terreno"]
_TIPO_PAT = {
    "casa de vila": r"casas?-(?:de|em|na|no)-vilas?|casas?-vilas?",
    "casa de condomínio": r"casas?-(?:de|em)-condominios?|casas?-condominios?",
    "sobrado": r"sobrados?",
    "cobertura": r"coberturas?",
    "studio/kitnet": r"kitnets?|kitchenettes?|studios?|estudios?|conjugados?",
    "apartamento": r"apartamentos?|aptos?|apts?|flats?",
    "casa": r"casas?",
    "terreno": r"terrenos?|lotes?",
}
TIPO_RE = {k: re.compile(r"(?<![a-z0-9])(?:" + v + r")(?![a-z0-9])") for k, v in _TIPO_PAT.items()}
TIPO_ACEITA = {
    "casa de vila": {"casa de vila"}, "casa de condomínio": {"casa de condomínio"}, "sobrado": {"sobrado"},
    "casa": {"casa", "casa de vila", "casa de condomínio", "sobrado"}, "apartamento": {"apartamento", "cobertura"},
    "cobertura": {"cobertura"}, "studio/kitnet": {"studio/kitnet"}, "terreno": {"terreno"},
}
TIPO_SUB = {"casa de vila", "casa de condomínio", "sobrado"}
TIPO_QUERY = {"studio/kitnet": "studio", "casa de condomínio": "casa em condomínio"}


def canon_tipo(t):
    n = "-" + norm(t) + "-"
    if "casa" in n and "vila" in n:
        return "casa de vila"
    if "casa" in n and "condominio" in n:
        return "casa de condomínio"
    for key, rx in (("sobrado", r"sobrado"), ("cobertura", r"cobertura"), ("studio/kitnet", r"kitnet|studio|estudio|conjugado"),
                    ("apartamento", r"apartamento|apto|flat"), ("casa", r"casa"), ("terreno", r"terreno|lote")):
        if re.search(rx, n):
            return key
    return None


def detect_tipos(c):
    """Tipos citados no anúncio. Usa o primeiro campo que citar algum: caminho da URL > título > título da página > trecho."""
    meta = c.get("meta") or {}
    for field in (urlparse(c["url"]).path, c["titulo"], meta.get("title", ""), c["trecho"]):
        t = "-" + norm(field) + "-"
        found = {k for k, rx in TIPO_RE.items() if rx.search(t)}
        if found:
            return found
    return set()


def main_tipo(found):
    return next((k for k in TIPO_PRIORIDADE if k in found), "")


def tipo_status(c, p):
    """ok | unknown | generic | mismatch. Define c['tipos'], c['tipo_ok'] e c['tipo_flag']."""
    want, found = p.get("tipo"), detect_tipos(c)
    c["tipos"], c["tipo_ok"], c["tipo_flag"] = sorted(found), False, ""
    if not want:
        return "ok"
    if not found:
        c["tipo_flag"] = f"Tipo não confirmado — verifique se é {want}"
        return "unknown"
    if want in TIPO_SUB:
        if want in found:
            c["tipo_ok"] = True
            return "ok"
        if found <= {"casa"}:
            c["tipo_flag"] = f"O anúncio indica apenas “casa” — confirme se é {want}"
            return "generic"
        return "mismatch"
    if found & TIPO_ACEITA.get(want, {want}):
        c["tipo_ok"] = True
        return "ok"
    return "mismatch"


def reject_reason(c, p, final=False):
    if c.get("dead"):
        return "anúncio fora do ar (404/410)"
    if tipo_status(c, p) == "mismatch":
        return f"tipo diferente do pedido ({main_tipo(set(c['tipos']))} ≠ {p['tipo']})"
    a, pr = c.get("area"), c.get("preco")
    min_area = int(p["area_m2"]) - AREA_TOL_DOWN
    if a and a < min_area:
        return f"área {a} m² abaixo do mínimo ({min_area} m²)"
    if pr and pr > p["valor_max"] * (1 + PRICE_TOL):
        return f"preço {brl(pr)} acima do limite"
    if final and not city_confirmed(c, p):
        return f"cidade não confirmada (esperado: {p['cidade']})"
    return ""


def money_html(v, fin):
    """Valor abreviado e sem corte: R$ 1,1MM / R$ 850 mil / R$ 4.500."""
    if not v:
        return '<span class="muted">Não informado</span>'
    exato = brl(v)
    if v >= 1_000_000:
        n = f"{v / 1e6:.2f}".rstrip("0").rstrip(".").replace(".", ",")
        body = f"R$&nbsp;{n}<small>MM</small>"
    elif v >= 10_000:
        n = f"{v / 1e3:.1f}".rstrip("0").rstrip(".").replace(".", ",")
        body = f"R$&nbsp;{n}<small>mil</small>"
    else:
        body = f"R$&nbsp;{v:,.0f}".replace(",", ".")
    return f'<span title="{exato}">{body}</span>'


def card_html(i, o, fin):
    e = lambda x: htmllib.escape(str(x), quote=True)
    area = o.get("area_m2")
    score = min(max(int(o.get("aderencia") or 0), 0), 100)
    quartos = o.get("quartos")
    parts = [
        '<div class="card"><div class="card-top">',
        f'<div class="rank">{i}</div><h4>{e(o.get("titulo") or "Imóvel")}</h4>',
        f'<span class="pill">{e(o["fonte"])}</span></div>',
        '<div class="stats">',
        f'<div class="stat"><div class="lbl">{"Aluguel" if fin == "alugar" else "Valor"}</div>'
        f'<div class="val">{money_html(o.get("preco"), fin)}</div></div>',
        '<div class="stat"><div class="lbl">Área</div><div class="val">'
        + (f"{area}<small>m²</small>" if area else '<span class="muted">Não confirmada</span>') + "</div></div>",
        '<div class="stat"><div class="lbl">Quartos</div><div class="val">'
        + (f"{int(quartos)}" if quartos else '<span class="muted">—</span>') + "</div></div>",
        "</div>",
        f'<div class="score">Aderência ao pedido · <b>{score}%</b></div><div class="bar"><i style="width:{score}%"></i></div>',
        f'<div class="why">{e(o.get("motivo", ""))}</div>',
    ]
    if o.get("tipo_detectado"):
        parts.append(f'<div class="tags"><span class="pill">🏷️ {e(o["tipo_detectado"])}</span></div>')
    if o.get("tipo_flag"):
        parts.append(f'<div class="alert">⚠️ {e(o["tipo_flag"])}</div>')
    if not o.get("preco"):
        parts.append('<div class="warn">💲 Valor não identificado no anúncio — confirme no link.</div>')
    for a in o.get("alertas") or []:
        parts.append(f'<div class="alert">⚠️ {e(a)}</div>')
    parts.append(f'<a class="cta" href="{e(o["url"])}" target="_blank" rel="noopener">Abrir anúncio ↗</a></div>')
    return "".join(parts)


def usage_load():
    d = {"budget": {}, "since": {}, "runs": []}
    try:
        d.update(json.loads(USAGE_FILE.read_text()))
    except Exception:
        pass
    return d


def usage_save(d):
    try:
        d["runs"] = d["runs"][-500:]
        USAGE_FILE.write_text(json.dumps(d, ensure_ascii=False))
    except Exception:
        pass  # filesystem somente leitura: segue sem persistir


PHASES = {"extract": "IA · entender pedido", "queries": "Busca web", "search": "Busca web",
          "native": "Busca nativa (IA)", "rank": "Ranking", "enrich": "Leitura dos anúncios",
          "select": "IA · escolher as 3", "answer": "IA · responder", "ask": "Resposta", "compose": "Resposta",
          "fail": "Resposta"}


def compute_eff(state, pin, pout):
    log = state["llm_log"]
    tin, tout = sum(x["tokens_in"] for x in log), sum(x["tokens_out"] for x in log)
    searches = sum(x.get("web_searches", 0) for x in log)
    # OpenAI cobra um bloco fixo de ~8.000 tokens de entrada por busca nos modelos "mini" (estimativa)
    extra_in = 8000 * searches if state["provider"] == "OpenAI" and "mini" in state["model"].lower() else 0
    cost_tok = (tin + extra_in) * pin / 1e6 + tout * pout / 1e6
    cost_search = searches * SEARCH_FEE
    phases = {}
    for t in state["trace"]:
        k = PHASES.get(t["nó"], t["nó"])
        phases[k] = phases.get(k, 0) + t["ms"]
    busca = sum(phases.get(k, 0) for k in ("Busca web", "Busca nativa (IA)", "Leitura dos anúncios"))
    return {"ts": time.time(), "provider": state["provider"], "model": state["model"],
            "cost_usd": cost_tok + cost_search, "cost_tokens": cost_tok, "cost_search": cost_search,
            "tin": tin, "tout": tout, "searches": searches, "llm_calls": len([x for x in log if x["nó"] != "native"]),
            "wall_ms": state.get("wall_ms") or sum(phases.values()), "busca_ms": busca, "phases": phases,
            "opcoes": len(state["top3"]), "ddg_queries": len(state["search_log"]),
            "cache_hits": sum(1 for x in state["search_log"] if x.get("cache"))}


def usd(v):
    return "US$&nbsp;" + (f"{v:.4f}" if v < 0.1 else f"{v:.2f}").replace(".", ",")


def brl2(v):
    return "R$&nbsp;" + (f"{v:.3f}" if v < 0.1 else f"{v:.2f}").replace(".", ",")


def secs(ms):
    return f"{ms / 1000:.1f}<small>s</small>"


def tiles(items):
    h = '<div class="stats">'
    for lbl, val, sub in items:
        h += f'<div class="stat"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
        h += (f'<div class="score">{sub}</div>' if sub else "") + "</div>"
    return h + "</div>"


# ---------------------------------------------------------------------------
# Avaliação de pagamento (à vista, financiamento, consórcio)
# ---------------------------------------------------------------------------
RATES_DATE_DEFAULT = "out/2026 (reportagens)"
BANKS_DEFAULT = [  # taxas mínimas nominais a.a. + TR; dependem de renda, relacionamento e entrada
    {"Banco": "Caixa", "Taxa a.a. (%)": 10.26, "Obs": "mín. c/ relacionamento, jun/2026; reajuste não confirmado"},
    {"Banco": "Banco do Brasil", "Taxa a.a. (%)": 11.60, "Obs": "mín., jun/2026; reajuste não confirmado"},
    {"Banco": "Santander", "Taxa a.a. (%)": 12.49, "Obs": "reajuste de out/2026"},
    {"Banco": "Itaú", "Taxa a.a. (%)": 12.95, "Obs": "reajuste de out/2026"},
    {"Banco": "Bradesco", "Taxa a.a. (%)": 15.00, "Obs": "reajuste de out/2026"},
]
CONS_DEFAULT = [  # taxa de administração total; fundo de reserva típico de 1% a 3%
    {"Administradora": "Caixa Consórcios", "Taxa adm. (%)": 18.0, "Prazo máx. (meses)": 200, "Fundo reserva (%)": 2.0},
    {"Administradora": "Porto Seguro", "Taxa adm. (%)": 20.0, "Prazo máx. (meses)": 200, "Fundo reserva (%)": 2.0},
    {"Administradora": "Ademicon", "Taxa adm. (%)": 18.0, "Prazo máx. (meses)": 180, "Fundo reserva (%)": 2.0},
    {"Administradora": "Itaú", "Taxa adm. (%)": 19.0, "Prazo máx. (meses)": 200, "Fundo reserva (%)": 2.0},
    {"Administradora": "Sicredi", "Taxa adm. (%)": 17.0, "Prazo máx. (meses)": 180, "Fundo reserva (%)": 2.0},
]


def _npv(flows, r):
    return sum(cf / (1 + r) ** t for t, cf in flows)


def irr_monthly(flows):
    """Menor taxa mensal > 0 que zera o fluxo [(mês, valor)]. Varre uma grade e refina por bisseção
    (o fluxo do consórcio tem várias trocas de sinal, então a bisseção direta não serve)."""
    if _npv(flows, 0.0) >= 0:
        return 0.0
    prev = 0.0
    for k in range(1, 401):  # até 20% ao mês, passos de 0,05%
        r = k * 0.0005
        if _npv(flows, r) >= 0:
            lo, hi = prev, r
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if _npv(flows, mid) < 0 else (lo, mid)
            return (lo + hi) / 2
        prev = r
    return 0.0


def annualize(r_month):
    return ((1 + r_month) ** 12 - 1) * 100


def fin_calc(pv, n, taxa_aa, sistema, seg_pct_m, adm_m, tarifa):
    """Financiamento SAC ou Price. Taxa nominal a.a. (parcelas mensais com juros = taxa/12). Retorna resumo."""
    i = taxa_aa / 100 / 12
    pmt = (pv * i / (1 - (1 + i) ** -n)) if (sistema == "Price" and i > 0) else pv / n
    saldo, parcelas, juros_tot, seg_tot = pv, [], 0.0, 0.0
    for _ in range(n):
        juros, seg = saldo * i, saldo * seg_pct_m / 100
        amort = pv / n if sistema == "SAC" else pmt - juros
        parcelas.append(amort + juros + seg + adm_m)
        juros_tot += juros
        seg_tot += seg
        saldo -= amort
    flows = [(0, pv - tarifa)] + [(t + 1, -p) for t, p in enumerate(parcelas)]
    return {"primeira": parcelas[0], "ultima": parcelas[-1], "total_parcelas": sum(parcelas), "juros": juros_tot,
            "seguros": seg_tot + adm_m * n, "cet": annualize(irr_monthly(flows))}


def cons_calc(carta, n, adm_pct, fr_pct, seg_pct_m, mes_cont):
    """Consórcio: parcela = (crédito + taxa adm + fundo de reserva)/prazo + seguro. CET considera a contemplação no mês informado."""
    if carta <= 0 or n < 2:
        return None
    adm, fr = carta * adm_pct / 100, carta * fr_pct / 100
    parcela = (carta + adm + fr) / n + carta * seg_pct_m / 100
    m = min(max(int(mes_cont), 1), n - 1)
    flows = [(m, carta)] + [(t, -parcela) for t in range(1, n + 1)]
    return {"parcela": parcela, "total": parcela * n, "adm": adm, "fr": fr, "mes": m,
            "cet": annualize(irr_monthly(flows))}


CART_FILE = Path(__file__).with_name("cart.json")


def cart_load():
    try:
        return json.loads(CART_FILE.read_text())
    except Exception:
        return []


def cart_save(items):
    try:
        CART_FILE.write_text(json.dumps(items, ensure_ascii=False))
    except Exception:
        pass  # filesystem somente leitura: o carrinho vale só na sessão (use o CSV para guardar)


def cart_key(url):
    return "cart_" + hashlib.md5(url.encode()).hexdigest()[:10]


CSS = """<style>
:root{--acc:#6366f1;--acc2:#0ea5e9;--line:rgba(128,128,128,.25);--soft:rgba(128,128,128,.07)}
.block-container{padding-top:2.2rem;max-width:1400px}
.hero{display:flex;align-items:center;gap:.9rem;margin:0 0 .8rem 0}
.hero .logo{font-size:1.6rem;width:3rem;height:3rem;display:grid;place-items:center;border-radius:14px;
 background:linear-gradient(135deg,var(--acc),var(--acc2));box-shadow:0 6px 18px rgba(99,102,241,.35)}
.hero h1{font-size:1.65rem;margin:0;padding:0;font-weight:700;letter-spacing:-.02em;line-height:1.2}
.hero p{margin:.1rem 0 0 0;opacity:.65;font-size:.9rem}
.card{border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;margin:.2rem 0 .9rem 0;background:var(--soft);
 transition:box-shadow .2s,transform .2s}
.card:hover{box-shadow:0 8px 24px rgba(0,0,0,.14);transform:translateY(-1px)}
.card-top{display:flex;align-items:center;gap:.6rem}
.card h4{margin:0;font-size:1rem;line-height:1.3;font-weight:600;padding:0;flex:1;min-width:0}
.rank{width:1.7rem;height:1.7rem;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:.85rem;
 color:#fff;background:linear-gradient(135deg,var(--acc),var(--acc2));flex:none}
.pill{font-size:.7rem;padding:.15rem .55rem;border-radius:999px;border:1px solid var(--line);opacity:.85;white-space:nowrap;flex:none}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:.5rem;margin:.8rem 0}
.stat{border:1px solid var(--line);border-radius:12px;padding:.5rem .65rem;background:rgba(128,128,128,.05);min-width:0}
.stat .lbl{font-size:.66rem;text-transform:uppercase;letter-spacing:.07em;opacity:.6}
.stat .val{font-size:1.08rem;font-weight:700;line-height:1.35;overflow-wrap:anywhere}
.val small{font-size:.62em;font-weight:600;opacity:.75;margin-left:.1rem}
.muted{font-weight:500;opacity:.55;font-size:.82rem}
.score{font-size:.75rem;opacity:.75}
.tags{margin:.3rem 0}
.bar{height:6px;border-radius:99px;background:var(--line);overflow:hidden;margin:.25rem 0 .1rem}
.bar>i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--acc),var(--acc2))}
.why{font-size:.9rem;margin:.6rem 0 .3rem 0;line-height:1.45}
.warn{font-size:.82rem;margin:.4rem 0;padding:.45rem .65rem;border-radius:10px;background:rgba(245,158,11,.14);
 border:1px solid rgba(245,158,11,.4)}
.alert{font-size:.8rem;opacity:.8;margin:.2rem 0}
.cta{display:block;text-align:center;margin-top:.8rem;padding:.55rem;border-radius:10px;text-decoration:none!important;
 font-weight:600;color:#fff!important;background:linear-gradient(135deg,var(--acc),var(--acc2));transition:filter .2s}
.cta:hover{filter:brightness(1.1)}
.empty{border:1.5px dashed var(--line);border-radius:16px;padding:2.2rem 1rem;text-align:center;opacity:.7}
[data-testid="stChatMessage"]{border-radius:14px}
button[role="tab"]{font-weight:600}
[data-testid="stSidebar"] h2{font-size:1.1rem}
</style>"""


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


def ddg(query, n=8):
    """Busca no DuckDuckGo. 'No results found' = busca vazia (não é falha): devolve [] para o chamador tentar outra variante."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    last, answered = None, False
    for backend in ("auto", "bing"):  # se um backend vier vazio/der timeout, tenta o próximo
        try:
            with DDGS(timeout=10) as d:
                try:
                    res = list(d.text(query, region="br-pt", max_results=n, backend=backend))
                except TypeError:  # versão antiga da lib, sem parâmetro backend
                    res = list(d.text(query, region="br-pt", max_results=n))
            answered = True
            if res:
                return res
        except Exception as e:
            if "no results" in str(e).lower():
                answered = True
                continue
            last = e
            time.sleep(1)
    if last and not answered:
        raise last  # só é erro se nenhum backend chegou a responder
    return []


JSON_PRICE = [re.compile(x, re.I) for x in (
    r'"(?:rentPrice|salePrice|price|totalCost|rent|listingPrice)"\s*:\s*"?(\d{3,9})',
    r'product:price:amount"[^>]*content="(\d[\d.]*)"', r'itemprop="price"[^>]*content="(\d[\d.]*)"')]
JSON_AREA = [re.compile(x, re.I) for x in (
    r'"(?:usableAreas?|area|totalArea|livingArea|useableArea)"\s*:\s*\[?\s*"?(\d{2,4})',
    r'itemprop="floorSize"[^>]*content="(\d{2,4})')]


def fetch_meta(url):
    r = requests.get(url, headers=UA, timeout=8)
    out = {"status": r.status_code, "title": "", "desc": "", "preco": None, "area": None, "origem": ""}
    if r.status_code >= 400:
        return out
    html = r.text[:700_000]
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    desc = re.search(r'<meta[^>]+(?:og:description|name="description")[^>]+content="([^"]+)"', html, re.I)
    out["title"] = (title.group(1).strip() if title else "")[:160]
    out["desc"] = (desc.group(1).strip() if desc else "")[:300]
    head = f"{out['title']} {out['desc']}"
    body = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", html, flags=re.S | re.I)
    body = re.sub(r"\s+", " ", body)[:3000]
    # ordem de confiança: título/descrição -> JSON embutido -> corpo da página
    pr = first_price(head)
    out["origem"] = "título/descrição" if pr else ""
    if not pr:
        for rx in JSON_PRICE:
            m = rx.search(html)
            if m:
                pr, out["origem"] = int(float(m.group(1))), "JSON da página"
                break
    if not pr:
        pr = first_price(body)
        out["origem"] = "corpo da página" if pr else ""
    ar = first_area(head)
    if not ar:
        for rx in JSON_AREA:
            m = rx.search(html)
            if m:
                ar = int(m.group(1))
                break
    out["preco"], out["area"] = pr, ar or first_area(body)
    return out


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
        "'3 mil' = 3000), finalidade ('alugar' ou 'comprar' ou null), area_m2 (número), quartos (número), vagas (número), tipo ('apartamento', 'casa', 'casa de vila', 'sobrado', 'casa de condomínio', 'cobertura', 'studio/kitnet', 'terreno' ou null), "
        "extras (texto curto com outras preferências, ex: aceita pet, andar alto). Use null para o que não foi informado."
    )
    try:
        data = parse_json(llm(s, "extract", system, user, max_tokens=1500, as_json=True))
    except Exception as e:
        s["errors"].append(f"[extract] fallback regex ({e}){hint(e)}")
        data = {"intent": "busca", "valor_max": first_price(msg), "area_m2": first_area(msg), "tipo": canon_tipo(msg)}
    for k in ["bairro", "cidade", "valor_max", "finalidade", "area_m2", "quartos", "vagas", "extras"]:
        v = data.get(k)
        if v not in (None, "", 0):
            prof[k] = v
    if canon_tipo(data.get("tipo")):
        prof["tipo"] = canon_tipo(data["tipo"])
    if data.get("finalidade"):
        prof["_finalidade_inferida"] = False
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
    if "_finalidade_inferida" not in p:
        p["_finalidade_inferida"] = not p.get("finalidade")
    fin = p.get("finalidade") or ("alugar" if p["valor_max"] <= 20000 else "comprar")
    p["finalidade"] = fin
    tq = TIPO_QUERY.get(p.get("tipo"), p.get("tipo") or "imóvel")
    base = f"{fin} {tq} {p['bairro']} {p['cidade']}"
    area, val = int(p["area_m2"]), int(p["valor_max"])
    extras = f" {int(p['quartos'])} quartos" if p.get("quartos") else ""
    qs = []
    for name in s["sites"]:
        dom = SITES[name]
        path = SITE_PATH.get(dom, "")
        # variantes da mais específica para a mais simples: se uma vier vazia, tenta a próxima
        v = ([f"site:{dom}{path} {base} {area} m2{extras}"] if mode == 0 else [])
        v += [f"site:{dom}{path} {base}", f"site:{dom} {base}", f"{base} {name}"]
        qs.append((name, list(dict.fromkeys(v))))
    web = ([f"{base} {area} m2 até R$ {val}{extras}"] if mode == 0 else []) + [f"{base} até R$ {val}", f"{base} anúncio"]
    qs.append(("Web", web))
    s["queries"] = qs
    s["attempt"] += 1
    return f"{len(qs)} buscas, até {max(len(v) for _, v in qs)} variantes cada (modo {'relaxado' if mode else 'estrito'})"


def n_search(s):
    def one(item):
        name, variants = item
        res, log = [], []
        for i, q in enumerate(variants):
            try:
                res, hit = cached("ddg:" + q, lambda q=q: ddg(q))
                if not res:
                    CACHE.pop("ddg:" + q, None)  # não guarda busca vazia no cache
                log.append({"site": name, "tentativa": i + 1, "query": q, "resultados": len(res), "cache": hit, "erro": ""})
                if res:
                    break  # achou resultados: não precisa das variantes mais simples
            except Exception as e:  # erro real (timeout/bloqueio): não insiste nesta busca
                log.append({"site": name, "tentativa": i + 1, "query": q, "resultados": 0, "cache": False,
                            "erro": f"{type(e).__name__}: {str(e)[:120]}"})
                break
        return res, log

    hits, errs = 0, []
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for res, log in ex.map(one, s["queries"]):
            s["search_log"].extend(log)
            hits += sum(1 for x in log if x["cache"])
            errs += [x["erro"] for x in log if x["erro"]]
            for r in res:
                url = r.get("href", "")
                if url and url not in s["raw"]:
                    s["raw"][url] = {"url": url, "fonte": site_name(url), "titulo": r.get("title", ""),
                                     "trecho": r.get("body", "")}
    if errs and not s["raw"] and not (s["native_fb"] and not s["native_used"]):
        raise RuntimeError(f"Todas as buscas falharam ({len(errs)}). Primeiro erro: {errs[0]}")
    return f"{len(s['raw'])} URLs únicas ({hits} do cache, {len(errs)} falhas, {len(s['search_log'])} tentativas)"


def r_after_search(s):
    # DuckDuckGo não trouxe nada (bloqueio/timeout) -> busca nativa da IA, no máximo 1 vez
    return "native" if (not s["raw"] and s["native_fb"] and not s["native_used"]) else "rank"


def n_native(s):
    s["native_used"] = True
    p = s["profile"]
    domains = [SITES[n] for n in s["sites"]]
    crit = {k: v for k, v in p.items() if not k.startswith("_")}
    min_area = int(p["area_m2"]) - AREA_TOL_DOWN
    system = ("Você é um assistente de busca imobiliária no Brasil. Use a ferramenta de busca web para achar anúncios "
              "REAIS nos sites indicados. Cada item deve ser a URL da página de UM imóvel específico (nunca página de "
              "busca, listagem, bairro ou home). Nunca invente URLs ou dados. Responda SOMENTE com JSON, sem texto extra.")
    user = (f"Critérios: {json.dumps(crit, ensure_ascii=False)}\nCidade: {p['cidade']} (somente imóveis nesta cidade). Tipo: {p.get('tipo') or 'qualquer'}. Área mínima: {min_area} m². "
            f"Valor máximo: R$ {int(p['valor_max'])}.\nSites: {', '.join(domains)}\n\n"
            'Retorne até 8 anúncios: {"anuncios":[{"url":str,"titulo":str,"preco":number|null,'
            '"area_m2":number|null,"trecho":"resumo curto"}]}. Use null quando não souber.')
    data = parse_json(llm_web(s, "native", system, user, domains))
    n = 0
    for a in data.get("anuncios", [])[:10]:
        url = str(a.get("url", ""))
        if not url.startswith("http") or url in s["raw"] or not is_listing_url(url):
            continue
        s["raw"][url] = {"url": url, "fonte": site_name(url), "titulo": str(a.get("titulo", ""))[:160],
                         "trecho": str(a.get("trecho") or "")[:350], "preco": num(a.get("preco")),
                         "area": num(a.get("area_m2"))}
        n += 1
    return f"{n} anúncios via busca nativa ({s['provider']})"


def n_rank(s):
    p = s["profile"]
    bairro = p["bairro"].lower()
    s["dropped"], cands = [], []
    for c in s["raw"].values():
        if not is_listing_url(c["url"]):
            s["dropped"].append({"url": c["url"], "motivo": "página de busca/listagem, não é um anúncio específico"})
            continue
        text = f"{c['titulo']} {c['trecho']}"
        c["preco"] = ok_price(first_price(text) or c.get("preco"), p["finalidade"])
        c["area"] = first_area(text) or c.get("area")
        why = reject_reason(c, p)
        if why:
            s["dropped"].append({"url": c["url"], "motivo": why})
            continue
        c["score"] = (2 * (bairro in (text + c["url"]).lower()) + 2 * bool(c["preco"]) + 2 * bool(c["area"])
                      + (1 if c["area"] and c["area"] >= p["area_m2"] else 0) + (2 if c.get("tipo_ok") else 0))
        cands.append(c)
    cands.sort(key=lambda c: c["score"], reverse=True)
    s["cands"] = cands[:10]
    return f"{len(s['cands'])} candidatos ({len(s['dropped'])} descartados de {len(s['raw'])})"


def n_enrich(s):
    p = s["profile"]
    todo = [c for c in s["cands"] if not (c.get("preco") and c.get("area") and city_confirmed(c, p)) and "meta" not in c]

    def one(c):
        try:
            meta, _ = cached("meta2:" + c["url"], lambda: fetch_meta(c["url"]))
            return c, meta
        except Exception as e:
            return c, {"erro": str(e)[:80]}

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for c, meta in ex.map(one, todo):
            c["meta"] = meta
            if meta.get("status") in (404, 410):
                c["dead"] = True
            c["preco"] = c.get("preco") or ok_price(meta.get("preco"), p["finalidade"])
            c["area"] = c.get("area") or meta.get("area")
            c["trecho"] = (c["trecho"] + " " + meta.get("desc", ""))[:500]
    keep = []
    for c in s["cands"]:  # reaplica os filtros com os dados novos (área/preço lidos da página)
        why = reject_reason(c, p, final=True)
        if why:
            s["dropped"].append({"url": c["url"], "motivo": why})
        else:
            keep.append(c)
    s["cands"] = keep
    ok = sum(1 for c in todo if "erro" not in c["meta"] and c["meta"].get("status", 0) < 400)
    return f"{ok}/{len(todo)} páginas lidas; {len(keep)} candidatos após filtros"


def n_select(s):
    p = s["profile"]
    pool = s["cands"]
    if not pool:
        s["top3"] = []
        return "sem candidatos após os filtros"
    min_area = int(p["area_m2"]) - AREA_TOL_DOWN
    compact = [{"id": i, "fonte": c["fonte"], "titulo": c["titulo"][:120], "trecho": c["trecho"][:350],
                "preco_detectado": c.get("preco"), "area_detectada": c.get("area"),
                "tipo_detectado": main_tipo(set(c.get("tipos", [])))} for i, c in enumerate(pool)]
    crit = {k: v for k, v in p.items() if not k.startswith("_")}
    system = ("Você é um corretor especialista. Escolha até 3 anúncios que melhor atendem aos critérios, SOMENTE entre "
              "os candidatos fornecidos. Nunca invente dados: use null quando não souber. Responda só JSON.")
    user = (f"Critérios: {json.dumps(crit, ensure_ascii=False)}\n"
            f"REGRAS: área mínima {min_area} m² (nunca escolha área menor); valor máximo R$ {int(p['valor_max'])}; cidade obrigatória: {p['cidade']}; tipo obrigatório: {p.get('tipo') or 'qualquer'}; "
            "prefira anúncios com preço e área confirmados; se não aparecerem nos dados, use null.\n"
            f"Candidatos: {json.dumps(compact, ensure_ascii=False)}\n\n"
            'Formato: {"opcoes":[{"id":int,"titulo":str,"preco":number|null,"area_m2":number|null,"quartos":number|null,'
            '"aderencia":0-100,"motivo":"1-2 frases","alertas":["divergências com o pedido"]}]} ordenado por aderência.')
    data = parse_json(llm(s, "select", system, user, max_tokens=4000, as_json=True))
    out = []
    for o in data.get("opcoes", [])[:3]:
        i = o.get("id")
        if not (isinstance(i, int) and 0 <= i < len(pool)):
            continue
        c = pool[i]  # link e valores detectados vêm do anúncio real, nunca do LLM
        area = c.get("area") or num(o.get("area_m2"))
        preco = c.get("preco") or ok_price(num(o.get("preco")), p["finalidade"])
        if area and area < min_area:
            continue
        out.append({**o, "preco": preco, "area_m2": area, "url": c["url"], "fonte": c["fonte"],
                    "tipo_detectado": main_tipo(set(c.get("tipos", []))), "tipo_flag": c.get("tipo_flag", "")})
    s["top3"] = out
    return f"{len(out)} opções escolhidas"


def r_after_select(s):
    if len(s["top3"]) >= 3:
        return "compose"
    if s["attempt"] < 2:
        return "queries"  # 1ª tentativa insuficiente -> busca relaxada
    if s["native_fb"] and not s["native_used"]:
        return "native"  # ainda faltam opções -> busca nativa da IA (1x)
    return "compose"


def n_compose(s):
    p, n = s["profile"], len(s["top3"])
    min_area = int(p["area_m2"]) - AREA_TOL_DOWN
    if n == 0:
        s["reply"] = (f"Não encontrei anúncios específicos com **mínimo de {min_area} m²** e valor até "
                      f"{brl(p['valor_max'])}. Tente ampliar o valor ou o bairro "
                      "(a aba **Debug** mostra o que foi descartado e por quê).")
        return "nenhuma opção"
    s["reply"] = (f"Encontrei **{n} opç{'ão' if n == 1 else 'ões'}**{(' de **' + p['tipo'] + '**') if p.get('tipo') else ''} em **{p['bairro']}/{p['cidade']}** para "
                  f"**{p['finalidade']}**, até {brl(p['valor_max'])} e **no mínimo {min_area} m²** "
                  f"(pedido: {int(p['area_m2'])} m², tolerância de {AREA_TOL_DOWN} m² abaixo). Detalhes ao lado →")
    sem_preco = sum(1 for o in s["top3"] if not o.get("preco"))
    if sem_preco:
        s["reply"] += (f"\n\n💲 **{sem_preco} de {n} sem valor identificado** — muitos portais carregam o preço via "
                       "JavaScript e não consegui lê-lo. Confirme no link do anúncio.")
    if p.get("_finalidade_inferida"):
        s["reply"] += f"\n\n_Assumi **{p['finalidade']}** pelo valor informado; me diga se for o contrário._"
    if n < 3:
        s["reply"] += "\n\n_Só achei menos de 3 opções confiáveis, mesmo ampliando a busca._"
    s["reply"] += "\n\n⚠️ Dados vêm de busca pública: confirme preço e disponibilidade no anúncio."
    return f"{n} opções, {sem_preco} sem preço"


def n_fail(s):
    s["reply"] = "Tive um problema ao processar seu pedido. Abra a aba **Debug** para ver o erro."
    return "falha"


def build_graph():
    g = Graph(entry="extract")
    g.node("extract", n_extract).edge("extract", r_after_extract)
    g.node("ask", n_ask).edge("ask", "END")
    g.node("answer", n_answer).edge("answer", "END")
    g.node("queries", n_queries).edge("queries", "search")
    g.node("search", n_search).edge("search", r_after_search)
    g.node("native", n_native).edge("native", "rank")
    g.node("rank", n_rank).edge("rank", "enrich")
    g.node("enrich", n_enrich).edge("enrich", "select")
    g.node("select", n_select).edge("select", r_after_select)
    g.node("compose", n_compose).edge("compose", "END")
    g.node("fail", n_fail).edge("fail", "END")
    return g


LABELS = {"extract": "Entendendo o pedido", "queries": "Montando buscas", "search": "Buscando nos sites", "native": "Busca nativa da IA (fallback)",
          "rank": "Filtrando candidatos", "enrich": "Lendo anúncios", "select": "Escolhendo as 3 melhores",
          "answer": "Respondendo", "ask": "Verificando dados", "compose": "Finalizando", "fail": "Tratando erro"}

# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
H = 560  # altura fixa e igual para conversa e painel direito (ambos com scroll próprio)
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("profile", {})
ss.setdefault("top3", [])
ss.setdefault("last_state", None)
ss.setdefault("last_eff", None)
ss.setdefault("cart", cart_load())


def test_key(provider):
    try:
        if provider == "OpenAI":
            from openai import OpenAI

            OpenAI(api_key=secret("OPENAI_API_KEY")).models.list()
        else:
            import anthropic

            anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY")).models.list(limit=1)
        return "✅ chave válida"
    except Exception as e:
        return f"❌ {type(e).__name__}: {str(e)[:200]}{hint(e)}"


with st.sidebar:
    st.header("⚙️ Configurações")
    provider = st.radio("Provedor de IA", ["OpenAI", "Claude"], horizontal=True)
    _sel = st.selectbox(
        "Modelo", MODEL_OPTIONS[provider] + [OTHER], key=f"modelsel_{provider}",
        format_func=lambda m: (f"{m}  ·  US$ {PRICES[m][0]:g} / {PRICES[m][1]:g} por 1M" if m in PRICES else m),
        help="Preços: entrada / saída por 1M de tokens. Modelos menores são mais baratos e rápidos.")
    model = (st.text_input("ID do modelo", key=f"modelcustom_{provider}", placeholder="cole o ID exato do modelo")
             if _sel == OTHER else _sel)
    cidade = st.text_input("Cidade padrão", "São Paulo")
    sites = st.multiselect("Sites", list(SITES), default=list(SITES)[:5])
    with st.expander("🌐 Domínios dos sites"):
        for _n in ("Tamaras", "Pilar Home", "Thura"):
            _d = st.text_input(_n, SITES[_n], key=f"dom_{_n}")
            SITES[_n] = re.sub(r"/.*$", "", re.sub(r"^(https?://)?(www\.)?", "", _d.strip().lower())) or SITES[_n]
        st.caption("O domínio da Thura não foi verificado por mim. Se a busca não achar nada nos sites, confirme o endereço oficial.")
    native_fb = st.toggle("Fallback: busca nativa da IA", value=True,
                          help="Usada se o DuckDuckGo falhar ou faltarem opções. ~US$ 0,01 por busca + tokens.")
    if st.button("🗑️ Nova conversa", use_container_width=True):
        for k in ("messages", "profile", "top3", "last_state"):
            ss.pop(k, None)
        st.rerun()
    with st.expander("💰 Custos e saldo"):
        _m = model.strip() or DEFAULT_MODELS[provider]
        _pi, _po = PRICES.get(_m, (0.0, 0.0))
        pin = st.number_input("Preço entrada (US$/1M tokens)", min_value=0.0, value=float(_pi), step=0.01,
                              format="%.3f", key=f"pin_{provider}_{_m}")
        pout = st.number_input("Preço saída (US$/1M tokens)", min_value=0.0, value=float(_po), step=0.01,
                               format="%.3f", key=f"pout_{provider}_{_m}")
        if _m not in PRICES:
            st.caption("⚠️ Modelo sem preço cadastrado: informe os valores acima para o custo ficar correto.")
        fx = st.number_input("Cotação US$ → R$", min_value=0.0, value=5.50, step=0.05,
                             help="Edite conforme a cotação do dia.")
        _u = usage_load()
        saldo = st.number_input(f"Saldo atual na {provider} (US$)", min_value=0.0, step=1.0,
                                value=float(_u["budget"].get(provider, 0.0)),
                                help="Copie do console de cobrança do provedor. O app desconta cada consulta a partir daí.")
        if saldo != float(_u["budget"].get(provider, 0.0)):
            _u["budget"][provider], _u["since"][provider] = saldo, time.time()
            usage_save(_u)
        if st.button("Zerar contador de gasto", use_container_width=True):
            _u["since"][provider] = time.time()
            usage_save(_u)
            st.toast("Contador zerado")
    with st.expander("🔑 Diagnóstico das chaves"):
        for label, name, prov in (("OpenAI", "OPENAI_API_KEY", "OpenAI"), ("Claude", "ANTHROPIC_API_KEY", "Claude")):
            v, src = secret_info(name)
            st.write(f"**{label}:** " + (f"`{v[:7]}…{v[-4:]}` · {len(v)} caracteres · fonte: {src}" if v
                                         else "❌ não encontrada"))
        if st.button("Testar chaves", use_container_width=True):
            st.write("OpenAI:", test_key("OpenAI"))
            st.write("Claude:", test_key("Claude"))
        if st.button("Limpar cache de buscas", use_container_width=True):
            CACHE.clear()
            st.toast("Cache limpo")

def toggle_cart(k, o):
    """Callback do checkbox de cada card: adiciona/remove do carrinho e persiste."""
    if ss.get(k):
        if all(c["url"] != o["url"] for c in ss.cart):
            pf = ss.profile
            item = {f: o.get(f) for f in ("titulo", "fonte", "preco", "area_m2", "quartos", "aderencia", "motivo", "url",
                                          "tipo_detectado", "tipo_flag")}
            item.update({"finalidade": pf.get("finalidade"), "salvo_em": time.strftime("%d/%m %H:%M"),
                         "busca": f"{pf.get('bairro', '')}/{pf.get('cidade', '')} · {pf.get('finalidade', '')}"})
            ss.cart.append(item)
    else:
        ss.cart = [c for c in ss.cart if c["url"] != o["url"]]
    cart_save(ss.cart)


def pay_scenarios(preco, entrada, banks, cons, anos, n_cons, mes_cont, cota, itbi, desc, seg_f, tarifa, adm_m, tr,
                  seg_c, incluir, inc_cons=0.0):
    """Monta as linhas do comparativo: à vista, financiamento (SAC/Price por banco) e consórcio (por administradora)."""
    import pandas as pd

    custos = preco * itbi / 100 if incluir else 0.0
    preco_vista = preco * (1 - desc / 100)
    rows = [{"Opção": "À vista", "Detalhe": f"desconto de {desc:g}%" if desc else "pagamento integral",
             "Entrada": preco_vista, "1ª parcela": None, "Última": None, "Total": preco_vista + custos, "CET": None,
             "Renda mín.": None, "Posse": "Imediata",
             "Alerta": f"Faltam {brl(preco_vista + custos - entrada)}" if entrada < preco_vista + custos else ""}]
    n_fin, ent_min = anos * 12, preco * (1 - cota / 100)
    ent_fin = max(entrada, ent_min)
    if ent_fin < preco:
        for _, b in banks.dropna(subset=["Banco", "Taxa a.a. (%)"]).iterrows():
            taxa = float(b["Taxa a.a. (%)"]) + tr
            for sis in ("SAC", "Price"):
                r = fin_calc(preco - ent_fin, n_fin, taxa, sis, seg_f, adm_m, tarifa)
                rows.append({"Opção": f"Financiamento {b['Banco']} · {sis}", "Detalhe": f"{taxa:.2f}% a.a. + TR · {anos} anos",
                             "Entrada": ent_fin, "1ª parcela": r["primeira"], "Última": r["ultima"],
                             "Total": ent_fin + r["total_parcelas"] + tarifa + custos, "CET": r["cet"],
                             "Renda mín.": r["primeira"] / 0.30, "Posse": "Imediata",
                             "Alerta": f"Entrada ajustada ao mínimo ({brl(ent_min)})" if entrada < ent_min else ""})
    carta = preco - entrada
    if carta > 0:
        for _, c in cons.dropna(subset=["Administradora", "Taxa adm. (%)"]).iterrows():
            pm, fr = c.get("Prazo máx. (meses)"), c.get("Fundo reserva (%)")
            n = int(min(n_cons, pm if pd.notna(pm) and pm else n_cons))
            r = cons_calc(carta, n, float(c["Taxa adm. (%)"]), float(fr) if pd.notna(fr) else 0.0, seg_c, mes_cont)
            if r:
                rows.append({"Opção": f"Consórcio {c['Administradora']}", "Detalhe": f"adm. {c['Taxa adm. (%)']:g}% · {n} meses",
                             "Entrada": entrada, "1ª parcela": r["parcela"], "Última": r["parcela"],
                             "Total": entrada + r["total"] + custos,
                             "CET": ((1 + r["cet"] / 100) * (1 + inc_cons / 100) - 1) * 100, "Renda mín.": r["parcela"] / 0.30,
                             "Posse": f"~mês {r['mes']} (não garantido)", "Alerta": ""})
    return rows


def refresh_rates(provider, model):
    """Busca taxas atuais na web (busca nativa da IA, paga por busca) e atualiza as tabelas de referência."""
    import pandas as pd

    t0 = time.perf_counter()
    mini = {"provider": provider, "model": model, "llm_log": [], "ratelimit": {}, "trace": [], "search_log": [],
            "top3": []}
    system = ("Você pesquisa condições ATUAIS de crédito imobiliário no Brasil usando a busca web. Nunca invente números: "
              "omita o que não encontrar. Responda SOMENTE com JSON, sem texto extra.")
    user = (f"Hoje é {time.strftime('%d/%m/%Y')}. Pesquise (1) as taxas mínimas atuais de financiamento imobiliário SBPE "
            "(juros nominais ao ano, além da TR) de Caixa, Banco do Brasil, Santander, Itaú e Bradesco e (2) para consórcio "
            "imobiliário a taxa de administração total (%), o prazo máximo (meses) e o fundo de reserva (%) de Caixa, "
            'Porto Seguro, Ademicon, Itaú e Sicredi. Formato: {"data_ref":"mês/ano das fontes","bancos":[{"banco":str,'
            '"taxa_aa":number,"obs":"fonte e mês"}],"consorcios":[{"administradora":str,"taxa_adm":number,'
            '"prazo_max":number,"fundo_reserva":number}]}')
    data = parse_json(llm_web(mini, "taxas", system, user, []))
    banks, cons = [], []
    for b in data.get("bancos", []):
        try:
            if 5 <= float(b["taxa_aa"]) <= 25:
                banks.append({"Banco": str(b["banco"]), "Taxa a.a. (%)": float(b["taxa_aa"]), "Obs": str(b.get("obs", ""))[:80]})
        except (KeyError, TypeError, ValueError):
            pass
    for c in data.get("consorcios", []):
        try:
            if 8 <= float(c["taxa_adm"]) <= 35:
                cons.append({"Administradora": str(c["administradora"]), "Taxa adm. (%)": float(c["taxa_adm"]),
                             "Prazo máx. (meses)": int(c.get("prazo_max") or 180),
                             "Fundo reserva (%)": float(c.get("fundo_reserva") or 2.0)})
        except (KeyError, TypeError, ValueError):
            pass
    if not banks and not cons:
        raise RuntimeError("A busca não retornou taxas utilizáveis. Mantive as taxas anteriores.")
    if banks:
        ss.rates_banks = pd.DataFrame(banks)
        ss.pop("ed_banks", None)
    if cons:
        ss.rates_cons = pd.DataFrame(cons)
        ss.pop("ed_cons", None)
    ss.rates_date = str(data.get("data_ref") or time.strftime("%m/%Y"))
    mini["wall_ms"] = round((time.perf_counter() - t0) * 1000)
    eff = compute_eff(mini, pin, pout)
    eff["kind"] = "taxas"
    u = usage_load()
    u["runs"].append(eff)
    usage_save(u)
    return eff


@st.dialog("💳 Avaliação de pagamento", width="large")
def pay_dialog(item):
    import pandas as pd

    e = htmllib.escape
    kx = cart_key(item["url"])
    st.markdown(f"**{e(item['titulo'])}**  \n`{item['fonte']}` · {item.get('busca', '')}")
    st.link_button("Abrir anúncio ↗", item["url"])
    if item.get("finalidade") == "alugar":
        st.info("Este imóvel é para **aluguel**. À vista, financiamento e consórcio se aplicam à compra.")
        if item.get("preco"):
            st.markdown(tiles([("Aluguel mensal", money_html(item["preco"], None), ""),
                               ("Em 12 meses", money_html(item["preco"] * 12, None), "sem reajuste")]),
                        unsafe_allow_html=True)
        return

    preco0 = float(item.get("preco") or 0)
    c1, c2 = st.columns(2)
    preco = c1.number_input("Valor do imóvel (R$)", min_value=0.0, value=preco0, step=10000.0, format="%.0f",
                            key=f"{kx}_pv", help="Pode ajustar para simular uma negociação.")
    entrada = c2.number_input("Valor de entrada (R$)", min_value=0.0, value=float(round(preco0 * 0.2)), step=5000.0,
                              format="%.0f", key=f"{kx}_ent")
    if preco <= 0:
        st.warning("O anúncio não trouxe o valor. Informe o valor do imóvel acima para simular.")
        return
    st.caption(f"Entrada = {entrada / preco * 100:.1f}% do valor do imóvel.")

    with st.expander("⚙️ Premissas (editáveis)"):
        a1, a2, a3 = st.columns(3)
        anos = a1.slider("Prazo do financiamento (anos)", 5, 35, 30, key="pay_anos")
        n_cons = a2.slider("Prazo do consórcio (meses)", 60, 240, 180, step=12, key="pay_ncons")
        mes_cont = a3.slider("Contemplação estimada (mês)", 1, 60, 12, key="pay_cont",
                             help="Mês em que você recebe a carta de crédito (lance ou sorteio). Não é garantido.")
        b1, b2, b3 = st.columns(3)
        cota = b1.number_input("Financia até (% do imóvel)", 50.0, 100.0, 80.0, 5.0, key="pay_cota")
        itbi = b2.number_input("ITBI + escritura/registro (%)", 0.0, 10.0, 4.5, 0.1, key="pay_itbi")
        desc = b3.number_input("Desconto à vista (%)", 0.0, 30.0, 0.0, 0.5, key="pay_desc")
        d1, d2, d3 = st.columns(3)
        seg_f = d1.number_input("Seguros do financ. (% saldo/mês)", 0.0, 0.5, 0.035, 0.005, format="%.3f", key="pay_segf")
        tarifa = d2.number_input("Avaliação e tarifas (R$)", 0.0, 20000.0, 3500.0, 100.0, key="pay_tar")
        adm_m = d3.number_input("Taxa mensal do contrato (R$)", 0.0, 200.0, 25.0, 5.0, key="pay_admm")
        f1, f2 = st.columns(2)
        tr = f1.number_input("TR estimada (% a.a.)", 0.0, 10.0, 0.0, 0.1, key="pay_tr")
        seg_c = f2.number_input("Seguro do consórcio (% crédito/mês)", 0.0, 0.2, 0.04, 0.005, format="%.3f", key="pay_segc")
        inc_cons = st.number_input("Reajuste anual do consórcio — INCC/IPCA (% a.a., estimativa)", 0.0, 20.0, 5.0, 0.5,
                                   key="pay_inccons", help="Parcelas e carta de crédito são corrigidas todo ano. Entra só no CET, "
                                                           "para comparar com o financiamento em termos nominais.")
        incluir = st.checkbox("Incluir ITBI e escritura/registro no total", value=True, key="pay_inc")

    ss.setdefault("rates_banks", pd.DataFrame(BANKS_DEFAULT))
    ss.setdefault("rates_cons", pd.DataFrame(CONS_DEFAULT))
    ss.setdefault("rates_date", RATES_DATE_DEFAULT)
    with st.expander(f"📊 Taxas de referência ({ss.rates_date})"):
        if st.button("🔄 Atualizar taxas via web (IA)", help="Usa a busca nativa do provedor: ~US$ 0,01 por busca + tokens."):
            try:
                with st.spinner("Pesquisando taxas atuais..."):
                    eff = refresh_rates(provider, (model.strip() or DEFAULT_MODELS[provider]))
                st.success(f"Taxas atualizadas. Custo da atualização: ~{usd(eff['cost_usd'])} ({brl2(eff['cost_usd'] * fx)}).".replace("&nbsp;", " "))
            except Exception as ex:
                st.error(f"Não consegui atualizar: {type(ex).__name__}: {str(ex)[:200]}{hint(ex)}")
        banks = st.data_editor(ss.rates_banks, num_rows="dynamic", hide_index=True, use_container_width=True, key="ed_banks",
                               column_config={"Taxa a.a. (%)": st.column_config.NumberColumn(min_value=0.0, max_value=40.0,
                                                                                            step=0.01, format="%.2f")})
        cons = st.data_editor(ss.rates_cons, num_rows="dynamic", hide_index=True, use_container_width=True, key="ed_cons",
                              column_config={"Taxa adm. (%)": st.column_config.NumberColumn(min_value=0.0, max_value=40.0, step=0.1),
                                             "Prazo máx. (meses)": st.column_config.NumberColumn(min_value=12, max_value=300, step=12),
                                             "Fundo reserva (%)": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.5)})
        st.caption("Financiamento: taxas mínimas (nominal a.a. + TR) em reportagens de out/2026 (Exame/MySide) e jun/2026; "
                   "Caixa e Banco do Brasil podem ter reajustado depois. Consórcio: médias divulgadas pelo Serasa. "
                   "As taxas reais dependem do seu perfil, da renda e da entrada.")

    rows = pay_scenarios(preco, entrada, banks, cons, anos, n_cons, mes_cont, cota, itbi, desc, seg_f, tarifa, adm_m, tr,
                         seg_c, incluir, inc_cons)
    if entrada >= preco:
        st.success("Sua entrada cobre o valor do imóvel: dá para comprar à vista.")

    # ------------------------- resultado
    st.markdown("##### Comparativo")
    pool = [r for r in rows if r["Opção"] != "À vista"]
    best = [("Menor custo total", min(rows, key=lambda r: r["Total"]), "Total")]
    if pool:
        best += [("Menor parcela inicial", min(pool, key=lambda r: r["1ª parcela"]), "1ª parcela"),
                 ("Menor CET a.a.", min(pool, key=lambda r: r["CET"]), "CET")]
    st.markdown(tiles([(lbl, (f"{r[k]:.1f}<small>%</small>" if k == "CET" else money_html(r[k], None)), e(r["Opção"]))
                       for lbl, r, k in best]), unsafe_allow_html=True)
    show = pd.DataFrame([{
        "Opção": r["Opção"], "Detalhe": r["Detalhe"], "Entrada": brl(r["Entrada"]),
        "1ª parcela": brl(r["1ª parcela"]) if r["1ª parcela"] else "—", "Última": brl(r["Última"]) if r["Última"] else "—",
        "Total pago": brl(r["Total"]), "Custo extra": brl(r["Total"] - preco),
        "CET a.a.": f"{r['CET']:.1f}%" if r["CET"] is not None else "—",
        "Renda mín. (30%)": brl(r["Renda mín."]) if r["Renda mín."] else "—", "Posse": r["Posse"], "Alerta": r["Alerta"]}
        for r in sorted(rows, key=lambda r: r["Total"])])
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.bar_chart(pd.DataFrame({"Total pago (R$)": [r["Total"] for r in sorted(rows, key=lambda r: r["Total"])]},
                              index=[r["Opção"] for r in sorted(rows, key=lambda r: r["Total"])]), horizontal=True)
    st.caption("Total pago = entrada + parcelas + tarifas + ITBI/escritura (se marcado), em valores de hoje, sem reajustes "
               "futuros. Custo extra = total pago − valor do imóvel. CET aproximado (o do consórcio inclui o reajuste anual informado; o do financiamento usa a TR estimada). Consórcio não tem juros, mas tem taxa de "
               "administração, fundo de reserva e seguro, e a contemplação (lance ou sorteio) não é garantida. "
               "Simulação com taxas de referência: não substitui a proposta oficial nem aconselhamento financeiro. "
               "Confirme as condições com bancos e administradoras.")


def remove_selected():
    sel = set(ss.get("cart_rm", []))
    ss.cart = [c for i, c in enumerate(ss.cart) if f"{i + 1}. {c['titulo'][:50]}" not in sel]
    ss.cart_rm = []
    cart_save(ss.cart)


def clear_cart():
    ss.cart, ss.cart_rm = [], []
    cart_save([])


st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="logo">🏠</div><div><h1>Agente Imobiliário</h1>'
            '<p>Informe bairro, valor e tamanho — busco as 3 melhores opções nos principais portais.</p></div></div>',
            unsafe_allow_html=True)
left, right = st.columns(2, gap="large")

with left:
    with st.tabs(["💬 Conversa"])[0]:
        box = st.container(height=H)
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
                "native_fb": native_fb, "native_used": False,
                "ratelimit": {}, "raw": {}, "dropped": [], "cands": [], "queries": [], "search_log": [], "llm_log": [],
                "trace": [], "errors": [], "reply": "", "intent": "", "missing": [],
            }
            with st.status("Trabalhando...", expanded=False) as status:
                _t0 = time.perf_counter()
                build_graph().run(state, on_step=lambda n: status.update(label=LABELS.get(n, n) + "..."))
                state["wall_ms"] = round((time.perf_counter() - _t0) * 1000)
                status.update(label="Pronto", state="error" if state["errors"] else "complete", expanded=False)
            st.markdown(state["reply"])
        ss.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": state["reply"]}]
        ss.profile = {k: v for k, v in state["profile"].items()}
        if state["intent"] == "busca" and not state["missing"]:
            ss.top3 = state["top3"]
        ss.last_state = state
        ss.last_eff = compute_eff(state, pin, pout)
        _u = usage_load()
        _u["runs"].append(ss.last_eff)
        usage_save(_u)
        if state.get("ratelimit"):
            ss.setdefault("rate", {})[state["provider"]] = state["ratelimit"]

with right:
    tab_res, tab_cart, tab_dbg, tab_eff = st.tabs(["🏘️ Resultado", "🛒 Carrinho", "🐞 Debug", "⚡ Eficiência"])

    with tab_res:
        with st.container(height=H):
            if not ss.top3:
                st.markdown('<div class="empty">As 3 melhores opções aparecerão aqui.</div>', unsafe_allow_html=True)
            fin = ss.profile.get("finalidade")
            in_cart = {c["url"] for c in ss.cart}
            for i, o in enumerate(ss.top3, 1):
                st.markdown(card_html(i, o, fin), unsafe_allow_html=True)
                _k = cart_key(o["url"])
                ss[_k] = o["url"] in in_cart  # mantém o checkbox sincronizado com o carrinho
                st.checkbox("🛒 Guardar no carrinho", key=_k, on_change=toggle_cart, args=(_k, o))
                st.write("")

    with tab_cart:
        with st.container(height=H):
            import pandas as pd

            cart = ss.cart
            st.markdown(f"##### Carrinho · {len(cart)} imóve{'l' if len(cart) == 1 else 'is'}")
            if not cart:
                st.markdown('<div class="empty">Marque “Guardar no carrinho” nos resultados para comparar depois.</div>',
                            unsafe_allow_html=True)
            else:
                rows = []
                for c in cart:
                    pr, ar = c.get("preco"), c.get("area_m2")
                    rows.append({"Imóvel": c["titulo"], "Site": c["fonte"], "Tipo": c.get("tipo_detectado") or "—",
                                 "Preço": pr, "Área (m²)": ar, "R$/m²": round(pr / ar) if pr and ar else None,
                                 "Quartos": c.get("quartos"), "Aderência": c.get("aderencia"), "Busca": c.get("busca"),
                                 "Salvo em": c.get("salvo_em"), "Link": c["url"]})
                df = pd.DataFrame(rows)
                esc = lambda t: htmllib.escape(str(t)[:38])
                best = []
                for lbl, col, fn in (("Menor preço", "Preço", min), ("Maior área", "Área (m²)", max),
                                     ("Menor R$/m²", "R$/m²", min)):
                    ok = [(r[col], r["Imóvel"]) for r in rows if r[col]]
                    if ok:
                        v, t = fn(ok, key=lambda x: x[0])
                        val = money_html(v, None) if col != "Área (m²)" else f"{v}<small>m²</small>"
                        best.append((lbl, val, esc(t)))
                if best:
                    st.markdown(tiles(best), unsafe_allow_html=True)
                st.caption("Compare preços apenas entre imóveis da mesma finalidade (aluguel x venda), conforme a coluna Busca.")
                ev = st.dataframe(df, hide_index=True, use_container_width=True, on_select="rerun",
                                  selection_mode="single-row", key="cart_table", column_config={
                    "Preço": st.column_config.NumberColumn(format="R$ %d"),
                    "Área (m²)": st.column_config.NumberColumn(format="%d m²"),
                    "R$/m²": st.column_config.NumberColumn(format="R$ %d"),
                    "Aderência": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                    "Link": st.column_config.LinkColumn("Link", display_text="Abrir ↗"),
                })
                sel = ev.selection.rows if ev else []
                if sel and sel[0] < len(cart):
                    _it = cart[sel[0]]
                    if ss.get("pay_last") != _it["url"]:  # abre ao selecionar (e não reabre a cada rerun)
                        ss.pay_last = _it["url"]
                        pay_dialog(_it)
                else:
                    ss.pay_last = None
                if st.button("💳 Avaliar pagamento do imóvel selecionado", disabled=not sel, use_container_width=True,
                             key="btn_pay"):
                    pay_dialog(cart[sel[0]])
                if not sel:
                    st.caption("Clique em uma linha da tabela para avaliar o pagamento (à vista, financiamento ou consórcio).")
                m2 = [(f"{i + 1}. {r['Imóvel'][:22]}", r["R$/m²"]) for i, r in enumerate(rows) if r["R$/m²"]]
                if len(m2) > 1:
                    st.markdown("##### R$ por m²")
                    st.bar_chart(pd.DataFrame({"R$/m²": [v for _, v in m2]}, index=[k for k, _ in m2]))
                with st.expander("Ver os cartões dos imóveis"):
                    for i, c in enumerate(cart, 1):
                        st.markdown(card_html(i, c, c.get("finalidade")), unsafe_allow_html=True)
                st.multiselect("Remover do carrinho", [f"{i + 1}. {c['titulo'][:50]}" for i, c in enumerate(cart)],
                               key="cart_rm")
                b1, b2 = st.columns(2)
                b1.button("Remover selecionados", on_click=remove_selected, use_container_width=True,
                          disabled=not ss.get("cart_rm"))
                b2.button("Limpar carrinho", on_click=clear_cart, use_container_width=True)
                st.download_button("⬇️ Baixar comparação (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                                   "carrinho_imoveis.csv", "text/csv", use_container_width=True)

    with tab_dbg:
        with st.container(height=H):
            s = ss.last_state
            if not s:
                st.caption("Faça uma busca para ver o debug.")
            else:
                for e in s["errors"]:
                    st.error(e)
                t = st.tabs(["Grafo", "LLM", "Buscas", "Candidatos", "Descartados", "Estado"])
                with t[0]:
                    st.dataframe(s["trace"], use_container_width=True)
                    st.caption(f"Tempo total: {sum(x['ms'] for x in s['trace'])} ms")
                with t[1]:
                    ws = sum(x.get("web_searches", 0) for x in s["llm_log"])
                    st.caption(f"Tokens: in={sum(x['tokens_in'] for x in s['llm_log'])} "
                               f"out={sum(x['tokens_out'] for x in s['llm_log'])} · "
                               f"buscas nativas: {ws} (~US$ {ws * 0.01:.2f} em taxa de busca)")
                    for x in s["llm_log"]:
                        with st.expander(f"{x['nó']} · {x['modelo']} · {x['ms']} ms"):
                            st.json(x)
                with t[2]:
                    st.dataframe(s["search_log"], use_container_width=True)
                with t[3]:
                    st.json([{k: v for k, v in c.items() if k != "trecho"} for c in s["cands"]], expanded=False)
                with t[4]:
                    st.caption("Links descartados e o motivo (listagem, área abaixo do mínimo, preço, fora do ar).")
                    st.dataframe(s["dropped"], use_container_width=True)
                with t[5]:
                    st.json({"perfil": s["profile"], "intent": s["intent"], "faltando": s["missing"],
                             "tentativas": s["attempt"], "native_usada": s["native_used"]}, expanded=False)

    with tab_eff:
        with st.container(height=H):
            import pandas as pd

            e = ss.last_eff
            st.markdown("##### Última consulta")
            if not e:
                st.markdown('<div class="empty">Faça uma busca para ver custo e tempo.</div>', unsafe_allow_html=True)
            else:
                st.markdown(tiles([
                    ("Custo da consulta", usd(e["cost_usd"]), brl2(e["cost_usd"] * fx)),
                    ("Tempo total", secs(e["wall_ms"]), f"{e['llm_calls']} chamada(s) de IA"),
                    ("Tempo de busca", secs(e["busca_ms"]), "web + leitura dos anúncios"),
                    ("Tokens", f"{e['tin'] + e['tout']:,}".replace(",", "."), f"{e['tin']:,} in · {e['tout']:,} out".replace(",", ".")),
                    ("Busca nativa", f"{e['searches']}", usd(e["cost_search"]) + " em taxas"),
                    ("Cache de buscas", f"{e['cache_hits']}/{e['ddg_queries']}", "queries sem custo/espera"),
                ]), unsafe_allow_html=True)
                st.caption(f"Modelo: {e['provider']} · {e['model']} · custo de tokens {usd(e['cost_tokens'])}".replace("&nbsp;", " "))
                ph = e["phases"]
                if ph:
                    st.markdown("##### Onde o tempo foi gasto (ms)")
                    st.bar_chart(pd.DataFrame({"ms": list(ph.values())}, index=list(ph.keys())), horizontal=True)

            _u = usage_load()
            since = _u["since"].get(provider, 0)
            _since_runs = [r for r in _u["runs"] if r["provider"] == provider and r["ts"] >= since]
            runs = [r for r in _since_runs if r.get("kind") != "taxas"]
            extra_taxas = sum(r["cost_usd"] for r in _since_runs if r.get("kind") == "taxas")
            allp = [r for r in _u["runs"] if r["provider"] == provider]
            st.markdown(f"##### Acumulado · {provider}")
            if runs:
                gasto = sum(r["cost_usd"] for r in runs) + extra_taxas
                st.markdown(tiles([
                    ("Consultas", f"{len(runs)}", f"{len(allp)} no histórico"),
                    ("Gasto contabilizado", usd(gasto), brl2(gasto * fx)),
                    ("Custo médio", usd(gasto / len(runs)), "por consulta"),
                    ("Tempo médio", secs(sum(r["wall_ms"] for r in runs) / len(runs)), "por consulta"),
                ]), unsafe_allow_html=True)
                st.bar_chart(pd.DataFrame({"US$ por consulta": [r["cost_usd"] for r in runs[-20:]]}))
            else:
                gasto = extra_taxas
                st.caption("Nenhuma consulta contabilizada para este provedor ainda.")

            st.markdown(f"##### Saldo disponível · {provider}")
            budget = float(_u["budget"].get(provider, 0.0))
            if budget > 0:
                resta = budget - gasto
                st.markdown(tiles([
                    ("Saldo estimado", usd(max(resta, 0)), brl2(max(resta, 0) * fx)),
                    ("Saldo informado", usd(budget), "ponto de partida"),
                    ("Consultas restantes", f"~{int(max(resta, 0) / (gasto / len(runs))):,}".replace(",", ".") if runs and gasto > 0 else "—",
                     "pelo custo médio"),
                ]), unsafe_allow_html=True)
                if resta < budget * 0.1:
                    st.warning("Saldo estimado abaixo de 10% do informado.")
            else:
                st.info("Informe seu saldo atual em **💰 Custos e saldo** (barra lateral). Até onde sei, OpenAI e "
                        "Anthropic não expõem o saldo por chave de API: o app desconta o custo de cada consulta "
                        "do valor que você informar. O saldo real fica no console de cobrança do provedor.")
            rl = ss.get("rate", {}).get(provider)
            if rl:
                st.markdown("##### Limite de uso da API (janela atual)")
                st.markdown(tiles([(k.capitalize(), str(v), "") for k, v in rl.items()]), unsafe_allow_html=True)
                st.caption("Vem dos headers da última chamada; é o limite de taxa, não o saldo financeiro.")
