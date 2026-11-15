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
            r = client.responses.create(tools=[{"type": "web_search"}], **kw)
        except Exception:  # modelos antigos só aceitam a versão preview da ferramenta
            r = client.responses.create(tools=[{"type": "web_search_preview"}], **kw)
        text = r.output_text or ""
        nsearch = sum(1 for o in r.output if getattr(o, "type", "") == "web_search_call")
        stop = str(r.status)
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
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
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3,
                    "allowed_domains": domains}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        nsearch = getattr(getattr(r.usage, "server_tool_use", None), "web_search_requests", 0) or 0
        stop = f"{r.stop_reason} | blocos={[getattr(b, 'type', '?') for b in r.content]}"
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
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


def reject_reason(c, p, final=False):
    if c.get("dead"):
        return "anúncio fora do ar (404/410)"
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
    if not o.get("preco"):
        parts.append('<div class="warn">💲 Valor não identificado no anúncio — confirme no link.</div>')
    for a in o.get("alertas") or []:
        parts.append(f'<div class="alert">⚠️ {e(a)}</div>')
    parts.append(f'<a class="cta" href="{e(o["url"])}" target="_blank" rel="noopener">Abrir anúncio ↗</a></div>')
    return "".join(parts)


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
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    last = None
    for backend in ("auto", "bing"):  # se um backend der timeout/bloqueio, tenta o próximo
        try:
            with DDGS(timeout=10) as d:
                try:
                    res = list(d.text(query, region="br-pt", max_results=n, backend=backend))
                except TypeError:  # versão antiga da lib, sem parâmetro backend
                    res = list(d.text(query, region="br-pt", max_results=n))
            if res:
                return res
        except Exception as e:
            last = e
            time.sleep(1)
    if last:
        raise last
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
        "'3 mil' = 3000), finalidade ('alugar' ou 'comprar' ou null), area_m2 (número), quartos (número), vagas (número), "
        "extras (texto curto com outras preferências, ex: aceita pet, andar alto). Use null para o que não foi informado."
    )
    try:
        data = parse_json(llm(s, "extract", system, user, max_tokens=1500, as_json=True))
    except Exception as e:
        s["errors"].append(f"[extract] fallback regex ({e}){hint(e)}")
        data = {"intent": "busca", "valor_max": first_price(msg), "area_m2": first_area(msg)}
    for k in ["bairro", "cidade", "valor_max", "finalidade", "area_m2", "quartos", "vagas", "extras"]:
        v = data.get(k)
        if v not in (None, "", 0):
            prof[k] = v
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
    base = f"{fin} apartamento {p['bairro']} {p['cidade']}"
    extras = f" {int(p['quartos'])} quartos" if p.get("quartos") else ""
    qs = []
    for name in s["sites"]:
        dom = SITES[name]
        site = f"site:{dom}{SITE_PATH.get(dom, '')}"  # foca em páginas de imóvel específico
        qs.append((name, f"{site} {base} {int(p['area_m2'])} m²{extras}" if mode == 0 else f"{site} {base}"))
    if mode == 0:
        qs.append(("Web", f"{base} {int(p['area_m2'])} m² até R$ {int(p['valor_max'])}{extras}"))
    else:
        qs.append(("Web", f"imóvel {fin} {p['bairro']} {p['cidade']} até R$ {int(p['valor_max'])}"))
    s["queries"] = qs
    s["attempt"] += 1
    return f"{len(qs)} queries (modo {'relaxado' if mode else 'estrito'})"


def n_search(s):
    def one(item):
        name, q = item
        try:
            res, hit = cached("ddg:" + q, lambda: ddg(q))
            if not res:
                CACHE.pop("ddg:" + q, None)  # não guarda busca vazia no cache
            return name, q, res, hit, ""
        except Exception as e:
            return name, q, [], False, f"{type(e).__name__}: {str(e)[:120]}"

    hits, errs = 0, []
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for name, q, res, hit, err in ex.map(one, s["queries"]):
            hits += hit
            if err:
                errs.append(err)
            s["search_log"].append({"query": q, "resultados": len(res), "cache": hit, "erro": err})
            for r in res:
                url = r.get("href", "")
                if url and url not in s["raw"]:
                    s["raw"][url] = {"url": url, "fonte": site_name(url), "titulo": r.get("title", ""),
                                     "trecho": r.get("body", "")}
    if errs and not s["raw"] and not (s["native_fb"] and not s["native_used"]):
        raise RuntimeError(f"Todas as buscas falharam ({len(errs)}). Primeiro erro: {errs[0]}")
    return f"{len(s['raw'])} URLs únicas ({hits} do cache, {len(errs)} falhas)"


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
    user = (f"Critérios: {json.dumps(crit, ensure_ascii=False)}\nCidade: {p['cidade']} (somente imóveis nesta cidade). Área mínima: {min_area} m². "
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
                      + (1 if c["area"] and c["area"] >= p["area_m2"] else 0))
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
                "preco_detectado": c.get("preco"), "area_detectada": c.get("area")} for i, c in enumerate(pool)]
    crit = {k: v for k, v in p.items() if not k.startswith("_")}
    system = ("Você é um corretor especialista. Escolha até 3 anúncios que melhor atendem aos critérios, SOMENTE entre "
              "os candidatos fornecidos. Nunca invente dados: use null quando não souber. Responda só JSON.")
    user = (f"Critérios: {json.dumps(crit, ensure_ascii=False)}\n"
            f"REGRAS: área mínima {min_area} m² (nunca escolha área menor); valor máximo R$ {int(p['valor_max'])}; cidade obrigatória: {p['cidade']}; "
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
        out.append({**o, "preco": preco, "area_m2": area, "url": c["url"], "fonte": c["fonte"]})
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
    s["reply"] = (f"Encontrei **{n} opç{'ão' if n == 1 else 'ões'}** em **{p['bairro']}/{p['cidade']}** para "
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
    model = st.text_input("Modelo", DEFAULT_MODELS[provider], key=f"model_{provider}",
                          help="Use um modelo rápido/barato; o grafo faz poucas chamadas por busca.")
    cidade = st.text_input("Cidade padrão", "São Paulo")
    sites = st.multiselect("Sites", list(SITES), default=list(SITES)[:5])
    native_fb = st.toggle("Fallback: busca nativa da IA", value=True,
                          help="Usada se o DuckDuckGo falhar ou faltarem opções. ~US$ 0,01 por busca + tokens.")
    if st.button("🗑️ Nova conversa", use_container_width=True):
        for k in ("messages", "profile", "top3", "last_state"):
            ss.pop(k, None)
        st.rerun()
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
                "raw": {}, "dropped": [], "cands": [], "queries": [], "search_log": [], "llm_log": [],
                "trace": [], "errors": [], "reply": "", "intent": "", "missing": [],
            }
            with st.status("Trabalhando...", expanded=False) as status:
                build_graph().run(state, on_step=lambda n: status.update(label=LABELS.get(n, n) + "..."))
                status.update(label="Pronto", state="error" if state["errors"] else "complete", expanded=False)
            st.markdown(state["reply"])
        ss.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": state["reply"]}]
        ss.profile = {k: v for k, v in state["profile"].items()}
        if state["intent"] == "busca" and not state["missing"]:
            ss.top3 = state["top3"]
        ss.last_state = state

with right:
    tab_res, tab_dbg = st.tabs(["🏘️ Resultado", "🐞 Debug"])

    with tab_res:
        with st.container(height=H):
            if not ss.top3:
                st.markdown('<div class="empty">As 3 melhores opções aparecerão aqui.</div>', unsafe_allow_html=True)
            fin = ss.profile.get("finalidade")
            for i, o in enumerate(ss.top3, 1):
                st.markdown(card_html(i, o, fin), unsafe_allow_html=True)

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
