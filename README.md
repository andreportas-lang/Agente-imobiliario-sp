# Agente Imobiliário (Streamlit + grafo de execução)

App em Python que funciona como um corretor virtual. O usuário informa **bairro, valor e tamanho** do imóvel (e, opcionalmente, quartos, vagas, pet etc.). O agente busca anúncios em portais (QuintoAndar, Loft, Viva Real, Tamaras, ZAP, Imovelweb, Chaves na Mão, OLX, Pilar Home, Thura e outros que encontrar na web) e devolve as **3 opções mais aderentes**, com link direto para o anúncio.

O provedor de IA pode ser **OpenAI** ou **Claude (Anthropic)**, escolhido na barra lateral.

---

## 1. Arquivos

| Arquivo | Função |
|---|---|
| `app.py` | Todo o app: grafo, busca, filtros, chamadas de IA e interface |
| `requirements.txt` | Dependências (`streamlit`, `openai`, `anthropic`, `requests`, `ddgs`) |
| `usage_log.json` | Criado automaticamente. Guarda histórico de custos e saldo informado |
| `cart.json` | Criado automaticamente. Guarda os imóveis do carrinho |

## 2. Como rodar

```
pip install -r requirements.txt
streamlit run app.py
```

**Chaves de API** (Streamlit Community Cloud > Settings > Secrets, ou `.streamlit/secrets.toml` local):

```
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

Se não houver secrets, o app tenta variáveis de ambiente com os mesmos nomes. Espaços e aspas coladas por engano são removidos.

---

## 3. Como o agente funciona (grafo)

O fluxo é um **grafo de nós**: cada nó faz uma tarefa e uma função de roteamento decide o próximo. Só alguns nós usam IA; o resto é código puro, o que reduz custo e tempo.

```
extract ──► (conversa) ──► answer ──► FIM
   │
   ├─► (faltam dados) ──► ask ──► FIM
   │
   └─► queries ──► search ──► (sem resultados?) ──► native ─┐
                      │                                      │
                      └──────────────────────────────────────┴─► rank ─► enrich ─► select
                                                                                   │
                         ┌─ menos de 3 e 1ª tentativa ─► queries (busca relaxada)  │
                         ├─ ainda menos de 3 e fallback ligado ─► native (1x)      │
                         └─ caso contrário ─► compose ─► FIM  ◄────────────────────┘
```

Qualquer erro em um nó desvia para o nó `fail`, que mostra uma mensagem amigável; o erro completo aparece na aba **Debug**.

### Nós

| Nó | O que faz | Usa IA? |
|---|---|---|
| `extract` | Entende a mensagem e extrai bairro, cidade, valor máximo, finalidade (alugar/comprar), **tipo do imóvel**, área, quartos, vagas e extras. Classifica a intenção: nova busca ou conversa sobre as opções atuais. Se a IA falhar, usa regex para valor e área | Sim (curta) |
| `ask` | Se faltar bairro, valor ou tamanho, pergunta ao usuário | Não |
| `answer` | Responde dúvidas sobre as opções já mostradas, sem nova busca | Sim |
| `queries` | Monta as buscas por site (`site:dominio/imovel ...`) mais uma busca aberta na web. Na 1ª tentativa é estrita (inclui tamanho); na 2ª é relaxada (sem tamanho) | Não |
| `search` | Executa as buscas no DuckDuckGo em paralelo (3 simultâneas), com timeout, troca de backend e cache. Cada site tem **variantes** de busca, da mais específica para a mais simples; se uma vier vazia, tenta a próxima | Não |
| `native` | **Fallback**: usa a busca web nativa do provedor (OpenAI ou Claude) para achar anúncios. Roda no máximo 1 vez por pedido | Sim (com busca paga) |
| `rank` | Descarta o que não é anúncio específico, aplica filtros e pontua os candidatos. Mantém até 10 | Não |
| `enrich` | Abre **todas** as páginas dos candidatos (em paralelo) para confirmar que o anúncio está no ar e ler preço, área e título. Reaplica os filtros com os dados novos | Não |
| `select` | Uma chamada de IA que escolhe as 3 melhores entre os candidatos e escreve motivo e alertas | Sim |
| `compose` | Monta a resposta final do chat | Não |
| `fail` | Mensagem de erro amigável | Não |

---

## 4. Regras de qualidade dos resultados

- **Link exato:** só passam URLs de um imóvel específico (ex.: `/imovel/123...`). Páginas de busca ou listagem são descartadas. O link exibido vem sempre do anúncio real encontrado, nunca da IA.
- **Área:** mínimo = área pedida − 10 m² (constante `AREA_TOL_DOWN`). Não há limite superior.
- **Valor:** máximo = valor pedido + 10% (constante `PRICE_TOL`).
- **Tipo de imóvel:** reconhece apartamento, casa, casa de vila, sobrado, casa de condomínio, cobertura, studio/kitnet e terreno. A busca usa o tipo pedido (sem tipo, busca "imóvel"). Anúncios de tipo diferente são descartados (ex.: apartamento numa busca por casa de vila). Se o anúncio indicar só "casa" (sem confirmar vila/sobrado/condomínio) ou não indicar o tipo, ele é mantido com um aviso no card.
- **Cidade:** a cidade pedida precisa aparecer na URL, no título ou na descrição do anúncio. Senão é descartado como "cidade não confirmada".
- **Anúncio fora do ar:** o app abre cada candidato e descarta o que deu 404/410, o que responde "200 OK" com mensagem de *página não encontrada/anúncio expirado/indisponível* e o que redireciona para uma página de busca ou home (comum em ZAP e Viva Real). Anúncios confirmados no ar recebem o selo ✅ e passam na frente no ranking; se houver 3 ou mais confirmados, os não verificados ficam de fora. Alguns sites bloqueiam a checagem automática (403/anti-robô): nesses casos o anúncio pode aparecer sem confirmação, com um aviso no card.
- **Preço plausível:** ignora condomínio e IPTU ao extrair o valor; ignora valores incoerentes com a finalidade (ex.: preço de venda numa busca de aluguel).
- **Sem valor ou área:** o card mostra "Não informado" ou "Não confirmada" e um aviso. A resposta do chat informa quantas opções vieram sem valor.
- **Finalidade:** se o usuário não disser, é inferida pelo valor (até R$ 20 mil = aluguel; acima = compra) e isso é avisado na resposta.
- Todos os descartes, com o motivo, aparecem em **Debug > Descartados**.

## 5. Otimização de custo e velocidade

- Poucas chamadas de IA no caminho normal: `extract` e `select` (2 chamadas), com modelo barato.
- Buscas e leitura de páginas em paralelo.
- **Cache em memória** (1 hora) de buscas e páginas lidas; buscas vazias não são guardadas.
- Pergunta com dados faltando e resposta final não usam IA.
- A busca nativa (paga por busca) só entra como fallback.

---

## 6. Interface

**Barra lateral**
- Provedor (OpenAI ou Claude) e **lista de seleção de modelos** (OpenAI: `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`; Claude: `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-4-5`), com o preço por 1M de tokens ao lado. A opção *Outro (digitar ID)* aceita qualquer modelo. Padrões: `gpt-4o-mini` e `claude-haiku-5-5`.
- Cidade padrão e seleção de sites (por padrão vêm marcados os 5 primeiros; Pilar Home e Thura estão disponíveis na lista). O bloco **🌐 Domínios dos sites** permite editar o endereço de Tamaras, Pilar Home e Thura.
- Liga/desliga do fallback de busca nativa.
- **Nova conversa**.
- **💰 Custos e saldo:** preços por 1M de tokens (editáveis), cotação US$→R$ e saldo atual informado.
- **🔑 Diagnóstico das chaves:** mostra de onde vem cada chave (secrets ou variável de ambiente), tamanho e prefixo/final; o botão *Testar chaves* valida cada uma.

**Esquerda:** conversa, com altura fixa e scroll próprio.

**Direita (abas: Resultado, Carrinho, Debug e Eficiência; mesma altura da conversa, com scroll):**
- **🏘️ Resultado** (padrão): até 3 cards com posição, site de origem, valor abreviado (ex.: R$ 1,1MM, R$ 850 mil), área, quartos, barra de aderência, motivo, alertas e botão "Abrir anúncio".
- **🐞 Debug:** erros; trace do grafo com tempo por nó; chamadas de IA (prompt enviado, tokens, motivo da parada, resposta); buscas e erros; candidatos; descartados com motivo; estado.
- **🛒 Carrinho:** cada card tem a opção *Guardar no carrinho*. O carrinho compara os imóveis salvos em tabela (preço, área, R$/m², quartos, aderência, link), destaca menor preço, maior área e menor R$/m², tem gráfico de R$/m², permite remover itens e exporta CSV. **Ao clicar em uma linha da tabela, abre a janela de avaliação de pagamento** (veja a seção 6.1). É salvo em `cart.json` (pode ser apagado quando o app reinicia no Streamlit Cloud; use o CSV como backup).
- **⚡ Eficiência:**
  - *Última consulta:* custo (US$ e R$), tempo total, tempo de busca, tokens, buscas nativas, cache e gráfico de tempo por etapa.
  - *Acumulado do provedor:* consultas, gasto, custo e tempo médios, gráfico por consulta.
  - *Saldo disponível:* saldo estimado, saldo informado e consultas restantes pelo custo médio.
  - *Limite de uso da API:* requisições e tokens restantes, lidos dos headers da última chamada.

### 6.1 Avaliação de pagamento (janela aberta pelo carrinho)

Vale para imóveis **à venda**. Para aluguel, a janela mostra só o aluguel mensal e o de 12 meses.

Entradas: valor do imóvel (editável, útil para simular negociação ou quando o anúncio não traz preço) e valor da entrada. Premissas editáveis: prazo do financiamento (5 a 35 anos), **amortização extra anual fixa** (R$/ano, paga no 12º mês de cada ano sobre o saldo devedor, com efeito *reduzir prazo* ou *reduzir parcela*; vale só para o financiamento e mostra prazo final e economia em relação ao financiamento sem amortização), prazo do consórcio, mês estimado de contemplação, % financiado (padrão 80%), ITBI + escritura/registro (padrão 4,5%), desconto à vista, seguros, tarifa de avaliação, taxa mensal do contrato, TR estimada, seguro do consórcio e reajuste anual do consórcio (INCC/IPCA).

Cenários calculados e comparados:
- **À vista:** valor com desconto + custos de aquisição; mostra quanto falta em relação à entrada informada.
- **Financiamento** por banco (Caixa, Banco do Brasil, Santander, Itaú, Bradesco), em **SAC** e **Price**: 1ª e última parcela, total pago, custo extra sobre o valor do imóvel, **CET a.a.** e renda mínima (parcela ≤ 30% da renda). Se a entrada for menor que o mínimo exigido (100% − % financiado), ela é ajustada e isso é avisado.
- **Consórcio** por administradora (Caixa, Porto Seguro, Ademicon, Itaú, Sicredi): crédito = valor do imóvel − entrada; parcela = (crédito + taxa de administração + fundo de reserva) / prazo + seguro. O CET considera a contemplação no mês informado e o reajuste anual. A posse do imóvel só ocorre após a contemplação, que **não é garantida** (lance ou sorteio).

A janela destaca o menor custo total, a menor parcela inicial e o menor CET, e mostra tabela e gráfico do total pago.

**Taxas de referência** (editáveis na própria janela): financiamento com taxas mínimas nominais a.a. + TR, segundo reportagens de out/2026 (Exame/MySide: Bradesco 15%, Itaú 12,95%, Santander 12,49%) e de jun/2026 (Caixa 10,26%, Banco do Brasil 11,60%; os dois podem ter reajustado depois). Consórcio: taxas de administração médias divulgadas pelo Serasa (17% a 20%) e fundo de reserva típico de 2%. O botão **Atualizar taxas via web (IA)** pesquisa valores atuais com a busca nativa do provedor (custo de ~US$ 0,01 por busca + tokens, registrado na aba Eficiência).

Limites: valores em moeda de hoje, sem reajustes futuros; as taxas reais dependem de renda, score, relacionamento e entrada; é uma simulação, não substitui a proposta oficial nem aconselhamento financeiro.

### Sobre o saldo
OpenAI e Anthropic, até onde sei, não informam o saldo financeiro por chave de API. O app desconta o custo de cada consulta do saldo que você digitar. É uma **estimativa** (só considera consultas feitas por este app). O saldo real fica no console de cobrança do provedor.

### Como o custo é calculado
`tokens de entrada × preço de entrada + tokens de saída × preço de saída` (por 1M de tokens) `+ US$ 0,01 por busca nativa`. Para modelos "mini" da OpenAI, soma-se uma estimativa de 8.000 tokens de entrada por busca nativa. Os preços ficam na constante `PRICES` e podem ser ajustados na barra lateral.

---

## 7. Configurações no código (`app.py`)

| Constante | Significado |
|---|---|
| `SITES` | Portais disponíveis e seus domínios (Tamaras = `maramores.com.br`, Pilar Home = `pilarhomes.com.br`; o domínio da Thura não foi verificado) |
| `MODEL_OPTIONS` | Modelos exibidos na lista de seleção de cada provedor |
| `DEFAULT_MODELS` | Modelo padrão de cada provedor |
| `AREA_TOL_DOWN` | Tolerância de área abaixo do pedido (10 m²) |
| `PRICE_TOL` | Tolerância de valor acima do pedido (10%) |
| `LISTING_RE` e `SITE_PATH` | Padrões de URL que identificam um anúncio específico, por site |
| `PRICES` | Preços de modelos (US$ por 1M de tokens) |
| `SEARCH_FEE` | Taxa por busca nativa (US$ 0,01) |
| `CACHE_TTL` | Tempo do cache (3600 s) |
| `H` | Altura dos painéis (560 px) |

---

## 8. Limitações conhecidas

- Muitos portais carregam preço e detalhes via JavaScript; o app lê apenas o HTML, então alguns anúncios ficam sem valor (sinalizado no card).
- O DuckDuckGo pode atrasar ou bloquear IPs de nuvem; nesse caso entra o fallback de busca nativa.
- A confirmação de cidade usa URL, título e descrição; anúncios legítimos sem a cidade nesses campos podem ser descartados.
- Os dados vêm de busca pública: **confirme preço e disponibilidade no anúncio**.
- No Streamlit Cloud, `usage_log.json` pode ser apagado quando o app reinicia e é compartilhado entre quem usa o app.
- A busca web nativa da Claude precisa estar habilitada no Console da Anthropic.

## 9. Solução de problemas

| Sintoma | O que fazer |
|---|---|
| Erro 401 | Chave recusada. Use *Diagnóstico das chaves* > *Testar chaves*; confira se a chave é do projeto certo e se há variável de ambiente antiga no Codespaces |
| Resposta vazia da IA | O modelo gastou os tokens "pensando". Veja a aba Debug > LLM (motivo da parada) e aumente `max_tokens` ou troque o modelo |
| `DDGSException: No results found` | Significa busca vazia (não é falha de rede). O app tenta variantes mais simples da busca e, se nada vier, aciona o fallback de busca nativa. Veja Debug > Descartados e a aba Buscas |
| Timeout no DuckDuckGo | Reduza o número de sites, atualize `pip install -U ddgs` ou mantenha o fallback de busca nativa ligado |
| Poucos resultados | Veja Debug > Descartados para entender os filtros; amplie valor, bairro ou tamanho |
| Preços ou modelo sem custo | Informe os preços em *Custos e saldo* |
