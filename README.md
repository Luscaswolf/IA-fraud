# AI Fraud Investigator (POC)

Não é apenas "um modelo que detecta fraude".
É um **sistema que investiga fraude**.

O modelo de ML é só o primeiro filtro. Ele varre ~100 mil transações e aponta:

```
Transaction #182731   Risk = 94%
```

A partir daí entra o **Agente Investigador**, que percorre uma cadeia de
investigação e produz um laudo explicável:

```
Transaction
     ↓
Customer history
     ↓
Previous transactions
     ↓
Device
     ↓
Location
     ↓
Merchant
     ↓
Behavior pattern
     ↓
Similar fraud cases
```

E entrega:

```
Fraud probability: 94%

Principais evidências:
  - valor 6.3x superior à média
  - novo dispositivo
  - localização incomum
  - 4 transações em 2 minutos
  - padrão semelhante a 137 casos anteriores
```

## Por que dados sintéticos?

Em vez de um dataset "caixa-preta" do Kaggle/HuggingFace, geramos **dados
sintéticos com padrões de fraude injetados e rotulados**. Cada fraude carrega o
seu **motivo real** (`fraud_pattern`), então é possível avaliar objetivamente
se o agente encontrou a **causa correta** — não só se acertou o rótulo binário.

> Você pode trocar o gerador por qualquer dataset real de fraude
> (HuggingFace `?other=fraud`, Kaggle credit-card, IEEE-CIS etc.).
> Basta mapear as colunas em `src/features.py`.

### Padrões de fraude injetados (ground truth)

| Padrão              | Descrição                                                    |
|---------------------|--------------------------------------------------------------|
| `high_amount`       | valor muito acima da média do cliente                        |
| `new_device`        | dispositivo nunca visto                                      |
| `impossible_travel` | localização geograficamente impossível no tempo             |
| `velocity_burst`    | várias transações em poucos minutos                          |
| `card_testing`      | rajada de micro-transações (teste de cartão)                |
| `account_takeover`  | novo device + nova localização + valor alto (tomada de conta)|
| `risky_merchant`    | categoria de alto risco em horário atípico                  |

## Arquitetura

```
src/
  config.py         parâmetros, catálogos e lista de padrões de fraude
  generator.py      gera clientes + 100k transações com fraudes rotuladas
  features.py       engenharia de features contextuais (por cliente/tempo)
  feature_store.py  ESTADO online por cliente (janela deslizante) p/ scoring em tempo real
  model.py          modelo de detecção (RandomForest) -> risk score 0-100%
  investigator.py   AGENTE: percorre a cadeia e coleta evidências ponderadas
  calibration.py    calibra os limiares de decisão por CUSTO (fraude × fricção)
  backtest.py       backtest temporal (treino=passado, teste=futuro) + dataset real
  feedback.py       feedback loop: decisão do analista -> rótulo para re-treino
  report.py         renderiza o laudo (texto / rich)
  narrative.py      camada LLM: redige o laudo em linguagem natural
main.py             CLI
api.py              API FastAPI (POST /score em tempo real, POST /api/feedback, ...)
tests/              pytest (features, feature store, agente, calibração, API, backtest)
```

O agente combina o **score do ML** com o **raciocínio sobre evidências**:

```
fraud_probability = 0.5 * risco_ML + 0.5 * score_do_agente
motivo_provável   = padrão com maior contribuição acumulada de evidências
```

## Como rodar

```bash
pip install -r requirements.txt

python main.py generate          # gera dados sintéticos rotulados
python main.py train             # treina o modelo + salva os scores
python main.py demo 5            # investiga as 5 transações de maior risco
python main.py investigate 102914  # investiga uma transação específica
python main.py narrate 102914    # laudo em linguagem natural (LLM/template)
python main.py evaluate 500      # mede se o agente acerta o MOTIVO da fraude
python main.py backtest          # backtest temporal no dataset sintético
python main.py backtest data/real_creditcard.csv   # idem, em dados reais (Sparkov)
python main.py calibrate         # limiares de decisão ótimos por custo -> data/thresholds.json
python main.py retrain           # re-treina incorporando o feedback dos analistas
```

### Scoring em tempo real (`POST /score`)

Além do modo batch, a API pontua **uma transação** contra o estado do cliente
(feature store online) e devolve `aprovar` / `revisar` / `bloquear` em ~50 ms:

```bash
curl -X POST http://127.0.0.1:8000/score -H 'content-type: application/json' -d '{
  "customer_id": 771001, "amount": 1200.00, "merchant_category": "cripto_exchange",
  "device_id": "dev_x", "city": "Miami", "lat": 25.76, "lon": -80.19
}'
```

O estado (média de gasto, dispositivos, localização, janela de velocity) é
mantido em memória por cliente e é **incrementado a cada chamada**
(`update_state: false` desliga). Hoje o store é um `dict`; a interface
`snapshot` / `update` foi desenhada para trocar por Redis sem mexer no resto.

### Feedback loop (`POST /api/feedback`)

```bash
curl -X POST http://127.0.0.1:8000/api/feedback -H 'content-type: application/json' \
  -d '{"transaction_id": 102914, "label": "fraud"}'   # ou "legit"
```

Grava em `data/feedback.jsonl`. `python main.py retrain` reescreve os rótulos
das transações com feedback e retreina o modelo.

### Limiares calibrados por custo

`python main.py calibrate` varre uma grade de limiares sobre o dataset rotulado
e escolhe o par `(revisar, bloquear)` que **minimiza o custo esperado por
transação**, dado o custo de deixar passar uma fraude (valor), o de bloquear um
cliente legítimo (fricção) e o de uma revisão manual. O resultado vai para
`data/thresholds.json` e é consumido automaticamente pelo `narrate` / `/score`.

### Autenticação e rate limit (opcionais)

```bash
export FRAUD_API_KEY=troque-isto     # passa a exigir header  x-api-key  em /api e /score
export FRAUD_RATE_LIMIT=120          # req/min por cliente (default 120; 0 desliga)
```

Uploads passam a ser **persistidos** em `data/uploads/` e recarregados no restart.

### Testes

```bash
pip install -r requirements.txt
pytest -q
```

### Camada LLM (opcional)

O `narrate` gera um laudo em linguagem natural. Sem chave de API ele usa um
**template determinístico** (custo zero). Com uma chave, redige via LLM — mas
o prompt recebe **sempre** as evidências estruturadas, então o modelo nunca
inventa fatos, apenas redige o que o agente apurou:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...      # opcional: OPENAI_MODEL=gpt-4o-mini
# ou Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

python main.py narrate 102914
```

### Datasets reais (Hugging Face)

Além do dataset sintético, o projeto pode baixar e adaptar **dados reais de
fraude** do Hugging Face. Basta rodar:

```bash
python scripts/fetch_datasets.py
```

Isso baixa a base **Sparkov** (`pointe77/credit-card-transaction`) — 200 mil
transações reais de cartão de crédito com clientes recorrentes, categoria,
valor, localização e rótulo `is_fraud` — mapeia para o esquema do projeto,
calcula o risco e salva em `data/real_creditcard_scored.csv`.

Qualquer arquivo `data/real_*_scored.csv` aparece **automaticamente** no seletor
de base de dados da interface. Assim você compara a investigação em dados
sintéticos e em dados reais lado a lado.

### Interface Web (front-end)

```bash
python api.py     # sobe em http://127.0.0.1:8000
```

Abra **http://127.0.0.1:8000** no navegador. A interface permite:

1. Escolher o **dataset** (o de exemplo com 100k transações já vem carregado).
2. Ver a tabela de **transações de maior risco** — clique em qualquer linha
   para o agente investigar e exibir o laudo completo (cadeia de evidências,
   probabilidade, motivo e recomendação).
3. **Enviar seu próprio CSV** (arrastar e soltar). O sistema deriva o perfil de
   cada cliente dos próprios dados, pontua o risco com o modelo treinado e
   compara com a biblioteca de fraudes conhecidas.

**Formato do CSV** (colunas obrigatórias):

```
transaction_id, timestamp, customer_id, amount, merchant_category,
device_id, city, lat, lon
```

Colunas opcionais `is_fraud` e `fraud_pattern` — se presentes, a interface
valida se o agente acertou o motivo real. Baixe um modelo pronto em
`http://127.0.0.1:8000/api/template.csv`.

**Arquivo de teste pronto:** já existe um `sample_transactions.csv` na raiz do
projeto (401 transações reais com 57 fraudes rotuladas). Basta arrastá-lo para
a área de importação. Para gerar um novo:

```bash
python scripts/make_sample.py
```

### API REST

```bash
python api.py     # sobe em http://127.0.0.1:8000  (docs em /docs)
```

| Método | Rota                        | Descrição                              |
|--------|-----------------------------|----------------------------------------|
| GET    | `/health`                   | status + total de transações           |
| GET    | `/transactions/top?n=10`    | transações de maior risco              |
| GET    | `/investigate/{id}`         | laudo estruturado + narrativa          |
| POST   | `/investigate`              | idem, via corpo JSON                   |

Exemplo de resposta (`GET /investigate/102914`):

```json
{
  "transaction_id": 102914,
  "fraud_probability": 1.0,
  "inferred_pattern": "account_takeover",
  "true_pattern": "account_takeover",
  "correct_motive": true,
  "recommendation": "BLOQUEAR a transacao e acionar o cliente.",
  "evidences": [ ... ],
  "similar_cases": [ ... ],
  "narrative": "LAUDO DE INVESTIGACAO ...",
  "narrative_engine": "template"
}
```

## Resultados da POC

Modelo de detecção:

```
ROC-AUC: 0.999    PR-AUC: 0.980
```

Agente (acerto do **motivo real** da fraude, top-500 por risco):

```
Acerto do motivo: 99.8%
  account_takeover   100%
  card_testing       100%
  impossible_travel  100%
  velocity_burst      99%
```

> A métrica-chave desta POC não é "detectou fraude?", e sim
> **"a IA encontrou o motivo correto?"** — que é o que um investigador humano
> precisa para tomar uma decisão.

## Próximos passos

Feito nesta iteração:

- [x] **Scoring em tempo real** (`POST /score`) com feature store online por cliente
  (`src/feature_store.py`) — decisão em ~50 ms, sem varrer o CSV.
- [x] **Janela deslizante** de transações por cliente para velocity / impossible-travel.
- [x] **Backtest temporal** (`main.py backtest`) + **métricas em dataset real** (Sparkov):
  ROC-AUC ~0.985 / PR-AUC ~0.64 no holdout futuro (vs. 0.999 do sintético).
- [x] **Feedback loop** (`POST /api/feedback` + `main.py retrain`).
- [x] **Limiares calibrados por custo** (`main.py calibrate` -> `data/thresholds.json`).
- [x] **Autenticação por API key + rate limit + persistência de uploads**.
- [x] **Suíte de testes** (`pytest`, `tests/`).

Ainda em aberto:

- Trocar o `FeatureStore` em memória por Redis/Cassandra (interface já pronta).
- Persistir o estado do store entre reinícios (hoje só o histórico via `bootstrap`).
- Consumir eventos (Kafka) para atualizar o store fora do caminho da requisição.
- Modelo mais leve / ONNX para derrubar a latência de ~50 ms para 1 dígito.
- Re-treino agendado e monitoramento de drift.
