# VULGARIS — Company demo playbook

Use this to show a customer or investor that **you built a full industrial AI stack**, not a single notebook model.

---

## Before the meeting (15 minutes)

```powershell
cd D:\Microsoft\Vulgaris
python demo/company_showcase.py
```

You will get:

| Artifact | Purpose |
|----------|---------|
| `demo/output/showcase_report.md` | Leave-behind PDF/print |
| `demo/output/showcase_report.json` | Technical appendix |
| `demo/output/showcase_checkpoint/` | Proof of train + save + SHA-256 weights |

Optional live API:

```powershell
docker compose up
# Browser: http://localhost:8000/health
# Metrics: http://localhost:9090 (Prometheus)
```

---

## 10-minute live script

### 1. Problem (1 min)

> Plants and networks produce millions of multivariate samples per day. Cloud forecast APIs cannot run on the factory floor, cannot explain causality, and cannot adapt per site without months of retraining.

### 2. What we built (2 min)

> **VULGARIS** — one ~1M-parameter model on **CPU**, streaming inference, **causal graph**, **memory for rare faults**, **per-site adapters**, **rules + explainability**, optional **safety head**. No PyTorch. Data stays on-prem.

Run:

```powershell
python demo/company_showcase.py --quick
```

Point at the **DIFFERENTIATORS** table in the terminal.

### 3. Proof it trains and runs (3 min)

- Training completes on **synthetic router/NOC telemetry** (latency, loss, CPU, jitter, etc.).
- **Test accuracy** prints on held-out data.
- **Streaming latency** (ms per `step()`) — edge-realistic path.
- **Causal graph** ASCII — “which latent nodes drive which” (root-cause story).

### 4. What competitors usually cannot demo in one box (2 min)

| We show live | They often need separate products |
|--------------|-----------------------------------|
| `model.step()` O(1) streaming | Batch-only or cloud API |
| `register_domain()` | New model per site |
| CRG causal edges | Anomaly score only |
| ESE rules | Dashboard alert, no IF-THEN |
| Checkpoint + SHA-256 | Ad-hoc pickle files |
| Docker + Prometheus | Custom glue |

Say clearly: **“We are not claiming billion-parameter pretrain; we are claiming the production shape industrial teams ask for and rarely get in one system.”**

### 5. Close (2 min)

> Next step: 4-week pilot on **your** SNMP/PI/OTel channels — same API, your `input_dim`, one domain per site. We integrate ingestion; you keep data inside your network.

Hand over `demo/output/showcase_report.md`.

---

## Answers to hard questions

**“Is this GPT-scale?”**  
No — intentionally **~1M params** for edge. Value is architecture + on-prem + causal + multi-domain, not parameter count.

**“How is this different from Azure/AWS time-series?”**  
They optimize **forecast API in cloud**. We optimize **streaming, causal, explainable, multi-site, air-gap**.

**“Where is SNMP / gRPC?”**  
Ingestion layer (Telegraf, OTel, your collector) → numeric tensor → VULGARIS. Demo uses the same math as router KPIs.

**“Is it production-certified?”**  
This demo proves **engineering capability**. Pilot + your compliance process = production path.

---

## Longer demo (30 min)

1. `python demo/company_showcase.py` (full, not `--quick`)
2. `python tests/network_telecom_test.py` (full train + confusion matrix)
3. `docker compose up` + hit `/health`, `/predict` (if checkpoint loaded)
4. Walk `OVERVIEW.md` architecture diagram

---

## What to build next for enterprise buyers

1. **One real customer dataset** (even anonymized) — replaces synthetic-only story.
2. **OPC-UA / SNMP → channel mapper** script — shows integration, not slides.
3. **2-page case study** from `showcase_report.md` metrics.
4. **Side-by-side**: VULGARIS vs threshold alerts on their KPIs (false positive rate).

---

*Run `python demo/company_showcase.py` before every external meeting so numbers in the report are fresh.*
